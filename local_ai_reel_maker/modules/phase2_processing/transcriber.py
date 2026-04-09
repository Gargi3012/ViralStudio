"""
=============================================================
  Local AI Reel Maker — Phase 2: Transcription & Audio Intelligence
  Module : transcriber.py
  Author : AI Engineer
  Purpose: Transcribe audio locally with OpenAI Whisper (word-level
           timestamps), analyse per-frame dB levels with librosa,
           then map audio intensity to every transcript segment,
           producing an enriched transcript.json ready for Phase 3.
=============================================================

Dependencies:
    pip install openai-whisper librosa numpy soundfile tqdm

Notes:
    • Whisper requires ffmpeg on the system PATH.
    • First run downloads the chosen model weights (~74 MB for 'base',
      ~461 MB for 'small') and caches them in ~/.cache/whisper.
    • Word-level timestamps require whisper >= 20230314.

Usage — standalone:
    python transcriber.py --audio data/raw_data/<session_id>/audio.mp3

Usage — chained from Phase 1 metadata:
    python transcriber.py \
        --metadata data/raw_data/<session_id>/metadata.json

Usage — programmatic (from main.py or Phase 3):
    from transcriber import AudioIntelligenceEngine
    engine = AudioIntelligenceEngine(model_name="small")
    result = engine.run(audio_path="data/raw_data/.../audio.mp3",
                        session_dir=Path("data/processed_data/..."))
"""

import os
import sys
import json
import logging
import argparse
import warnings
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import Optional

import numpy as np

# Suppress noisy FP16 warning on CPU-only machines
warnings.filterwarnings("ignore", message="FP16 is not supported on CPU")


# ─────────────────────────────────────────────
#  Logging
# ─────────────────────────────────────────────

def setup_logger(name: str, log_path: Optional[Path] = None) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    if log_path:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_path, encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger


# ─────────────────────────────────────────────
#  Data models
# ─────────────────────────────────────────────

@dataclass
class WordToken:
    """A single whisper word with timing."""
    word:  str
    start: float   # seconds
    end:   float   # seconds
    probability: float  # whisper confidence 0.0–1.0


@dataclass
class TranscriptSegment:
    """
    One whisper segment enriched with audio intensity data.
    This is the atom used by Phase 3 (script generation).
    """
    segment_id:   int
    text:         str
    start:        float          # segment start (seconds)
    end:          float          # segment end (seconds)
    duration:     float          # end - start
    avg_db:       float          # mean dB across the segment's frames
    max_db:       float          # peak dB in the segment
    min_db:       float          # lowest dB in the segment
    db_variance:  float          # variance — high = dynamic, low = monotone
    energy_score: float          # normalised 0–1 intensity score (for Phase 3)
    words:        list[dict]     # list of WordToken dicts
    is_energy_spike: bool = False  # set True if energy_score > threshold


@dataclass
class TranscriptResult:
    """Full output written to transcript.json."""
    session_id:      str
    audio_path:      str
    whisper_model:   str
    language:        str
    full_text:       str
    total_duration:  float
    segment_count:   int
    avg_db_overall:  float
    max_db_overall:  float
    energy_spike_count: int
    segments:        list[dict]   # list of TranscriptSegment dicts
    processing_notes: list[str] = field(default_factory=list)


# ─────────────────────────────────────────────
#  Audio Intensity Analyser
# ─────────────────────────────────────────────

class AudioIntensityAnalyser:
    """
    Uses librosa to compute per-frame RMS energy converted to dB,
    then exposes a method to query the intensity statistics for any
    [start, end] time window.
    """

    # Hop between STFT frames in samples — controls time resolution.
    # At 22050 Hz, HOP_LENGTH=512 → ~23 ms per frame (≈43 frames/sec).
    HOP_LENGTH = 512
    SAMPLE_RATE = 22_050   # librosa default resampling target
    REF_DB = 1.0            # reference amplitude for dB conversion

    def __init__(self, audio_path: str | Path, logger: logging.Logger):
        self.audio_path = Path(audio_path)
        self.logger = logger
        self._times: Optional[np.ndarray] = None   # frame centre times (seconds)
        self._db: Optional[np.ndarray] = None       # dB values per frame

    def load_and_analyse(self) -> None:
        """
        Load the audio file, compute RMS energy, convert to dB.
        Must be called before any db_stats_for_window() calls.
        """
        import librosa  # deferred import — not available at module level on all machines

        self.logger.info("Loading audio for intensity analysis: %s", self.audio_path.name)
        y, sr = librosa.load(str(self.audio_path), sr=self.SAMPLE_RATE, mono=True)
        self.logger.debug(
            "Audio loaded — duration: %.2fs | sample rate: %d Hz | samples: %d",
            len(y) / sr, sr, len(y),
        )

        # RMS energy per short-time frame
        rms = librosa.feature.rms(y=y, hop_length=self.HOP_LENGTH)[0]

        # Convert to dB (amplitude, not power — matches human loudness perception)
        # np.maximum avoids log(0); floor at -80 dB (perceptual silence)
        self._db = librosa.amplitude_to_db(
            np.maximum(rms, 1e-10), ref=self.REF_DB
        )

        # Centre time of each RMS frame
        self._times = librosa.frames_to_time(
            np.arange(len(self._db)),
            sr=sr,
            hop_length=self.HOP_LENGTH,
        )

        self.logger.info(
            "Intensity analysis complete — %d frames | dB range: [%.1f, %.1f]",
            len(self._db), float(self._db.min()), float(self._db.max()),
        )

    def db_stats_for_window(self, start: float, end: float) -> dict:
        """
        Return dB statistics for the audio window [start, end] in seconds.

        Returns dict with keys: avg_db, max_db, min_db, db_variance, frame_count.
        Returns zeros if the window is empty or data hasn't been loaded.
        """
        if self._times is None or self._db is None:
            return {"avg_db": 0.0, "max_db": 0.0, "min_db": 0.0,
                    "db_variance": 0.0, "frame_count": 0}

        mask = (self._times >= start) & (self._times < end)
        window = self._db[mask]

        if len(window) == 0:
            # Edge case: very short segment with no frames — interpolate nearest
            idx = int(np.argmin(np.abs(self._times - (start + end) / 2)))
            window = self._db[max(0, idx - 1): idx + 2]

        if len(window) == 0:
            return {"avg_db": 0.0, "max_db": 0.0, "min_db": 0.0,
                    "db_variance": 0.0, "frame_count": 0}

        return {
            "avg_db":      float(np.mean(window)),
            "max_db":      float(np.max(window)),
            "min_db":      float(np.min(window)),
            "db_variance": float(np.var(window)),
            "frame_count": int(len(window)),
        }

    @property
    def global_db_range(self) -> tuple[float, float]:
        """(min_db, max_db) across the entire audio file."""
        if self._db is None:
            return (0.0, 0.0)
        return (float(self._db.min()), float(self._db.max()))


# ─────────────────────────────────────────────
#  Whisper Transcription Wrapper
# ─────────────────────────────────────────────

class WhisperTranscriber:
    """
    Thin wrapper around openai-whisper that enforces word-level timestamps
    and returns a clean list of segment dicts.
    """

    SUPPORTED_MODELS = ("tiny", "base", "small", "medium", "large")

    def __init__(self, model_name: str = "tiny", device: Optional[str] = None,
                 logger: Optional[logging.Logger] = None):
        """
        Args:
            model_name : Whisper model size.  'base' is good for short clips;
                         'small' noticeably improves accuracy with modest extra RAM.
            device     : 'cpu' | 'cuda' | None (auto-detect).
            logger     : Optional logger instance.
        """
        if model_name not in self.SUPPORTED_MODELS:
            raise ValueError(
                f"Unknown model '{model_name}'. "
                f"Choose from: {self.SUPPORTED_MODELS}"
            )
        self.model_name = model_name
        self.device = device
        self.logger = logger or logging.getLogger(__name__)
        self._model = None

    def _load_model(self):
        """Lazy-load Whisper model on first use."""
        if self._model is None:
            import whisper  # deferred import
            self.logger.info(
                "Loading Whisper model '%s' (first run downloads weights)…",
                self.model_name,
            )
            import os
            from pathlib import Path
            models_dir = str(Path("models/whisper").resolve())
            os.makedirs(models_dir, exist_ok=True)
            self._model = whisper.load_model(
                self.model_name,
                device=self.device,
                download_root=models_dir
            )
            self.logger.info("Whisper model loaded on device: %s",
                             next(self._model.parameters()).device)
        return self._model

    def transcribe(self, audio_path: str | Path) -> dict:
        """
        Transcribe *audio_path* and return the raw Whisper result dict,
        which includes .segments with per-word timing data.

        Args:
            audio_path: Path to the audio file (mp3, wav, m4a, etc.)

        Returns:
            Whisper result dict with keys: text, segments, language.

        Raises:
            FileNotFoundError : If the audio file doesn't exist.
            RuntimeError      : If Whisper encounters a processing error.
        """
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        model = self._load_model()
        self.logger.info("Transcribing: %s", audio_path.name)

        try:
            result = model.transcribe(
                str(audio_path),
                word_timestamps=True,    # ← enables per-word timing
                verbose=False,           # suppress whisper's own progress output
                task="transcribe",       # 'transcribe' | 'translate'
                fp16=False,              # use fp32 for CPU compatibility
                # Greedy decoding — ~5x faster than beam search on CPU
                beam_size=1,
                best_of=1,
                temperature=0.0,
                condition_on_previous_text=False,
            )
        except Exception as exc:
            raise RuntimeError(f"Whisper transcription failed: {exc}") from exc

        segment_count = len(result.get("segments", []))
        word_count    = sum(
            len(seg.get("words", [])) for seg in result.get("segments", [])
        )
        self.logger.info(
            "Transcription complete — language: %s | segments: %d | words: %d",
            result.get("language", "?"), segment_count, word_count,
        )
        return result

    @staticmethod
    def extract_word_tokens(whisper_segment: dict) -> list[WordToken]:
        """Parse the words list from a single Whisper segment dict."""
        tokens = []
        for w in whisper_segment.get("words", []):
            tokens.append(
                WordToken(
                    word=w.get("word", "").strip(),
                    start=float(w.get("start", 0.0)),
                    end=float(w.get("end", 0.0)),
                    probability=float(w.get("probability", 1.0)),
                )
            )
        return tokens


# ─────────────────────────────────────────────
#  Energy Spike Detector
# ─────────────────────────────────────────────

def compute_energy_score(avg_db: float, global_min: float, global_max: float) -> float:
    """
    Normalise avg_db to a 0–1 energy score relative to the file's dB range.
    Score of 1.0 = loudest segment in the file; 0.0 = quietest.
    """
    db_range = global_max - global_min
    if db_range == 0:
        return 0.5
    return float(np.clip((avg_db - global_min) / db_range, 0.0, 1.0))


def flag_energy_spikes(
    segments: list[TranscriptSegment],
    threshold: float = 0.70,
) -> int:
    """
    Mark segments with energy_score >= threshold as energy spikes.
    Returns the count of spikes found.
    """
    count = 0
    for seg in segments:
        if seg.energy_score >= threshold:
            seg.is_energy_spike = True
            count += 1
    return count


# ─────────────────────────────────────────────
#  Main Engine
# ─────────────────────────────────────────────

class AudioIntelligenceEngine:
    """
    Phase-2 orchestrator for the Local AI Reel Maker.

    Pipeline:
        1. Load + analyse audio intensity with librosa   (AudioIntensityAnalyser)
        2. Transcribe audio locally with Whisper         (WhisperTranscriber)
        3. Map dB stats to every Whisper segment
        4. Compute normalised energy scores
        5. Flag energy spikes
        6. Save enriched transcript.json
    """

    ENERGY_SPIKE_THRESHOLD = 0.70   # top 30% loudest segments get flagged

    def __init__(
        self,
        model_name: str = "tiny",
        device: Optional[str] = None,
        spike_threshold: float = 0.70,
    ):
        self.model_name = model_name
        self.spike_threshold = spike_threshold
        self._whisper = WhisperTranscriber(model_name=model_name, device=device)
        # Logger is session-scoped; set up in run()
        self.logger: Optional[logging.Logger] = None

    # ──────────────────────────────────────────
    #  Step 3 — Merge transcription + intensity
    # ──────────────────────────────────────────

    def _build_enriched_segments(
        self,
        whisper_result: dict,
        analyser: AudioIntensityAnalyser,
    ) -> list[TranscriptSegment]:
        """
        For every Whisper segment, query the AudioIntensityAnalyser and
        assemble a fully-enriched TranscriptSegment.
        """
        global_min, global_max = analyser.global_db_range
        segments: list[TranscriptSegment] = []

        for idx, raw_seg in enumerate(whisper_result.get("segments", [])):
            start = float(raw_seg.get("start", 0.0))
            end   = float(raw_seg.get("end",   0.0))
            text  = raw_seg.get("text", "").strip()

            db_stats = analyser.db_stats_for_window(start, end)
            energy   = compute_energy_score(db_stats["avg_db"], global_min, global_max)
            words    = WhisperTranscriber.extract_word_tokens(raw_seg)

            seg = TranscriptSegment(
                segment_id   = idx,
                text         = text,
                start        = round(start, 3),
                end          = round(end,   3),
                duration     = round(end - start, 3),
                avg_db       = round(db_stats["avg_db"],      2),
                max_db       = round(db_stats["max_db"],      2),
                min_db       = round(db_stats["min_db"],      2),
                db_variance  = round(db_stats["db_variance"], 4),
                energy_score = round(energy, 4),
                words        = [asdict(w) for w in words],
                is_energy_spike = False,
            )
            segments.append(seg)

        return segments

    # ──────────────────────────────────────────
    #  Save
    # ──────────────────────────────────────────

    @staticmethod
    def save_transcript(result: TranscriptResult, output_dir: Path) -> Path:
        """Write the TranscriptResult to transcript.json."""
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / "transcript.json"
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(asdict(result), fh, indent=2, ensure_ascii=False)
        return out_path

    # ──────────────────────────────────────────
    #  Public API
    # ──────────────────────────────────────────

    def run(
        self,
        audio_path: str | Path,
        session_dir: Optional[Path] = None,
        session_id: Optional[str] = None,
    ) -> TranscriptResult:
        """
        Execute the full Phase-2 pipeline.

        Args:
            audio_path  : Path to the audio file from Phase 1.
            session_dir : Directory to write transcript.json (and logs).
                          Defaults to data/processed_data/<stem_of_audio>/
            session_id  : Human-readable ID embedded in metadata.
                          Defaults to the audio file's parent directory name.

        Returns:
            A TranscriptResult with all enriched segments.
        """
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        # ── Session bookkeeping ────────────────────────────────────────
        if session_id is None:
            session_id = audio_path.parent.name   # e.g. "session_20240409_..."
        if session_dir is None:
            session_dir = Path("data/processed_data") / session_id

        log_path = Path("logs/session_logs") / f"{session_id}_phase2.log"
        self.logger = setup_logger(f"transcriber.{session_id}", log_path)
        self._whisper.logger = self.logger

        self.logger.info("Phase 2 start — session: %s", session_id)
        self.logger.info("Audio file   : %s (%.1f KB)",
                         audio_path, audio_path.stat().st_size / 1024)
        self.logger.info("Whisper model: %s", self.model_name)

        notes: list[str] = []

        # ── Step 1: Audio intensity ────────────────────────────────────
        analyser = AudioIntensityAnalyser(audio_path, self.logger)
        analyser.load_and_analyse()
        global_min, global_max = analyser.global_db_range

        # ── Step 2: Whisper transcription ──────────────────────────────
        whisper_result = self._whisper.transcribe(audio_path)
        language = whisper_result.get("language", "unknown")
        full_text = whisper_result.get("text", "").strip()

        if not full_text:
            notes.append("WARNING: Whisper returned empty transcription — "
                         "check audio quality and file format.")
            self.logger.warning("Empty transcription returned by Whisper.")

        # ── Step 3: Merge transcription + intensity ────────────────────
        self.logger.info("Mapping audio intensity to transcript segments…")
        segments = self._build_enriched_segments(whisper_result, analyser)

        # ── Step 4: Detect energy spikes ──────────────────────────────
        spike_count = flag_energy_spikes(segments, self.spike_threshold)
        self.logger.info(
            "Energy spikes (score ≥ %.0f%%): %d / %d segments",
            self.spike_threshold * 100, spike_count, len(segments),
        )
        if spike_count == 0:
            notes.append(
                "No energy spikes detected. Consider lowering spike_threshold "
                f"(current: {self.spike_threshold:.0%})."
            )

        # ── Step 5: Assemble result ────────────────────────────────────
        avg_db_all = float(np.mean([s.avg_db for s in segments])) if segments else 0.0

        result = TranscriptResult(
            session_id       = session_id,
            audio_path       = str(audio_path),
            whisper_model    = self.model_name,
            language         = language,
            full_text        = full_text,
            total_duration   = round(segments[-1].end if segments else 0.0, 3),
            segment_count    = len(segments),
            avg_db_overall   = round(avg_db_all, 2),
            max_db_overall   = round(global_max, 2),
            energy_spike_count = spike_count,
            segments         = [asdict(s) for s in segments],
            processing_notes = notes,
        )

        # ── Step 6: Save ───────────────────────────────────────────────
        out_path = self.save_transcript(result, session_dir)
        self.logger.info("transcript.json saved → %s", out_path)

        # ── Summary ────────────────────────────────────────────────────
        self._print_summary(result, out_path)
        return result

    def _print_summary(self, result: TranscriptResult, out_path: Path) -> None:
        bar = "=" * 58
        # Use a safe print that handles Windows encoding issues gracefully
        def p(s=""):
            try:
                # We attempt to print to a UTF-8 stream if possible, or fallback to standard print
                # which uses 'errors=replace' in many modern IDEs/Terminals.
                print(s)
            except UnicodeEncodeError:
                # Fallback for environments with strict non-UTF-8 encoding (e.g. legacy Windows CMD)
                print(s.encode('ascii', 'replace').decode('ascii'))

        p(f"\n{bar}")
        p("  PHASE 2 - TRANSCRIPTION & AUDIO INTELLIGENCE RESULT")
        p(bar)
        p(f"  Session      : {result.session_id}")
        p(f"  Language     : {result.language}")
        p(f"  Model        : {result.whisper_model}")
        p(f"  Duration     : {result.total_duration}s")
        p(f"  Segments     : {result.segment_count}")
        p(f"  Energy spikes: {result.energy_spike_count}")
        p(f"  Avg dB       : {result.avg_db_overall:.1f} dB")
        p(f"  Peak dB      : {result.max_db_overall:.1f} dB")
        p(f"\n  Full text preview:")
        preview = result.full_text[:200]
        if len(result.full_text) > 200:
            preview += "..."
        p(f"    \"{preview}\"")
        if result.processing_notes:
            p(f"\n  Notes:")
            for note in result.processing_notes:
                p(f"    [!] {note}")
        p(f"\n  Output -> {out_path}")
        p(bar + "\n")


# ─────────────────────────────────────────────
#  Utility: Load Phase 1 metadata
# ─────────────────────────────────────────────

def resolve_audio_from_metadata(metadata_path: Path) -> Path:
    """
    Read a Phase-1 metadata.json and return the audio_path it recorded.
    Raises ValueError if the metadata doesn't contain a valid audio_path.
    """
    with open(metadata_path, "r", encoding="utf-8") as fh:
        meta = json.load(fh)

    audio_path_str = meta.get("audio_path")
    if not audio_path_str:
        raise ValueError(
            f"metadata.json at '{metadata_path}' has no 'audio_path' key. "
            "Was Phase 1 completed successfully?"
        )

    audio_path = Path(audio_path_str)
    if not audio_path.exists():
        raise FileNotFoundError(
            f"audio_path '{audio_path}' from metadata does not exist. "
            "Check that Phase 1 output hasn't been moved."
        )
    return audio_path


# ─────────────────────────────────────────────
#  CLI Entry Point
# ─────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Local AI Reel Maker — Phase 2: Transcription & Audio Intelligence",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--audio", "-a",
        type=str,
        help="Direct path to audio file (mp3, wav, m4a, …).",
    )
    source.add_argument(
        "--metadata", "-m",
        type=str,
        help="Path to Phase-1 metadata.json (audio_path is read from it).",
    )

    parser.add_argument(
        "--model",
        type=str,
        default="base",
        choices=["tiny", "base", "small", "medium", "large"],
        help="Whisper model size.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cpu", "cuda"],
        help="Compute device for Whisper (default: auto-detect).",
    )
    parser.add_argument(
        "--spike_threshold",
        type=float,
        default=0.70,
        help="Energy score threshold (0–1) above which segments are flagged as spikes.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to write transcript.json. Defaults to data/processed_data/<session>/",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Resolve audio path
    if args.audio:
        audio_path = Path(args.audio)
    else:
        audio_path = resolve_audio_from_metadata(Path(args.metadata))

    session_dir = Path(args.output_dir) if args.output_dir else None

    engine = AudioIntelligenceEngine(
        model_name      = args.model,
        device          = args.device,
        spike_threshold = args.spike_threshold,
    )

    engine.run(audio_path=audio_path, session_dir=session_dir)


if __name__ == "__main__":
    main()