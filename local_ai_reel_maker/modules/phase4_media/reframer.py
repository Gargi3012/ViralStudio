"""
=============================================================
  Local AI Reel Maker — Phase 4: Computer Vision Layer
  Module : reframer.py
  Author : AI Engineer
  Status : NEW FILE
  Purpose: Take a source 16:9 video and a list of viral clip
           timestamps (from Phase 3 viral_clips.json), detect
           the speaker's face in every frame using MediaPipe,
           compute a smooth 9:16 crop window that follows the
           face, and write 1080×1920 output clips via OpenCV.

  Pipeline (per clip):
    1.  Extract the raw segment from the source video
    2.  Per-frame face detection  (MediaPipe BlazeFace)
    3.  Derive ideal crop-center  (face bbox center-x,
        upper-third y — "talking head" convention)
    4.  Smooth crop trajectory    (exponential moving average
        + optional Gaussian pass for extra silkiness)
    5.  Apply crop & resize to 1080 × 1920
    6.  Write output clip         (H.264 via OpenCV VideoWriter)

  Output layout:
    outputs/reels/
      {session_id}_rank1_{start}s_{end}s.mp4
      {session_id}_rank2_{start}s_{end}s.mp4
      …

Dependencies:
    pip install mediapipe opencv-python numpy tqdm

    MediaPipe ≥ 0.10 requires Python 3.8-3.11 and a
    reasonably modern numpy (≥ 1.23).

Usage — standalone:
    python reframer.py \
        --video   path/to/source.mp4 \
        --clips   data/processed_data/<session>/viral_clips.json

Usage — programmatic:
    from reframer import ReframerEngine
    engine = ReframerEngine(source_video="source.mp4")
    engine.run("data/processed_data/<session>/viral_clips.json")
=============================================================
"""

from __future__ import annotations

import json
import logging
import argparse
import warnings
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from tqdm import tqdm
import os
try:
    import ffmpeg
except ImportError:
    pass

warnings.filterwarnings("ignore", category=UserWarning)

# ─────────────────────────────────────────────
#  Constants
# ─────────────────────────────────────────────

OUTPUT_W   = 1080
OUTPUT_H   = 1920
ASPECT     = OUTPUT_W / OUTPUT_H          # 9/16  ≈ 0.5625

# Face detection confidence
MIN_DETECTION_CONFIDENCE = 0.55

# Exponential moving average α  (0 = frozen, 1 = raw/jittery)
# 0.08 gives a very smooth, "cinematic" follow feel
EMA_ALPHA  = 0.08

# Gaussian smoothing window (frames) applied AFTER EMA for a 2nd pass
GAUSS_WINDOW = 15

# Fallback: if face not detected for this many consecutive frames,
# drift back to frame-center at 10 % per frame
FALLBACK_DRIFT = 0.10

# Output codec — H.264 via mp4v (broad compatibility)
FOURCC     = cv2.VideoWriter_fourcc(*"mp4v")


# ─────────────────────────────────────────────
#  Logging
# ─────────────────────────────────────────────

def setup_logger(name: str, log_path: Optional[Path] = None) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger          # already configured (re-entrant safe)
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
#  Data Models
# ─────────────────────────────────────────────

@dataclass
class CropWindow:
    """
    Defines a crop rectangle in source-video pixel coordinates.
    cx, cy = centre of the crop window (floats for sub-pixel precision)
    w, h   = fixed dimensions of the crop (set once per video source).
    """
    cx: float
    cy: float
    w:  int
    h:  int

    # ── derived pixel rect ──────────────────
    def to_rect(self, frame_w: int, frame_h: int) -> tuple[int, int, int, int]:
        """
        Return (x1, y1, x2, y2) clamped to frame bounds.
        All values are integers, ready for OpenCV slicing.
        """
        x1 = int(round(self.cx - self.w / 2))
        y1 = int(round(self.cy - self.h / 2))
        x2 = x1 + self.w
        y2 = y1 + self.h

        # Clamp to frame
        x1 = max(0, min(x1, frame_w - self.w))
        y1 = max(0, min(y1, frame_h - self.h))
        x2 = x1 + self.w
        y2 = y1 + self.h
        return x1, y1, x2, y2


@dataclass
class ClipSpec:
    """One viral clip to be processed."""
    rank:    int
    start:   float     # seconds in source video
    end:     float
    text:    str
    score:   float


@dataclass
class ReframeResult:
    """Summary of one processed clip."""
    rank:        int
    start:       float
    end:         float
    output_path: str
    fps:         float
    total_frames: int
    face_detected_frames: int
    face_detection_rate:  float   # 0–1
    status:      str              # "success" | "failed" | "skipped"
    error:       Optional[str] = None


# ─────────────────────────────────────────────
#  1. MediaPipe Face Detector wrapper
# ─────────────────────────────────────────────

class FaceDetector:
    """
    Thin wrapper around MediaPipe's BlazeFace short-range detector.

    Returns the *primary* face bounding box as (cx, cy, w, h)
    normalised to [0, 1] relative to frame dimensions, or None
    if no face is found.
    """

class FaceDetector:
    """
    Abstractions for Face Detection using MediaPipe (best)
    with a fallback to OpenCV Haar Cascades (legacy).
    """

    def __init__(
        self,
        min_confidence: float = MIN_DETECTION_CONFIDENCE,
        logger: Optional[logging.Logger] = None,
    ):
        self.min_confidence = min_confidence
        self.logger = logger or logging.getLogger(__name__)
        self._detector = None
        self._mode = "NONE"

    def get_detector(self):
        if self._detector is None:
            self._detector = self._load()
        return self._detector

    def _load(self):
        """
        Lazy-load the MediaPipe FaceDetection solution or fallback to OpenCV.
        """
        try:
            import mediapipe as mp
            # Check for legacy solutions (might be missing in newer Python builds)
            if hasattr(mp, "solutions") and hasattr(mp.solutions, "face_detection"):
                 self.logger.debug("Initializing MediaPipe FaceDetection...")
                 self._mp_face = mp.solutions.face_detection
                 detector = self._mp_face.FaceDetection(
                     model_selection=1,  
                     min_detection_confidence=self.min_confidence
                 )
                 self._mode = "MEDIAPIPE"
                 return detector
            else:
                 raise AttributeError("mediapipe.solutions.face_detection not found")
        except (ImportError, AttributeError) as e:
            self.logger.warning("MediaPipe FaceDetection unavailable (%s). Falling back to OpenCV...", e)
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            detector = cv2.CascadeClassifier(cascade_path)
            if detector.empty():
                 self.logger.error("OpenCV Haar Cascade failed to load from: %s", cascade_path)
                 self._mode = "NONE"
            else:
                 self._mode = "OPENCV"
            return detector

    def detect(self, frame_rgb: np.ndarray) -> Optional[tuple[float, float, float, float]]:
        """
        Detect the most prominent face in *frame_rgb*.
        Returns (cx, cy, w, h) in normalized [0, 1] coordinates.
        """
        detector = self.get_detector()
        if detector is None or self._mode == "NONE":
             return None

        # ── MediaPipe Path ───────────────────────────────────────
        if self._mode == "MEDIAPIPE":
            result = detector.process(frame_rgb)
            if not result.detections:
                return None
            
            # Pick detection with highest confidence
            best = max(result.detections, key=lambda d: d.score[0])
            bbox = best.location_data.relative_bounding_box
            cx = bbox.xmin + bbox.width  / 2.0
            cy = bbox.ymin + bbox.height / 2.0
            return (
                float(np.clip(cx,        0.0, 1.0)),
                float(np.clip(cy,        0.0, 1.0)),
                float(np.clip(bbox.width,  0.0, 1.0)),
                float(np.clip(bbox.height, 0.0, 1.0)),
            )
            
        # ── OpenCV Fallback Path ─────────────────────────────────
        elif self._mode == "OPENCV":
            gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
            faces = detector.detectMultiScale(gray, 1.1, 4)
            if len(faces) == 0:
                return None
            
            # Pick largest face
            (x, y, w, h) = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)[0]
            img_h, img_w = frame_rgb.shape[:2]
            cx_n = (x + w/2) / img_w
            cy_n = (y + h/2) / img_h
            w_n  = w / img_w
            h_n  = h / img_h
            return (
                float(np.clip(cx_n, 0.0, 1.0)),
                float(np.clip(cy_n, 0.0, 1.0)),
                float(np.clip(w_n,  0.0, 1.0)),
                float(np.clip(h_n,  0.0, 1.0)),
            )
        
        return None

    def close(self):
        if self._detector and self._mode == "MEDIAPIPE":
            try: self._detector.close()
            except: pass
        self._detector = None


# ─────────────────────────────────────────────
#  2. Crop Window Calculator
# ─────────────────────────────────────────────

def compute_crop_size(frame_w: int, frame_h: int) -> tuple[int, int]:
    """
    Calculate the largest 9:16 crop rectangle that fits inside the frame.

    For a typical 1920×1080 source:
        crop_h = 1080,  crop_w = int(1080 × 9/16) = 607
    For a 1280×720 source:
        crop_h = 720,   crop_w = int(720  × 9/16) = 405
    """
    crop_h = frame_h
    crop_w = int(frame_h * ASPECT)
    if crop_w > frame_w:
        crop_w = frame_w
        crop_h = int(frame_w / ASPECT)
    return crop_w, crop_h


def face_to_crop_center(
    face_cx_norm: float,
    face_cy_norm: float,
    frame_w: int,
    frame_h: int,
    crop_w: int,
    crop_h: int,
) -> tuple[float, float]:
    """
    Convert a normalised face centre to the desired crop-window centre.

    Strategy: horizontally track the face; vertically position the face
    in the upper-third of the frame (classic "talking head" composition).
    The face centre sits at ~35 % down from the top of the crop window.
    """
    face_px_x = face_cx_norm * frame_w
    face_px_y = face_cy_norm * frame_h

    # Horizontal: centre crop on face x-position
    target_cx = face_px_x

    # Vertical: face should appear at ~35 % down inside the crop window
    target_cy = face_px_y - crop_h * (0.35 - 0.5)   # shift window up

    # Clamp so the window stays inside the frame
    half_w = crop_w / 2
    half_h = crop_h / 2
    target_cx = float(np.clip(target_cx, half_w, frame_w - half_w))
    target_cy = float(np.clip(target_cy, half_h, frame_h - half_h))

    return target_cx, target_cy


# ─────────────────────────────────────────────
#  3. Trajectory Smoother
# ─────────────────────────────────────────────

class TrajectorySmoother:
    """
    Two-stage trajectory smoother for the crop-window centre.

    Stage 1 — Exponential Moving Average (online, causal):
        Smooths the raw per-frame target centre in real-time.
        α controls responsiveness vs. smoothness.

    Stage 2 — Gaussian convolution (offline, non-causal):
        Applied to the full trajectory array after all frames
        are processed, for a second silky-smooth pass.
    """

    def __init__(self, alpha: float = EMA_ALPHA, gauss_window: int = GAUSS_WINDOW):
        self.alpha        = alpha
        self.gauss_window = gauss_window
        self._ema_cx: Optional[float] = None
        self._ema_cy: Optional[float] = None

    def reset(self):
        self._ema_cx = None
        self._ema_cy = None

    def ema_step(self, cx: float, cy: float) -> tuple[float, float]:
        """Apply one EMA step and return the smoothed position."""
        if self._ema_cx is None:
            self._ema_cx = cx
            self._ema_cy = cy
        else:
            self._ema_cx = self.alpha * cx + (1 - self.alpha) * self._ema_cx
            self._ema_cy = self.alpha * cy + (1 - self.alpha) * self._ema_cy
        return self._ema_cx, self._ema_cy

    @staticmethod
    def gaussian_smooth(
        trajectory: list[tuple[float, float]],
        window: int = GAUSS_WINDOW,
    ) -> list[tuple[float, float]]:
        """
        Apply a 1-D Gaussian filter to (cx, cy) trajectory arrays.

        Args:
            trajectory : list of (cx, cy) floats, one per frame.
            window     : Gaussian kernel half-width in frames.

        Returns:
            Smoothed trajectory of the same length.
        """
        if len(trajectory) < 3:
            return trajectory

        arr_cx = np.array([p[0] for p in trajectory], dtype=np.float64)
        arr_cy = np.array([p[1] for p in trajectory], dtype=np.float64)

        # Build Gaussian kernel
        ksize  = window * 2 + 1
        x      = np.arange(ksize) - window
        sigma  = window / 3.0
        kernel = np.exp(-0.5 * (x / sigma) ** 2)
        kernel /= kernel.sum()

        # Reflect-pad the signal at both ends before convolution
        pad_cx = np.pad(arr_cx, window, mode="reflect")
        pad_cy = np.pad(arr_cy, window, mode="reflect")

        smooth_cx = np.convolve(pad_cx, kernel, mode="valid")
        smooth_cy = np.convolve(pad_cy, kernel, mode="valid")

        return list(zip(smooth_cx.tolist(), smooth_cy.tolist()))


# ─────────────────────────────────────────────
#  4. Per-frame crop pipeline (Pass 1 + Pass 2)
# ─────────────────────────────────────────────

def build_smooth_trajectory(
    cap:         cv2.VideoCapture,
    start_frame: int,
    end_frame:   int,
    frame_w:     int,
    frame_h:     int,
    crop_w:      int,
    crop_h:      int,
    detector:    FaceDetector,
    smoother:    TrajectorySmoother,
    logger:      logging.Logger,
) -> tuple[list[tuple[float, float]], int]:
    """
    Pass 1 — scrub through the clip frame-by-frame, detect faces, run EMA.
    Pass 2 — apply Gaussian smoothing to the collected trajectory.

    Returns:
        (smooth_trajectory, face_detected_count)
    """
    default_cx = frame_w / 2.0
    default_cy = frame_h / 2.0

    smoother.reset()
    raw_traj: list[tuple[float, float]] = []
    face_count     = 0
    no_face_streak = 0

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    total = end_frame - start_frame
    last_target = (default_cx, default_cy)

    for i in tqdm(range(total), desc="  Pass 1 face-detect", leave=False, unit="fr"):
        ret, frame = cap.read()
        if not ret:
            # Video ended early — pad with last known position
            last = raw_traj[-1] if raw_traj else (default_cx, default_cy)
            raw_traj.append(last)
            continue

        if i % 3 == 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            detection = detector.detect(frame_rgb)

            if detection is not None:
                fx, fy, _, _ = detection
                target_cx, target_cy = face_to_crop_center(
                    fx, fy, frame_w, frame_h, crop_w, crop_h
                )
                face_count     += 1
                no_face_streak  = 0
                last_target = (target_cx, target_cy)
            else:
                # No face detected — drift toward frame centre
                no_face_streak += 1
                last_cx, last_cy = raw_traj[-1] if raw_traj else (default_cx, default_cy)
                target_cx = last_cx + FALLBACK_DRIFT * (default_cx - last_cx)
                target_cy = last_cy + FALLBACK_DRIFT * (default_cy - last_cy)
                last_target = (target_cx, target_cy)
        else:
            # Reuse target to speed up processing
            target_cx, target_cy = last_target
            if no_face_streak == 0:
                face_count += 1

        smooth_cx, smooth_cy = smoother.ema_step(target_cx, target_cy)
        raw_traj.append((smooth_cx, smooth_cy))

    # Pass 2 — Gaussian smoothing over the whole trajectory
    logger.debug(
        "  Pass 2 Gaussian smooth (window=%d)…", smoother.gauss_window
    )
    final_traj = TrajectorySmoother.gaussian_smooth(raw_traj, smoother.gauss_window)

    return final_traj, face_count


# ─────────────────────────────────────────────
#  5. Clip Writer
# ─────────────────────────────────────────────

def write_clip(
    cap:         cv2.VideoCapture,
    start_frame: int,
    end_frame:   int,
    trajectory:  list[tuple[float, float]],
    crop_w:      int,
    crop_h:      int,
    frame_w:     int,
    frame_h:     int,
    fps:         float,
    output_path: Path,
    source_video: Path,
    start_time:  float,
    end_time:    float,
    logger:      logging.Logger,
) -> bool:
    """
    Pass 3 — seek back to start, apply the pre-computed crop at each frame,
    resize to OUTPUT_W × OUTPUT_H, and write to *output_path*.
    Pass 4 — merge original audio with ffmpeg.

    Returns True on success.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_name(f"temp_{output_path.name}")
    
    writer = cv2.VideoWriter(
        str(temp_path),
        FOURCC,
        fps,
        (OUTPUT_W, OUTPUT_H),
    )

    if not writer.isOpened():
        logger.error("VideoWriter failed to open: %s", output_path)
        return False

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    total = end_frame - start_frame

    for i in tqdm(range(total), desc="  Pass 3 write  ", leave=False, unit="fr"):
        ret, frame = cap.read()
        if not ret:
            break

        cx, cy = trajectory[min(i, len(trajectory) - 1)]
        cw = CropWindow(cx=cx, cy=cy, w=crop_w, h=crop_h)
        x1, y1, x2, y2 = cw.to_rect(frame_w, frame_h)

        cropped = frame[y1:y2, x1:x2]

        # Safety — handle edge cases where crop is unexpectedly small
        if cropped.shape[0] == 0 or cropped.shape[1] == 0:
            cropped = frame[
                max(0, frame_h // 2 - crop_h // 2): frame_h // 2 + crop_h // 2,
                max(0, frame_w // 2 - crop_w // 2): frame_w // 2 + crop_w // 2,
            ]

        resized = cv2.resize(cropped, (OUTPUT_W, OUTPUT_H), interpolation=cv2.INTER_LANCZOS4)
        writer.write(resized)

    writer.release()
    
    # Pass 4: FFmpeg muxing
    try:
        import ffmpeg
        # Sanity check: ensure we have the correct ffmpeg-python module
        if not hasattr(ffmpeg, 'input') or not hasattr(ffmpeg, 'output'):
             raise ImportError("Imported 'ffmpeg' module is missing 'input'/'output'. Is 'ffmpeg-python' correctly installed?")
             
        logger.debug("Muxing audio using ffmpeg-python...")
        video_input = ffmpeg.input(str(temp_path))
        # We need the source audio sliced by start_time, end_time
        audio_input = ffmpeg.input(str(source_video), ss=start_time, t=end_time - start_time)
        
        # Merge them
        out = ffmpeg.output(
            video_input.video,
            audio_input.audio,
            str(output_path),
            vcodec='libx264',   # high quality encoding
            acodec='aac',
            strict='experimental'
        )
        ffmpeg.run(out, overwrite_output=True, quiet=True)
        if temp_path.exists():
            try: os.remove(temp_path)
            except: pass
        logger.debug("Clip written with audio → %s", output_path)
    except Exception as e:
        logger.error("FFmpeg muxing failed: %s", e)
        # fallback onto the mute video
        if temp_path.exists():
             try:
                 if output_path.exists(): os.remove(output_path)
                 temp_path.replace(output_path)
                 logger.warning("Falling back to mute video for %s", output_path.name)
             except Exception as rename_err:
                 logger.error("Failed to rename temp video after muxing failure: %s", rename_err)
        return False

    return True


# ─────────────────────────────────────────────
#  6. Main Engine
# ─────────────────────────────────────────────

class ReframerEngine:
    """
    Phase-4 orchestrator for the Local AI Reel Maker.

    For each viral clip:
      1. Open the source video & seek to the clip window.
      2. Run Pass 1: per-frame face detection + EMA smoothing.
      3. Run Pass 2: Gaussian smoothing over the trajectory.
      4. Run Pass 3: apply crop & write 1080 × 1920 output.
      5. Collect stats and persist reframe_report.json.
    """

    def __init__(
        self,
        source_video:           str | Path,
        ema_alpha:              float = EMA_ALPHA,
        gauss_window:           int   = GAUSS_WINDOW,
        min_face_confidence:    float = MIN_DETECTION_CONFIDENCE,
        output_dir:             str | Path = "outputs/reels",
    ):
        self.source_video        = Path(source_video)
        self.ema_alpha           = ema_alpha
        self.gauss_window        = gauss_window
        self.min_face_confidence = min_face_confidence
        self.output_dir          = Path(output_dir)
        self.logger: Optional[logging.Logger] = None

    # ── helpers ──────────────────────────────

    @staticmethod
    def _load_clips(viral_clips_path: Path) -> tuple[str, list[ClipSpec]]:
        """Load viral_clips.json and return (session_id, list[ClipSpec])."""
        with open(viral_clips_path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        session_id = data.get("session_id", "unknown_session")
        specs = []
        for item in data.get("top_clips", []):
            specs.append(
                ClipSpec(
                    rank  = item.get("rank",  0),
                    start = float(item.get("start", 0.0)),
                    end   = float(item.get("end",   0.0)),
                    text  = item.get("text",  ""),
                    score = float(item.get("viral_score", 0.0)),
                )
            )
        return session_id, specs

    @staticmethod
    def _open_video(path: Path) -> cv2.VideoCapture:
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise IOError(f"Cannot open video: {path}")
        return cap

    @staticmethod
    def _video_meta(cap: cv2.VideoCapture) -> tuple[int, int, float, int]:
        """Return (frame_w, frame_h, fps, total_frames)."""
        w     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps   = cap.get(cv2.CAP_PROP_FPS)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        return w, h, fps, total

    @staticmethod
    def _save_report(results: list[ReframeResult], output_dir: Path, session_id: str):
        report_path = output_dir / f"{session_id}_reframe_report.json"
        report = {
            "session_id": session_id,
            "clips": [
                {
                    "rank":                 r.rank,
                    "start":                r.start,
                    "end":                  r.end,
                    "output_path":          r.output_path,
                    "fps":                  r.fps,
                    "total_frames":         r.total_frames,
                    "face_detected_frames": r.face_detected_frames,
                    "face_detection_rate":  round(r.face_detection_rate, 3),
                    "status":               r.status,
                    "error":                r.error,
                }
                for r in results
            ],
        }
        with open(report_path, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2)
        return report_path

    # ── core per-clip processor ───────────────

    def _process_clip(
        self,
        cap:        cv2.VideoCapture,
        clip:       ClipSpec,
        frame_w:    int,
        frame_h:    int,
        crop_w:     int,
        crop_h:     int,
        fps:        float,
        total_src:  int,
        detector:   FaceDetector,
        session_id: str,
    ) -> ReframeResult:

        start_frame = int(clip.start * fps)
        end_frame   = min(int(clip.end   * fps), total_src - 1)
        n_frames    = end_frame - start_frame

        if n_frames <= 0:
            return ReframeResult(
                rank=clip.rank, start=clip.start, end=clip.end,
                output_path="", fps=fps, total_frames=0,
                face_detected_frames=0, face_detection_rate=0.0,
                status="skipped",
                error="Computed frame range is empty — check timestamps vs source video.",
            )

        slug = f"{session_id}_rank{clip.rank}_{clip.start:.1f}s_{clip.end:.1f}s"
        out_path = self.output_dir / f"{slug}.mp4"

        self.logger.info(
            "Processing Rank #%d  [%.2fs → %.2fs | %d frames]",
            clip.rank, clip.start, clip.end, n_frames,
        )

        smoother = TrajectorySmoother(
            alpha=self.ema_alpha,
            gauss_window=self.gauss_window,
        )

        # ── Passes 1 & 2: face detect + smooth trajectory ──────────────
        trajectory, face_count = build_smooth_trajectory(
            cap         = cap,
            start_frame = start_frame,
            end_frame   = end_frame,
            frame_w     = frame_w,
            frame_h     = frame_h,
            crop_w      = crop_w,
            crop_h      = crop_h,
            detector    = detector,
            smoother    = smoother,
            logger      = self.logger,
        )

        detection_rate = face_count / n_frames if n_frames > 0 else 0.0
        self.logger.info(
            "  Face detection rate: %.1f%%  (%d / %d frames)",
            detection_rate * 100, face_count, n_frames,
        )
        if detection_rate < 0.30:
            self.logger.warning(
                "  Low face detection rate (%.0f%%). "
                "Clip will use centre-fallback for most frames.",
                detection_rate * 100,
            )

        # ── Pass 3: apply crop and write ───────────────────────────────
        success = write_clip(
            cap         = cap,
            start_frame = start_frame,
            end_frame   = end_frame,
            trajectory  = trajectory,
            crop_w      = crop_w,
            crop_h      = crop_h,
            frame_w     = frame_w,
            frame_h     = frame_h,
            fps         = fps,
            output_path = out_path,
            source_video= self.source_video,
            start_time  = clip.start,
            end_time    = clip.end,
            logger      = self.logger,
        )

        status = "success" if success else "failed"
        error  = None if success else "VideoWriter.write() loop failed — check codec."

        file_kb = out_path.stat().st_size / 1024 if out_path.exists() else 0
        self.logger.info(
            "  Output: %s  (%.1f KB)  status=%s",
            out_path.name, file_kb, status,
        )

        return ReframeResult(
            rank                 = clip.rank,
            start                = clip.start,
            end                  = clip.end,
            output_path          = str(out_path),
            fps                  = fps,
            total_frames         = n_frames,
            face_detected_frames = face_count,
            face_detection_rate  = detection_rate,
            status               = status,
            error                = error,
        )

    # ── public run() ─────────────────────────

    def run(
        self,
        viral_clips_path: str | Path,
        ranks: Optional[list[int]] = None,
    ) -> list[ReframeResult]:
        """
        Execute Phase-4 for all (or selected) clips.

        Args:
            viral_clips_path : Path to Phase-3 viral_clips.json.
            ranks            : Optional list of ranks to process
                               (e.g. [1, 2] skips ranks 3–5).
                               Default: process all.

        Returns:
            list[ReframeResult] — one per processed clip.
        """
        viral_clips_path = Path(viral_clips_path)
        if not viral_clips_path.exists():
            raise FileNotFoundError(
                f"viral_clips.json not found: {viral_clips_path}"
            )
        if not self.source_video.exists():
            raise FileNotFoundError(
                f"Source video not found: {self.source_video}"
            )

        session_id = viral_clips_path.parent.name
        log_path   = Path("logs/session_logs") / f"{session_id}_phase4.log"
        self.logger = setup_logger(f"reframer.{session_id}", log_path)

        self.logger.info("Phase 4 start — session: %s", session_id)
        self.logger.info("Source video  : %s", self.source_video)
        self.logger.info("Output dir    : %s", self.output_dir)

        session_id_meta, clip_specs = self._load_clips(viral_clips_path)
        self.logger.info("Loaded %d viral clips.", len(clip_specs))

        if ranks:
            clip_specs = [c for c in clip_specs if c.rank in ranks]
            self.logger.info(
                "Filtered to ranks %s → %d clips.", ranks, len(clip_specs)
            )

        if not clip_specs:
            self.logger.warning("No clips to process after filtering.")
            return []

        # ── Open source video ─────────────────────────────────────────
        cap = self._open_video(self.source_video)
        frame_w, frame_h, fps, total_src = self._video_meta(cap)
        self.logger.info(
            "Source: %d×%d @ %.2f fps | %d frames (%.1f s)",
            frame_w, frame_h, fps, total_src, total_src / fps,
        )

        # ── Compute 9:16 crop dimensions ─────────────────────────────
        crop_w, crop_h = compute_crop_size(frame_w, frame_h)
        self.logger.info(
            "Crop window: %d×%d (9:16 → resize to %d×%d)",
            crop_w, crop_h, OUTPUT_W, OUTPUT_H,
        )

        # ── Shared face detector (one instance, many clips) ───────────
        detector = FaceDetector(
            min_confidence=self.min_face_confidence,
            logger=self.logger,
        )

        self.output_dir.mkdir(parents=True, exist_ok=True)
        results: list[ReframeResult] = []

        for clip in clip_specs:
            try:
                res = self._process_clip(
                    cap        = cap,
                    clip       = clip,
                    frame_w    = frame_w,
                    frame_h    = frame_h,
                    crop_w     = crop_w,
                    crop_h     = crop_h,
                    fps        = fps,
                    total_src  = total_src,
                    detector   = detector,
                    session_id = session_id_meta,
                )
            except Exception as exc:          # pylint: disable=broad-except
                self.logger.exception("Unexpected error on Rank #%d: %s", clip.rank, exc)
                res = ReframeResult(
                    rank=clip.rank, start=clip.start, end=clip.end,
                    output_path="", fps=fps, total_frames=0,
                    face_detected_frames=0, face_detection_rate=0.0,
                    status="failed", error=str(exc),
                )
            results.append(res)

        cap.release()
        detector.close()

        # ── Save report ───────────────────────────────────────────────
        report_path = self._save_report(results, self.output_dir, session_id_meta)
        self.logger.info("Reframe report saved → %s", report_path)

        self._print_summary(results)
        return results

    def _print_summary(self, results: list[ReframeResult]) -> None:
        bar = "═" * 62
        def p(s=""):
            try:
                print(s)
            except UnicodeEncodeError:
                print(s.encode('ascii', 'replace').decode('ascii'))

        p(f"\n{bar}")
        p("  PHASE 4 — FACE-AWARE REFRAMING RESULT")
        p(bar)
        for r in results:
            status_icon = "✓" if r.status == "success" else "✗"
            p(
                f"  [{status_icon}] Rank #{r.rank}  "
                f"{r.start:.1f}s→{r.end:.1f}s  |  "
                f"face: {r.face_detection_rate*100:.0f}%  |  "
                f"{Path(r.output_path).name if r.output_path else 'FAILED'}"
            )
            if r.error:
                p(f"       ERROR: {r.error}")
        p(bar + "\n")


# ─────────────────────────────────────────────
#  CLI Entry Point
# ─────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Local AI Reel Maker — Phase 4: Face-Aware 9:16 Reframing",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--video", "-v",
        required=True,
        help="Path to the source 16:9 video file.",
    )
    parser.add_argument(
        "--clips", "-c",
        required=True,
        help="Path to Phase-3 viral_clips.json.",
    )
    parser.add_argument(
        "--output_dir", "-o",
        default="outputs/reels",
        help="Directory to write 9:16 output clips.",
    )
    parser.add_argument(
        "--ranks",
        nargs="+",
        type=int,
        default=None,
        help="Only process these ranks (e.g. --ranks 1 2 3). Default: all.",
    )
    parser.add_argument(
        "--ema_alpha",
        type=float,
        default=EMA_ALPHA,
        help="EMA smoothing factor (0=frozen, 1=raw). Lower = smoother.",
    )
    parser.add_argument(
        "--gauss_window",
        type=int,
        default=GAUSS_WINDOW,
        help="Gaussian smoothing half-window in frames.",
    )
    parser.add_argument(
        "--min_face_confidence",
        type=float,
        default=MIN_DETECTION_CONFIDENCE,
        help="Minimum MediaPipe face detection confidence.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    engine = ReframerEngine(
        source_video         = args.video,
        ema_alpha            = args.ema_alpha,
        gauss_window         = args.gauss_window,
        min_face_confidence  = args.min_face_confidence,
        output_dir           = args.output_dir,
    )
    engine.run(
        viral_clips_path = args.clips,
        ranks            = args.ranks,
    )


if __name__ == "__main__":
    main()