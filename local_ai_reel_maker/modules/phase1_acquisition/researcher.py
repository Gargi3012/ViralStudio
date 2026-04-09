"""
=============================================================
  Local AI Reel Maker — Phase 1: Data Acquisition Engine
  Module : researcher.py
  Author : AI Engineer
  Purpose: Search YouTube for high-performing videos matching
           a keyword, filter by duration & view count, then
           download the best-quality audio track via yt-dlp.
=============================================================

Dependencies:
    pip install google-api-python-client yt-dlp python-dotenv

Usage:
    python researcher.py --query "Motivational Podcasts" \
                         --max_results 10 \
                         --min_views 50000 \
                         --min_duration 20 \
                         --max_duration 60
"""

import os
import re
import json
import uuid
import logging
import argparse
from datetime import datetime, timezone
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional

import yt_dlp
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from dotenv import load_dotenv


# ─────────────────────────────────────────────
#  Load environment variables (.env file)
# ─────────────────────────────────────────────
load_dotenv()


# ─────────────────────────────────────────────
#  Logging configuration
# ─────────────────────────────────────────────
def setup_logger(session_id: str, log_dir: Path) -> logging.Logger:
    """Configure a session-scoped logger that writes to file + console."""
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{session_id}.log"

    logger = logging.getLogger(f"researcher.{session_id}")
    logger.setLevel(logging.DEBUG)

    fmt = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # File handler (DEBUG+)
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)

    # Console handler (INFO+)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


# ─────────────────────────────────────────────
#  Data models
# ─────────────────────────────────────────────
@dataclass
class VideoCandidate:
    """Represents a YouTube video that passed the filter stage."""
    video_id: str
    title: str
    channel: str
    duration_seconds: int
    view_count: int
    like_count: int
    published_at: str
    url: str
    thumbnail_url: str


@dataclass
class SessionResult:
    """Full result written to metadata.json after a session."""
    session_id: str
    query: str
    timestamp: str
    filters: dict
    candidates_found: int
    selected_video: Optional[dict]
    audio_path: Optional[str]
    video_path: Optional[str]
    status: str          # "success" | "no_candidates" | "download_failed"
    error: Optional[str]


# ─────────────────────────────────────────────
#  ISO 8601 Duration Parser
# ─────────────────────────────────────────────
def parse_iso8601_duration(duration_str: str) -> int:
    """
    Convert YouTube ISO-8601 duration (e.g. 'PT1M30S') to total seconds.

    Examples:
        'PT30S'      →  30
        'PT1M20S'    →  80
        'PT1H2M3S'   →  3723
    """
    pattern = r"PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?"
    match = re.match(pattern, duration_str)
    if not match:
        return 0

    hours   = int(match.group(1) or 0)
    minutes = int(match.group(2) or 0)
    seconds = int(match.group(3) or 0)
    return hours * 3600 + minutes * 60 + seconds


# ─────────────────────────────────────────────
#  Core Engine
# ─────────────────────────────────────────────
class DataAcquisitionEngine:
    """
    Phase-1 engine for the Local AI Reel Maker.

    Responsibilities:
      1. Search YouTube via Data API v3 (keyword-based).
      2. Batch-fetch video statistics + content details.
      3. Filter videos by view count and duration window.
      4. Download the best-quality audio of the top candidate.
      5. Persist raw audio + structured metadata under /raw_data/{session_id}/.
    """

    BASE_RAW_DIR  = Path("data/raw_data")
    BASE_LOG_DIR  = Path("logs/session_logs")
    YT_BASE_URL   = "https://www.youtube.com/watch?v="

    def __init__(
        self,
        api_key: Optional[str] = None,
        min_duration: int = 20,
        max_duration: int = 60,
        min_views: int = 10_000,
        max_results: int = 25,
    ):
        """
        Args:
            api_key      : YouTube Data API v3 key.
                           Falls back to YOUTUBE_API_KEY env variable.
            min_duration : Minimum video length in seconds (inclusive).
            max_duration : Maximum video length in seconds (inclusive).
            min_views    : Minimum view count threshold.
            max_results  : Max results to request from YouTube search.
        """
        self.api_key      = api_key or os.getenv("YOUTUBE_API_KEY")
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.min_views    = min_views
        self.max_results  = max_results

        # We lazily check for API key only if a search is performed
        # to allow run_from_url to work without a Google Cloud account.
        self._youtube = None

        # Session bookkeeping
        self.session_id  = self._new_session_id()
        self.session_dir = self.BASE_RAW_DIR / self.session_id
        self.session_dir.mkdir(parents=True, exist_ok=True)

        self.logger = setup_logger(self.session_id, self.BASE_LOG_DIR)
        self.logger.info("Session started  : %s", self.session_id)
        self.logger.info(
            "Filters — duration: %ds–%ds | min views: %s",
            self.min_duration,
            self.max_duration,
            f"{self.min_views:,}",
        )

    # ──────────────────────────────────────────
    #  Internal helpers
    # ──────────────────────────────────────────

    @staticmethod
    def _new_session_id() -> str:
        """Generate a timestamped unique session ID."""
        ts   = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        uid  = uuid.uuid4().hex[:6]
        return f"session_{ts}_{uid}"

    def _get_youtube_client(self):
        """Lazy-build the Google API client (avoids slow import at module level)."""
        if self._youtube is None:
            if not self.api_key:
                 raise EnvironmentError(
                    "YouTube Data API key is required for search operations. "
                    "Please set YOUTUBE_API_KEY in your .env file."
                )
            self._youtube = build(
                "youtube",
                "v3",
                developerKey=self.api_key,
                cache_discovery=False,   # suppress file_cache warnings
            )
        return self._youtube

    # ──────────────────────────────────────────
    #  Step 1 — YouTube Search
    # ──────────────────────────────────────────

    def search_videos(self, query: str) -> list[str]:
        """
        Run a YouTube search and return a list of video IDs.

        Args:
            query: Search keyword(s), e.g. 'Motivational Podcasts'.

        Returns:
            List of YouTube video ID strings.

        Raises:
            HttpError: On API quota exhaustion or invalid key.
        """
        self.logger.info("Searching YouTube for: '%s'", query)
        youtube = self._get_youtube_client()

        try:
            response = (
                youtube.search()
                .list(
                    q=query,
                    part="id,snippet",
                    type="video",
                    videoDuration="short",   # YouTube pre-filter: clips < 4 min
                    order="viewCount",       # surface popular videos first
                    maxResults=self.max_results,
                    relevanceLanguage="en",
                    safeSearch="none",
                )
                .execute()
            )
        except HttpError as exc:
            status = exc.resp.status
            if status == 403:
                self.logger.error(
                    "API quota exceeded or key invalid (HTTP 403). "
                    "Check your quota at https://console.cloud.google.com"
                )
            elif status == 400:
                self.logger.error("Bad API request (HTTP 400): %s", exc)
            else:
                self.logger.error("YouTube API error (HTTP %s): %s", status, exc)
            raise

        video_ids = [
            item["id"]["videoId"]
            for item in response.get("items", [])
            if item["id"].get("kind") == "youtube#video"
        ]
        self.logger.info("Raw search returned %d video IDs.", len(video_ids))
        return video_ids

    # ──────────────────────────────────────────
    #  Step 2 — Batch Fetch Details & Filter
    # ──────────────────────────────────────────

    def fetch_and_filter(self, video_ids: list[str]) -> list[VideoCandidate]:
        """
        Batch-fetch statistics + content details for up to 50 video IDs,
        then apply duration and view-count filters.

        Args:
            video_ids: List of YouTube video ID strings.

        Returns:
            Filtered list of VideoCandidate objects, sorted by view count desc.
        """
        if not video_ids:
            self.logger.warning("No video IDs to fetch details for.")
            return []

        youtube = self._get_youtube_client()
        self.logger.info("Fetching details for %d videos…", len(video_ids))

        try:
            response = (
                youtube.videos()
                .list(
                    id=",".join(video_ids),
                    part="snippet,statistics,contentDetails",
                )
                .execute()
            )
        except HttpError as exc:
            self.logger.error("Failed to fetch video details: %s", exc)
            raise

        candidates: list[VideoCandidate] = []

        for item in response.get("items", []):
            vid_id    = item["id"]
            snippet   = item.get("snippet", {})
            stats     = item.get("statistics", {})
            content   = item.get("contentDetails", {})

            # ── Duration filter ──────────────────────
            duration_sec = parse_iso8601_duration(content.get("duration", "PT0S"))
            if not (self.min_duration <= duration_sec <= self.max_duration):
                self.logger.debug(
                    "SKIP (duration %ds out of [%d–%d]s): %s",
                    duration_sec, self.min_duration, self.max_duration, vid_id,
                )
                continue

            # ── View count filter ────────────────────
            view_count = int(stats.get("viewCount", 0))
            if view_count < self.min_views:
                self.logger.debug(
                    "SKIP (views %s < %s): %s",
                    f"{view_count:,}", f"{self.min_views:,}", vid_id,
                )
                continue

            like_count = int(stats.get("likeCount", 0))
            thumbnails = snippet.get("thumbnails", {})
            thumbnail  = (
                thumbnails.get("maxres")
                or thumbnails.get("high")
                or thumbnails.get("default")
                or {}
            ).get("url", "")

            candidates.append(
                VideoCandidate(
                    video_id       = vid_id,
                    title          = snippet.get("title", "Unknown Title"),
                    channel        = snippet.get("channelTitle", "Unknown Channel"),
                    duration_seconds = duration_sec,
                    view_count     = view_count,
                    like_count     = like_count,
                    published_at   = snippet.get("publishedAt", ""),
                    url            = f"{self.YT_BASE_URL}{vid_id}",
                    thumbnail_url  = thumbnail,
                )
            )

        # Sort best candidate first (highest views)
        candidates.sort(key=lambda c: c.view_count, reverse=True)
        self.logger.info(
            "%d candidate(s) passed filters (duration %d–%ds, views ≥ %s).",
            len(candidates), self.min_duration, self.max_duration,
            f"{self.min_views:,}",
        )
        return candidates

    # ──────────────────────────────────────────
    #  Step 3 — Audio Download (yt-dlp)
    # ──────────────────────────────────────────

    def download_audio(self, video: VideoCandidate) -> Optional[Path]:
        """
        Download the best-quality audio stream of *video* using yt-dlp.

        The file is saved to:
            data/raw_data/{session_id}/audio.{ext}

        Args:
            video: A VideoCandidate that passed the filter stage.

        Returns:
            Path to the downloaded audio file, or None on failure.
        """
        output_template_audio = str(self.session_dir / "audio.%(ext)s")
        output_template_video = str(self.session_dir / "video.%(ext)s")

        ydl_opts = {
            # Prefer mp4/webm with both video and audio.
            # Best single file, or best video+audio composite if needed.
            "format"         : "best[ext=mp4]/best",
            "outtmpl"        : output_template_video,
            "noplaylist"     : True,
            "quiet"          : False,
            "no_warnings"    : False,
            "logger"         : _YtdlpLogger(self.logger),

            # Post-process: extract audio to mp3 and keep the video.
            "postprocessors" : [
                {
                    "key"            : "FFmpegExtractAudio",
                    "preferredcodec" : "mp3",
                    "preferredquality": "320",
                }
            ],
            # Keep the original video file along with the extracted audio
            "keepvideo": True,

            # Retry logic
            "retries"        : 3,
            "fragment_retries": 3,
        }

        self.logger.info("Downloading video+audio: '%s' [%s]", video.title, video.url)

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([video.url])
        except yt_dlp.utils.DownloadError as exc:
            self.logger.error("yt-dlp DownloadError: %s", exc)
            return None, None
        except yt_dlp.utils.ExtractorError as exc:
            self.logger.error("yt-dlp ExtractorError (invalid URL or removed video): %s", exc)
            return None, None
        except Exception as exc:  # pylint: disable=broad-except
            self.logger.error("Unexpected download error: %s", exc)
            return None, None

        # Locate the written audio file
        audio_file = self.session_dir / "audio.mp3"
        
        # It's actually put with outtmpl which is video.mp3 because ffmpeg replaces ext
        potential_audio = self.session_dir / "video.mp3"
        if potential_audio.exists():
             try:
                 potential_audio.replace(audio_file)
             except Exception as e:
                 self.logger.warning("Could not rename video.mp3 to audio.mp3: %s", e)
                 audio_file = potential_audio
        
        if audio_file.exists():
            size_kb = audio_file.stat().st_size / 1024
            self.logger.info(
                "Audio saved → %s  (%.1f KB)", audio_file, size_kb
            )
        else:
             self.logger.error("Audio file not found after download.")
             audio_file = None
             
        # Locate the written video file
        video_file = None
        for ext in ("mp4", "webm", "mkv"):
             fallback = self.session_dir / f"video.{ext}"
             if fallback.exists():
                 video_file = fallback
                 self.logger.info("Video saved → %s", video_file)
                 break

        if not video_file:
             self.logger.error("Video file not found after download.")
             
        return audio_file, video_file

    # ──────────────────────────────────────────
    #  Step 4 — Persist Metadata
    # ──────────────────────────────────────────

    def save_metadata(self, result: SessionResult) -> Path:
        """Write the SessionResult to metadata.json inside the session directory."""
        metadata_path = self.session_dir / "metadata.json"
        with open(metadata_path, "w", encoding="utf-8") as fh:
            json.dump(asdict(result), fh, indent=2, ensure_ascii=False)
        self.logger.info("Metadata saved → %s", metadata_path)
        return metadata_path

    # ──────────────────────────────────────────
    #  Public Orchestrator
    # ──────────────────────────────────────────

    def run_from_url(self, url: str) -> SessionResult:
        """
        Execute the Phase-1 pipeline using a direct YouTube URL.
        Bypasses search and directly downloads/records metadata.
        """
        filters = {
            "min_duration_sec" : 0,
            "max_duration_sec" : 0,
            "min_views"        : 0,
            "max_results"      : 0,
        }
        result = SessionResult(
            session_id        = self.session_id,
            query             = f"DIRECT_URL: {url}",
            timestamp         = datetime.now(timezone.utc).isoformat(),
            filters           = filters,
            candidates_found  = 1,
            selected_video    = None,
            audio_path        = None,
            video_path        = None,
            status            = "no_candidates",
            error             = None,
        )

        try:
            self.logger.info("Extracting info from direct URL: %s", url)
            ydl_opts = {
                "quiet": True,
                "no_warnings": True,
            }
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=False)

            best = VideoCandidate(
                video_id       = info.get("id", "Unknown"),
                title          = info.get("title", "Unknown Title"),
                channel        = info.get("uploader", "Unknown Channel"),
                duration_seconds = info.get("duration", 0),
                view_count     = info.get("view_count", 0),
                like_count     = info.get("like_count", 0),
                published_at   = info.get("upload_date", ""),
                url            = url,
                thumbnail_url  = info.get("thumbnail", ""),
            )
            result.selected_video = asdict(best)

            # 5. Download audio and video
            audio_path, video_path = self.download_audio(best)
            if audio_path is None or video_path is None:
                result.status = "download_failed"
                result.error  = "yt-dlp could not download from the provided URL."
            else:
                result.audio_path = str(audio_path)
                result.video_path = str(video_path)
                result.status     = "success"

        except Exception as exc:  # pylint: disable=broad-except
            result.status = "unexpected_error"
            result.error  = str(exc)
            self.logger.exception("Unexpected pipeline error for URL %s: %s", url, exc)

        # 6. Always persist metadata
        self.save_metadata(result)

        self.logger.info(
            "Session complete — status: %s | session_id: %s",
            result.status.upper(), self.session_id,
        )
        return result

    def run(self, query: str) -> SessionResult:
        """
        Execute the full Phase-1 pipeline:
            search → filter → download → save metadata.

        Args:
            query: Search keyword(s) to use on YouTube.

        Returns:
            A SessionResult summarising the outcome of the session.
        """
        filters = {
            "min_duration_sec" : self.min_duration,
            "max_duration_sec" : self.max_duration,
            "min_views"        : self.min_views,
            "max_results"      : self.max_results,
        }
        result = SessionResult(
            session_id        = self.session_id,
            query             = query,
            timestamp         = datetime.now(timezone.utc).isoformat(),
            filters           = filters,
            candidates_found  = 0,
            selected_video    = None,
            audio_path        = None,
            video_path        = None,
            status            = "no_candidates",
            error             = None,
        )

        try:
            # 1. Search
            video_ids = self.search_videos(query)
            if not video_ids:
                self.logger.warning("Search returned zero results for '%s'.", query)
                self.save_metadata(result)
                return result

            # 2. Fetch details & filter
            candidates = self.fetch_and_filter(video_ids)
            result.candidates_found = len(candidates)

            if not candidates:
                self.logger.warning(
                    "No videos passed the filters. "
                    "Try lowering --min_views or widening the duration range."
                )
                self.save_metadata(result)
                return result

            # 3. Log top-3 candidates
            self.logger.info("── Top candidates ──────────────────────────────")
            for rank, c in enumerate(candidates[:3], start=1):
                self.logger.info(
                    " #%d  %s  [%ds | %s views]",
                    rank, c.title[:60], c.duration_seconds, f"{c.view_count:,}",
                )
            self.logger.info("────────────────────────────────────────────────")

            # 4. Pick the best candidate (highest views)
            best = candidates[0]
            result.selected_video = asdict(best)

            # 5. Download audio and video
            audio_path, video_path = self.download_audio(best)
            if audio_path is None or video_path is None:
                result.status = "download_failed"
                result.error  = "yt-dlp could not download the video/audio."
            else:
                result.audio_path = str(audio_path)
                result.video_path = str(video_path)
                result.status     = "success"

        except HttpError as exc:
            result.status = "api_error"
            result.error  = str(exc)
            self.logger.error("Pipeline aborted due to API error: %s", exc)

        except Exception as exc:  # pylint: disable=broad-except
            result.status = "unexpected_error"
            result.error  = str(exc)
            self.logger.exception("Unexpected pipeline error: %s", exc)

        # 6. Always persist metadata
        self.save_metadata(result)

        self.logger.info(
            "Session complete — status: %s | session_id: %s",
            result.status.upper(), self.session_id,
        )
        return result


# ─────────────────────────────────────────────
#  yt-dlp Logger Adapter
# ─────────────────────────────────────────────
class _YtdlpLogger:
    """
    Pipe yt-dlp's internal log output into the session logger
    so everything ends up in the same log file.
    """

    def __init__(self, logger: logging.Logger):
        self._logger = logger

    def debug(self, msg: str):
        # yt-dlp uses debug for progress lines — keep at DEBUG level
        self._logger.debug("[yt-dlp] %s", msg)

    def info(self, msg: str):
        self._logger.info("[yt-dlp] %s", msg)

    def warning(self, msg: str):
        self._logger.warning("[yt-dlp] %s", msg)

    def error(self, msg: str):
        self._logger.error("[yt-dlp] %s", msg)


# ─────────────────────────────────────────────
#  CLI Entry Point
# ─────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Local AI Reel Maker — Phase 1: Data Acquisition Engine",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--query", "-q",
        type=str,
        default="Motivational Podcasts",
        help="YouTube search keyword(s).",
    )
    parser.add_argument(
        "--min_duration",
        type=int,
        default=20,
        help="Minimum video duration in seconds.",
    )
    parser.add_argument(
        "--max_duration",
        type=int,
        default=60,
        help="Maximum video duration in seconds.",
    )
    parser.add_argument(
        "--min_views",
        type=int,
        default=10_000,
        help="Minimum view count for a video to qualify.",
    )
    parser.add_argument(
        "--max_results",
        type=int,
        default=25,
        help="Max number of search results to fetch from YouTube.",
    )
    parser.add_argument(
        "--api_key",
        type=str,
        default=None,
        help="YouTube Data API v3 key (overrides YOUTUBE_API_KEY env var).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    engine = DataAcquisitionEngine(
        api_key      = args.api_key,
        min_duration = args.min_duration,
        max_duration = args.max_duration,
        min_views    = args.min_views,
        max_results  = args.max_results,
    )

    result = engine.run(query=args.query)

    # ── Summary print ──────────────────────────────────────────────────
    def p(s=""):
        try:
            print(s)
        except UnicodeEncodeError:
            print(s.encode('ascii', 'replace').decode('ascii'))

    p("\n" + "═" * 56)
    p("  PHASE 1 — DATA ACQUISITION RESULT")
    p("═" * 56)
    p(f"  Session ID   : {result.session_id}")
    p(f"  Status       : {result.status.upper()}")
    p(f"  Query        : {result.query}")
    p(f"  Candidates   : {result.candidates_found}")

    if result.selected_video:
        sv = result.selected_video
        p(f"  Selected     : {sv['title'][:55]}")
        p(f"  Channel      : {sv['channel']}")
        p(f"  Views        : {sv['view_count']:,}")
        p(f"  Duration     : {sv['duration_seconds']}s")
        p(f"  URL          : {sv['url']}")

    if result.audio_path:
        p(f"  Audio saved  : {result.audio_path}")

    if result.video_path:
        p(f"  Video saved  : {result.video_path}")

    if result.error:
        p(f"  Error        : {result.error}")

    p("═" * 56 + "\n")


if __name__ == "__main__":
    main()