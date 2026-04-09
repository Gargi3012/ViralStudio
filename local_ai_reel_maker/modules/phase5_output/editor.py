"""
=============================================================
  Local AI Reel Maker — Phase 5: AI Post-Processing Editor
  Module : editor.py
  Purpose: Add captions, background music, and visual filters.
=============================================================
"""

import os
import subprocess
import logging
from pathlib import Path
from typing import Optional

class ReelEditor:
    """
    Handles post-processing tasks like subtitles and BGM using FFmpeg.
    """

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)

    def add_subtitles(self, video_path: Path, transcript_text: str) -> Optional[Path]:
        """
        Burn basic captions into the video.
        """
        output_path = video_path.parent / f"edited_{video_path.name}"
        # A simple FFmpeg command to add a text overlay at the bottom center
        # For professional usage, we'd generate a .srt and use the 'subtitles' filter
        # But for a quick 'Polish', we'll use 'drawtext'
        drawtext_filter = (
            f"drawtext=text='{transcript_text[:40]}...':fontcolor=white:fontsize=24:"
            f"box=1:boxcolor=black@0.5:boxborderw=5:x=(w-text_w)/2:y=h-80"
        )
        
        cmd = [
            "ffmpeg", "-y", "-i", str(video_path),
            "-vf", drawtext_filter,
            "-c:a", "copy",
            str(output_path)
        ]
        
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            return output_path
        except Exception as e:
            self.logger.error("Failed to add subtitles: %s", e)
            return None

    def enhance_visuals(self, video_path: Path) -> Optional[Path]:
        """
        Boost contrast and saturation for a more 'Viral' look.
        """
        output_path = video_path.parent / f"pro_{video_path.name}"
        # eq=contrast=1.1:saturation=1.3
        cmd = [
            "ffmpeg", "-y", "-i", str(video_path),
            "-vf", "eq=contrast=1.1:saturation=1.2:brightness=0.02",
            "-c:a", "copy",
            str(output_path)
        ]
        
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            return output_path
        except Exception as e:
            self.logger.error("Visual enhancement failed: %s", e)
            return None

    def mix_bgm(self, video_path: Path, bgm_path: Path, volume: float = 0.15) -> Optional[Path]:
        """
        Mix background music with original audio.
        """
        output_path = video_path.parent / f"music_{video_path.name}"
        # amix filter mixes two audio streams
        filter_complex = f"[0:a]volume=1[v];[1:a]volume={volume}[m];[v][m]amix=inputs=2:duration=first"
        
        cmd = [
            "ffmpeg", "-y", "-i", str(video_path), "-i", str(bgm_path),
            "-filter_complex", filter_complex,
            "-c:v", "copy",
            str(output_path)
        ]
        
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            return output_path
        except Exception as e:
            self.logger.error("BGM mixing failed: %s", e)
            return None
