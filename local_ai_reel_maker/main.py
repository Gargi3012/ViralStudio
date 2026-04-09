"""
=============================================================
  Local AI Reel Maker — Main Orchestrator
  Module : main.py
  Author : AI Engineer
  Purpose: Parse CLI arguments and run Phase 1, 2, 3, 4 sequentially.
           Maintains session state and connects inputs/outputs.
=============================================================
"""

import argparse
import sys
from pathlib import Path
import logging

from modules.phase1_acquisition.researcher import DataAcquisitionEngine
from modules.phase2_processing.transcriber import AudioIntelligenceEngine
from modules.phase3_ai_script.scorer import ScoringEngine
from modules.phase4_media.reframer import ReframerEngine

def setup_logger():
    logger = logging.getLogger("orchestrator")
    logger.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    fmt = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    return logger

def parse_args():
    parser = argparse.ArgumentParser(
        description="Local AI Reel Maker — End-to-End Orchestrator"
    )
    parser.add_argument("--query", "-q", required=False, help="YouTube search query")
    parser.add_argument("--url", "-u", required=False, help="Direct YouTube URL to bypass search")
    parser.add_argument("--min_duration", type=int, default=20)
    parser.add_argument("--max_duration", type=int, default=180)
    parser.add_argument("--min_views", type=int, default=10000)
    parser.add_argument("--top_n", type=int, default=3, help="Number of viral clips to extract")
    parser.add_argument("--dry_run", action="store_true", help="Stop after phase 3 scoring")
    return parser.parse_args()

def main():
    args = parse_args()
    logger = setup_logger()
    logger.info("Starting Local AI Reel Maker pipeline...")
    
    if not args.query and not args.url:
        logger.error("Must provide either --query or --url")
        sys.exit(1)

    if args.url:
        logger.info(f"Using DIRECT URL: '{args.url}'")
    else:
        logger.info(f"Using SEARCH Query: '{args.query}'")

    # 1. Phase 1: Data Acquisition
    logger.info("\n" + "="*40 + "\n PHASE 1: ACQUISITION \n" + "="*40)
    phase1 = DataAcquisitionEngine(
        min_duration=args.min_duration,
        max_duration=args.max_duration,
        min_views=args.min_views
    )
    
    if args.url:
        res1 = phase1.run_from_url(args.url)
    else:
        res1 = phase1.run(args.query)

    if res1.status != "success":
        logger.error(f"Phase 1 failed: {res1.error}")
        sys.exit(1)
        
    audio_path = Path(res1.audio_path)
    video_path = Path(res1.video_path)
    session_id = res1.session_id
    
    # Generate common paths
    processed_dir = Path("data/processed_data") / session_id
    transcript_path = processed_dir / "transcript.json"
    viral_clips_path = processed_dir / "viral_clips.json"

    # 2. Phase 2: Transcription & Intelligence
    logger.info("\n" + "="*40 + "\n PHASE 2: TRANSCRIPTION \n" + "="*40)
    phase2 = AudioIntelligenceEngine(model_name="tiny")
    res2 = phase2.run(
        audio_path=audio_path,
        session_dir=processed_dir,
        session_id=session_id
    )
    
    if not res2.full_text:
        logger.error("Phase 2 failed: Empty transcript")
        sys.exit(1)

    # 3. Phase 3: Scoring
    logger.info("\n" + "="*40 + "\n PHASE 3: NEURAL SCORING \n" + "="*40)
    phase3 = ScoringEngine(top_n=args.top_n)
    res3 = phase3.run(
        transcript_path=transcript_path,
        output_dir=processed_dir
    )

    if not res3.top_clips:
        logger.error("Phase 3 failed: No viral clips generated")
        sys.exit(1)

    if args.dry_run:
        logger.info("Dry run requested, stopping before Phase 4. Pipeline successful.")
        sys.exit(0)

    # 4. Phase 4: Reframing
    logger.info("\n" + "="*40 + "\n PHASE 4: REFRAMING \n" + "="*40)
    phase4 = ReframerEngine(
        source_video=video_path,
        output_dir="outputs/reels"
    )
    res4 = phase4.run(viral_clips_path=viral_clips_path)
    
    success_count = sum(1 for r in res4 if r.status == "success")
    logger.info("\n" + "="*40 + "\n PIPELINE COMPLETE \n" + "="*40)
    logger.info(f"Successfully created {success_count}/{len(res4)} viral reels in outputs/reels/")

if __name__ == "__main__":
    main()
