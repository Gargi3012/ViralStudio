# ViralStudio Pro: Enterprise Edition

**ViralStudio Pro** is a high-performance, local-first engine designed to transform long-form YouTube content into high-engagement vertical reels. Built for privacy and speed, it handles the entire production pipeline on-device without relying on external cloud APIs.

---

## 🚀 Key Features

*   **100% Local & Secure:** No data leaves your machine. Transcription, scoring, and rendering happen locally.
*   **Neural Analysis Engine:** Uses semantic chunking and sentiment analysis to identify viral-worthy segments.
*   **Proprietary Face-Tracking:** Automatic 9:16 reframing using computer vision fallbacks for maximum reliability.
*   **Overlap Prevention:** Advanced logic ensures that multiple clips from the same source are distinct and unique.
*   **Built-in Studio Tools:** Post-processing suite for visual enhancement and automated captions.

---

## 🛠️ Machine Learning Architecture

The pipeline is divided into 5 specialized modules:

1.  **Phase 1: Acquisition (Smart Engine)**
    *   Uses `yt-dlp` for high-quality source fetching.
    *   Metadata filtering for view counts and engagement.
2.  **Phase 2: Processing (Transcription)**
    *   **OpenAI Whisper:** Locally hosted speech-reconstruction for timestamped transcripts.
3.  **Phase 3: Intelligence (Neural Core)**
    *   **Semantic Chunking:** Groups segments based on topical coherence using `Sentence-Transformers`.
    *   **Viral Scoring:** Combined metrics from `DistilBERT` (Sentiment) and Audio Energy mapping.
4.  **Phase 4: Synthesis (Visual Reframer)**
    *   **MediaPipe + OpenCV:** Face detection and smooth tracking to maintain vertical focus.
    *   **FFmpeg Integration:** High-speed muxing for final export.
5.  **Phase 5: Output (Studio Polish)**
    *   AI-driven color grading and automated burn-in captions.

---

## 📦 Installation

### 1. System Dependencies
Ensure you have [FFmpeg](https://ffmpeg.org/) installed and added to your system PATH.

### 2. Python Environment
Install the required libraries:
```bash
pip install -r requirements.txt
```

### 3. Environment Variables
Create a `.env` file in the root directory:
```env
YOUTUBE_API_KEY=your_api_key_here
```

---

## 🖥️ Usage

### Launch the Enterprise Dashboard
```bash
streamlit run dashboard.py
```

### Direct CLI Access (Orchestrator)
```bash
python main.py --url "https://youtube.com/watch?v=..." --top_n 3
```

---

## 📁 Project Structure
*   `modules/`: Core backend processing phases.
*   `data/`: Local storage for raw and processed assets.
*   `outputs/`: Final high-definition vertical reels.
*   `logs/`: Detailed session execution logs.
*   `models/`: Cached local ML weights (HuggingFace/TFLite).

---

<img width="1887" height="861" alt="Screenshot 2026-04-14 194625" src="https://github.com/user-attachments/assets/3c4b5362-670a-483a-82c6-276dfe18d0b8" />
<img width="1287" height="745" alt="Screenshot 2026-04-14 194526" src="https://github.com/user-attachments/assets/f09b4038-aa83-46da-9eb1-0680e57558ba" />
<img width="1283" height="894" alt="Screenshot 2026-04-14 194535" src="https://github.com/user-attachments/assets/748d9071-303b-4569-98cf-ae248418fd42" />

