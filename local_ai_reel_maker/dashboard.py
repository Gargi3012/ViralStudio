"""
=============================================================
  Local AI Reel Maker — Streamlit Dashboard v3
  Module : dashboard.py
  Purpose: Dual-mode UI — full-width content, sidebar = tasks only
=============================================================
"""

import streamlit as st
import subprocess
import os
from pathlib import Path
from modules.phase1_acquisition.researcher import DataAcquisitionEngine
from modules.phase5_output.editor import ReelEditor

# ─── Page Config ────────────────────────────────────────────
st.set_page_config(
    page_title="ViralStudio Pro",
    layout="wide",
    page_icon="🎬",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ─────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

  html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

  .stApp {
    background: linear-gradient(135deg, #0d0d1a 0%, #0f1625 50%, #0a0f1e 100%);
    color: #e2e8f0;
  }

  /* ── Sidebar ── */
  [data-testid="stSidebar"] {
    background: linear-gradient(180deg, #111827 0%, #0f172a 100%);
    border-right: 1px solid rgba(99,102,241,0.2);
    box-shadow: 4px 0 24px rgba(0,0,0,0.5);
  }

  /* ── Logo ── */
  .logo-badge {
    display: flex; align-items: center; gap: 10px;
    padding: 18px 0 22px; border-bottom: 1px solid rgba(255,255,255,0.07);
    margin-bottom: 20px;
  }
  .logo-icon {
    width: 42px; height: 42px; border-radius: 11px;
    background: linear-gradient(135deg,#6366f1,#8b5cf6);
    display: flex; align-items: center; justify-content: center; font-size: 22px;
  }
  .logo-text { font-size: 1.05rem; font-weight: 700; color: #e2e8f0; }
  .logo-sub  { font-size: 0.68rem; color: #475569; }

  /* ── Dashboard header ── */
  .dash-header {
    background: linear-gradient(135deg,rgba(99,102,241,0.1),rgba(139,92,246,0.07));
    border: 1px solid rgba(99,102,241,0.18);
    border-radius: 16px; padding: 26px 32px; margin-bottom: 28px;
  }
  .dash-header h1 {
    font-size: 2rem; font-weight: 800; margin: 0 0 6px;
    background: linear-gradient(135deg,#e2e8f0,#94a3b8);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
  }
  .dash_header p { color: #475569; font-size: 0.92rem; margin: 0; }

  /* ── Search bar row ── */
  .search-row {
    display: flex; gap: 12px; align-items: flex-end; margin-bottom: 20px;
  }

  /* ── Filter chips row ── */
  .filter-row {
    display: flex; flex-wrap: wrap; gap: 16px; align-items: center;
    background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.07);
    border-radius: 12px; padding: 16px 20px; margin-bottom: 24px;
  }

  /* ── Stat chip ── */
  .stat-chip {
    display: inline-flex; align-items: center; gap: 5px;
    background: rgba(99,102,241,0.1); border: 1px solid rgba(99,102,241,0.25);
    border-radius: 50px; padding: 4px 12px; font-size: 0.78rem; color: #a78bfa; margin: 2px;
  }

  /* ── Video result card ── */
  .vid-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 12px; padding: 16px; margin-bottom: 12px;
    transition: border-color 0.2s, background 0.2s;
  }
  .vid-card:hover {
    border-color: rgba(99,102,241,0.3);
    background: rgba(99,102,241,0.05);
  }

  /* ── Control card ── */
  .ctrl-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 12px; padding: 22px 24px; margin-bottom: 16px;
  }
  .ctrl-card-title {
    font-size: 0.72rem; font-weight: 600; text-transform: uppercase;
    letter-spacing: 0.09em; color: #475569; margin: 0 0 14px;
  }

  /* ── Pipeline phase steps ── */
  .phase-step {
    display: flex; align-items: center; gap: 12px;
    padding: 9px 0; border-bottom: 1px solid rgba(255,255,255,0.04);
  }
  .phase-dot { width: 10px; height: 10px; border-radius: 50%; flex-shrink: 0; }
  .phase-dot.done   { background: #22c55e; box-shadow: 0 0 8px #22c55e66; }
  .phase-dot.active { background: #a78bfa; box-shadow: 0 0 8px #a78bfa66; animation: pulse 1.4s infinite; }
  .phase-dot.idle   { background: #1e293b; border: 1px solid #334155; }
  @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.35} }
  .phase-lbl        { font-size: 0.84rem; color: #64748b; }
  .phase-lbl.done   { color: #4ade80; }
  .phase-lbl.active { color: #c4b5fd; font-weight: 600; }

  /* ── Progress label ── */
  .prog-lbl {
    font-size: 0.85rem; font-weight: 600; color: #a78bfa; margin-bottom: 5px;
  }

  /* ── Empty state ── */
  .empty-state {
    text-align: center; padding: 70px 20px; color: #334155;
  }
  .empty-state .icon { font-size: 3.5rem; margin-bottom: 14px; }
  .empty-state .title { font-size: 1.05rem; font-weight: 600; color: #475569; margin-bottom: 6px; }
  .empty-state .sub   { font-size: 0.84rem; }

  /* ── Step card (how it works) ── */
  .step-row {
    display:flex; gap:14px; align-items:flex-start;
    padding:12px 0; border-bottom:1px solid rgba(255,255,255,0.04);
  }
  .step-icon { font-size:1.4rem; width:34px; text-align:center; flex-shrink:0; }
  .step-title { font-weight:600; color:#e2e8f0; font-size:0.88rem; }
  .step-desc  { color:#64748b; font-size:0.8rem; margin-top:2px; }

  /* ── Video Reel Gallery ── */
  .reel-container {
    background: rgba(0,0,0,0.2);
    border: 1px solid rgba(255,255,255,0.05);
    border-radius: 12px;
    padding: 10px;
    margin-bottom: 20px;
    text-align: center;
  }
  .reel-container video {
    max-height: 480px !important; /* Limit height for 9:16 reels */
    border-radius: 8px;
    box-shadow: 0 4px 15px rgba(0,0,0,0.4);
  }

  /* ── Streamlit widget overrides ── */
  .stTextInput>div>div>input {
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 10px !important; color: #e2e8f0 !important;
  }
  .stTextInput>div>div>input:focus {
    border-color: rgba(99,102,241,0.5) !important;
    box-shadow: 0 0 0 3px rgba(99,102,241,0.1) !important;
  }
  .stSlider [data-baseweb="slider"] div[role="slider"] { background: #8b5cf6 !important; }
  .stButton>button {
    background: linear-gradient(135deg,#6366f1,#8b5cf6) !important;
    border: none !important; border-radius: 10px !important;
    color: white !important; font-weight: 600 !important;
    transition: all 0.2s !important;
  }
  .stButton>button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 8px 24px rgba(99,102,241,0.35) !important;
  }
  div[data-testid="stProgress"]>div { border-radius: 10px !important; }
  div[data-testid="stProgress"]>div>div {
    background: linear-gradient(90deg,#6366f1,#a78bfa) !important;
    border-radius: 10px !important;
  }
  [data-testid="stMetric"] label { color:#64748b !important; font-size:0.75rem !important; }
  [data-testid="stMetric"] [data-testid="stMetricValue"] { color:#a78bfa !important; font-size:1.5rem !important; }
  hr { border-color: rgba(255,255,255,0.07) !important; }
</style>
""", unsafe_allow_html=True)


# ─── Session State ───────────────────────────────────────────
for k, v in [("mode","market_research"), ("search_results",[])]:
    if k not in st.session_state:
        st.session_state[k] = v


# ══════════════════════════════════════════════════════════════
#  SIDEBAR — Logo + 2 Task Buttons ONLY
# ══════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div class="logo-badge">
      <div class="logo-icon">🎬</div>
      <div>
        <div class="logo-text">ViralStudio Pro</div>
        <div class="logo-sub">ENTERPRISE EDITION</div>
      </div>
    </div>
    <p style="font-size:0.72rem;font-weight:600;text-transform:uppercase;letter-spacing:0.09em;color:#475569;margin:0 0 12px;">Modules</p>
    """, unsafe_allow_html=True)

    is_mr = st.session_state.mode == "market_research"
    if st.button(
        "🔍  Market Research",
        use_container_width=True,
        key="btn_market",
        type="primary" if is_mr else "secondary",
    ):
        st.session_state.mode = "market_research"
        st.session_state.search_results = []
        st.rerun()

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    is_rm = st.session_state.mode == "reel_maker"
    if st.button(
        "🎞️  Reel Shorts Maker",
        use_container_width=True,
        key="btn_reel",
        type="primary" if is_rm else "secondary",
    ):
        st.session_state.mode = "reel_maker"
        st.rerun()

    st.markdown("---")
    st.markdown("""
    <div style="font-size:0.7rem;color:#334155;line-height:1.8;">
      🔒 100% Secure & Local<br>
      ⚙️ Neural Processing Core<br>
      ⚡ High Performance Engine
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════
PIPELINE_PHASES = [
    (0,  12,  "📥 Downloading video…"),
    (12, 38,  "🎙️ Transcribing audio with Whisper…"),
    (38, 70,  "🧠 Neural scoring with BERT…"),
    (70, 95,  "✂️  Reframing & rendering reels…"),
    (95, 100, "✅ Finalizing output…"),
]
PHASE_TRIGGERS = ["PHASE 1", "PHASE 2", "PHASE 3", "PHASE 4", "PIPELINE COMPLETE"]


def render_phases(active: int):
    labels = [
        "Phase 1 · Acquisition",
        "Phase 2 · Transcription",
        "Phase 3 · Neural Scoring",
        "Phase 4 · Reframing",
        "Complete",
    ]
    html = ""
    for i, lbl in enumerate(labels):
        if i < active:
            d, t = "done", "done"
        elif i == active:
            d, t = "active", "active"
        else:
            d, t = "idle", ""
        html += f'<div class="phase-step"><div class="phase-dot {d}"></div><span class="phase-lbl {t}">{lbl}</span></div>'
    return html


def run_pipeline_with_progress(cmd):
    prog_lbl  = st.empty()
    prog_bar  = st.progress(0)
    phase_ph  = st.empty()

    phase_idx   = 0
    current_pct = 0

    phase_ph.markdown(
        f'<div class="ctrl-card">{render_phases(0)}</div>',
        unsafe_allow_html=True
    )

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        encoding="utf-8",
        errors="replace",
        cwd=str(Path(__file__).parent),
    )

    for line in process.stdout:
        upper = line.strip().upper()
        for i, trigger in enumerate(PHASE_TRIGGERS):
            if trigger in upper and i >= phase_idx:
                phase_idx = i
                phase_ph.markdown(
                    f'<div class="ctrl-card">{render_phases(phase_idx)}</div>',
                    unsafe_allow_html=True
                )
                break

        if phase_idx < len(PIPELINE_PHASES):
            _, end_pct, label = PIPELINE_PHASES[phase_idx]
            if current_pct < end_pct - 2:
                current_pct = min(current_pct + 1, end_pct - 2)
            prog_bar.progress(current_pct)
            prog_lbl.markdown(
                f'<p class="prog-lbl">{label} &nbsp;<strong>{current_pct}%</strong></p>',
                unsafe_allow_html=True
            )

    process.wait()

    if process.returncode == 0:
        prog_bar.progress(100)
        prog_lbl.markdown('<p class="prog-lbl">✅ Done! &nbsp;<strong>100%</strong></p>', unsafe_allow_html=True)
        phase_ph.markdown(
            f'<div class="ctrl-card">{render_phases(4)}</div>',
            unsafe_allow_html=True
        )
        st.success("🎉 Pipeline complete! Your reels are ready below.")
        st.balloons()
    else:
        prog_lbl.markdown('<p class="prog-lbl">❌ Pipeline failed</p>', unsafe_allow_html=True)
        st.error("Pipeline encountered an error. Check the logs/ directory for details.")


def show_reels_gallery():
    st.markdown("---")
    st.markdown("### 🎬 Generated Reels")
    reels_dir = Path("outputs/reels")
    if reels_dir.exists():
        mp4_files = sorted(reels_dir.glob("*.mp4"), key=os.path.getctime, reverse=True)
        if not mp4_files:
            st.info("No reels yet — run the pipeline to generate some!")
        else:
            cols = st.columns(min(len(mp4_files[:6]), 3))
            for i, mp4_path in enumerate(mp4_files[:6]):
                with cols[i % 3]:
                    st.markdown('<div class="reel-container">', unsafe_allow_html=True)
                    st.video(str(mp4_path))
                    size_mb = mp4_path.stat().st_size / (1024 * 1024)
                    
                    with st.expander("🪄 AI Polish & Tools"):
                        ed_col1, ed_col2 = st.columns(2)
                        editor = ReelEditor()
                        
                        if ed_col1.button("✨ Enhance Colors", key=f"enh_{i}_{mp4_path.stem}"):
                            with st.spinner("Boosting visuals..."):
                                res = editor.enhance_visuals(mp4_path)
                                if res: st.success("Enhanced! Refresh to see Pro version.")
                        
                        if ed_col2.button("📝 Add Caption", key=f"cap_{i}_{mp4_path.stem}"):
                             with st.spinner("Burning captions..."):
                                res = editor.add_subtitles(mp4_path, "LOCAL AI REEL")
                                if res: st.success("Caption added!")

                    st.markdown(f"""
                        <div style="font-size:0.75rem; color:#64748b; margin-top:8px; font-weight:600;">
                           🎬 {mp4_path.name[:25]}...<br>
                           📦 {size_mb:.1f} MB
                        </div>
                    """, unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.info("No reels yet — `outputs/reels/` folder not found.")


# ══════════════════════════════════════════════════════════════
#  MODE 1 — MARKET RESEARCH  (full-width)
# ══════════════════════════════════════════════════════════════
if st.session_state.mode == "market_research":

    st.markdown("""
    <div class="dash-header">
      <h1>🔍 Market Research</h1>
      <p>Search YouTube for top-performing videos and queue them for reel generation.</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Search bar (full width) ──────────────────────────────
    s_col, btn_col = st.columns([5, 1])
    with s_col:
        query = st.text_input(
            "Search Query",
            value="Top Business Podcast",
            placeholder="e.g. Motivational speeches, Tech talks…",
            label_visibility="collapsed",
            key="mr_query",
        )
    with btn_col:
        search_btn = st.button("🔍 Search", use_container_width=True, key="mr_search")

    # ── Filters row (full width, inline) ────────────────────
    f1, f2, f3, f4 = st.columns(4)
    with f1:
        min_duration = st.number_input("Min Duration (s)", 5, 600, 20, key="mr_mindur")
    with f2:
        max_duration = st.number_input("Max Duration (s)", 30, 3600, 300, key="mr_maxdur")
    with f3:
        max_results = st.slider("Max Results", 1, 25, 8, key="mr_maxres")
    with f4:
        top_n = st.slider("Reels per Video", 1, 5, 2, key="mr_topn")

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    # ── Trigger search ───────────────────────────────────────
    if search_btn:
        if not query.strip():
            st.error("Please enter a search query.")
        else:
            with st.spinner("Searching YouTube…"):
                try:
                    engine = DataAcquisitionEngine(
                        min_duration=int(min_duration),
                        max_duration=int(max_duration),
                        max_results=max_results,
                    )
                    ids        = engine.search_videos(query)
                    candidates = engine.fetch_and_filter(ids)
                    st.session_state.search_results = candidates
                except Exception as e:
                    st.error(f"Search failed: {e}")
                    st.session_state.search_results = []

    candidates = st.session_state.get("search_results", [])

    # ── Results (full width) ─────────────────────────────────
    if not candidates and not search_btn:
        st.markdown("""
        <div class="empty-state">
          <div class="icon">🔍</div>
          <div class="title">No search yet</div>
          <div class="sub">Enter a query above and click <strong>Search</strong> to discover videos.</div>
        </div>
        """, unsafe_allow_html=True)

    elif not candidates and search_btn:
        st.warning("No videos found matching your criteria. Try different filters.")

    else:
        st.markdown(f"""
        <div style="display:flex;align-items:center;gap:12px;margin-bottom:18px;">
          <span style="font-size:1.05rem;font-weight:700;color:#e2e8f0;">Results</span>
          <span class="stat-chip">📹 {len(candidates)} videos found</span>
        </div>
        """, unsafe_allow_html=True)

        for idx, c in enumerate(candidates):
            img_col, info_col, action_col = st.columns([1, 4, 1])

            with img_col:
                if c.thumbnail_url:
                    st.image(c.thumbnail_url, use_container_width=True)

            with info_col:
                st.markdown(f"**{c.title[:90]}{'…' if len(c.title)>90 else ''}**")
                st.markdown(f"""
                <div style="display:flex;flex-wrap:wrap;gap:4px;margin:8px 0 0;">
                  <span class="stat-chip">📺 {c.channel[:30]}</span>
                  <span class="stat-chip">👁 {c.view_count:,}</span>
                  <span class="stat-chip">⏱ {c.duration_seconds}s</span>
                  <span class="stat-chip">👍 {c.like_count:,}</span>
                </div>
                """, unsafe_allow_html=True)

            with action_col:
                def make_trigger(url=c.url, n=top_n):
                    st.session_state["trigger_url"]   = url
                    st.session_state["trigger_top_n"] = n

                st.button(
                    "⚡ Generate",
                    key=f"gen_{idx}_{c.video_id}",
                    on_click=make_trigger,
                    use_container_width=True,
                )

            st.markdown("<hr style='margin:10px 0;'>", unsafe_allow_html=True)

    # ── Handle pipeline trigger ──────────────────────────────
    if "trigger_url" in st.session_state:
        url = st.session_state.pop("trigger_url")
        t_n = st.session_state.pop("trigger_top_n", 2)
        st.info(f"⚡ Starting pipeline for: **{url}**")
        cmd = ["python", "main.py", "--url", url, "--top_n", str(t_n)]
        run_pipeline_with_progress(cmd)

    show_reels_gallery()


# ══════════════════════════════════════════════════════════════
#  MODE 2 — REEL SHORTS MAKER  (full-width)
# ══════════════════════════════════════════════════════════════
elif st.session_state.mode == "reel_maker":

    st.markdown("""
    <div class="dash-header">
      <h1>🎞️ Professional Clip Flow</h1>
      <p>Transform YouTube sources into high-engagement vertical assets using proprietary neural reframing.</p>
    </div>
    """, unsafe_allow_html=True)

    # ── URL input full width ─────────────────────────────────
    url_col, btn_col = st.columns([5, 1])
    with url_col:
        target_url = st.text_input(
            "YouTube URL",
            placeholder="https://youtube.com/watch?v=…",
            label_visibility="collapsed",
            key="rm_url",
        )
    with btn_col:
        generate_btn = st.button("🚀 Generate", use_container_width=True, key="rm_generate")

    # ── Settings row ─────────────────────────────────────────
    s1, s2, _ = st.columns([1, 1, 3])
    with s1:
        top_n = st.slider("Reels to generate", 1, 5, 2, key="rm_topn")

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

    # ── How it works — full width steps ─────────────────────
    st.markdown("""
    <div class="ctrl-card">
      <p class="ctrl-card-title">How It Works</p>
      <div class="step-row">
        <div class="step-icon">📥</div>
        <div><div class="step-title">Download</div><div class="step-desc">yt-dlp fetches the full video and audio track locally. No cloud storage.</div></div>
      </div>
      <div class="step-row">
        <div class="step-icon">🎙️</div>
        <div><div class="step-title">Transcribe</div><div class="step-desc">OpenAI Whisper (running 100% on-device) converts speech to timestamped text.</div></div>
      </div>
      <div class="step-row">
        <div class="step-icon">🧠</div>
        <div><div class="step-title">Score</div><div class="step-desc">BERT sentiment models + Librosa audio energy identify the most viral-worthy segments.</div></div>
      </div>
      <div class="step-row">
        <div class="step-icon">✂️</div>
        <div><div class="step-title">Reframe</div><div class="step-desc">Proprietary tracking crops and exports each clip to a 9:16 vertical format.</div></div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    reels_dir   = Path("outputs/reels")
    total_reels = len(list(reels_dir.glob("*.mp4"))) if reels_dir.exists() else 0
    m1, m2, m3 = st.columns(3)
    m1.metric("Reels Generated", total_reels)
    m2.metric("AI Engine", "Local / Offline")
    m3.metric("Output Format", "9:16 Vertical")

    # ── Run pipeline ─────────────────────────────────────────
    if generate_btn:
        if not target_url.strip():
            st.error("Please enter a valid YouTube URL.")
        else:
            st.markdown("---")
            cmd = ["python", "main.py", "--url", target_url.strip(), "--top_n", str(top_n)]
            run_pipeline_with_progress(cmd)

    show_reels_gallery()
