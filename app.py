import streamlit as st
from pathlib import Path
import joblib
import numpy as np

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="EmotionAI · Text Emotion Detector",
    page_icon="🧠",
    layout="centered",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* Gradient hero */
.hero {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    border-radius: 20px;
    padding: 2.5rem 2rem 2rem;
    text-align: center;
    margin-bottom: 2rem;
    box-shadow: 0 20px 60px rgba(102,126,234,0.35);
}
.hero h1 { color: #fff; font-size: 2.8rem; font-weight: 700; margin: 0; letter-spacing: -1px; }
.hero p  { color: rgba(255,255,255,0.85); font-size: 1.05rem; margin: 0.5rem 0 0; }

/* Emotion result card */
.emotion-card {
    background: linear-gradient(135deg, var(--c1), var(--c2));
    border-radius: 16px;
    padding: 1.6rem 2rem;
    text-align: center;
    margin: 1.5rem 0;
    box-shadow: 0 12px 40px rgba(0,0,0,0.18);
    animation: pop 0.4s cubic-bezier(.175,.885,.32,1.275);
}
@keyframes pop { from { transform: scale(0.8); opacity: 0; } to { transform: scale(1); opacity: 1; } }
.emotion-card .emoji { font-size: 4rem; }
.emotion-card .label { color: #fff; font-size: 1.8rem; font-weight: 700; margin-top: 0.3rem; text-transform: capitalize; }
.emotion-card .conf  { color: rgba(255,255,255,0.82); font-size: 0.95rem; margin-top: 0.2rem; }

/* Bar chart rows */
.bar-row { margin: 0.4rem 0; }
.bar-label { display: flex; justify-content: space-between; font-size: 0.82rem; color: #ccc; margin-bottom: 3px; }
.bar-outer { background: rgba(255,255,255,0.08); border-radius: 99px; height: 10px; overflow: hidden; }
.bar-inner { height: 100%; border-radius: 99px; transition: width 0.8s ease; }

/* Input area */
textarea { border-radius: 12px !important; font-family: 'Inter', sans-serif !important; }
button[kind="primary"] { border-radius: 12px !important; font-weight: 600 !important; }

/* Examples */
.example-chip {
    display: inline-block;
    background: rgba(102,126,234,0.15);
    border: 1px solid rgba(102,126,234,0.4);
    border-radius: 99px;
    padding: 0.25rem 0.9rem;
    font-size: 0.82rem;
    color: #a8b4ff;
    margin: 0.2rem;
    cursor: pointer;
}
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
EMOTIONS = {
    0: {"name": "sadness",  "emoji": "😢", "c1": "#2193b0", "c2": "#6dd5ed"},
    1: {"name": "anger",    "emoji": "😠", "c1": "#c0392b", "c2": "#e74c3c"},
    2: {"name": "love",     "emoji": "❤️", "c1": "#ee0979", "c2": "#ff6a00"},
    3: {"name": "surprise", "emoji": "😲", "c1": "#f7971e", "c2": "#ffd200"},
    4: {"name": "fear",     "emoji": "😨", "c1": "#4b1248", "c2": "#f10711"},
    5: {"name": "joy",      "emoji": "😄", "c1": "#11998e", "c2": "#38ef7d"},
}

BAR_COLORS = ["#6dd5ed", "#e74c3c", "#ff6a00", "#ffd200", "#f10711", "#38ef7d"]

EXAMPLES = [
    "I can't believe how amazing this day has been!",
    "I miss you so much, everything feels empty.",
    "I am furious at how they treated me!",
    "I love you more than words can express.",
    "There's something in the dark that scares me.",
    "Oh wow, I never saw that coming!",
]

# ── Load model ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    BASE_DIR = Path(__file__).resolve().parent
    return joblib.load(BASE_DIR / "emotion_model.pkl")

model = load_model()

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <h1>🧠 EmotionAI</h1>
  <p>Discover the emotion hidden inside your words — powered by machine learning.</p>
</div>
""", unsafe_allow_html=True)

# ── Try an example ────────────────────────────────────────────────────────────
st.markdown("**✨ Try an example:**")
cols = st.columns(3)
for i, ex in enumerate(EXAMPLES):
    if cols[i % 3].button(ex[:30] + "…", key=f"ex_{i}", use_container_width=True):
        st.session_state["input_text"] = ex

# ── Input ─────────────────────────────────────────────────────────────────────
text = st.text_area(
    "📝 Enter your text:",
    value=st.session_state.get("input_text", ""),
    placeholder="Type something like: I feel so happy today!",
    height=130,
    key="text_input",
)

analyze_btn = st.button("🔍 Analyze Emotion", type="primary", use_container_width=True)

# ── Analysis ──────────────────────────────────────────────────────────────────
if analyze_btn:
    if not text.strip():
        st.warning("⚠️ Please enter some text first.")
    else:
        with st.spinner("Analyzing…"):
            pred_idx   = int(model.predict([text])[0])
            probs      = model.predict_proba([text])[0]
            confidence = float(np.max(probs))
            emotion    = EMOTIONS[pred_idx]

        # Result card
        st.markdown(f"""
        <div class="emotion-card" style="--c1:{emotion['c1']};--c2:{emotion['c2']};">
          <div class="emoji">{emotion['emoji']}</div>
          <div class="label">{emotion['name']}</div>
          <div class="conf">Confidence: {confidence*100:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)

        # Probability breakdown
        st.markdown("#### 📊 Probability Breakdown")
        sorted_idx = np.argsort(probs)[::-1]
        for rank, idx in enumerate(sorted_idx):
            e    = EMOTIONS[idx]
            pct  = probs[idx] * 100
            col  = BAR_COLORS[idx]
            st.markdown(f"""
            <div class="bar-row">
              <div class="bar-label">
                <span>{e['emoji']} {e['name'].capitalize()}</span>
                <span>{pct:.1f}%</span>
              </div>
              <div class="bar-outer">
                <div class="bar-inner" style="width:{pct:.1f}%;background:{col};"></div>
              </div>
            </div>
            """, unsafe_allow_html=True)

        # Fun fact
        st.markdown("---")
        tips = {
            "joy":      "😄 Joy detected! Spread those good vibes.",
            "sadness":  "😢 It's okay to feel sad — emotions are valid.",
            "anger":    "😠 Anger can signal unmet needs. Take a breath.",
            "love":     "❤️ Love makes the world go round!",
            "fear":     "😨 Fear is the mind's alarm system — you're okay.",
            "surprise": "😲 Surprise! Life loves to catch us off guard.",
        }
        st.info(tips.get(emotion["name"], ""))

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<hr style="border:none;border-top:1px solid rgba(255,255,255,0.1);margin-top:3rem;">
<p style="text-align:center;color:#666;font-size:0.8rem;">
  Built with ❤️ using Streamlit · TF-IDF + Logistic Regression pipeline
</p>
""", unsafe_allow_html=True)