import streamlit as st
import streamlit.components.v1 as components
import time, html, re, math
from datetime import datetime
from collections import Counter

try:
    import torch
    from transformers import T5Tokenizer, T5ForConditionalGeneration
except ImportError as e:
    st.error(f"Missing required libraries: {e}")
    st.stop()

try:
    from langdetect import detect, LangDetectException
    LANG_AVAILABLE = True
except ImportError:
    LANG_AVAILABLE = False

try:
    from textblob import TextBlob
    SENTIMENT_AVAILABLE = True
except ImportError:
    SENTIMENT_AVAILABLE = False

# ── Page Config ──
st.set_page_config(
    page_title="SummarizeAI | Intelligent Text Summarization",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ══════════════════════════════════════════════════════════════════════
# CSS DESIGN SYSTEM
# ══════════════════════════════════════════════════════════════════════
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Sora:wght@300;400;500;600;700&family=Playfair+Display:wght@400;500;600;700;800&display=swap');

:root {
    --bg-primary: #0F1117;
    --bg-secondary: #1A1D2E;
    --bg-card: rgba(26, 29, 46, 0.7);
    --bg-glass: rgba(26, 29, 46, 0.45);
    --accent-blue: #4F8EF7;
    --accent-purple: #7C5CBF;
    --accent-teal: #00D4AA;
    --accent-pink: #E84393;
    --text-primary: #E8ECF4;
    --text-secondary: #8B95A8;
    --text-muted: #5A6478;
    --border-glass: rgba(79, 142, 247, 0.15);
    --border-glow: rgba(79, 142, 247, 0.3);
    --positive: #00D4AA;
    --neutral: #8B95A8;
    --critical: #FF6B6B;
    --warning: #F7B955;
    --shadow-lg: 0 20px 60px rgba(0,0,0,0.4);
    --shadow-glow: 0 0 30px rgba(79,142,247,0.15);
    --radius-lg: 20px;
    --radius-md: 14px;
    --radius-sm: 8px;
    --space-xs: 4px; --space-sm: 8px; --space-md: 16px; --space-lg: 24px; --space-xl: 32px; --space-2xl: 48px;
}

/* ── Base Reset ── */
html, body, [class*="css"] { font-family: 'Inter', 'Sora', sans-serif !important; }
.stApp { background: var(--bg-primary) !important; }
.stApp::before {
    content: ''; position: fixed; top: 0; left: 0; width: 100%; height: 100%; z-index: 0; pointer-events: none;
    background:
        radial-gradient(ellipse at 20% 20%, rgba(79,142,247,0.08) 0%, transparent 50%),
        radial-gradient(ellipse at 80% 80%, rgba(124,92,191,0.06) 0%, transparent 50%),
        radial-gradient(ellipse at 50% 50%, rgba(0,212,170,0.04) 0%, transparent 60%);
}
header[data-testid="stHeader"] { background: rgba(15,17,23,0.85) !important; backdrop-filter: blur(20px) !important; border-bottom: 1px solid var(--border-glass) !important; }
section[data-testid="stSidebar"] {
    background: rgba(15,17,23,0.92) !important; backdrop-filter: blur(24px) !important;
    border-right: 1px solid var(--border-glass) !important;
}
section[data-testid="stSidebar"] * { color: var(--text-primary) !important; }

/* ── Animations ── */
@keyframes fadeInUp { from { opacity:0; transform:translateY(20px); } to { opacity:1; transform:translateY(0); } }
@keyframes pulse { 0%,100% { opacity:1; } 50% { opacity:0.6; } }
@keyframes shimmer { 0% { background-position:-200% 0; } 100% { background-position:200% 0; } }
@keyframes gradientRotate { 0% { background-position:0% 50%; } 50% { background-position:100% 50%; } 100% { background-position:0% 50%; } }
@keyframes borderGlow {
    0%,100% { border-color: rgba(79,142,247,0.2); box-shadow: 0 0 15px rgba(79,142,247,0.05); }
    50% { border-color: rgba(79,142,247,0.5); box-shadow: 0 0 25px rgba(79,142,247,0.15); }
}
@keyframes progressBar { 0% { width:0%; } 100% { width:100%; } }

.fade-in { animation: fadeInUp 0.6s ease-out forwards; }
.fade-in-delay { animation: fadeInUp 0.6s ease-out 0.15s forwards; opacity: 0; }
.fade-in-delay2 { animation: fadeInUp 0.6s ease-out 0.3s forwards; opacity: 0; }

/* ── Progress Bar ── */
.progress-bar-container { position:fixed; top:0; left:0; width:100%; height:3px; z-index:9999; background:transparent; }
.progress-bar-fill {
    height:100%; border-radius:0 2px 2px 0;
    background: linear-gradient(90deg, var(--accent-blue), var(--accent-purple), var(--accent-teal));
    background-size: 200% 100%; animation: progressBar 3s ease-in-out, gradientRotate 2s ease infinite;
}

/* ── Title ── */
.app-title {
    font-family: 'Playfair Display', serif !important; font-size: clamp(2rem, 4vw, 3.2rem); font-weight: 800;
    background: linear-gradient(135deg, var(--accent-blue), var(--accent-purple), var(--accent-teal));
    background-size: 200% 200%; animation: gradientRotate 4s ease infinite;
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin-bottom: var(--space-xs); line-height:1.15;
}
.app-subtitle { font-family:'Sora',sans-serif; font-size:clamp(0.9rem,1.5vw,1.15rem); color:var(--text-secondary); font-weight:400; margin-bottom:var(--space-xl); }

/* ── Glass Card ── */
.glass-card {
    background: var(--bg-glass); backdrop-filter: blur(20px); -webkit-backdrop-filter: blur(20px);
    border: 1px solid var(--border-glass); border-radius: var(--radius-lg);
    padding: var(--space-lg); box-shadow: var(--shadow-lg), var(--shadow-glow);
    transition: all 0.3s cubic-bezier(0.4,0,0.2,1);
}
.glass-card:hover { transform: translateY(-2px); border-color: var(--border-glow); box-shadow: var(--shadow-lg), 0 0 40px rgba(79,142,247,0.2); }
.glass-card-active { animation: borderGlow 2.5s ease-in-out infinite; }

/* ── Panel Headers ── */
.panel-header {
    display:flex; align-items:center; justify-content:space-between; padding-bottom:var(--space-md);
    border-bottom:1px solid var(--border-glass); margin-bottom:var(--space-md);
}
.panel-title { font-family:'Playfair Display',serif; font-size:1.3rem; font-weight:700; color:var(--text-primary); display:flex; align-items:center; gap:var(--space-sm); }
.panel-badge {
    display:inline-flex; align-items:center; gap:4px; padding:4px 12px; border-radius:20px; font-size:0.75rem; font-weight:600;
    font-family:'Sora',sans-serif; background:rgba(79,142,247,0.12); color:var(--accent-blue); border:1px solid rgba(79,142,247,0.2);
}
.panel-badge.teal { background:rgba(0,212,170,0.12); color:var(--accent-teal); border-color:rgba(0,212,170,0.2); }
.panel-badge.purple { background:rgba(124,92,191,0.12); color:var(--accent-purple); border-color:rgba(124,92,191,0.2); }

/* ── Article Panel ── */
.article-content {
    font-size:clamp(0.95rem,1.2vw,1.05rem); line-height:1.8; color:var(--text-primary);
    white-space:pre-wrap; word-wrap:break-word; max-height:65vh; overflow-y:auto; padding:var(--space-md);
    scrollbar-width:thin; scrollbar-color:var(--accent-blue) transparent;
}
.article-content::-webkit-scrollbar { width:6px; }
.article-content::-webkit-scrollbar-thumb { background:var(--accent-blue); border-radius:3px; }
.article-content::-webkit-scrollbar-track { background:transparent; }

/* ── Summary Structures ── */
.exec-summary {
    background: linear-gradient(135deg, rgba(79,142,247,0.08), rgba(124,92,191,0.08));
    border: 1px solid rgba(79,142,247,0.2); border-radius: var(--radius-md);
    padding: var(--space-lg); margin-bottom: var(--space-lg); position:relative; overflow:hidden;
}
.exec-summary::before { content:''; position:absolute; top:0; left:0; width:4px; height:100%; background:linear-gradient(to bottom, var(--accent-blue), var(--accent-purple)); }
.exec-summary-title { font-family:'Playfair Display',serif; font-size:1.1rem; font-weight:700; color:var(--accent-blue); margin-bottom:var(--space-sm); display:flex; align-items:center; gap:8px; }
.exec-summary-text { font-size:1rem; line-height:1.75; color:var(--text-primary); padding-left:var(--space-md); }

.takeaway-chip {
    display:inline-flex; align-items:center; gap:6px; padding:8px 16px; border-radius:24px; margin:4px;
    font-size:0.85rem; font-weight:500; background:rgba(0,212,170,0.1); color:var(--accent-teal);
    border:1px solid rgba(0,212,170,0.2); transition:all 0.2s ease;
}
.takeaway-chip:hover { background:rgba(0,212,170,0.2); transform:scale(1.03); }

.insight-item {
    display:flex; gap:var(--space-md); padding:var(--space-md); border-radius:var(--radius-sm);
    background:rgba(255,255,255,0.02); margin-bottom:var(--space-sm); border-left:3px solid var(--accent-purple);
    transition: background 0.2s ease;
}
.insight-item:hover { background:rgba(255,255,255,0.05); }
.insight-num { font-family:'Sora',sans-serif; font-size:0.8rem; font-weight:700; color:var(--accent-purple); min-width:24px; height:24px; display:flex; align-items:center; justify-content:center; background:rgba(124,92,191,0.15); border-radius:6px; }
.insight-text { font-size:0.95rem; line-height:1.6; color:var(--text-primary); flex:1; }

/* ── Confidence Bar ── */
.confidence-bar-bg { width:100%; height:6px; background:rgba(255,255,255,0.05); border-radius:3px; margin-top:6px; overflow:hidden; }
.confidence-bar-fill { height:100%; border-radius:3px; transition:width 0.8s ease; }
.conf-high { background:linear-gradient(90deg, var(--accent-teal), #00E5BB); }
.conf-med { background:linear-gradient(90deg, var(--warning), #F7C96B); }
.conf-low { background:linear-gradient(90deg, var(--critical), #FF8A8A); }

/* ── Sentiment Tags ── */
.sentiment-tag {
    display:inline-flex; align-items:center; gap:4px; padding:3px 10px; border-radius:12px;
    font-size:0.72rem; font-weight:600; font-family:'Sora',sans-serif; text-transform:uppercase; letter-spacing:0.04em;
}
.sentiment-positive { background:rgba(0,212,170,0.12); color:var(--positive); border:1px solid rgba(0,212,170,0.25); }
.sentiment-neutral { background:rgba(139,149,168,0.12); color:var(--neutral); border:1px solid rgba(139,149,168,0.25); }
.sentiment-critical { background:rgba(255,107,107,0.12); color:var(--critical); border:1px solid rgba(255,107,107,0.25); }

/* ── Keyword Strip ── */
.keyword-strip { display:flex; flex-wrap:wrap; gap:6px; padding:var(--space-md) 0; }
.keyword-tag {
    padding:5px 14px; border-radius:16px; font-size:0.78rem; font-weight:500;
    background:rgba(79,142,247,0.08); color:var(--accent-blue); border:1px solid rgba(79,142,247,0.15);
    transition:all 0.2s ease; cursor:default;
}
.keyword-tag:hover { background:rgba(79,142,247,0.18); transform:translateY(-1px); }
.keyword-tag:nth-child(3n+2) { background:rgba(124,92,191,0.08); color:var(--accent-purple); border-color:rgba(124,92,191,0.15); }
.keyword-tag:nth-child(3n+2):hover { background:rgba(124,92,191,0.18); }
.keyword-tag:nth-child(3n) { background:rgba(0,212,170,0.08); color:var(--accent-teal); border-color:rgba(0,212,170,0.15); }
.keyword-tag:nth-child(3n):hover { background:rgba(0,212,170,0.18); }

/* ── Metric Cards ── */
.metric-card {
    background:var(--bg-glass); backdrop-filter:blur(16px); border:1px solid var(--border-glass);
    border-radius:var(--radius-md); padding:var(--space-lg) var(--space-md); text-align:center;
    transition:all 0.3s ease;
}
.metric-card:hover { transform:translateY(-4px); border-color:var(--border-glow); }
.metric-label { font-family:'Sora',sans-serif; font-size:0.72rem; color:var(--text-muted); font-weight:600; text-transform:uppercase; letter-spacing:0.06em; margin-bottom:6px; }
.metric-value {
    font-family:'Sora',sans-serif; font-size:1.6rem; font-weight:700;
    background:linear-gradient(135deg, var(--accent-blue), var(--accent-purple));
    -webkit-background-clip:text; -webkit-text-fill-color:transparent;
}

/* ── Skeleton Loader ── */
.skeleton { background:linear-gradient(90deg, rgba(255,255,255,0.03) 25%, rgba(255,255,255,0.08) 50%, rgba(255,255,255,0.03) 75%); background-size:200% 100%; animation:shimmer 1.5s infinite; border-radius:var(--radius-sm); }
.skeleton-line { height:14px; margin-bottom:10px; border-radius:6px; }
.skeleton-line.w75 { width:75%; } .skeleton-line.w90 { width:90%; } .skeleton-line.w60 { width:60%; } .skeleton-line.w100 { width:100%; }
.skeleton-card { height:120px; border-radius:var(--radius-md); margin-bottom:var(--space-md); }

/* ── Floating Toolbar ── */
.floating-toolbar {
    display:flex; flex-direction:column; gap:8px; padding:10px; border-radius:var(--radius-md);
    background:rgba(15,17,23,0.9); backdrop-filter:blur(20px); border:1px solid var(--border-glass);
    box-shadow:0 8px 32px rgba(0,0,0,0.4);
}
.toolbar-btn {
    width:42px; height:42px; border-radius:10px; border:1px solid var(--border-glass);
    background:var(--bg-glass); color:var(--text-primary); cursor:pointer;
    display:flex; align-items:center; justify-content:center; font-size:1.1rem;
    transition:all 0.25s ease; position:relative;
}
.toolbar-btn:hover { background:rgba(79,142,247,0.15); border-color:var(--accent-blue); transform:scale(1.08); }
.toolbar-btn .tooltip {
    position:absolute; left:52px; top:50%; transform:translateY(-50%); padding:6px 12px;
    background:var(--bg-secondary); color:var(--text-primary); font-size:0.75rem; font-weight:500;
    border-radius:6px; white-space:nowrap; opacity:0; pointer-events:none; transition:opacity 0.2s ease;
    border:1px solid var(--border-glass);
}
.toolbar-btn:hover .tooltip { opacity:1; }

/* ── Text Area ── */
.stTextArea textarea {
    background:var(--bg-glass) !important; color:var(--text-primary) !important; border:1px solid var(--border-glass) !important;
    border-radius:var(--radius-md) !important; font-size:1rem !important; line-height:1.7 !important;
    padding:var(--space-lg) !important; transition:all 0.3s ease !important;
}
.stTextArea textarea:focus { border-color:var(--accent-blue) !important; box-shadow:0 0 0 3px rgba(79,142,247,0.15) !important; }
.stTextArea textarea::placeholder { color:var(--text-muted) !important; }

/* ── Buttons ── */
.stButton > button {
    background:linear-gradient(135deg, var(--accent-blue), var(--accent-purple)) !important; color:white !important;
    border:none !important; border-radius:var(--radius-md) !important; padding:0.7rem 0 !important;
    font-size:1rem !important; font-weight:600 !important; font-family:'Sora',sans-serif !important;
    transition:all 0.3s cubic-bezier(0.4,0,0.2,1) !important;
    box-shadow:0 4px 20px rgba(79,142,247,0.3) !important;
}
.stButton > button:hover { transform:translateY(-2px) !important; box-shadow:0 8px 30px rgba(79,142,247,0.4) !important; }
.stButton > button:active { transform:translateY(0) !important; }

/* ── Download Button ── */
.stDownloadButton > button {
    background:rgba(0,212,170,0.1) !important; color:var(--accent-teal) !important;
    border:1px solid rgba(0,212,170,0.25) !important; border-radius:var(--radius-sm) !important;
    font-family:'Sora',sans-serif !important; font-weight:600 !important;
}
.stDownloadButton > button:hover { background:rgba(0,212,170,0.2) !important; }

/* ── Expander ── */
.streamlit-expanderHeader { background:var(--bg-glass) !important; border:1px solid var(--border-glass) !important; border-radius:var(--radius-sm) !important; color:var(--text-primary) !important; }
.streamlit-expanderContent { background:rgba(15,17,23,0.5) !important; border:1px solid var(--border-glass) !important; border-top:none !important; }

/* ── Slider ── */
.stSlider > div > div { color:var(--text-secondary) !important; }
.stSlider [data-testid="stThumbValue"] { color:var(--accent-blue) !important; }

/* ── Footer ── */
.app-footer {
    margin-top:var(--space-2xl); padding:var(--space-lg); text-align:center; font-size:0.8rem;
    color:var(--text-muted); border-top:1px solid var(--border-glass);
    font-family:'Sora',sans-serif;
}
.app-footer a { color:var(--accent-blue); text-decoration:none; }
.app-footer a:hover { text-decoration:underline; }

/* ── History ── */
.history-card {
    background:var(--bg-glass); backdrop-filter:blur(12px); border:1px solid var(--border-glass);
    border-radius:var(--radius-md); padding:var(--space-md); margin-bottom:var(--space-sm);
    transition:all 0.2s ease;
}
.history-card:hover { border-color:var(--border-glow); }
.history-meta { display:flex; justify-content:space-between; align-items:center; margin-bottom:8px; font-size:0.8rem; }
.history-time { color:var(--text-muted); font-family:'Sora',sans-serif; }
.history-badge { padding:3px 10px; border-radius:12px; font-size:0.72rem; font-weight:600; background:rgba(0,212,170,0.12); color:var(--accent-teal); }
.history-snippet { color:var(--text-secondary); font-size:0.85rem; font-style:italic; border-left:2px solid var(--accent-purple); padding-left:12px; margin:8px 0; }
.history-body { color:var(--text-primary); font-size:0.9rem; line-height:1.5; }

/* ── Section Divider ── */
.section-divider { height:1px; background:linear-gradient(90deg, transparent, var(--border-glass), transparent); margin:var(--space-xl) 0; }

/* ── Responsive ── */
@media (max-width: 768px) {
    .app-title { font-size:1.8rem !important; }
    .metric-card { padding:var(--space-md) var(--space-sm); }
    .metric-value { font-size:1.2rem; }
}

/* Hide Streamlit branding */
#MainMenu { visibility:hidden; } footer { visibility:hidden; }
div[data-testid="stToolbar"] { visibility:hidden; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════
# MODEL LOADING
# ══════════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner=False)
def load_model():
    tokenizer = T5Tokenizer.from_pretrained("t5-small")
    model = T5ForConditionalGeneration.from_pretrained("t5-small")
    return tokenizer, model

if "model_loaded" not in st.session_state:
    with st.spinner("🤖 Warming up AI Engine..."):
        tokenizer, model = load_model()
        st.session_state["model_loaded"] = True
else:
    tokenizer, model = load_model()

# ══════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════════════
STOP_WORDS = set("the a an and or but in on at to for of is it this that with as by from be are was were been has have had do does did will would could should may might shall can".split())

def count_words(text):
    return len(str(text).split())

def estimate_read_time(text):
    return max(1, round(count_words(text) / 200))

def calculate_compression(original, summary):
    o = count_words(original)
    return round(((o - count_words(summary)) / o) * 100) if o else 0

def estimate_tokens(text):
    return max(1, int(len(text.split()) * 1.3))

def detect_language(text):
    if not LANG_AVAILABLE:
        return "Unknown"
    try:
        code = detect(text[:500])
        lang_map = {"en":"English","es":"Spanish","fr":"French","de":"German","it":"Italian","pt":"Portuguese","nl":"Dutch","ru":"Russian","zh-cn":"Chinese","zh-tw":"Chinese","ja":"Japanese","ko":"Korean","ar":"Arabic","hi":"Hindi","ur":"Urdu","tr":"Turkish","pl":"Polish","sv":"Swedish","da":"Danish","fi":"Finnish","no":"Norwegian"}
        return lang_map.get(code, code.upper())
    except LangDetectException:
        return "Unknown"

def extract_keywords(text, n=12):
    words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
    filtered = [w for w in words if w not in STOP_WORDS]
    freq = Counter(filtered)
    return [w for w, _ in freq.most_common(n)]

def analyze_sentiment(text):
    if not SENTIMENT_AVAILABLE:
        return "neutral", 0.0
    polarity = TextBlob(text).sentiment.polarity
    if polarity > 0.1:
        return "positive", polarity
    elif polarity < -0.1:
        return "critical", polarity
    return "neutral", polarity

def summarize_text(text, _model, _tokenizer, settings):
    try:
        inputs = _tokenizer("summarize: " + text.strip(), return_tensors="pt", max_length=512, truncation=True)
        ids = _model.generate(inputs["input_ids"], max_length=settings["max_length"], min_length=settings["min_length"], num_beams=settings["num_beams"], length_penalty=settings["length_penalty"], early_stopping=True)
        return _tokenizer.decode(ids[0], skip_special_tokens=True), None
    except Exception as e:
        return None, str(e)

def format_structured_summary(summary_text, original_text=""):
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', summary_text) if s.strip()]
    executive = sentences[0] if sentences else summary_text
    takeaways = sentences[1:4] if len(sentences) > 1 else []
    detailed = sentences[4:] if len(sentences) > 4 else []
    scores = []
    if original_text:
        orig_lower = original_text.lower()
        for s in sentences:
            words = set(s.lower().split())
            overlap = sum(1 for w in words if w in orig_lower) / max(len(words), 1)
            scores.append(min(round(overlap * 100), 98))
    else:
        scores = [85] * len(sentences)
    return {"executive": executive, "takeaways": takeaways, "detailed": detailed, "scores": scores, "all_sentences": sentences}

def skeleton_html():
    return """<div class="glass-card" style="margin-top:16px;">
        <div class="skeleton skeleton-line w75"></div>
        <div class="skeleton skeleton-line w90"></div>
        <div class="skeleton skeleton-line w60"></div>
        <div style="height:16px;"></div>
        <div class="skeleton skeleton-card"></div>
        <div class="skeleton skeleton-line w100"></div>
        <div class="skeleton skeleton-line w75"></div>
    </div>"""

def format_timestamp():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# ══════════════════════════════════════════════════════════════════════
# SESSION STATE
# ══════════════════════════════════════════════════════════════════════
for key, default in [("history", []), ("is_processing", False), ("current_summary", None), ("current_stats", None), ("show_full_analysis", False)]:
    if key not in st.session_state:
        st.session_state[key] = default

# ══════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""<div style="display:flex;align-items:center;gap:12px;margin-bottom:8px;">
        <div style="width:40px;height:40px;border-radius:12px;background:linear-gradient(135deg,#4F8EF7,#7C5CBF);display:flex;align-items:center;justify-content:center;font-size:1.3rem;">✨</div>
        <div><div style="font-family:'Playfair Display',serif;font-size:1.4rem;font-weight:700;color:#E8ECF4;">SummarizeAI</div>
        <div style="font-size:0.72rem;color:#5A6478;font-family:'Sora',sans-serif;">INTELLIGENT ENGINE v2.0</div></div>
    </div>""", unsafe_allow_html=True)
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    st.markdown("#### ⚙️ Summary Tuning")
    max_length = st.slider("Max Summary Length", 50, 300, 150)
    min_length = st.slider("Min Summary Length", 20, 100, 40)
    num_beams = st.slider("Beam Search Width", 1, 8, 4, help="Higher = better quality but slower")
    length_penalty = st.slider("Length Penalty", 0.5, 3.0, 2.0, 0.1, help="> 1.0 → longer summaries")
    settings = {"max_length": max_length, "min_length": min_length, "num_beams": num_beams, "length_penalty": length_penalty}

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.markdown("#### 📥 Input Mode")
    input_mode = st.radio("Select input method", ["📋 Paste Text", "📁 Upload .txt File"], label_visibility="collapsed")

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.markdown("#### 📚 Session History")
    if not st.session_state["history"]:
        st.caption("No summaries yet. Results will appear here.")
    else:
        if st.button("🗑️ Clear All", use_container_width=True):
            st.session_state["history"] = []
            st.rerun()
        for h in st.session_state["history"]:
            safe_snip = html.escape(h["original_snippet"])
            safe_summ = html.escape(h["summary"][:100])
            st.markdown(f"""<div class="history-card">
                <div class="history-meta"><span class="history-time">🕒 {h['timestamp']}</span><span class="history-badge">-{h['compression']}%</span></div>
                <div class="history-snippet">"{safe_snip}"</div>
                <div class="history-body">{safe_summ}{'…' if len(h['summary'])>100 else ''}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    with st.expander("ℹ️ About Engine"):
        st.markdown("Powered by **T5 encoder-decoder** (Google).  \n**Model:** `T5-Small` · **Params:** 60M")
    st.markdown("""<div style="margin-top:24px;padding:10px 16px;border-radius:10px;background:rgba(0,212,170,0.08);border:1px solid rgba(0,212,170,0.2);display:flex;align-items:center;gap:8px;">
        <div style="width:8px;height:8px;border-radius:50%;background:#00D4AA;animation:pulse 2s infinite;"></div>
        <span style="font-size:0.82rem;font-weight:600;color:#00D4AA;font-family:'Sora',sans-serif;">Engine Active & Ready</span>
    </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════
# MAIN CONTENT — HEADER
# ══════════════════════════════════════════════════════════════════════
if st.session_state["is_processing"]:
    st.markdown('<div class="progress-bar-container"><div class="progress-bar-fill"></div></div>', unsafe_allow_html=True)

st.markdown('<div class="fade-in"><h1 class="app-title">Intelligent Summarization</h1></div>', unsafe_allow_html=True)
st.markdown('<div class="fade-in-delay"><p class="app-subtitle">Distill articles, research papers, and documents into structured, actionable insights — powered by T5 neural architecture.</p></div>', unsafe_allow_html=True)

# ── Input Collection ──
user_input = ""
if input_mode == "📋 Paste Text":
    user_input = st.text_area("Paste your text", height=180, placeholder="Paste your article, news, research paper, or any long-form text here...", disabled=st.session_state["is_processing"], label_visibility="collapsed")
else:
    uploaded = st.file_uploader("Upload a .txt file", type=["txt"], disabled=st.session_state["is_processing"])
    if uploaded:
        user_input = uploaded.getvalue().decode("utf-8").strip()
        if user_input:
            st.success(f"📄 Loaded: {uploaded.name}")

user_input = user_input.strip() if user_input else ""

# ── Generate Button ──
_, btn_c, _ = st.columns([1.5, 2, 1.5])
with btn_c:
    generate_clicked = st.button("✨ Generate Summary", disabled=st.session_state["is_processing"] or not user_input, use_container_width=True)

if generate_clicked and user_input:
    st.session_state["is_processing"] = True
    with st.spinner("🧠 Analyzing content and generating summary..."):
        start_t = time.time()
        summary_result, error = summarize_text(user_input, model, tokenizer, settings)
        elapsed = round(time.time() - start_t, 2)
    st.session_state["is_processing"] = False
    if error:
        st.error(f"❌ Error: {error}")
    elif summary_result:
        snippet = user_input[:150] + "…" if len(user_input) > 150 else user_input
        st.session_state["current_summary"] = summary_result
        st.session_state["current_stats"] = {
            "orig_words": count_words(user_input), "summ_words": count_words(summary_result),
            "compression": calculate_compression(user_input, summary_result), "time_taken": elapsed,
            "original_snippet": snippet, "original_text": user_input,
            "tokens": estimate_tokens(user_input), "language": detect_language(user_input),
        }
        st.session_state["history"].insert(0, {"timestamp": format_timestamp(), "original_snippet": str(snippet), "summary": str(summary_result), "compression": st.session_state["current_stats"]["compression"]})
        st.session_state["history"] = st.session_state["history"][:10]
        st.rerun()

# ══════════════════════════════════════════════════════════════════════
# TWO-COLUMN LAYOUT — ARTICLE + SUMMARY
# ══════════════════════════════════════════════════════════════════════
if st.session_state["current_summary"] is not None:
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    stats = st.session_state["current_stats"]
    summary_text = st.session_state["current_summary"]
    original_text = stats.get("original_text", "")

    # ── Metrics Row ──
    st.markdown('<div class="fade-in">', unsafe_allow_html=True)
    mc1, mc2, mc3, mc4, mc5, mc6 = st.columns(6)
    metrics = [
        ("Original Words", f"{stats['orig_words']:,}"),
        ("Summary Words", f"{stats['summ_words']:,}"),
        ("Compression", f"{stats['compression']}%"),
        ("Processing", f"{stats['time_taken']}s"),
        ("Tokens Used", f"{stats.get('tokens','—'):,}" if isinstance(stats.get('tokens'), int) else "—"),
        ("Language", stats.get("language", "—")),
    ]
    for col, (label, value) in zip([mc1,mc2,mc3,mc4,mc5,mc6], metrics):
        with col:
            st.markdown(f'<div class="metric-card"><div class="metric-label">{label}</div><div class="metric-value">{value}</div></div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div style="height:24px;"></div>', unsafe_allow_html=True)

    # ── Two Column Panels ──
    art_col, toolbar_col, sum_col = st.columns([5, 0.6, 5])

    # ──── ARTICLE PANEL (LEFT) ────
    with art_col:
        word_cnt = stats["orig_words"]
        read_t = estimate_read_time(original_text)
        lang = stats.get("language", "—")
        st.markdown(f"""<div class="glass-card glass-card-active fade-in">
            <div class="panel-header">
                <div class="panel-title">📄 Article Content</div>
                <div style="display:flex;gap:8px;flex-wrap:wrap;">
                    <span class="panel-badge">{word_cnt:,} words</span>
                    <span class="panel-badge teal">~{read_t} min read</span>
                    <span class="panel-badge purple">{lang}</span>
                </div>
            </div>
            <div class="article-content">{html.escape(original_text)}</div>
        </div>""", unsafe_allow_html=True)

    # ──── FLOATING TOOLBAR (CENTER) ────
    with toolbar_col:
        st.markdown('<div style="height:60px;"></div>', unsafe_allow_html=True)
        st.markdown("""<div class="floating-toolbar">
            <div class="toolbar-btn" onclick="navigator.clipboard.writeText(document.querySelector('.exec-summary-text')?.innerText||'')">📋<div class="tooltip">Copy Summary</div></div>
            <div class="toolbar-btn">🔄<div class="tooltip">Regenerate</div></div>
            <div class="toolbar-btn">🖍️<div class="tooltip">Highlight</div></div>
            <div class="toolbar-btn">📥<div class="tooltip">Export</div></div>
        </div>""", unsafe_allow_html=True)

    # ──── SUMMARY PANEL (RIGHT) ────
    with sum_col:
        structured = format_structured_summary(summary_text, original_text)
        sentiment, polarity = analyze_sentiment(summary_text)
        keywords = extract_keywords(original_text)
        sent_class = f"sentiment-{sentiment}"
        sent_icon = {"positive": "🟢", "neutral": "⚪", "critical": "🔴"}.get(sentiment, "⚪")

        # Executive Summary Card
        exec_html = f"""<div class="glass-card fade-in">
            <div class="panel-header">
                <div class="panel-title">🧠 AI Summary</div>
                <span class="sentiment-tag {sent_class}">{sent_icon} {sentiment.upper()}</span>
            </div>
            <div class="exec-summary">
                <div class="exec-summary-title">📌 Executive Summary</div>
                <div class="exec-summary-text">{html.escape(structured['executive'])}</div>
                <div class="confidence-bar-bg" style="margin-top:12px;">
                    <div class="confidence-bar-fill {'conf-high' if structured['scores'][0] > 70 else 'conf-med' if structured['scores'][0] > 40 else 'conf-low'}" style="width:{structured['scores'][0]}%;"></div>
                </div>
                <div style="font-size:0.7rem;color:var(--text-muted);margin-top:4px;text-align:right;">Relevance: {structured['scores'][0]}%</div>
            </div>"""

        # Key Takeaways
        if structured["takeaways"]:
            exec_html += '<div style="margin-bottom:16px;"><div style="font-family:\'Sora\',sans-serif;font-size:0.85rem;font-weight:600;color:var(--text-secondary);margin-bottom:8px;text-transform:uppercase;letter-spacing:0.05em;">💡 Key Takeaways</div><div style="display:flex;flex-wrap:wrap;">'
            for t in structured["takeaways"]:
                exec_html += f'<div class="takeaway-chip">• {html.escape(t)}</div>'
            exec_html += '</div></div>'

        # Detailed Breakdown
        if structured["detailed"]:
            exec_html += '<div style="margin-bottom:16px;"><div style="font-family:\'Sora\',sans-serif;font-size:0.85rem;font-weight:600;color:var(--text-secondary);margin-bottom:8px;text-transform:uppercase;letter-spacing:0.05em;">📊 Detailed Breakdown</div>'
            for i, d in enumerate(structured["detailed"]):
                sc_idx = min(i + len(structured["takeaways"]) + 1, len(structured["scores"]) - 1)
                score = structured["scores"][sc_idx] if sc_idx < len(structured["scores"]) else 75
                conf_class = "conf-high" if score > 70 else ("conf-med" if score > 40 else "conf-low")
                exec_html += f"""<div class="insight-item">
                    <div class="insight-num">{i+1}</div>
                    <div class="insight-text">{html.escape(d)}
                        <div class="confidence-bar-bg"><div class="confidence-bar-fill {conf_class}" style="width:{score}%;"></div></div>
                    </div>
                </div>"""
            exec_html += '</div>'

        # Keyword Strip
        if keywords:
            exec_html += '<div style="margin-top:8px;"><div style="font-family:\'Sora\',sans-serif;font-size:0.85rem;font-weight:600;color:var(--text-secondary);margin-bottom:8px;text-transform:uppercase;letter-spacing:0.05em;">🏷️ Extracted Keywords</div><div class="keyword-strip">'
            for kw in keywords:
                exec_html += f'<span class="keyword-tag">{html.escape(kw)}</span>'
            exec_html += '</div></div>'

        exec_html += '</div>'  # close glass-card
        st.markdown(exec_html, unsafe_allow_html=True)

    # ── Full Analysis Expander ──
    st.markdown('<div style="height:24px;"></div>', unsafe_allow_html=True)
    with st.expander("📖 Full Analysis & Raw Output", expanded=False):
        st.markdown(f"""<div class="glass-card">
            <div style="font-family:'Playfair Display',serif;font-size:1.1rem;font-weight:700;color:var(--accent-blue);margin-bottom:12px;">Complete Generated Summary</div>
            <div style="font-size:1rem;line-height:1.8;color:var(--text-primary);white-space:pre-wrap;">{html.escape(summary_text)}</div>
        </div>""", unsafe_allow_html=True)
        st.markdown(f"""<div class="glass-card" style="margin-top:16px;">
            <div style="font-family:'Playfair Display',serif;font-size:1.1rem;font-weight:700;color:var(--accent-purple);margin-bottom:12px;">Sentiment Analysis Details</div>
            <div style="display:flex;gap:24px;flex-wrap:wrap;">
                <div><span style="color:var(--text-muted);font-size:0.85rem;">Polarity Score:</span> <span style="color:var(--text-primary);font-weight:600;">{polarity:.3f}</span></div>
                <div><span style="color:var(--text-muted);font-size:0.85rem;">Classification:</span> <span class="sentiment-tag {sent_class}">{sent_icon} {sentiment.upper()}</span></div>
                <div><span style="color:var(--text-muted);font-size:0.85rem;">Model:</span> <span style="color:var(--text-primary);font-weight:600;">T5-Small (60M params)</span></div>
            </div>
        </div>""", unsafe_allow_html=True)

    # ── Action Row ──
    st.markdown('<div style="height:16px;"></div>', unsafe_allow_html=True)
    dl_col, code_col, reset_col = st.columns([1, 1, 1])
    with dl_col:
        download_content = f"=== SummarizeAI Output ===\nGenerated: {format_timestamp()}\n\n[ METRICS ]\nOriginal Words: {stats['orig_words']}\nSummary Words: {stats['summ_words']}\nCompression: {stats['compression']}%\nLanguage: {stats.get('language','—')}\nTokens: {stats.get('tokens','—')}\n\n[ SUMMARY ]\n{summary_text}\n\n[ KEYWORDS ]\n{', '.join(keywords)}\n"
        st.download_button("⬇️ Download Summary", download_content, f"summary_{datetime.now().strftime('%Y%m%d_%H%M')}.txt", "text/plain", use_container_width=True)
    with code_col:
        st.code(summary_text, language="text")
    with reset_col:
        if st.button("🔄 New Summarization", use_container_width=True):
            st.session_state["current_summary"] = None
            st.session_state["current_stats"] = None
            st.rerun()

# ══════════════════════════════════════════════════════════════════════
# FOOTER
# ══════════════════════════════════════════════════════════════════════
st.markdown(f"""<div class="app-footer">
    <div style="margin-bottom:8px;">
        <span style="font-weight:600;">SummarizeAI</span> v2.0 · Powered by <a href="https://huggingface.co/t5-small" target="_blank">T5-Small</a> · Built with <a href="https://streamlit.io" target="_blank">Streamlit</a>
    </div>
    <div>Processed: {format_timestamp()} · Model: T5-Small (60M params) · Max Tokens: 512</div>
</div>""", unsafe_allow_html=True)
