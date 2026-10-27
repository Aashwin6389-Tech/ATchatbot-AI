
# AiChatbot2.py
# ATchatbot.AI — Created by Aashwin Tripathi

import ast
import hashlib
import io
import json
import operator
import os
import re
import uuid
from datetime import datetime
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st
from groq import Groq
from pypdf import PdfReader
from gtts import gTTS

APP_NAME = "ATchatbot.AI"
MODEL_NAME = "openai/gpt-oss-20b"
HISTORY_DIR = Path("chat_history")
HISTORY_DIR.mkdir(exist_ok=True)

st.set_page_config(
    page_title=APP_NAME,
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded",
)

SYSTEM_PROMPT = """
You are ATchatbot.AI, an AI assistant created by Aashwin Tripathi.

If asked who you are or who created you, reply exactly:
I am Chatbot for help created by Aashwin Tripathi

Be helpful, accurate, friendly, and clear.
Explain coding questions step by step when useful.
Never invent facts or sources.
"""

# -------------------- SESSION STATE --------------------

if "show_recent_sidebar" not in st.session_state:
    st.session_state.show_recent_sidebar = True

if "chat_id" not in st.session_state:
    st.session_state.chat_id = str(uuid.uuid4())

if "chat_title" not in st.session_state:
    st.session_state.chat_title = "New chat"

if "messages" not in st.session_state:
    st.session_state.messages = []

if "pending_prompt" not in st.session_state:
    st.session_state.pending_prompt = None

if "last_voice_hash" not in st.session_state:
    st.session_state.last_voice_hash = None

if "voice_transcript" not in st.session_state:
    st.session_state.voice_transcript = ""

# -------------------- PROFESSIONAL THEME --------------------
# Ivory + charcoal + copper. No blue background.

CUSTOM_CSS = r"""
<style>
:root {
    color-scheme: light;
    --bg: #F5F2EB;
    --sidebar: #EAE6DD;
    --surface: #FFFFFF;
    --border: #DED7CB;
    --ink: #292A27;
    --muted: #706D65;
    --accent: #B86635;
    --accent-hover: #98502A;
}

.stApp,
[data-testid="stAppViewContainer"],
[data-testid="stMain"],
[data-testid="stMainBlockContainer"] {
    background: var(--bg) !important;
    color: var(--ink) !important;
}

.stApp {
    font-family: Inter, "Segoe UI", Arial, sans-serif;
}

[data-testid="stHeader"] {
    background: rgba(245,242,235,.97) !important;
}

[data-testid="stSidebar"] {
    background: var(--sidebar) !important;
    border-right: 1px solid var(--border) !important;
}

[data-testid="stSidebar"] * {
    color: var(--ink);
}

.stMarkdown,
.stMarkdown p,
.stMarkdown li,
label,
[data-testid="stCaptionContainer"] p {
    color: var(--ink);
}

a {
    color: var(--accent) !important;
}

/* Main title card */
.hero {
    padding: 26px 30px;
    margin-bottom: 22px;
    background: #FFFFFF;
    border: 1px solid var(--border);
    border-top: 4px solid var(--accent);
    border-radius: 16px;
    box-shadow: 0 5px 18px rgba(41,42,39,.05);
}

.hero h1 {
    margin: 0 0 8px;
    color: var(--ink) !important;
    font-size: clamp(1.8rem, 3vw, 2.5rem);
    font-weight: 760;
    letter-spacing: -1px;
}

.hero p {
    color: var(--muted) !important;
    margin: 0;
    line-height: 1.7;
}

/* Tabs */
[data-testid="stTabs"] [data-baseweb="tab-list"] {
    gap: 6px;
    background: #ECE7DE;
    padding: 6px;
    border-radius: 11px;
    border: 1px solid var(--border);
}

[data-testid="stTabs"] button[data-baseweb="tab"] {
    color: #55534C !important;
    background: transparent !important;
    border-radius: 8px;
    border-bottom: none !important;
    font-weight: 600;
}

[data-testid="stTabs"] button[data-baseweb="tab"][aria-selected="true"] {
    color: #FFFFFF !important;
    background: #292A27 !important;
}

/* Buttons */
div.stButton > button,
div[data-testid="stDownloadButton"] button {
    background: var(--accent) !important;
    color: #FFFFFF !important;
    border: 1px solid var(--accent) !important;
    border-radius: 9px !important;
    min-height: 2.4rem;
    font-weight: 600;
}

div.stButton > button:hover,
div[data-testid="stDownloadButton"] button:hover {
    background: var(--accent-hover) !important;
    border-color: var(--accent-hover) !important;
}

/* Sidebar history buttons */
[data-testid="stSidebar"] div.stButton > button {
    background: #F7F4EE !important;
    color: var(--ink) !important;
    border-color: #D7D0C4 !important;
}

[data-testid="stSidebar"] div.stButton > button:hover {
    background: #E4D8C9 !important;
}

/* Input fields */
input,
textarea,
[data-baseweb="input"] input,
[data-baseweb="textarea"] textarea {
    color: var(--ink) !important;
    background: #FFFFFF !important;
    border-color: #D1C9BD !important;
    caret-color: var(--accent) !important;
}

input::placeholder,
textarea::placeholder {
    color: #777168 !important;
    opacity: 1 !important;
}

/* Dropdowns and uploader */
[data-baseweb="select"] > div,
[data-testid="stFileUploader"] section {
    background: #FFFFFF !important;
    color: var(--ink) !important;
    border-color: #D1C9BD !important;
}

/* Chat message cards */
[data-testid="stChatMessage"] {
    background: #FFFFFF !important;
    color: var(--ink) !important;
    border: 1px solid var(--border) !important;
    border-radius: 13px !important;
    padding: 15px 17px !important;
    box-shadow: 0 2px 8px rgba(41,42,39,.035);
    margin-bottom: 12px !important;
}

[data-testid="stChatMessage"] p,
[data-testid="stChatMessage"] li,
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] {
    color: var(--ink) !important;
    line-height: 1.75;
}

/* Data tables and metrics */
[data-testid="stDataFrame"],
[data-testid="stTable"] {
    border: 1px solid var(--border);
    border-radius: 10px;
    overflow: hidden;
}

[data-testid="stMetric"] {
    background: #FFFFFF !important;
    color: var(--ink) !important;
    border: 1px solid var(--border);
    border-radius: 11px;
    padding: 13px;
}

[data-testid="stMetricLabel"],
[data-testid="stMetricValue"] {
    color: var(--ink) !important;
}

[data-testid="stExpander"] {
    background: #FFFFFF !important;
    border-radius: 10px;
    border-color: var(--border) !important;
}

[data-testid="stExpander"] summary,
[data-testid="stExpander"] summary p {
    color: var(--ink) !important;
}

hr {
    border-color: var(--border) !important;
}

/* Fixed, compact message box */
[data-testid="stMain"] {
    padding-bottom: 110px !important;
}

[data-testid="stChatInput"] {
    position: fixed !important;
    left: var(--composer-left, 365px) !important;
    bottom: 12px !important;
    width: var(--composer-width, min(900px, calc(100vw - 390px))) !important;
    max-width: 900px !important;
    z-index: 1001 !important;
    background: #FFFFFF !important;
    border: 1px solid #D6CFC3 !important;
    border-radius: 12px !important;
    box-shadow: 0 4px 16px rgba(35,35,30,.09) !important;
}

[data-testid="stChatInput"] > div,
[data-testid="stChatInput"] textarea {
    background: #FFFFFF !important;
    color: var(--ink) !important;
    -webkit-text-fill-color: var(--ink) !important;
    border-color: #D6CFC3 !important;
    font-size: 15px !important;
}

/* Voice recorder remains in its own tab and won't cover answers */
[data-testid="stAudioInput"] {
    position: relative !important;
    left: auto !important;
    bottom: auto !important;
    width: 100% !important;
    max-width: 100% !important;
    z-index: auto !important;
    padding: 8px !important;
    background: #F5F2EB !important;
    color: var(--ink) !important;
    border: 1px solid var(--border) !important;
    border-radius: 11px !important;
    box-shadow: none !important;
}

[data-testid="stAudioInput"] label,
[data-testid="stAudioInput"] p,
[data-testid="stAudioInput"] span {
    color: var(--ink) !important;
}

/* Sidebar hidden */
body.sidebar-hidden {
    --composer-left: max(16px, calc((100vw - 900px) / 2));
    --composer-width: min(900px, calc(100vw - 32px));
}

@media (max-width: 700px) {
    .hero {
        padding: 20px;
        border-radius: 13px;
    }

    [data-testid="stChatMessage"] {
        padding: 12px !important;
    }

    [data-testid="stMain"] {
        padding-bottom: 105px !important;
    }

    [data-testid="stChatInput"] {
        left: 10px !important;
        bottom: 8px !important;
        width: calc(100vw - 20px) !important;
    }
}
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# -------------------- GROQ API --------------------

def get_api_key():
    try:
        key = st.secrets.get("GROQ_API_KEY", "")
    except Exception:
        key = ""
    return key or os.getenv("GROQ_API_KEY", "")


api_key = get_api_key()

if not api_key:
    st.error(
        "Groq API key nahi mili. Streamlit Cloud → Settings → Secrets "
        "mein GROQ_API_KEY set karo."
    )
    st.code('GROQ_API_KEY = "YOUR_GROQ_API_KEY_HERE"', language="toml")
    st.stop()

client = Groq(api_key=api_key)


# -------------------- SAFE CALCULATOR --------------------

_ALLOWED_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def safe_calculate(expression):
    """Evaluate basic arithmetic without eval()."""
    tree = ast.parse(expression, mode="eval")

    def visit(node):
        if isinstance(node, ast.Expression):
            return visit(node.body)

        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value

        if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_OPERATORS:
            left = visit(node.left)
            right = visit(node.right)

            if isinstance(node.op, ast.Pow) and abs(right) > 100:
                raise ValueError("Exponent bahut bada hai.")

            return _ALLOWED_OPERATORS[type(node.op)](left, right)

        if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_OPERATORS:
            return _ALLOWED_OPERATORS[type(node.op)](visit(node.operand))

        raise ValueError("Unsupported mathematical expression.")

    result = visit(tree)

    if isinstance(result, complex):
        raise ValueError("Complex numbers supported nahi hain.")

    return result


def extract_calculation(text):
    candidate = text.strip().replace(",", "")
    candidate = re.sub(
        r"^(calculate|compute|solve)\s+",
        "",
        candidate,
        flags=re.I,
    )
    candidate = candidate.rstrip(" ?=")

    if re.fullmatch(r"[\d\s.+\-*/()%]+", candidate):
        return candidate

    return None


# -------------------- CHAT HISTORY --------------------

def history_path(chat_id):
    return HISTORY_DIR / f"{chat_id}.json"


def save_chat(chat_id, title, messages):
    data = {
        "id": chat_id,
        "title": title,
        "updated": datetime.now().isoformat(timespec="seconds"),
        "messages": messages,
    }

    history_path(chat_id).write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def load_all_chats():
    chats = []

    for file in HISTORY_DIR.glob("*.json"):
        try:
            data = json.loads(file.read_text(encoding="utf-8"))
            chats.append(data)
        except (OSError, json.JSONDecodeError):
            continue

    return sorted(
        chats,
        key=lambda item: item.get("updated", ""),
        reverse=True,
    )


def start_new_chat():
    st.session_state.chat_id = str(uuid.uuid4())
    st.session_state.chat_title = "New chat"
    st.session_state.messages = []
    st.session_state.pending_prompt = None
    st.session_state.last_voice_hash = None
    st.session_state.voice_transcript = ""


# -------------------- PDF PROCESSING --------------------

def extract_pdf_text(file_bytes):
    reader = PdfReader(io.BytesIO(file_bytes))
    return "\n\n".join(
        page.extract_text() or "" for page in reader.pages
    ).strip()


# -------------------- GROQ CHAT --------------------

def ask_groq(messages, enable_web_search=False):
    request = {
        "model": MODEL_NAME,
        "messages": messages,
        "temperature": 0.4,
        "max_tokens": 3000,
    }

    # Try web search only if the current API supports it.
    if enable_web_search:
        try:
            return client.chat.completions.create(
                **request,
                tools=[{"type": "browser_search"}],
                tool_choice="auto",
            )
        except Exception:
            pass

    return client.chat.completions.create(**request)


def get_answer(user_prompt, context=None, enable_web_search=False):
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    for message in st.session_state.messages[-16:]:
        if message["role"] in ("user", "assistant"):
            messages.append({
                "role": message["role"],
                "content": message["content"],
            })

    combined_prompt = user_prompt

    if context:
        combined_prompt = (
            f"{user_prompt}\n\n"
            "Use the following provided content as context. "
            "If it is incomplete, say so.\n\n"
            f"--- CONTEXT START ---\n{context[:45000]}\n"
            "--- CONTEXT END ---"
        )

    messages.append({
        "role": "user",
        "content": combined_prompt,
    })

    response = ask_groq(
        messages,
        enable_web_search=enable_web_search,
    )

    answer = response.choices[0].message.content

    return answer or "Is request ke liye text response nahi mila."
