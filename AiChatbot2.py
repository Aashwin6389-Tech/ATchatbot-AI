# AiChatbot2.py
# ATchatbot.AI — AI Chatbot by Aashwin Tripathi

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


# -------------------- CONFIGURATION --------------------

APP_NAME = "ATchatbot.AI"
HISTORY_DIR = Path("chat_history")
HISTORY_DIR.mkdir(exist_ok=True)

MODEL_NAME = "openai/gpt-oss-20b"

st.set_page_config(
    page_title=APP_NAME,
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

SYSTEM_PROMPT = """
You are ATchatbot.AI, a helpful AI assistant created by Aashwin Tripathi.

If the user asks who you are, who created you, or your identity, reply exactly:
I am Chatbot for help created by Aashwin Tripathi

Be helpful, accurate, friendly, and clear.
If you are unsure about something, say so. Do not invent facts or sources.
Explain coding questions step by step when useful.
"""


# -------------------- STYLING --------------------

st.markdown("""
<style>
/* ATchatbot.AI: clean, high-contrast, professional light theme */
:root {
    color-scheme: light;
    --at-ink: #202b27;
    --at-muted: #5f6b65;
    --at-paper: #f7f7f2;
    --at-card: #ffffff;
    --at-border: #dfe5df;
    --at-green: #176b52;
    --at-green-hover: #10543f;
    --at-soft-green: #eaf4ee;
}

.stApp {
    background: var(--at-paper);
    color: var(--at-ink);
}

/* Keep the main page readable regardless of Streamlit's selected theme */
[data-testid="stAppViewContainer"],
[data-testid="stMain"],
[data-testid="stMainBlockContainer"] {
    background: var(--at-paper);
    color: var(--at-ink);
}

[data-testid="stHeader"] {
    background: rgba(247, 247, 242, 0.96);
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: #eef1eb;
    border-right: 1px solid var(--at-border);
}
[data-testid="stSidebar"] * {
    color: var(--at-ink);
}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
[data-testid="stSidebar"] .stCaption {
    color: var(--at-muted);
}
