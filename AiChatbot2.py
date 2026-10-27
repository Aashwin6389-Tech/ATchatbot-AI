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

CUSTOM_CSS = r"""
<style>
:root {
    color-scheme: light;
    --ink: #202b27;
    --muted: #5f6b65;
    --paper: #f7f7f2;
    --card: #ffffff;
    --border: #dfe5df;
    --green: #176b52;
    --green-hover: #10543f;
    --soft-green: #eaf4ee;
}
.stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"],
[data-testid="stMainBlockContainer"] {
    background: var(--paper) !important;
    color: var(--ink) !important;
}
[data-testid="stHeader"] { background: rgba(247,247,242,.96) !important; }
[data-testid="stSidebar"] {
    background: #eef1eb !important;
    border-right: 1px solid var(--border);
}
[data-testid="stSidebar"] * { color: var(--ink); }
.stMarkdown, .stMarkdown p, .stMarkdown li,
[data-testid="stChatMessage"] p, [data-testid="stChatMessage"] li,
label, [data-testid="stCaptionContainer"] {
    color: var(--ink);
}
a { color: var(--green); }
.hero {
    padding: 28px 30px;
    border: 1px solid var(--border);
    border-left: 6px solid var(--green);
    border-radius: 18px;
    background: #fff;
    box-shadow: 0 5px 20px rgba(31,48,39,.06);
    margin-bottom: 22px;
}
.hero h1 {
    margin: 0 0 8px;
    color: #173d31 !important;
    font-size: clamp(1.8rem, 3vw, 2.5rem);
    font-weight: 750;
}
