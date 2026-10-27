
# ============================================================
# ATchatbot.AI
# Created by Aashwin Tripathi
# ============================================================

import ast
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


# ============================================================
# CONFIGURATION
# ============================================================

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

Be helpful, professional, accurate, friendly, and clear.
Explain coding questions step by step when useful.
Use headings, examples, and tables where helpful.
Never invent facts or sources.
"""


# ============================================================
# SESSION STATE
# ============================================================

defaults = {
    "messages": [],
    "chat_id": str(uuid.uuid4()),
    "chat_title": "New Chat",
    "show_recent": True,
    "spoken_answers": False,
    "response_language": "Auto detect",
    "pending_prompt": None,
    "pdf_text": "",
    "pdf_name": "",
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ============================================================
# PROFESSIONAL THEME
# Ivory + Charcoal + Copper — No Blue Background
# ============================================================

CUSTOM_CSS = r"""
<style>
:root {
    color-scheme: light;
    --bg: #F0EDE5;
    --sidebar: #E4DED2;
    --surface: #FAF8F3;
    --surface-alt: #F4F0E7;
    --border: #D6CDBE;
    --ink: #292A27;
    --muted: #69665F;
    --accent: #A85D35;
    --accent-hover: #874526;
}

html, body, .stApp,
[data-testid="stAppViewContainer"],
[data-testid="stMain"],
[data-testid="stMainBlockContainer"] {
    background: var(--bg) !important;
    color: var(--ink) !important;
    font-family: "Segoe UI", Arial, sans-serif !important;
}

[data-testid="stHeader"] {
    background: var(--bg) !important;
}

[data-testid="stSidebar"] {
    background: var(--sidebar) !important;
    border-right: 1px solid var(--border) !important;
}

[data-testid="stSidebar"] * {
    color: var(--ink);
}

.stMarkdown, .stMarkdown p, .stMarkdown li,
[data-testid="stCaptionContainer"] p,
label {
    color: var(--ink);
}

a {
    color: var(--accent) !important;
}

/* Main hero */
.hero {
    background: var(--surface) !important;
    padding: 26px 30px;
    margin-bottom: 22px;
    border: 1px solid var(--border);
    border-top: 4px solid var(--accent);
    border-radius: 15px;
    box-shadow: 0 5px 18px rgba(40,35,25,.06);
}

.hero h1 {
    margin: 0 0 8px;
    color: var(--ink) !important;
    font-size: clamp(1.8rem, 3vw, 2.4rem);
    font-weight: 750;
    letter-spacing: -.7px;
}

.hero p {
    color: var(--muted) !important;
    line-height: 1.65;
    margin: 0;
}

/* Buttons */
div.stButton > button,
div[data-testid="stDownloadButton"] button {
    background: var(--accent) !important;
    color: #FFFFFF !important;
    border: 1px solid var(--accent) !important;
    border-radius: 9px !important;
    font-weight: 600 !important;
    min-height: 38px;
}

div.stButton > button:hover,
div[data-testid="stDownloadButton"] button:hover {
    background: var(--accent-hover) !important;
    border-color: var(--accent-hover) !important;
}

/* Sidebar buttons */
[data-testid="stSidebar"] div.stButton > button {
    background: #F8F5EE !important;
    color: var(--ink) !important;
    border: 1px solid var(--border) !important;
    text-align: left !important;
}

[data-testid="stSidebar"] div.stButton > button:hover {
    background: #EDE4D6 !important;
}

/* Tabs */
[data-testid="stTabs"] [data-baseweb="tab-list"] {
    background: #E3DCCE !important;
    padding: 5px;
    gap: 5px;
    border-radius: 10px;
}

[data-testid="stTabs"] button[data-baseweb="tab"] {
    background: transparent !important;
    color: var(--ink) !important;
    border-radius: 8px !important;
}

[data-testid="stTabs"] button[data-baseweb="tab"][aria-selected="true"] {
    background: #34342F !important;
    color: #FFFFFF !important;
}

/* Chat messages */
[data-testid="stChatMessage"] {
    background: var(--surface) !important;
    color: var(--ink) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
    padding: 14px 17px !important;
    margin-bottom: 12px !important;
    box-shadow: 0 2px 7px rgba(40,35,25,.035);
}

[data-testid="stChatMessage"] p,
[data-testid="stChatMessage"] li,
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] {
    color: var(--ink) !important;
    line-height: 1.75 !important;
}

/* Inputs */
input, textarea,
[data-baseweb="input"] input,
[data-baseweb="textarea"] textarea,
[data-testid="stChatInput"] textarea {
    background: #FAF8F3 !important;
    color: var(--ink) !important;
    -webkit-text-fill-color: var(--ink) !important;
    border-color: var(--border) !important;
}

input::placeholder,
textarea::placeholder {
    color: #777168 !important;
    -webkit-text-fill-color: #777168 !important;
    opacity: 1 !important;
}

/* Selects and file upload */
[data-baseweb="select"] > div {
    background: #FAF8F3 !important;
    color: var(--ink) !important;
    border-color: var(--border) !important;
}

[data-testid="stFileUploader"] section {
    background: var(--surface) !important;
    border-color: var(--border) !important;
}

/* Cards and tables */
[data-testid="stMetric"],
[data-testid="stExpander"],
[data-testid="stDataFrame"],
[data-testid="stTable"] {
    background: var(--surface) !important;
    color: var(--ink) !important;
    border-color: var(--border) !important;
    border-radius: 10px !important;
}

[data-testid="stMetricLabel"],
[data-testid="stMetricValue"] {
    color: var(--ink) !important;
}

hr {
    border-color: var(--border) !important;
}

/* Compact fixed chat input at the bottom */
[data-testid="stMain"] {
    padding-bottom: 95px !important;
}

[data-testid="stChatInput"] {
    position: fixed !important;
    bottom: 12px !important;
    left: max(365px, calc((100vw - 900px + 350px) / 2)) !important;
    width: min(900px, calc(100vw - 390px)) !important;
    max-width: 900px !important;
    z-index: 1001 !important;
    background: #FAF8F3 !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
    box-shadow: 0 4px 16px rgba(40,35,25,.10) !important;
}

[data-testid="stChatInput"] > div {
    background: #FAF8F3 !important;
    border-color: var(--border) !important;
}

/* Voice input stays in normal document flow */
[data-testid="stAudioInput"] {
    position: relative !important;
    left: auto !important;
    bottom: auto !important;
    width: 100% !important;
    max-width: 100% !important;
    background: var(--surface-alt) !important;
    color: var(--ink) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    padding: 7px !important;
    box-shadow: none !important;
}

[data-testid="stAudioInput"] label,
[data-testid="stAudioInput"] p {
    color: var(--ink) !important;
}

@media (max-width: 700px) {
    .hero {
        padding: 19px;
    }

    [data-testid="stMain"] {
        padding-bottom: 90px !important;
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


# ============================================================
# API KEY
# ============================================================

def get_api_key():
    try:
        key = st.secrets.get("GROQ_API_KEY", "")
    except Exception:
        key = ""
    return key or os.getenv("GROQ_API_KEY", "")


api_key = get_api_key()

if not api_key:
    st.error("Groq API key missing hai.")
    st.info(
        "Streamlit Cloud → App Settings → Secrets mein "
        "GROQ_API_KEY set karo."
    )
    st.code('GROQ_API_KEY = "your_actual_key_here"', language="toml")
    st.stop()

client = Groq(api_key=api_key)


# ============================================================
# SAFE CALCULATOR
# ============================================================

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
    tree = ast.parse(expression, mode="eval")

    def visit(node):
        if isinstance(node, ast.Expression):
            return visit(node.body)

        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)):
                return node.value
            raise ValueError("Only numbers are allowed.")

        if isinstance(node, ast.BinOp):
            op_type = type(node.op)

            if op_type not in _ALLOWED_OPERATORS:
                raise ValueError("Operator not supported.")

            left = visit(node.left)
            right = visit(node.right)

            if isinstance(node.op, ast.Pow) and abs(right) > 100:
                raise ValueError("Exponent is too large.")

            return _ALLOWED_OPERATORS[op_type](left, right)

        if isinstance(node, ast.UnaryOp):
            op_type = type(node.op)

            if op_type not in _ALLOWED_OPERATORS:
                raise ValueError("Operator not supported.")

            return _ALLOWED_OPERATORS[op_type](visit(node.operand))

        raise ValueError("Only basic arithmetic is supported.")

    return visit(tree)


# ============================================================
# CHAT HISTORY
# ============================================================

def history_path(chat_id):
    return HISTORY_DIR / f"{chat_id}.json"


def save_chat():
    data = {
        "id": st.session_state.chat_id,
        "title": st.session_state.chat_title,
        "updated": datetime.now().isoformat(timespec="seconds"),
        "messages": st.session_state.messages,
    }

    history_path(st.session_state.chat_id).write_text(
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


def new_chat():
    st.session_state.chat_id = str(uuid.uuid4())
    st.session_state.chat_title = "New Chat"
    st.session_state.messages = []
    st.session_state.pending_prompt = None
    st.session_state.pdf_text = ""
    st.session_state.pdf_name = ""


def open_saved_chat(chat_data):
    st.session_state.chat_id = chat_data.get("id", str(uuid.uuid4()))
    st.session_state.chat_title = chat_data.get("title", "Chat")
    st.session_state.messages = chat_data.get("messages", [])
    st.session_state.pending_prompt = None


# ============================================================
# GROQ CHAT
# ============================================================

def get_answer(prompt, context=""):
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    for message in st.session_state.messages[-16:]:
        if message.get("role") in ("user", "assistant"):
            messages.append({
                "role": message["role"],
                "content": message.get("content", ""),
            })

    full_prompt = prompt

    if context:
        full_prompt += (
            "\n\nUse this supplied content as context. "
            "Be clear if information is missing.\n\n"
            + context[:45000]
        )

    messages.append({"role": "user", "content": full_prompt})

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        temperature=0.4,
        max_tokens=3000,
    )

    return response.choices[0].message.content or "No text response received."


def speak_text(text):
    try:
        audio_buffer = io.BytesIO()
        language = "en"

        if st.session_state.response_language == "Hindi":
            language = "hi"

        tts = gTTS(text=text[:3500], lang=language)
        tts.write_to_fp(audio_buffer)
        audio_buffer.seek(0)
        st.audio(audio_buffer, format="audio/mp3")
    except Exception as exc:
        st.caption(f"Spoken answer unavailable: {exc}")


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown("## ✦ ATchatbot.AI")
    st.caption("AI Workspace · Created by Aashwin Tripathi")
    st.divider()

    if st.button("＋ New Chat", use_container_width=True):
        new_chat()
        st.rerun()

    if st.button(
        "Hide Recent Chats" if st.session_state.show_recent
        else "Show Recent Chats",
        use_container_width=True,
    ):
        st.session_state.show_recent = not st.session_state.show_recent
        st.rerun()

    if st.session_state.show_recent:
        st.markdown("### Recent Chats")

        saved_chats = load_all_chats()

        if not saved_chats:
            st.caption("Your saved chats will appear here.")
        else:
            for chat in saved_chats[:10]:
                title = chat.get("title", "Chat")[:35]
                if st.button(
                    title,
                    key=f"history_{chat.get('id', title)}",
                    use_container_width=True,
                ):
                    open_saved_chat(chat)
                    st.rerun()

    st.divider()
    st.markdown("### Preferences")

    st.session_state.response_language = st.selectbox(
        "Response language",
        ["Auto detect", "English", "Hindi"],
        index=["Auto detect", "English", "Hindi"].index(
            st.session_state.response_language
        ),
    )

    st.session_state.spoken_answers = st.checkbox(
        "Enable spoken answers",
        value=st.session_state.spoken_answers,
    )

    st.caption("Keep your Groq API key private.")
    st.caption("Chat history is stored in this app's local files.")


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
    <div class="hero">
        <h1>✦ ATchatbot.AI</h1>
        <p>Your AI workspace for learning, coding, documents,
        and data analysis.</p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# MAIN TABS
# ============================================================

tab_chat, tab_pdf, tab_data, tab_calc = st.tabs([
    "💬 AI Chat",
    "📄 PDF Assistant",
    "📊 Data Analysis",
    "🧮 Calculator",
])


# ============================================================
# PDF ASSISTANT
# ============================================================

with tab_pdf:
    st.subheader("PDF Assistant")
    st.write("Upload a PDF to summarize it, make notes, or ask questions.")

    pdf_file = st.file_uploader(
        "Upload PDF",
        type=["pdf"],
        key="pdf_upload",
    )

    if pdf_file is not None:
        try:
            pdf_reader = PdfReader(io.BytesIO(pdf_file.getvalue()))
            pdf_text = "\n\n".join(
                page.extract_text() or ""
                for page in pdf_reader.pages
            ).strip()

            if pdf_text:
                st.session_state.pdf_text = pdf_text
                st.session_state.pdf_name = pdf_file.name

                st.success(f"Loaded: {pdf_file.name}")
                st.caption(f"Extracted characters: {len(pdf_text):,}")

                pdf_action = st.selectbox(
                    "What would you like to do?",
                    [
                        "Summarize PDF",
                        "Make study notes",
                        "Generate MCQs",
                        "Explain this PDF",
                    ],
                )

                if st.button("Process PDF", key="process_pdf"):
                    prompts = {
                        "Summarize PDF": (
                            "Summarize this PDF using clear headings, "
                            "key points, and a short conclusion."
                        ),
                        "Make study notes": (
                            "Create exam-ready study notes with definitions, "
                            "important points, and examples."
                        ),
                        "Generate MCQs": (
                            "Create 15 multiple-choice questions with four "
                            "options each. Give the correct answer and a "
                            "brief explanation."
                        ),
                        "Explain this PDF": (
                            "Explain the content step by step in simple "
                            "language. Include examples where useful."
                        ),
                    }

                    with st.spinner("Reading and processing PDF..."):
                        result = get_answer(
                            prompts[pdf_action],
                            context=pdf_text,
                        )

                    st.markdown("### Result")
                    st.markdown(result)

                    st.download_button(
                        "Download result as TXT",
                        data=result,
                        file_name="ATchatbot_PDF_result.txt",
                        mime="text/plain",
                    )
            else:
                st.warning(
                    "PDF se selectable text nahi mila. "
                    "Scanned PDF ke liye OCR required ho sakta hai."
                )

        except Exception as exc:
            st.error(f"PDF processing failed: {exc}")

    if st.session_state.pdf_text:
        st.divider()
        st.markdown("### Ask a question about the uploaded PDF")

        pdf_question = st.text_input(
            "Your PDF question",
            placeholder="Explain chapter 2...",
            key="pdf_question",
        )

        if st.button("Ask about PDF", key="ask_pdf"):
            if pdf_question.strip():
                with st.spinner("Finding the answer..."):
                    answer = get_answer(
                        pdf_question,
                        context=st.session_state.pdf_text,
                    )
                st.markdown(answer)


# ============================================================
# DATA ANALYSIS
# ============================================================

with tab_data:
    st.subheader("Data Analysis Workspace")
    st.write("Upload a CSV or Excel file to explore and visualize your data.")

    data_file = st.file_uploader(
        "Upload CSV or Excel",
        type=["csv", "xlsx", "xls"],
        key="data_upload",
    )

    if data_file is not None:
        try:
            if data_file.name.lower().endswith(".csv"):
                df = pd.read_csv(data_file)
            else:
                df = pd.read_excel(data_file)

            st.markdown("### Dataset Preview")
            st.dataframe(df.head(100), use_container_width=True)

            col1, col2, col3 = st.columns(3)
            col1.metric("Rows", f"{df.shape[0]:,}")
            col2.metric("Columns", f"{df.shape[1]:,}")
            col3.metric("Missing cells", f"{int(df.isna().sum().sum()):,}")

            with st.expander("Column information"):
                info_df = pd.DataFrame({
                    "Column": df.columns.astype(str),
                    "Data Type": df.dtypes.astype(str).values,
                    "Missing Values": df.isna().sum().values,
                    "Unique Values": df.nunique(dropna=True).values,
                })
                st.dataframe(info_df, use_container_width=True)

            numeric_cols = df.select_dtypes(include="number").columns.tolist()
            all_cols = df.columns.astype(str).tolist()

            if numeric_cols:
                st.markdown("### Visualize your data")

                chart_col1, chart_col2 = st.columns(2)

                with chart_col1:
                    chart_type = st.selectbox(
                        "Chart type",
                        ["Bar Chart", "Line Chart", "Histogram", "Scatter Plot"],
                    )

                with chart_col2:
                    x_col = st.selectbox(
                        "X-axis / category",
                        all_cols,
                    )

                y_col = st.selectbox(
                    "Numeric column",
                    numeric_cols,
                )

                try:
                    if chart_type == "Bar Chart":
                        chart_data = df.groupby(
                            x_col, dropna=False
                        )[y_col].sum().reset_index().head(50)

                        fig = px.bar(
                            chart_data,
                            x=x_col,
                            y=y_col,
                            title=f"{y_col} by {x_col}",
                        )

                    elif chart_type == "Line Chart":
                        chart_data = df.groupby(
                            x_col, dropna=False
                        )[y_col].mean().reset_index().head(100)

                        fig = px.line(
                            chart_data,
                            x=x_col,
                            y=y_col,
                            markers=True,
                            title=f"{y_col} trend",
                        )

                    elif chart_type == "Histogram":
                        fig = px.histogram(
                            df,
                            x=y_col,
                            title=f"Distribution of {y_col}",
                        )

                    else:
                        fig = px.scatter(
                            df,
                            x=x_col,
                            y=y_col,
                            title=f"{y_col} vs {x_col}",
                        )

                    fig.update_layout(
                        template="simple_white",
                        paper_bgcolor="#FAF8F3",
                        plot_bgcolor="#FAF8F3",
                        font_color="#292A27",
                    )

                    st.plotly_chart(fig, use_container_width=True)

                except Exception as exc:
                    st.warning(f"Chart could not be generated: {exc}")

            st.markdown("### Ask AI about your dataset")

            data_question = st.text_input(
                "Question about this data",
                placeholder="What are the main trends in this dataset?",
                key="data_question",
            )

            if st.button("Analyze with AI", key="analyze_data"):
                if data_question.strip():
                    data_context = (
                        f"Dataset columns: {list(df.columns)}\n"
                        f"Dataset shape: {df.shape}\n"
                        f"Summary statistics:\n"
                        f"{df.describe(include='all').to_string()}\n\n"
                        f"Sample rows:\n{df.head(20).to_string()}"
                    )

                    with st.spinner("Analyzing your data..."):
                        answer = get_answer(
                            data_question,
                            context=data_context,
                        )

                    st.markdown(answer)

            st.download_button(
                "Download cleaned dataset (CSV)",
                data=df.to_csv(index=False).encode("utf-8"),
                file_name="ATchatbot_dataset.csv",
                mime="text/csv",
            )

        except Exception as exc:
            st.error(f"Could not read this dataset: {exc}")


# ============================================================
# CALCULATOR
# ============================================================

with tab_calc:
    st.subheader("Smart Calculator")
    st.write("Calculate basic mathematical expressions.")

    expression = st.text_input(
        "Enter calculation",
        placeholder="Example: (125 * 4) + 50 / 2",
        key="calculator_expression",
    )

    if st.button("Calculate", key="calculate_button"):
        if expression.strip():
            try:
                result = safe_calculate(expression)
                st.success(f"Result: {result}")
            except Exception as exc:
                st.error(f"Could not calculate: {exc}")

    st.caption("Supported: +, -, *, /, //, %, **, and parentheses.")


# ============================================================
# AI CHAT
# ============================================================

with tab_chat:
    st.subheader("Chat with your AI assistant")
    st.caption("Ask questions, learn concepts, or get help with code.")

    st.markdown("#### 🎙️ Ask by voice (optional)")
    audio_value = st.audio_input(
        "Record your question",
        key="voice_recorder",
    )

    if audio_value is not None:
        audio_bytes = audio_value.getvalue()

        import hashlib

        audio_hash = hashlib.sha256(audio_bytes).hexdigest()

        if st.session_state.get("last_voice_hash") != audio_hash:
            st.session_state.last_voice_hash = audio_hash

            try:
                transcription = client.audio.transcriptions.create(
                    file=("voice.wav", audio_bytes),
                    model="whisper-large-v3",
                    response_format="text",
                )

                if isinstance(transcription, str):
                    voice_text = transcription
                else:
                    voice_text = getattr(transcription, "text", "")

                if voice_text.strip():
                    st.session_state.pending_prompt = voice_text.strip()
                    st.success(f"Recognized: {voice_text.strip()}")

            except Exception as exc:
                st.error(f"Voice transcription failed: {exc}")

    # Render previous conversation
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Process voice prompt once
    if st.session_state.pending_prompt:
        current_prompt = st.session_state.pending_prompt
        st.session_state.pending_prompt = None

        st.session_state.messages.append({
            "role": "user",
            "content": current_prompt,
        })

        if st.session_state.chat_title == "New Chat":
            st.session_state.chat_title = current_prompt[:35]

        with st.chat_message("user"):
            st.markdown(current_prompt)

        with st.chat_message("assistant"):
            with st.spinner("ATchatbot.AI is thinking..."):
                try:
                    answer = get_answer(current_prompt)
                    st.markdown(answer)

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                    })

                    if st.session_state.spoken_answers:
                        speak_text(answer)

                    save_chat()

                except Exception as exc:
                    st.error(f"AI request failed: {exc}")

        st.rerun()

    # Normal chat input remains at the bottom of the page.
    prompt = st.chat_input("Message ATchatbot.AI...")

    if prompt:
        st.session_state.messages.append({
            "role": "user",
            "content": prompt,
        })

        if st.session_state.chat_title == "New Chat":
            st.session_state.chat_title = prompt[:35]

        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("ATchatbot.AI is thinking..."):
                try:
                    answer = get_answer(prompt)
                    st.markdown(answer)

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                    })

                    if st.session_state.spoken_answers:
                        speak_text(answer)

                    save_chat()

                except Exception as exc:
                    st.error(f"AI request failed: {exc}")

        st.rerun()


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")
st.markdown(
    """
    <div style="
        text-align:center;
        color:#69665F;
        padding:10px 0 20px 0;
        font-size:13px;
    ">
        ATchatbot.AI · Built by Aashwin Tripathi
    </div>
    """,
    unsafe_allow_html=True,
)
