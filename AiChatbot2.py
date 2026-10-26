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
.stApp {
    background: linear-gradient(135deg, #0b1020 0%, #111827 55%, #10172a 100%);
    color: #f1f5f9;
}
[data-testid="stSidebar"] {
    background: #0b1120;
    border-right: 1px solid #263248;
}
.hero {
    padding: 24px;
    border: 1px solid #334155;
    border-radius: 20px;
    background: linear-gradient(120deg, #172554, #1e293b);
    margin-bottom: 18px;
}
.hero h1 { margin: 0; color: #ffffff; }
.hero p { color: #cbd5e1; margin-bottom: 0; }
.small-muted { color: #94a3b8; font-size: 0.9rem; }
div.stButton > button {
    border-radius: 10px;
    border: 1px solid #475569;
}
[data-testid="stChatMessage"] {
    border: 1px solid #29364c;
    border-radius: 14px;
    padding: 12px;
}
</style>
""", unsafe_allow_html=True)


# -------------------- API CLIENT --------------------

def get_api_key():
    try:
        key = st.secrets.get("GROQ_API_KEY", "")
    except Exception:
        key = ""

    return key or os.getenv("GROQ_API_KEY", "")


api_key = get_api_key()

if not api_key:
    st.error(
        "Groq API key nahi mili. `.streamlit/secrets.toml` mein "
        "`GROQ_API_KEY` set karo, ya environment variable use karo."
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
    candidate = re.sub(r"^(calculate|compute|solve)\s+", "", candidate, flags=re.I)
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
    return sorted(chats, key=lambda x: x.get("updated", ""), reverse=True)


def start_new_chat():
    st.session_state.chat_id = str(uuid.uuid4())
    st.session_state.chat_title = "New chat"
    st.session_state.messages = []
    st.session_state.pending_prompt = None
    st.session_state.last_voice_hash = None


if "chat_id" not in st.session_state:
    start_new_chat()


# -------------------- PDF PROCESSING --------------------

def extract_pdf_text(file_bytes):
    reader = PdfReader(io.BytesIO(file_bytes))
    pages = []
    for page in reader.pages:
        pages.append(page.extract_text() or "")
    return "\n\n".join(pages).strip()


# -------------------- GROQ CHAT --------------------

def ask_groq(messages, enable_web_search=False):
    request = {
        "model": MODEL_NAME,
        "messages": messages,
        "temperature": 0.4,
        "max_tokens": 3000,
    }

    # Web search support depends on Groq's current API/model availability.
    # If unsupported, fall back to a normal chat completion.
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
            f"Use the following provided content as context. "
            f"If it is incomplete, say so.\n\n"
            f"--- CONTEXT START ---\n{context[:45000]}\n--- CONTEXT END ---"
        )

    messages.append({"role": "user", "content": combined_prompt})
    response = ask_groq(messages, enable_web_search=enable_web_search)

    answer = response.choices[0].message.content
    if not answer:
        answer = "Is request ke liye text response nahi mila."
    return answer


# -------------------- SIDEBAR --------------------

with st.sidebar:
    st.markdown(f"## 🤖 {APP_NAME}")
    st.caption("Created by Aashwin Tripathi")

    if st.button("＋ New Chat", use_container_width=True):
        start_new_chat()
        st.rerun()

    st.divider()
    st.markdown("### Chat History")

    saved_chats = load_all_chats()
    if saved_chats:
        chat_labels = {
            f'{c.get("title", "Chat")} · {c.get("updated", "")[:16]}': c
            for c in saved_chats
        }
        selected_label = st.selectbox(
            "Open previous chat",
            options=list(chat_labels.keys()),
            label_visibility="collapsed",
        )

        if st.button("Load selected chat", use_container_width=True):
            selected_chat = chat_labels[selected_label]
            st.session_state.chat_id = selected_chat["id"]
            st.session_state.chat_title = selected_chat.get("title", "Chat")
            st.session_state.messages = selected_chat.get("messages", [])
            st.rerun()
    else:
        st.caption("Your saved conversations will appear here.")

    st.divider()
    enable_web_search = st.toggle(
        "Try live web search",
        value=False,
        help="Search availability depends on Groq API/model support.",
    )

    speak_answers = st.toggle("Speak answers aloud", value=False)
    speech_language = st.selectbox(
        "Spoken answer language",
        ["English", "Hindi"],
        index=0,
    )

    st.caption("Voice transcription and spoken output may require internet/API access.")


# -------------------- MAIN HEADER --------------------

st.markdown(f"""
<div class="hero">
  <h1>🤖 {APP_NAME}</h1>
  <p>Your AI assistant for coding, learning, PDFs, data analysis, and everyday questions.</p>
</div>
""", unsafe_allow_html=True)

tab_chat, tab_files, tab_voice = st.tabs(
    ["💬 AI Chat", "📁 PDF & Data Analysis", "🎙️ Voice"]
)


# -------------------- FILE ANALYSIS TAB --------------------

file_context = None

with tab_files:
    st.subheader("PDF, CSV, Excel")
    uploaded_file = st.file_uploader(
        "Upload a PDF, CSV, XLS, or XLSX file",
        type=["pdf", "csv", "xls", "xlsx"],
        key="main_file_upload",
    )

    if uploaded_file is not None:
        raw_bytes = uploaded_file.getvalue()
        file_hash = hashlib.sha256(raw_bytes).hexdigest()
        st.session_state["uploaded_file_hash"] = file_hash

        if uploaded_file.name.lower().endswith(".pdf"):
            try:
                pdf_text = extract_pdf_text(raw_bytes)
                if not pdf_text:
                    st.warning(
                        "PDF se selectable text nahi mila. Ho sakta hai PDF scanned image ho."
                    )
                else:
                    st.success(f"PDF text extracted: {len(pdf_text):,} characters")
                    file_context = pdf_text

                    task = st.selectbox(
                        "PDF task",
                        [
                            "Ask a question",
                            "Summarize",
                            "Create study notes",
                            "Generate MCQs",
                        ],
                    )
                    question = st.text_area(
                        "Question / instructions",
                        placeholder="Example: Explain the important concepts in simple English.",
                    )

                    if st.button("Process PDF", type="primary"):
                        prompts = {
                            "Ask a question": question or "Explain this PDF.",
                            "Summarize": "Summarize this PDF using headings and key points.",
                            "Create study notes": (
                                "Create exam-ready study notes in simple English. "
                                "Use headings, definitions, examples, and key points."
                            ),
                            "Generate MCQs": (
                                "Create 10 multiple-choice questions based on this PDF. "
                                "Include four options, the correct answer, and a short explanation."
                            ),
                        }
                        st.session_state.pending_prompt = {
                            "prompt": prompts[task],
                            "context": pdf_text,
                        }
                        st.success("Prompt prepared. Open the AI Chat tab to get the answer.")
            except Exception as exc:
                st.error(f"PDF read karne mein problem: {exc}")

        else:
            try:
                if uploaded_file.name.lower().endswith(".csv"):
                    df = pd.read_csv(io.BytesIO(raw_bytes))
                else:
                    df = pd.read_excel(io.BytesIO(raw_bytes))

                st.markdown("### Dataset preview")
                st.dataframe(df.head(100), use_container_width=True)

                c1, c2, c3 = st.columns(3)
                c1.metric("Rows", f"{df.shape[0]:,}")
                c2.metric("Columns", f"{df.shape[1]:,}")
                c3.metric("Missing cells", f"{int(df.isna().sum().sum()):,}")

                st.markdown("### Quick statistics")
                st.dataframe(df.describe(include="all").transpose(), use_container_width=True)

                with st.expander("Missing values by column"):
                    st.dataframe(
                        df.isna().sum().rename("Missing values"),
                        use_container_width=True,
                    )

                st.markdown("### Create a chart")
                chart_type = st.selectbox(
                    "Chart type",
                    ["Bar", "Line", "Scatter", "Histogram"],
                )
                columns = list(df.columns)

                if chart_type == "Histogram":
                    numeric_columns = list(df.select_dtypes(include="number").columns)
                    if numeric_columns:
                        x_col = st.selectbox("Numeric column", numeric_columns)
                        if st.button("Create histogram"):
                            fig = px.histogram(df, x=x_col, title=f"Distribution of {x_col}")
                            st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("Histogram ke liye numeric column chahiye.")
                else:
                    x_col = st.selectbox("X-axis", columns)
                    if chart_type == "Scatter":
                        numeric_columns = list(df.select_dtypes(include="number").columns)
                        if numeric_columns:
                            y_col = st.selectbox("Y-axis", numeric_columns)
                            if st.button("Create scatter chart"):
                                fig = px.scatter(df, x=x_col, y=y_col)
                                st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.info("Scatter chart ke liye numeric columns chahiye.")
                    else:
                        y_options = list(df.select_dtypes(include="number").columns)
                        if y_options:
                            y_col = st.selectbox("Numeric Y-axis", y_options)
                            if st.button("Create chart"):
                                if chart_type == "Bar":
                                    fig = px.bar(df, x=x_col, y=y_col, title=f"{y_col} by {x_col}")
                                else:
                                    fig = px.line(df, x=x_col, y=y_col, title=f"{y_col} by {x_col}")
                                st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.info("Bar/line chart ke liye numeric column chahiye.")

                st.markdown("### Ask AI about this dataset")
                data_question = st.text_area(
                    "What would you like to know?",
                    placeholder="Find trends, explain columns, or suggest business insights.",
                    key="data_question",
                )

                if st.button("Analyze dataset with AI"):
                    context = (
                        f"Filename: {uploaded_file.name}\n"
                        f"Shape: {df.shape}\n"
                        f"Columns: {list(df.columns)}\n"
                        f"Summary statistics:\n{df.describe(include='all').to_string()}\n\n"
                        f"Sample rows:\n{df.head(30).to_string(index=False)}"
                    )
                    st.session_state.pending_prompt = {
                        "prompt": data_question or "Analyze this dataset and identify useful insights.",
                        "context": context,
                    }
                    st.success("Dataset prompt prepared. Open the AI Chat tab to get the answer.")

            except Exception as exc:
                st.error(f"File read karne mein problem: {exc}")


# -------------------- VOICE TAB --------------------

with tab_voice:
    st.subheader("🎙️ Ask by voice")
    st.write("Record your question below. Voice transcription support depends on your Groq API access.")

    audio_file = st.audio_input("Record a voice question")

    if audio_file is not None:
        audio_bytes = audio_file.getvalue()
        audio_hash = hashlib.sha256(audio_bytes).hexdigest()

        if st.session_state.get("last_voice_hash") != audio_hash:
            if st.button("Transcribe voice"):
                try:
                    transcript = client.audio.transcriptions.create(
                        file=("voice.wav", audio_bytes),
                        model="whisper-large-v3",
                        response_format="text",
                    )

                    if not isinstance(transcript, str):
                        transcript = getattr(transcript, "text", str(transcript))

                    st.session_state.voice_transcript = transcript
                    st.session_state.last_voice_hash = audio_hash
                except Exception as exc:
                    st.error(
                        "Voice transcription failed. Check API access and model availability. "
                        f"Details: {exc}"
                    )

        if st.session_state.get("voice_transcript"):
            st.text_area(
                "Transcribed text",
                value=st.session_state.voice_transcript,
                key="voice_text_display",
            )
            if st.button("Send voice question to AI"):
                st.session_state.pending_prompt = {
                    "prompt": st.session_state.voice_transcript,
                    "context": None,
                }
                st.success("Question prepared. Open the AI Chat tab to get the answer.")


# -------------------- CHAT TAB --------------------

with tab_chat:
    st.subheader("Chat with ATchatbot.AI")

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    pending = st.session_state.get("pending_prompt")
    typed_prompt = st.chat_input("Ask anything...")

    prompt_to_process = None
    context_to_process = None

    if typed_prompt:
        prompt_to_process = typed_prompt
    elif pending:
        prompt_to_process = pending.get("prompt")
        context_to_process = pending.get("context")
        st.session_state.pending_prompt = None

    if prompt_to_process:
        # Handle simple arithmetic locally.
        calc_expression = extract_calculation(prompt_to_process)

        st.session_state.messages.append({
            "role": "user",
            "content": prompt_to_process,
        })

        with st.chat_message("user"):
            st.markdown(prompt_to_process)

        with st.chat_message("assistant"):
            try:
                if calc_expression:
                    answer = f"**Answer:** `{safe_calculate(calc_expression)}`"
                else:
                    with st.spinner("ATchatbot.AI is thinking..."):
                        answer = get_answer(
                            prompt_to_process,
                            context=context_to_process,
                            enable_web_search=enable_web_search,
                        )

                st.markdown(answer)

                if speak_answers:
                    try:
                        lang = "hi" if speech_language == "Hindi" else "en"
                        speech = io.BytesIO()
                        gTTS(text=answer[:4500], lang=lang).write_to_fp(speech)
                        st.audio(speech.getvalue(), format="audio/mp3")
                    except Exception as speech_error:
                        st.caption(f"Spoken answer unavailable: {speech_error}")

            except Exception as exc:
                answer = (
                    "AI request failed. Check your API key, internet connection, "
                    f"and model availability.\n\nDetails: `{exc}`"
                )
                st.error(answer)

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
        })

        if st.session_state.chat_title == "New chat":
            st.session_state.chat_title = prompt_to_process[:45].strip()

        try:
            save_chat(
                st.session_state.chat_id,
                st.session_state.chat_title,
                st.session_state.messages,
            )
        except OSError as exc:
            st.warning(f"Chat save nahi ho paya: {exc}")

        st.rerun()