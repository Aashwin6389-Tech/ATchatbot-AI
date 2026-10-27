
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
