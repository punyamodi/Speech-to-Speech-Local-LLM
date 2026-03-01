import os
import json
import logging
from typing import List, Tuple, Optional

import gradio as gr

from src.config import AppConfig, LLMConfig, STTConfig, TTSConfig
from src.pipeline import Pipeline
from src.tts.openvoice import OPENVOICE_STYLES

logger = logging.getLogger(__name__)


def _build_conversation_export(history: List[Tuple]) -> str:
    lines = []
    for user_msg, assistant_msg in history:
        if user_msg:
            lines.append(f"You: {user_msg}")
        if assistant_msg:
            lines.append(f"Assistant: {assistant_msg}")
        lines.append("")
    return "\n".join(lines)


def create_ui(pipeline: Pipeline) -> gr.Blocks:
    theme = gr.themes.Soft(
        primary_hue="indigo",
        secondary_hue="slate",
        neutral_hue="slate",
    )

    with gr.Blocks(
        title="VoiceChat — Local Speech-to-Speech AI",
        theme=theme,
        css="""
        .message-wrap { max-height: 520px; overflow-y: auto; }
        .status-bar { font-size: 0.85rem; color: #6b7280; }
        """,
    ) as app:
        gr.Markdown(
            """
            **Local speech-to-speech AI assistant** — powered by Whisper, your local LLM, and multi-backend TTS.
            Speak or type to start a conversation. All processing happens on your machine.
            """
        )

        with gr.Tabs():
            with gr.Tab("Voice Chat"):
                with gr.Row():
                    with gr.Column(scale=3):
                        chatbot = gr.Chatbot(
                            label="Conversation",
                            elem_id="chat-area",
                            bubble_full_width=False,
                            show_copy_button=True,
                            height=500,
                        )
                        transcript_box = gr.Textbox(
                            label="Last Transcription",
                            interactive=False,
                            lines=2,
                            placeholder="Transcription will appear here after recording...",
                        )

                    with gr.Column(scale=1):
                        gr.Markdown("### Voice Input")
                        audio_input = gr.Audio(
                            sources=["microphone"],
                            type="filepath",
                            label="Record your message",
                        )
                        gr.Markdown("### Audio Output")
                        audio_output = gr.Audio(
                            label="AI Response",
                            autoplay=True,
                            interactive=False,
                        )
                        status_display = gr.Textbox(
                            label="Status",
                            interactive=False,
                            lines=2,
                            elem_classes=["status-bar"],
                            value="Ready",
                        )

                with gr.Row():
                    text_input = gr.Textbox(
                        placeholder="Type your message here (or use the microphone above)...",
                        label="Text Input",
                        lines=2,
                        scale=5,
                    )
                    with gr.Column(scale=1, min_width=120):
                        send_btn = gr.Button("Send", variant="primary")
                        clear_btn = gr.Button("Clear Chat")
                        export_btn = gr.Button("Export Chat")

                export_output = gr.File(label="Exported Conversation", visible=False)
                conversation_state = gr.State([])

                def process_audio(audio_path, chat_history, conv_history):
                    if not audio_path:
                        return chat_history, conv_history, "", None, "No audio recorded."
                    try:
                        transcript = pipeline.transcribe(audio_path)
                        if not transcript.strip():
                            return chat_history, conv_history, "", None, "Could not transcribe audio."
                        return _process_message(transcript, chat_history, conv_history)
                    except Exception as exc:
                        logger.exception("Transcription error")
                        return chat_history, conv_history, "", None, f"Transcription error: {exc}"

                def _process_message(message, chat_history, conv_history):
                    if not message or not message.strip():
                        return chat_history, conv_history, message, None, "Please enter a message."
                    try:
                        status = "Generating response..."
                        full_response = pipeline.get_response(message, conv_history)
                        audio_path = None
                        try:
                            audio_path = pipeline.synthesize(full_response)
                            status = "Done"
                        except Exception as tts_exc:
                            logger.warning("TTS failed: %s", tts_exc)
                            status = f"TTS unavailable: {tts_exc}"

                        updated_history = chat_history + [[message, full_response]]
                        updated_conv = conv_history + [
                            {"role": "user", "content": message},
                            {"role": "assistant", "content": full_response},
                        ]
                        pipeline.log_exchange(message, full_response)
                        return updated_history, updated_conv, "", audio_path, status
                    except Exception as exc:
                        logger.exception("Pipeline error")
                        err_msg = f"Error: {exc}"
                        return chat_history + [[message, err_msg]], conv_history, "", None, err_msg

                def send_text(message, chat_history, conv_history):
                    return _process_message(message, chat_history, conv_history)

                def clear_chat():
                    return [], [], "", None, "Chat cleared."

                def export_chat(chat_history):
                    if not chat_history:
                        return gr.update(visible=False)
                    content = _build_conversation_export(chat_history)
                    export_path = "outputs/conversation_export.txt"
                    os.makedirs("outputs", exist_ok=True)
                    with open(export_path, "w", encoding="utf-8") as f:
                        f.write(content)
                    return gr.update(value=export_path, visible=True)

                audio_input.stop_recording(
                    process_audio,
                    inputs=[audio_input, chatbot, conversation_state],
                    outputs=[chatbot, conversation_state, transcript_box, audio_output, status_display],
                )

                send_btn.click(
                    send_text,
                    inputs=[text_input, chatbot, conversation_state],
                    outputs=[chatbot, conversation_state, text_input, audio_output, status_display],
                )

                text_input.submit(
                    send_text,
                    inputs=[text_input, chatbot, conversation_state],
                    outputs=[chatbot, conversation_state, text_input, audio_output, status_display],
                )

                clear_btn.click(
                    clear_chat,
                    outputs=[chatbot, conversation_state, text_input, audio_output, status_display],
                )

                export_btn.click(
                    export_chat,
                    inputs=[chatbot],
                    outputs=[export_output],
                )

            with gr.Tab("Settings"):
                gr.Markdown("### Configure the pipeline. Click **Apply** to save changes.")

                with gr.Row():
                    with gr.Column():
                        gr.Markdown("#### LLM Backend")
                        llm_backend = gr.Radio(
                            ["lmstudio", "ollama"],
                            value="lmstudio",
                            label="Backend",
                            info="LM Studio uses the OpenAI-compatible API. Ollama runs local models directly.",
                        )
                        lmstudio_url = gr.Textbox(
                            value="http://localhost:1234/v1",
                            label="LM Studio Server URL",
                        )
                        ollama_model = gr.Textbox(
                            value="llama2",
                            label="Ollama Model Name",
                            placeholder="e.g. llama2, mistral, phi3",
                        )
                        temperature = gr.Slider(
                            0.0, 2.0, value=0.7, step=0.05,
                            label="Temperature",
                            info="Higher values make the output more creative.",
                        )
                        max_history = gr.Slider(
                            2, 40, value=20, step=2,
                            label="Max Conversation History (messages)",
                        )

                    with gr.Column():
                        gr.Markdown("#### TTS Backend")
                        tts_backend = gr.Radio(
                            ["bark", "system", "openvoice"],
                            value="bark",
                            label="Backend",
                            info="Bark: high quality, slow. System: fast, robotic. OpenVoice: requires checkpoints.",
                        )
                        openvoice_style = gr.Dropdown(
                            OPENVOICE_STYLES,
                            value="default",
                            label="OpenVoice Style",
                            info="Only used with the OpenVoice backend.",
                        )
                        reference_voice = gr.Audio(
                            label="Reference Voice (OpenVoice only)",
                            type="filepath",
                            info="Upload a short voice sample for voice cloning.",
                        )
                        bark_voice = gr.Dropdown(
                            [f"v2/en_speaker_{i}" for i in range(10)],
                            value="v2/en_speaker_6",
                            label="Bark Voice Preset",
                            info="Only used with the Bark backend.",
                        )

                with gr.Row():
                    with gr.Column():
                        gr.Markdown("#### STT Settings")
                        whisper_model = gr.Dropdown(
                            ["tiny.en", "base.en", "small.en", "medium.en", "large"],
                            value="base.en",
                            label="Whisper Model",
                            info="Larger models are more accurate but slower to load.",
                        )

                gr.Markdown("#### System Prompt")
                system_prompt = gr.Textbox(
                    value=pipeline.config.system_prompt,
                    label="System Prompt",
                    lines=4,
                    placeholder="Describe the assistant's personality and instructions...",
                )

                with gr.Row():
                    apply_btn = gr.Button("Apply Settings", variant="primary")
                    reset_btn = gr.Button("Reset to Defaults")

                settings_status = gr.Textbox(
                    label="Status",
                    interactive=False,
                    elem_classes=["status-bar"],
                )

                def apply_settings(
                    llm_back, url, olm, temp, max_hist,
                    tts_back, ov_style, ref_voice, bk_voice,
                    wh_model, sys_prompt
                ):
                    try:
                        new_config = AppConfig(
                            llm=LLMConfig(
                                backend=llm_back,
                                base_url=url,
                                model="local-model",
                                ollama_model=olm,
                                temperature=float(temp),
                                max_history=int(max_hist),
                            ),
                            stt=STTConfig(model_size=wh_model),
                            tts=TTSConfig(
                                backend=tts_back,
                                style=ov_style,
                                reference_audio=ref_voice,
                                bark_voice=bk_voice,
                                system_rate=150,
                            ),
                            system_prompt=sys_prompt,
                        )
                        pipeline.update_config(new_config)
                        return "Settings applied successfully."
                    except Exception as exc:
                        logger.exception("Settings error")
                        return f"Error applying settings: {exc}"

                def reset_settings():
                    defaults = AppConfig()
                    return (
                        defaults.llm.backend,
                        defaults.llm.base_url,
                        defaults.llm.ollama_model,
                        defaults.llm.temperature,
                        defaults.llm.max_history,
                        defaults.tts.backend,
                        defaults.tts.style,
                        None,
                        defaults.tts.bark_voice,
                        defaults.stt.model_size,
                        defaults.system_prompt,
                        "Settings reset to defaults.",
                    )

                apply_btn.click(
                    apply_settings,
                    inputs=[
                        llm_backend, lmstudio_url, ollama_model, temperature, max_history,
                        tts_backend, openvoice_style, reference_voice, bark_voice,
                        whisper_model, system_prompt,
                    ],
                    outputs=[settings_status],
                )

                reset_btn.click(
                    reset_settings,
                    outputs=[
                        llm_backend, lmstudio_url, ollama_model, temperature, max_history,
                        tts_backend, openvoice_style, reference_voice, bark_voice,
                        whisper_model, system_prompt, settings_status,
                    ],
                )

            with gr.Tab("About"):
                gr.Markdown(
                    """

                    VoiceChat is a local, privacy-first speech-to-speech AI assistant. Everything runs
                    on your machine — no data is sent to external servers.


                    ```
                    Microphone → Whisper (STT) → Local LLM → TTS → Speaker
                    ```


                    | Component | Options |
                    |-----------|---------|
                    | STT | Whisper (tiny, base, small, medium, large) |
                    | LLM | LM Studio (any GGUF model), Ollama (llama2, mistral, phi3, ...) |
                    | TTS | Bark (HuggingFace), System (pyttsx3), OpenVoice (with checkpoints) |


                    To use the OpenVoice backend with voice cloning:

                    1. Create a `checkpoints/` folder in the project root.
                    2. Download checkpoints from [myshell-public-repo-hosting](https://myshell-public-repo-hosting.s3.amazonaws.com/checkpoints_1226.zip).
                    3. Extract and place `base_speakers/` and `converter/` inside `checkpoints/`.
                    4. Select **openvoice** as the TTS backend in Settings and optionally upload a reference voice.


                    1. Download [LM Studio](https://lmstudio.ai/).
                    2. Download a model (e.g. [Mistral 7B GGUF](https://huggingface.co/TheBloke/dolphin-2.2.1-mistral-7B-AWQ)).
                    3. Start the local server (default port 1234).


                    ```bash
                    ollama pull llama2
                    ```
                    """
                )

    return app
