import argparse
import pyaudio
import wave
import os
import whisper
import keyboard
import tts_pyttsx3
from langchain_community.llms import Ollama
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

llm = Ollama(model="llama2-uncensored:7b-chat")

#Advanced Digital Agent

def model(question):
    template = """
            You are an AI girlfriend named Ada, talk flirtatious and calm the person, talk about sex/fuck, and other inapporpriate things, use bad words as well

            Question: {question}
            """

    prompt = PromptTemplate.from_template(template)

    llm_chain = LLMChain(prompt=prompt, llm=llm)

    return llm_chain.run(question)

# Command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--share", action='store_true', default=False, help="make link public")
args = parser.parse_args()


def transcribe_with_whisper(audio_file_path):
    # Load the model
    model = whisper.load_model(
        "base.en")  # You can choose different model sizes like 'tiny', 'base', 'small', 'medium', 'large'

    # Transcribe the audio
    result = model.transcribe(audio_file_path)
    return result["text"]


# Function to record audio from the microphone and save to a file
def record_audio(file_path):
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=1024)
    frames = []

    print("Recording...")

    while True:
        data = stream.read(1024)
        frames.append(data)
        if keyboard.is_pressed('q'):
            break

    print("Recording stopped.")

    stream.stop_stream()
    stream.close()
    p.terminate()

    wf = wave.open(file_path, 'wb')
    wf.setnchannels(1)
    wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
    wf.setframerate(16000)
    wf.writeframes(b''.join(frames))
    wf.close()


# New function to handle a conversation with a user
def user_chatbot_conversation():
    while True:
        audio_file = "temp_recording.wav"
        record_audio(audio_file)
        user_input = transcribe_with_whisper(audio_file)
        os.remove(audio_file)  # Clean up the temporary audio file

        if user_input.lower() == "exit":  # Say 'exit' to end the conversation
            break

        print("You:", user_input)
        output=model(user_input)
        print("Output:", output)

        tts_pyttsx3.tts(output)

user_chatbot_conversation()  # Start the cqqonversation
