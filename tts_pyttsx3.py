import pyttsx3

Assistant = pyttsx3.init('sapi5')
voices = Assistant.getProperty('voices')
def tts(text, rate=150, vid=1):
    Assistant.setProperty('voices', voices[vid].id)
    Assistant.setProperty('rate', rate)
    Assistant.say(text)
    Assistant.runAndWait()

