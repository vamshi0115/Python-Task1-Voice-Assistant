import asyncio
import datetime
import io
import json
import os
import re
import smtplib
import threading
import time
import webbrowser
from email.message import EmailMessage

import speech_recognition as sr
import requests
import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav

# Low-latency voice and playback engines
import edge_tts
from playsound import playsound

# Conversational AI Brain
from google import genai

# ==========================================
# CONFIGURATION & CONSTANTS
# ==========================================
CONFIG_FILE = "assistant_config.json"
DEFAULT_CONFIG = {
    "gemini_api_key": "YOUR_GEMINI_API_KEY",
    "weather_api_key": "YOUR_OPENWEATHERMAP_API_KEY",
    "weather_city": "karimnagar",
    "smtp_server": "smtp.gmail.com",
    "smtp_port": 465,
    "sender_email": "your_test_email@gmail.com",
    "sender_password": "your_app_password",
    "user_name": "User"
}

if not os.path.exists(CONFIG_FILE):
    with open(CONFIG_FILE, "w") as f:
        json.dump(DEFAULT_CONFIG, f, indent=4)
    CONFIG = DEFAULT_CONFIG
else:
    with open(CONFIG_FILE, "r") as f:
        CONFIG = json.load(f)

# ==========================================
# ENGINE INITIALIZATION
# ==========================================
recognizer = sr.Recognizer()
ai_client = None
if CONFIG.get("gemini_api_key") and CONFIG["gemini_api_key"] != "YOUR_GEMINI_API_KEY":
    ai_client = genai.Client(api_key=CONFIG["gemini_api_key"])

conversation_history = []

# ==========================================
# AUDIO OUTPUT SYSTEM (COMPILATION FREE)
# ==========================================
def speak(text: str):
    """Generates lifelike audio from text and plays it back with zero backend drivers."""
    print(f"Assistant: {text}")
    audio_file = "response.mp3"
    
    async def generate_speech():
        # Premium natural voice profile
        communicate = edge_tts.Communicate(text, "en-US-AriaNeural")
        await communicate.save(audio_file)
        
    try:
        # Generate the audio file using edge-tts web service
        asyncio.run(generate_speech())
        
        # Play out loud (This automatically blocks execution until done speaking)
        playsound(audio_file)
        
        # Clean up the file so it can be cleanly re-written on the next loop
        if os.path.exists(audio_file):
            os.remove(audio_file)
            
    except Exception as e:
        print(f"[Audio Playback Error]: {e}")

# ==========================================
# SOUNDDEVICE MICROPHONE WITH AUTOMATIC SILENCE DETECTION
# ==========================================
def listen() -> str:
    """Listens continuously; automatically cuts off when you stop speaking."""
    sample_rate = 16000
    chunk_duration = 0.2  
    chunk_samples = int(sample_rate * chunk_duration)
    
    silence_threshold = 250     # Adjusted lower to catch quieter speaking environments
    max_silence_duration = 1.3  # Stops recording after 1.3 seconds of continuous silence
    max_total_duration = 7.0    
    
    print("\nListening...", end="", flush=True)
    
    audio_buffer = []
    speaking_started = False
    silence_chunks = 0
    total_chunks = 0
    
    try:
        with sd.InputStream(samplerate=sample_rate, channels=1, dtype='int16') as stream:
            while True:
                data, _ = stream.read(chunk_samples)
                audio_buffer.append(data)
                total_chunks += 1
                
                amplitude = np.max(np.abs(data))
                
                if amplitude > silence_threshold:
                    if not speaking_started:
                        speaking_started = True
                    silence_chunks = 0
                else:
                    if speaking_started:
                        silence_chunks += 1
                
                if speaking_started and (silence_chunks * chunk_duration >= max_silence_duration):
                    break
                if total_chunks * chunk_duration >= max_total_duration:
                    break
                    
        print(" Processing...")
        
        if not speaking_started or not audio_buffer:
            return ""
            
        full_audio = np.concatenate(audio_buffer, axis=0)
        wav_io = io.BytesIO()
        wav.write(wav_io, sample_rate, full_audio)
        wav_io.seek(0)
        
        with sr.AudioFile(wav_io) as source:
            audio_content = recognizer.record(source)
            
        query = recognizer.recognize_google(audio_content)
        print(f"You said: {query}")
        return query.lower()
        
    except sr.UnknownValueError:
        return ""
    except Exception as e:
        print(f"\n[Microphone Input Error]: {e}")
        return ""

# ==========================================
# CORE FEATURES & TOOL MATCHING
# ==========================================
def get_weather():
    api_key = CONFIG.get("weather_api_key")
    city = CONFIG.get("weather_city")
    if api_key == "YOUR_OPENWEATHERMAP_API_KEY":
        speak("Please complete setting up your OpenWeatherMap key.")
        return
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
    try:
        response = requests.get(url).json()
        if response.get("cod") == 200:
            speak(f"The weather in {city} is currently {response['main']['temp']}°C with {response['weather'][0]['description']}.")
    except:
        speak("I couldn't contact the weather server right now.")

def web_search(text: str):
    query = re.sub(r'\b(search for|search|google|look up)\b', '', text).strip()
    if query:
        speak(f"Opening a browser search window.")
        webbrowser.open(f"https://www.google.com/search?q={query}")

def chat_with_ai(user_input: str):
    """Routes conversational inputs directly to Gemini AI."""
    global conversation_history
    if not ai_client:
        speak("I am operating in tool-only mode. Please add a valid Gemini API key to my configuration file.")
        return

    if len(conversation_history) > 6:
        conversation_history = conversation_history[-6:]

    conversation_history.append(f"User: {user_input}")
    context_prompt = (
        "You are an empathetic, concise smart speaker named Alexa. "
        "CRITICAL: Keep your answers brief, friendly, and under two sentences so they sound good over audio.\n"
        + "\n".join(conversation_history) + "\nAssistant:"
    )

    try:
        response = ai_client.models.generate_content(
            model='gemini-2.5-flash',
            contents=context_prompt
        )
        reply = response.text.strip()
        conversation_history.append(f"Assistant: {reply}")
        speak(reply)
    except Exception as e:
        speak("I have trouble connecting to my cloud servers right now.")

def process_intent(text: str) -> bool:
    if not text:
        return True
        
    if any(k in text for k in ["stop", "exit", "quit", "goodbye"]):
        speak("Goodbye!")
        return False

    if "weather" in text or "temperature" in text:
        get_weather()
    elif any(k in text for k in ["search for", "google for", "look up"]):
        web_search(text)
    elif any(k in text for k in ["time", "date", "day is it"]):
        now = datetime.datetime.now()
        speak(f"It is {now.strftime('%I:%M %p')}.")
    else:
        chat_with_ai(text)
    return True

# ==========================================
# MAIN ROUTINE
# ==========================================
if __name__ == "__main__":
    speak("System active.")
    running = True
    while running:
        command = listen()
        if command:
            running = process_intent(command)
