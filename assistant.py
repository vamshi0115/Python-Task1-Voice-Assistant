import asyncio
import datetime
import io
import json
import os
import re
import threading
import time
import webbrowser

import speech_recognition as sr
import requests
import httpx
import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
from bs4 import BeautifulSoup
from ddgs import DDGS

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
# AUDIO OUTPUT SYSTEM
# ==========================================
def speak(text: str):
    """Generates lifelike audio from text and plays it back."""
    print(f"\nAssistant: {text}")
    audio_file = "response.mp3"

    async def generate_speech():
        communicate = edge_tts.Communicate(text, "en-US-AriaNeural")
        await communicate.save(audio_file)

    try:
        asyncio.run(generate_speech())
        playsound(audio_file)
        if os.path.exists(audio_file):
            os.remove(audio_file)
    except Exception as e:
        print(f"[Audio Playback Error]: {e}")

# ==========================================
# MICROPHONE WITH AUTOMATIC SILENCE DETECTION
# ==========================================
def listen() -> str:
    """Listens continuously; automatically cuts off when you stop speaking."""
    sample_rate = 16000
    chunk_duration = 0.2
    chunk_samples = int(sample_rate * chunk_duration)

    silence_threshold = 250
    max_silence_duration = 1.3
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
# REAL-TIME WEB ACCESS LAYER
# ==========================================

_http_headers = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0 Safari/537.36"
    )
}

def _fetch_page_text(url: str, char_limit: int = 3000) -> str:
    """Fetches a URL and returns cleaned visible text, capped at char_limit."""
    try:
        resp = httpx.get(url, headers=_http_headers, timeout=8, follow_redirects=True)
        soup = BeautifulSoup(resp.text, "lxml")
        # Remove non-content tags
        for tag in soup(["script", "style", "nav", "footer", "header", "aside", "form"]):
            tag.decompose()
        text = " ".join(soup.get_text(separator=" ").split())
        return text[:char_limit]
    except Exception as e:
        return ""


def web_search_live(query: str, max_results: int = 4) -> str:
    """
    Performs a real-time DuckDuckGo web search and returns a condensed
    summary string combining titles, snippets, and fetched page text.
    """
    print(f"[Web] Searching: {query}")
    context_parts = []
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
    except Exception as e:
        print(f"[Web Search Error]: {e}")
        return ""

    for i, r in enumerate(results):
        title   = r.get("title", "")
        snippet = r.get("body", "")
        url     = r.get("href", "")
        context_parts.append(f"[{i+1}] {title}\n{snippet}")
        # Fetch full page text from the top 2 results for deeper context
        if i < 2 and url:
            page_text = _fetch_page_text(url, char_limit=2000)
            if page_text:
                context_parts.append(f"  >> Page content: {page_text}")

    return "\n\n".join(context_parts)


def fetch_wikipedia_summary(topic: str) -> str:
    """Returns a Wikipedia introduction for the given topic."""
    print(f"[Wikipedia] Fetching: {topic}")
    try:
        url = "https://en.wikipedia.org/api/rest_v1/page/summary/" + topic.replace(" ", "_")
        resp = requests.get(url, headers=_http_headers, timeout=8)
        data = resp.json()
        return data.get("extract", "")
    except Exception as e:
        print(f"[Wikipedia Error]: {e}")
        return ""


def fetch_news_headlines(topic: str = "top news", max_results: int = 5) -> str:
    """Fetches the latest news headlines via DuckDuckGo News."""
    print(f"[News] Fetching headlines for: {topic}")
    headlines = []
    try:
        with DDGS() as ddgs:
            results = list(ddgs.news(topic, max_results=max_results))
        for r in results:
            headlines.append(f"- {r.get('title','')} ({r.get('source','')})")
    except Exception as e:
        print(f"[News Error]: {e}")
    return "\n".join(headlines)


def fetch_crypto_price(symbol: str) -> str:
    """Fetches current crypto price from CoinGecko (free, no key needed)."""
    print(f"[Crypto] Fetching: {symbol}")
    id_map = {
        "bitcoin": "bitcoin", "btc": "bitcoin",
        "ethereum": "ethereum", "eth": "ethereum",
        "dogecoin": "dogecoin", "doge": "dogecoin",
        "solana": "solana", "sol": "solana",
    }
    coin_id = id_map.get(symbol.lower().strip(), symbol.lower().strip())
    try:
        url = f"https://api.coingecko.com/api/v3/simple/price?ids={coin_id}&vs_currencies=usd"
        resp = requests.get(url, headers=_http_headers, timeout=8).json()
        price = resp.get(coin_id, {}).get("usd")
        if price:
            return f"{coin_id.capitalize()} is currently trading at ${price:,.2f} USD."
    except Exception as e:
        print(f"[Crypto Error]: {e}")
    return ""


# ==========================================
# AI BRAIN — WEB-GROUNDED RESPONSES
# ==========================================

def _build_grounded_prompt(user_input: str, web_context: str) -> str:
    today = datetime.datetime.now().strftime("%A, %B %d %Y, %I:%M %p")
    history_block = "\n".join(conversation_history[-6:]) if conversation_history else ""
    return (
        f"You are a smart, concise voice assistant named Alexa with live internet access. "
        f"Today is {today}.\n"
        f"IMPORTANT: Use the live web data below to answer accurately. "
        f"Keep your spoken answer under three sentences — clear and natural for audio.\n\n"
        f"=== LIVE WEB DATA ===\n{web_context}\n=== END WEB DATA ===\n\n"
        f"{history_block}\n"
        f"User: {user_input}\nAssistant:"
    )


def chat_with_ai(user_input: str, web_context: str = ""):
    """Routes conversational inputs to Gemini AI, optionally grounded with live web data."""
    global conversation_history
    if not ai_client:
        speak("Please add a valid Gemini API key to the configuration file.")
        return

    conversation_history.append(f"User: {user_input}")
    if len(conversation_history) > 12:
        conversation_history = conversation_history[-12:]

    if web_context:
        prompt = _build_grounded_prompt(user_input, web_context)
    else:
        today = datetime.datetime.now().strftime("%A, %B %d %Y, %I:%M %p")
        history_block = "\n".join(conversation_history[-6:])
        prompt = (
            f"You are a concise voice assistant named Alexa. Today is {today}. "
            f"Keep answers under two sentences for audio.\n\n"
            f"{history_block}\nAssistant:"
        )

    try:
        response = ai_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )
        reply = response.text.strip()
        # Strip markdown formatting that sounds bad over audio
        reply = re.sub(r'\*+', '', reply)
        reply = re.sub(r'#+\s*', '', reply)
        conversation_history.append(f"Assistant: {reply}")
        speak(reply)
    except Exception as e:
        print(f"[AI Error]: {e}")
        speak("I am having trouble reaching my AI servers right now.")


# ==========================================
# CORE FEATURE HANDLERS
# ==========================================

def get_weather():
    api_key = CONFIG.get("weather_api_key")
    city = CONFIG.get("weather_city", "karimnagar")
    if api_key == "YOUR_OPENWEATHERMAP_API_KEY":
        speak("Please complete setting up your OpenWeatherMap key in the config file.")
        return
    url = (
        f"http://api.openweathermap.org/data/2.5/weather"
        f"?q={city}&appid={api_key}&units=metric"
    )
    try:
        data = requests.get(url, timeout=8).json()
        if data.get("cod") == 200:
            temp  = data["main"]["temp"]
            feels = data["main"]["feels_like"]
            desc  = data["weather"][0]["description"]
            humidity = data["main"]["humidity"]
            speak(
                f"The weather in {city} is {temp}°C, feels like {feels}°C, "
                f"with {desc} and {humidity}% humidity."
            )
        else:
            speak(f"I could not find weather data for {city}.")
    except Exception:
        speak("I could not contact the weather server right now.")


def handle_news(text: str):
    """Fetches and reads top news headlines."""
    topic = re.sub(r'\b(news|headlines|latest|tell me|about|on)\b', '', text).strip() or "top news"
    headlines = fetch_news_headlines(topic, max_results=5)
    if headlines:
        speak(f"Here are the latest headlines on {topic}:")
        speak(headlines)
    else:
        speak("I could not fetch news at this moment.")


def handle_crypto(text: str):
    """Handles cryptocurrency price queries."""
    tokens = ["bitcoin", "btc", "ethereum", "eth", "dogecoin", "doge", "solana", "sol"]
    found = next((t for t in tokens if t in text), None)
    if found:
        result = fetch_crypto_price(found)
        if result:
            speak(result)
            return
    # Fall back to web search
    handle_web_query(text)


def handle_web_query(text: str):
    """
    Central handler for any question that needs live data.
    Fetches web context then asks Gemini to summarise it.
    """
    # Strip filler trigger words
    query = re.sub(
        r'\b(search for|search|google for|look up|find|tell me about|what is|'
        r'who is|when is|where is|how to|how do|explain)\b', '', text
    ).strip()
    query = query or text

    web_context = web_search_live(query)
    if not web_context:
        speak("I searched the web but could not retrieve any results.")
        return
    chat_with_ai(text, web_context=web_context)


def open_browser_search(text: str):
    """Opens a browser search window for the query."""
    query = re.sub(r'\b(open|browser|search for|search|google|look up)\b', '', text).strip()
    if query:
        speak("Opening a browser search window.")
        webbrowser.open(f"https://www.google.com/search?q={query}")


# ==========================================
# INTENT ROUTING
# ==========================================

def process_intent(text: str) -> bool:
    if not text:
        return True

    # ── Exit ──────────────────────────────────────────────────────────────
    if any(k in text for k in ["stop", "exit", "quit", "goodbye", "bye"]):
        speak("Goodbye! Have a great day.")
        return False

    # ── Time / Date ───────────────────────────────────────────────────────
    if any(k in text for k in ["what time", "current time", "what date", "today's date",
                                "what day", "day is it", "date today"]):
        now = datetime.datetime.now()
        speak(f"It is {now.strftime('%I:%M %p')} on {now.strftime('%A, %B %d, %Y')}.")

    # ── Weather ───────────────────────────────────────────────────────────
    elif "weather" in text or "temperature" in text or "forecast" in text:
        get_weather()

    # ── News ──────────────────────────────────────────────────────────────
    elif any(k in text for k in ["news", "headlines", "latest news"]):
        handle_news(text)

    # ── Crypto / Stock prices ─────────────────────────────────────────────
    elif any(k in text for k in ["bitcoin", "btc", "ethereum", "eth",
                                  "dogecoin", "doge", "solana", "crypto", "price of"]):
        handle_crypto(text)

    # ── Wikipedia lookup ─────────────────────────────────────────────────
    elif re.search(r'\b(wikipedia|wiki|who is|what is|biography of)\b', text):
        topic = re.sub(r'\b(wikipedia|wiki|who is|what is|biography of|tell me about)\b', '', text).strip()
        wiki = fetch_wikipedia_summary(topic)
        if wiki:
            chat_with_ai(text, web_context=f"Wikipedia summary:\n{wiki}")
        else:
            handle_web_query(text)

    # ── Open browser ──────────────────────────────────────────────────────
    elif any(k in text for k in ["open browser", "open google", "open search"]):
        open_browser_search(text)

    # ── Live web questions (anything that hints at current info) ──────────
    elif re.search(
        r'\b(who|what|when|where|how|why|latest|current|today|'
        r'right now|live|score|result|update|release|price|'
        r'stock|rate|fact|explain|define|tell me)\b', text
    ):
        handle_web_query(text)

    # ── Pure conversational chat ──────────────────────────────────────────
    else:
        chat_with_ai(text)

    return True


# ==========================================
# MAIN ROUTINE
# ==========================================
if __name__ == "__main__":
    speak("System active. I now have real-time internet access and can answer any question.")
    running = True
    while running:
        command = listen()
        if command:
            running = process_intent(command)
