📑 ADVANCED CONVERSATIONAL VOICE ASSISTANT

1. Executive Summary
This project outlines the architecture and implementation of a modern, low-latency, conversational voice assistant built natively to support modern Python runtime footprints (Python 3.14+). Moving past rigid, keyword-dependent legacy frameworks, this assistant implements Open-Ended Cognition using a Large Language Model (LLM) alongside Dynamic Functional Routing to interface with live system applications and third-party APIs.


2. System Architecture & Data Flow
The system decouples operational steps into a linear pipeline: Perception (Ingestion) -> Cognition (Intent Extraction & AI) -> Action (Execution & Speech Synthesis).

[User Acoustic Input]
       │
       ▼
[sounddevice VAD] ────> Evaluates live amplitude arrays in 200ms windows
       │
       ▼
[SpeechRecognition] ──> Transient in-memory transcription via STT API
       │
       ▼
[Intent Router]
       │
       ├─► (True Match) ─────► [Local Exec / API Tool Matches]
       │
       └─► (Fallback Match) ─► [Gemini-2.5-Flash LLM Brain]
                                     │
                                     ▼
                               [edge-tts Engine] (Generates high-fidelity MP3)
                                     │
                                     ▼
                               [playsound Native OS Playback Layer]


3. Subsystem Breakdown

3.1 Perception: Stream Ingestion & Voice Activity Detection (VAD)
To bypass C++ compilation blocks caused by legacy dependencies like PyAudio on modern Windows/Python 3.14 installations, the assistant interfaces directly with Windows audio streams via NumPy arrays using sounddevice.

* Sample Rate: 16000 Hz (Standardized acoustic matrix size for vocal transcription models).
* VAD Windowing: Processes inputs in 200 ms chunks.
* Early Terminate Condition: The system samples peak window amplitude. If the peak amplitude falls below a value of 250 consistently for more than 1.3 seconds after speech starts, the microphone actively clips the recording line. This completely eliminates the lag of standard fixed-window timers.

3.2 Cognition: Hybrid Semantic Router & Context Management
When a sentence is transcribed, it is evaluated by an explicit hierarchy processing system:

1. System Directives: Immediate local system exits (stop, exit).
2. Local Tool Matching: High-priority regular expressions map specific functional intents such as live meteorological calculations (OpenWeatherMap API via requests) or OS interface orchestration (Google Search via webbrowser).
3. Conversational Fallback Engine: If zero deterministic tools are triggered, the raw phrase passes to the gemini-2.5-flash model.

Memory Management: The assistant wraps the payload in a rolling 6-turn history buffer, allowing it to interpret relative conversational context and trailing pronouns (e.g., "Who is the president?" -> "How old is he?") without unbounded memory leaks or latency drops.

3.3 Action: Low-Latency Speech Synthesis (TTS)
Instead of executing local, machine-dependent text-to-speech engine drivers (pyttsx3/SAPI5) which frequently break on newer Python runtimes, this architecture offloads acoustic compilation to Microsoft’s asynchronous edge-tts API. 

The resulting stream generates a transient, in-memory structured .mp3 object which is targeted by playsound to invoke native Windows audio calls, bypassing any local multimedia compiler needs.


4. API Interface Metrics

4.1 Weather API (OpenWeatherMap)
* Endpoint: http://api.openweathermap.org/data/2.5/weather
* Data Protocol: REST JSON injection.
* Payload Variables: 'q' (Target City Profile), 'appid' (Secure Token Matrix), 'units' (Metric Standardized Systems).

4.2 Core LLM Engine (Google Gemini)
* Model Configuration: gemini-2.5-flash
* System Prompt Guardrails: "You are an empathetic, concise smart speaker named Alexa. CRITICAL: Keep your answers brief, friendly, and under two sentences so they sound good over audio."
* Optimization Goal: Reduces the generated token array size, maintaining conversational audio responses at a targeted latency threshold of less than 1.5 seconds from end-of-speech to initial vocal output.



5. Security & Isolation Profiling
* Credential Protection: Secrets are stored entirely out of line inside an isolated, untracked local configuration file (assistant_config.json).
* Memory Footprint Safety: Captured sound arrays are held in a virtual io.BytesIO() file buffer. Raw user audio is never written to disk, ensuring privacy and eliminating persistent ambient room tracking risks.