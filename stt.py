import asyncio
import base64
import inspect
import json
import os
import sys
from urllib.parse import urlencode

import sounddevice as sd
import websockets
from dotenv import load_dotenv

load_dotenv()

SAMPLE_RATE = 16000
CHUNK_SECONDS = 0.1
CHUNK_FRAMES = int(SAMPLE_RATE * CHUNK_SECONDS)
CHANNELS = 1
QUEUE_MAXSIZE = 50
API_KEY = os.getenv("ELEVENLABS_API_KEY")

if not API_KEY:
    raise RuntimeError("ELEVENLABS_API_KEY is missing from the environment.")


def build_ws_url() -> str:
    params = {
        "model_id": "scribe_v2_realtime",
        "audio_format": "pcm_16000",
        "include_timestamps": "true",
        "commit_strategy": "vad",
        "vad_silence_threshold_secs": "1.0",
    }
    return "wss://api.elevenlabs.io/v1/speech-to-text/realtime?" + urlencode(params)


async def transcribe_live() -> None:
    audio_q: asyncio.Queue = asyncio.Queue(maxsize=QUEUE_MAXSIZE)
    stop_event = asyncio.Event()
    ws_url = build_ws_url()

    def audio_callback(indata, frames, time, status):
        if status:
            print(status)
        try:
            audio_q.put_nowait(bytes(indata))
        except asyncio.QueueFull:
            pass

    async def sender(ws):
        while not stop_event.is_set():
            chunk = await audio_q.get()
            if chunk is None:
                break
            payload = {
                "message_type": "input_audio_chunk",
                "audio_base_64": base64.b64encode(chunk).decode("ascii"),
                "commit": False,
                "sample_rate": SAMPLE_RATE,
            }
            await ws.send(json.dumps(payload))

    async def receiver(ws):
        async for message in ws:
            try:
                data = json.loads(message)
            except Exception:
                print(message)
                continue

            message_type = data.get("message_type")
            if message_type == "session_started":
                print("Session started")
            elif message_type == "partial_transcript":
                text = data.get("text", "")
                if text:
                    print("Partial:", text)
            elif message_type == "committed_transcript":
                text = data.get("text", "")
                if text:
                    print("Final:", text)
            elif message_type == "committed_transcript_with_timestamps":
                text = data.get("text", "")
                if text:
                    print("Final:", text)
            elif isinstance(message_type, str) and message_type.endswith("error"):
                print("Error:", data)
            else:
                print(data)

    connect_sig = inspect.signature(websockets.connect)
    header_kw = "additional_headers" if "additional_headers" in connect_sig.parameters else "extra_headers"
    connect_kwargs = {
        header_kw: [("xi-api-key", API_KEY)],
        "ping_interval": 20,
        "ping_timeout": 20,
        "max_queue": 32,
    }

    print("Speak now. Press Ctrl+C to stop.")
    try:
        async with websockets.connect(ws_url, **connect_kwargs) as ws:
            sender_task = asyncio.create_task(sender(ws))
            receiver_task = asyncio.create_task(receiver(ws))

            with sd.RawInputStream(
                samplerate=SAMPLE_RATE,
                channels=CHANNELS,
                dtype="int16",
                blocksize=CHUNK_FRAMES,
                callback=audio_callback,
            ):
                try:
                    while True:
                        await asyncio.sleep(0.1)
                except KeyboardInterrupt:
                    print("\nStopping...")
                finally:
                    stop_event.set()
                    await audio_q.put(None)
                    sender_task.cancel()
                    receiver_task.cancel()
                    await ws.close()
    finally:
        stop_event.set()


if __name__ == "__main__":
    if sys.platform.startswith("win"):
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(transcribe_live())
