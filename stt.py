import os
import asyncio
import sounddevice as sd
import numpy as np
from dotenv import load_dotenv

from elevenlabs.client import ElevenLabs
from elevenlabs import RealtimeEvents

load_dotenv()

SAMPLE_RATE = 16000
CHUNK_SIZE = 1024

async def transcribe_live():
    client = ElevenLabs(api_key=os.getenv("ELEVENLABS_API_KEY"))

    connection = await client.speech_to_text.realtime.connect()

    @connection.on(RealtimeEvents.TRANSCRIPT_PARTIAL)
    def on_partial(data):
        print("Partial:", data.text)

    @connection.on(RealtimeEvents.TRANSCRIPT_COMMIT)
    def on_final(data):
        print("Final:", data.text)

    @connection.on(RealtimeEvents.ERROR)
    def on_error(err):
        print("Error:", err)

    print("🎤 Speak now...")

    loop = asyncio.get_event_loop()

    def callback(indata, frames, time, status):
        if status:
            print(status)

        audio = (indata * 32767).astype(np.int16)

        asyncio.run_coroutine_threadsafe(
            connection.send_audio(audio.tobytes()),
            loop
        )

    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
        blocksize=CHUNK_SIZE,
        callback=callback
    ):
        try:
            while True:
                await asyncio.sleep(0.1)
        except KeyboardInterrupt:
            print("\nStopping...")
        finally:
            await connection.close()

if __name__ == "__main__":
    asyncio.run(transcribe_live())