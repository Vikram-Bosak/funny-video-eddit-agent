import os
import sys
import asyncio
import subprocess
from loguru import logger
from memory_agent import async_get_latest_video_id, async_get_memory, async_update_memory

async def generate_voice():
    video_id = await async_get_latest_video_id()
    if not video_id:
        logger.error("No video_id found in memory.")
        sys.exit(1)
        
    memory = await async_get_memory(video_id)
    script = memory.generated_script
    
    if not script:
        logger.error("Error: Script not found in memory.")
        sys.exit(1)
        
    logger.info("Generating voiceover using piper-tts...")
    
    os.makedirs("audio", exist_ok=True)
    voiceover_path = f"audio/{video_id}_voice.wav"
    
    try:
        model_file = "kokoro-v1.0.onnx"
        voices_file = "voices-v1.0.bin"
        
        # Download files if they do not exist
        import urllib.request
        if not os.path.exists(model_file):
            logger.info(f"Downloading {model_file} (approx. 82MB)...")
            url = f"https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/{model_file}"
            urllib.request.urlretrieve(url, model_file)
            
        if not os.path.exists(voices_file):
            logger.info(f"Downloading {voices_file} (approx. 20MB)...")
            url = f"https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/{voices_file}"
            urllib.request.urlretrieve(url, voices_file)
            
        from kokoro_onnx import Kokoro
        import soundfile as sf
        import hashlib
        
        logger.info("Initializing Kokoro TTS model...")
        kokoro = Kokoro(model_file, voices_file)
        
        selected_voice = "af_sarah"
        lang_code = "en-us"
        
        logger.info(f"Synthesizing speech using flagship realistic female voice: {selected_voice}")
        samples, sample_rate = kokoro.create(script, voice=selected_voice, speed=1.0, lang=lang_code)
        
        sf.write(voiceover_path, samples, sample_rate)
        
        await async_update_memory(video_id, {"voiceover_file": voiceover_path})
        logger.success("Voice generation via Kokoro complete.")
        
    except Exception as e:
        logger.error(f"Error during voice generation: {e}")
        await async_update_memory(video_id, {"error": str(e)})
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(generate_voice())
