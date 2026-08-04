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
        # Download a tiny Piper model if not exists
        model_name = "en_US-lessac-low.onnx"
        model_url = f"https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/low/{model_name}"
        
        if not os.path.exists(model_name):
            logger.info(f"Downloading piper model {model_name}...")
            subprocess.run(["wget", "-q", model_url], check=True)
            subprocess.run(["wget", "-q", f"{model_url}.json"], check=True)
            
        # Run piper via CLI
        # echo "text" | piper --model en_US-lessac-low.onnx --output_file out.wav
        process = subprocess.Popen(['piper', '--model', model_name, '--output_file', voiceover_path], 
                                   stdin=subprocess.PIPE, 
                                   stdout=subprocess.PIPE, 
                                   stderr=subprocess.PIPE)
        stdout, stderr = process.communicate(input=script.encode('utf-8'))
        
        if process.returncode != 0:
            logger.error(f"Piper error: {stderr.decode()}")
            raise Exception("Piper TTS failed")
            
        await async_update_memory(video_id, {"voiceover_file": voiceover_path})
        logger.success("Voice generation complete.")
        
    except Exception as e:
        logger.error(f"Error during voice generation: {e}")
        await async_update_memory(video_id, {"error": str(e)})
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(generate_voice())
