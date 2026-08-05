import os
import sys
import json
import asyncio
from loguru import logger
from openai import OpenAI
from memory_agent import async_get_latest_video_id, async_get_memory, async_update_memory

async def write_script():
    video_id = await async_get_latest_video_id()
    if not video_id:
        logger.error("No video_id found in memory.")
        sys.exit(1)
        
    memory = await async_get_memory(video_id)
    
    transcript = memory.transcript or "No spoken words detected."
    ocr_text = memory.ocr_text or "No text detected."
    
    # Parse scene analysis to extract detected objects
    objects_detected = set()
    if memory.scene_analysis:
        try:
            scenes = json.loads(memory.scene_analysis)
            for scene in scenes:
                if "objects" in scene:
                    objects_detected.update(scene["objects"])
        except Exception:
            pass
            
    objects_str = ", ".join(list(objects_detected)) if objects_detected else "No specific objects detected."
    
    # Use Environment Variable or fallback to the provided key
    api_key = os.environ.get("NVIDIA_API_KEY", "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")

    logger.info("Generating script using NVIDIA Nemotron LLM...")
    
    client = OpenAI(
      base_url = "https://integrate.api.nvidia.com/v1",
      api_key = api_key
    )
    
    prompt = f"""
    You are an expert, highly engaging US YouTube/TikTok scriptwriter.
    Write a short, funny, and energetic voiceover script for a viral video specifically targeting a United States (US) audience:
    
    1. Spoken words in the video: {transcript}
    2. Text visible on screen (OCR): {ocr_text}
    3. Objects detected in the video: {objects_str}
    
    Requirements:
    - Write the script in engaging American English, utilizing US slang, idioms, and humor where appropriate.
    - The tone must be exciting, witty, and highly relatable to US social media viewers.
    - Don't use placeholders. Write the exact words to be spoken.
    - Keep it under 60 seconds (around 100-150 words).
    - Do not include stage directions or visual cues, ONLY the spoken words.
    """
    
    try:
        def generate():
            completion = client.chat.completions.create(
              model="nvidia/nemotron-3-ultra-550b-a55b",
              messages=[{"role":"user","content": prompt}],
              temperature=1,
              top_p=0.95,
              max_tokens=16384,
              extra_body={"chat_template_kwargs":{"enable_thinking":True},"reasoning_budget":16384},
              stream=False
            )
            return completion.choices[0].message.content.strip()
            
        generated_script = await asyncio.to_thread(generate)
        
        await async_update_memory(video_id, {"generated_script": generated_script})
        logger.success("Script generation via NVIDIA LLM complete.")
        
    except Exception as e:
        logger.error(f"Error during NVIDIA LLM script generation: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(write_script())
