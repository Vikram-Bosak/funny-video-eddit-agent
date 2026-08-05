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
    video_summary = memory.summary or "No visual summary available."
    original_title = memory.original_title or "N/A"
    original_description = memory.original_description or "N/A"
    
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
    You are an expert, highly engaging US social media scriptwriter and storyteller.
    Write a short, funny, and energetic voiceover script for a viral video that is DIRECTLY and TIGHTLY aligned with the actual visual content of the video.
    
    Here is the metadata and analysis of the video content:
    - Video Title: {original_title}
    - Video Description: {original_description}
    - Visual Scene Summary: {video_summary}
    - Detected Objects: {objects_str}
    - Spoken Words / Original Dialogue: {transcript}
    - On-screen Text (OCR): {ocr_text}
    
    CRITICAL INSTRUCTIONS:
    1. The voiceover script MUST match the visuals described in the Visual Scene Summary and Detected Objects. For example, if the video shows a person playing with cats, the script must talk about cats/playing/the interaction shown. Do NOT hallucinate an unrelated story.
    2. If the original dialogue is empty ("No dialogue detected" or "No spoken words"), write a completely new, engaging story or commentary describing the funny, exciting, or interesting actions happening in the video.
    3. If there is original dialogue, do not use it raw. Instead, rewrite it to be more natural, engaging, and better suited for viral storytelling, while keeping the same meaning and matching the video's context.
    4. Write only the spoken script in engaging English. Do not include stage directions, speaker names, brackets, or visual cues. Write only the exact words to be spoken.
    5. Keep it under 60 seconds (around 80-120 words maximum).
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
