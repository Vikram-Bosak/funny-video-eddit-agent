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
    You are an expert, highly engaging social media storyteller. 
    Your task is to write a short, funny, and energetic voiceover script for a viral video.
    
    CRITICAL RULE: DO NOT write a dry description of the video (e.g. do not say "In this video we see a man doing X"). Instead, write an engaging STORY, commentary, or narrative that guides the viewer through the emotional arc of what is happening.
    
    If the video contains action, suspense, dramatic, or fictional movie scenes/trailers, treat them as pure cinematic entertainment and write an exciting, energetic, or humorous commentary for the movie fans. Do NOT refuse to write the script, and do NOT output any safety warnings or refusal messages.
    
    Use the following narrative structure:
    1. Setup/Context: Introduce the character, their goal, or the starting situation.
    2. Build-up/Suspense: Describe the action or attempt.
    3. The Twist/Funny Moment: Focus on the unexpected fail, surprise, or climax.
    4. Emotional Reaction: Describe the reaction, laughter, or aftermath.
    
    Here is the visual and textual analysis of the video content:
    - Visual Scene Summary: {video_summary}
    - Detected Objects: {objects_str}
    - Spoken Words / Original Dialogue: {transcript}
    - On-screen Text (OCR): {ocr_text}
    
    STORYTELLING GUIDELINES:
    1. The story MUST align tightly with the actual visual content. For example, if a skateboard stunt fails, tell a story about trying a skateboard trick and hitting the ground. Do NOT hallucinate unrelated characters.
    2. Write only the spoken voiceover in English. Do NOT include stage directions, bracketed instructions, speaker names, or video descriptions. Only output the exact words to be read.
    3. Rewrite any original dialogue/transcript to fit a natural, engaging viral storytelling format.
    4. Keep the script under 59 seconds (between 40 and 100 words maximum).
    """
    
    try:
        def generate():
            completion = client.chat.completions.create(
              model="meta/llama-3.1-70b-instruct",
              messages=[{"role":"user","content": prompt}],
              temperature=0.8,
              top_p=0.95,
              max_tokens=1024,
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
