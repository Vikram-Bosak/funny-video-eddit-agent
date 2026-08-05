import os
import sys
import asyncio
import httpx
from datetime import datetime
from loguru import logger
from memory_agent import async_get_latest_video_id, async_get_memory

def truncate_str(text, max_len=300):
    if not text:
        return "N/A"
    return text[:max_len] + "..." if len(text) > max_len else text

async def send_report():
    video_id = await async_get_latest_video_id()
    if not video_id:
        logger.error("No video_id found in memory.")
        sys.exit(1)
        
    memory = await async_get_memory(video_id)
    webhook_url = os.environ.get("DISCORD_WEBHOOK_URL")
    
    if not webhook_url:
        logger.warning("DISCORD_WEBHOOK_URL not set. Skipping report.")
        return
        
    logger.info("Preparing detailed report for Discord...")
    
    # Calculate execution time
    duration_str = "N/A"
    if memory.start_time and memory.end_time:
        try:
            start_dt = datetime.fromisoformat(memory.start_time)
            end_dt = datetime.fromisoformat(memory.end_time)
            diff = end_dt - start_dt
            seconds = int(diff.total_seconds())
            mins, secs = divmod(seconds, 60)
            duration_str = f"{mins}m {secs}s" if mins > 0 else f"{secs}s"
        except Exception as e:
            logger.warning(f"Failed to calculate duration: {e}")
            
    # Format and truncate fields to fit Discord Embed limits
    title = truncate_str(memory.original_title, 100)
    desc = truncate_str(memory.original_description, 150)
    transcript = truncate_str(memory.transcript, 300)
    ocr = truncate_str(memory.ocr_text, 100)
    translation = truncate_str(memory.translation, 300)
    summary = truncate_str(memory.summary, 200)
    script = truncate_str(memory.generated_script, 400)
    drive_url = memory.google_drive_public_url or 'N/A'
    source_url = memory.source_url or 'N/A'
    
    has_failed = memory.error is not None and memory.error != ""
    if has_failed:
        embed_title = "❌ AI Video Automation Pipeline Failed!"
        embed_desc = f"The pipeline encountered an error: **{truncate_str(memory.error, 300)}**"
        embed_color = 15158332  # Red color code
    else:
        embed_title = "🚀 AI Video Automation Pipeline Completed!"
        embed_desc = "The video editing pipeline has finished executing successfully."
        embed_color = 5763719  # Green color code
        
    embed = {
        "title": embed_title,
        "description": embed_desc,
        "color": embed_color,
        "fields": [
            {
                "name": "📥 1. Downloaded Video Info",
                "value": f"**Title:** {title}\n**Desc:** {desc}\n**Source:** [Twitter/X Link]({source_url})"
            },
            {
                "name": "🔍 2. Video Analysis",
                "value": f"**Summary:** {summary}\n**OCR:** {ocr}\n**Transcript:**\n```\n{transcript}\n```"
            },
            {
                "name": "✍️ 3. AI Generated Script",
                "value": f"```\n{script}\n```"
            },
            {
                "name": "🎬 4. Final Output Link",
                "value": f"🔗 [Google Drive Public Link]({drive_url})"
            },
            {
                "name": "⚙️ 5. Metrics & Github",
                "value": f"**Time:** {duration_str}\n**Run:** [Github Action Run]({memory.github_run_url or 'https://github.com'})"
            }
        ],
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "footer": {
            "text": f"Run ID: {memory.github_run_id or 'N/A'}"
        }
    }
    
    payload = {"embeds": [embed]}
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(webhook_url, json=payload)
            response.raise_for_status()
            
        logger.success("Detailed embed report sent to Discord successfully.")
    except Exception as e:
        logger.error(f"Error sending report to Discord: {e}")
        logger.warning("Continuing despite Discord reporting failure.")

if __name__ == "__main__":
    asyncio.run(send_report())
