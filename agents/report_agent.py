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
            
    # Format and truncate fields to fit Discord's 2000 character limit
    title = truncate_str(memory.original_title, 150)
    desc = truncate_str(memory.original_description, 200)
    transcript = truncate_str(memory.transcript, 400)
    ocr = truncate_str(memory.ocr_text, 150)
    translation = truncate_str(memory.translation, 400)
    summary = truncate_str(memory.summary, 250)
    script = truncate_str(memory.generated_script, 500)
    
    report_message = f"""
## 🚀 AI Video automation pipeline completed!

### 📥 1. Downloaded Video Info
* **Title:** {title}
* **Description:** {desc}
* **Source Twitter/X URL:** {memory.source_url or 'N/A'}

### 🔍 2. Video Analysis
* **Transcript:** 
```
{transcript}
```
* **OCR Text (On-screen):** {ocr}
* **English Translation:** {translation}
* **Content Summary:** {summary}

### ✍️ 3. AI Generated Script
```
{script}
```

### 🎬 4. Final Output Links
* **Google Drive Public Share URL:** {memory.google_drive_public_url or 'N/A'}
* **Original Twitter/X Source URL:** {memory.source_url or 'N/A'}

### ⚙️ 5. Workflow Metrics
* **Start Time (UTC):** {memory.start_time or 'N/A'}
* **End Time (UTC):** {memory.end_time or 'N/A'}
* **Total Execution Time:** {duration_str}

### 💻 6. GitHub Actions Info
* **Repository:** {memory.github_repository or 'N/A'}
* **Run ID:** {memory.github_run_id or 'N/A'}
* **Run URL:** {memory.github_run_url or 'N/A'}
"""
    
    try:
        async with httpx.AsyncClient() as client:
            # Send text only (no video file attachment to avoid Discord size limit issues)
            response = await client.post(webhook_url, json={"content": report_message})
            response.raise_for_status()
            
        logger.success("Detailed report sent to Discord successfully.")
    except Exception as e:
        logger.error(f"Error sending report to Discord: {e}")
        logger.warning("Continuing despite Discord reporting failure.")

if __name__ == "__main__":
    asyncio.run(send_report())
