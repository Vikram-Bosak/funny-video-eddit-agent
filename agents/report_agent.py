import os
import sys
import asyncio
import httpx
from loguru import logger
from memory_agent import async_get_latest_video_id, async_get_memory

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
        
    logger.info("Preparing report for Discord...")
    
    report_message = f"""
**🚀 New AI Video Processed! (Local AI Architecture)**

**Original Source:** {memory.source_url or 'N/A'}
**Title:** {memory.original_title or 'N/A'}
**Objects Detected:** {memory.ocr_text or 'N/A'}
**Generated Script:** {memory.generated_script or 'N/A'}
**Google Drive Link:** {memory.google_drive_public_url or 'N/A'}

**Status:** Successfully completed.
"""
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(webhook_url, json={"content": report_message})
            response.raise_for_status()
            
        logger.success("Report sent to Discord successfully.")
    except Exception as e:
        logger.error(f"Error sending report to Discord: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(send_report())
