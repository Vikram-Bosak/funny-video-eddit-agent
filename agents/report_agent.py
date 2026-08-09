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
    source_url = memory.source_url or 'N/A'
    account = memory.instagram_account or 'Unknown'
    views = memory.views_count or 'N/A'
    likes = memory.likes_count or 'N/A'
    comments = memory.comments_count or 'N/A'
    shares = memory.shares_count or 'N/A'
    post_time = memory.post_time or 'N/A'
    sel_reason = memory.selection_reason or 'N/A'
    score = f"{memory.trending_score:.1f}" if memory.trending_score is not None else 'N/A'
    viral_reason = memory.virality_reason or 'N/A'
    
    dl_status = "✅ Success" if memory.download_success == 1 else "❌ Failed"
    edit_status = "✅ Success" if memory.edit_success == 1 else "❌ Failed"
    upload_dest = "Google Drive" if memory.google_drive_public_url else "N/A"
    drive_url = memory.google_drive_public_url or 'N/A'
    
    script = memory.generated_script or 'N/A'
    summary = memory.summary or 'N/A'
    final_desc = f"**Summary:** {summary}\n**Voiceover Script:** {script}"
    
    err_info = memory.error or "None"
    
    has_failed = memory.error is not None and memory.error != ""
    if has_failed:
        embed_title = "❌ AI Video Automation Pipeline Failed!"
        embed_desc = f"The pipeline encountered an error: **{truncate_str(err_info, 300)}**"
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
                "name": "📊 1. Instagram Discovery & Metrics",
                "value": f"**Original Link:** [Instagram Reel/Post]({source_url})\n**Account:** @{account}\n**Views:** {views}\n**Likes:** {likes}\n**Comments:** {comments}\n**Shares:** {shares}\n**Posted Time:** {post_time}"
            },
            {
                "name": "🎯 2. Selection & Virality Analysis",
                "value": f"**Virality Score:** {score}\n**Why Selected:** {sel_reason}\n**Key Virality Factor:** {viral_reason}"
            },
            {
                "name": "⚙️ 3. Execution Status",
                "value": f"**Download Success:** {dl_status}\n**Editing Success:** {edit_status}\n**Upload Destination:** {upload_dest}\n**Final Video Link:** [Public Link]({drive_url})\n**Duration:** {duration_str}"
            },
            {
                "name": "📝 4. Final Video Details",
                "value": truncate_str(final_desc, 1000)
            },
            {
                "name": "⚠️ 5. Errors / Warning Info",
                "value": f"```\n{truncate_str(err_info, 200)}\n```"
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
