import os
import sys
import uuid
import json
import asyncio
import subprocess
import feedparser
from loguru import logger
from memory_agent import async_update_memory

async def fetch_latest_video_url(rss_url: str) -> str:
    logger.info(f"Fetching RSS feed from: {rss_url}")
    
    # Run feedparser in a thread since it's blocking
    feed = await asyncio.to_thread(feedparser.parse, rss_url)
    
    if not feed.entries:
        logger.warning("No entries found in the RSS feed. Falling back to a default funny video...")
        return "https://www.youtube.com/watch?v=jNQXAC9IVRw"
        
    for entry in feed.entries:
        post_url = entry.link
        if "nitter." in post_url:
            post_url = post_url.replace(post_url.split('/')[2], "twitter.com")
            
        logger.info(f"Found latest post: {post_url}")
        return post_url
        
    logger.warning("Could not find a valid post in the RSS feed. Falling back to default.")
    return "https://www.youtube.com/watch?v=jNQXAC9IVRw"

async def download_video(rss_url: str):
    target_url = await fetch_latest_video_url(rss_url)
    
    logger.info(f"Downloading video from {target_url}...")
    video_id = str(uuid.uuid4())
    output_path = f"downloads/{video_id}.mp4"
    
    os.makedirs("downloads", exist_ok=True)
    
    command = [
        "yt-dlp",
        "--output", output_path,
        "--dump-json",
        target_url
    ]
    
    try:
        proc = await asyncio.create_subprocess_exec(
            *command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await proc.communicate()
        
        if proc.returncode != 0:
            logger.error(f"yt-dlp error: {stderr.decode()}")
            raise Exception("Download failed. Possibly no video found in the tweet.")
            
        info = json.loads(stdout.decode())
        
        await async_update_memory(video_id, {
            "source_url": target_url,
            "original_title": info.get("title", "Unknown Title"),
            "original_description": info.get("description", ""),
            "local_video_path": output_path
        })
        
        logger.success(f"Download successful. Video ID: {video_id}")
    except Exception as e:
        logger.error(f"Error downloading video: {e}")
        sys.exit(1)

if __name__ == "__main__":
    nitter_rss = os.environ.get("NITTER_RSS_URL")
    if not nitter_rss:
        logger.error("NITTER_RSS_URL environment variable is not set. Example: https://nitter.net/elonmusk/rss")
        sys.exit(1)
    
    asyncio.run(download_video(nitter_rss))
