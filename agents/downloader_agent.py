import os
import sys
import uuid
import json
import asyncio
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timezone, timedelta
from email.utils import parsedate_to_datetime
from loguru import logger
from memory_agent import async_update_memory

async def download_video(rss_url_arg: str = None):
    logger.info("Starting Multi-Profile Nitter Downloader...")
    
    profiles = [
        "9GAG", "Lmao", "pubity", "NoContextHumans", 
        "crazyclipsonly", "Fails_Vids", "AnimalsNoContext",
        "ComedyCentral", "Funnyhood", "WholesomeMeme"
    ]
    
    nitter_instances = [
        "https://nitter.poast.org",
        "https://nitter.net",
        "https://nitter.privacydev.net",
        "https://nitter.perennialte.ch"
    ]
    
    time_limit = datetime.now(timezone.utc) - timedelta(hours=24)
    valid_videos = []
    
    for username in profiles:
        logger.info(f"Checking profile: {username}")
        rss_fetched = False
        items = []
        
        for instance in nitter_instances:
            url = f"{instance}/{username}/rss"
            try:
                # Use to_thread for blocking urllib request
                def fetch_url(url):
                    headers = {'User-Agent': 'Mozilla/5.0'}
                    req = urllib.request.Request(url, headers=headers)
                    with urllib.request.urlopen(req, timeout=10) as response:
                        return response.read()
                        
                xml_data = await asyncio.to_thread(fetch_url, url)
                root = ET.fromstring(xml_data)
                items = root.findall('.//item')
                rss_fetched = True
                break
            except Exception as e:
                continue
                
        if not rss_fetched:
            logger.warning(f"Could not fetch RSS for {username}")
            continue
            
        for item in items:
            link = item.find('link').text if item.find('link') is not None else ""
            pubDate_str = item.find('pubDate').text if item.find('pubDate') is not None else ""
            desc = item.find('description').text if item.find('description') is not None else ""
            title = item.find('title').text if item.find('title') is not None else "Funny Video"
            
            if not link or not pubDate_str:
                continue
                
            if ">Video<" not in desc and "Video" not in desc:
                continue
                
            try:
                tweet_id = link.split("/status/")[1].split("#")[0].split("?")[0]
            except:
                continue
                
            try:
                post_time = parsedate_to_datetime(pubDate_str)
                if post_time.tzinfo is None:
                    post_time = post_time.replace(tzinfo=timezone.utc)
            except:
                continue
                
            if post_time < time_limit:
                break
                
            original_tweet_url = f"https://twitter.com/{username}/status/{tweet_id}"
            valid_videos.append({
                "url": original_tweet_url,
                "title": title,
                "description": desc
            })
            
    if not valid_videos:
        logger.error("No valid recent videos found across all profiles.")
        sys.exit(1)
        
    os.makedirs("downloads", exist_ok=True)
    video_id = str(uuid.uuid4())
    output_path = f"downloads/{video_id}.mp4"
    
    # Try downloading videos until one succeeds
    for video in valid_videos:
        target_url = video["url"]
        logger.info(f"Attempting to download {target_url}...")
        
        command = [
            "yt-dlp",
            "--output", output_path,
            "--quiet",
            target_url
        ]
        
        try:
            proc = await asyncio.create_subprocess_exec(
                *command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await proc.communicate()
            
            if proc.returncode == 0 and os.path.exists(output_path):
                logger.success(f"Download successful! Video ID: {video_id}")
                
                # HTML tag cleaning for description if needed, otherwise raw is fine
                clean_desc = video["description"]
                # A simple replacement to remove HTML tags from description
                import re
                clean_desc = re.sub('<[^<]+?>', '', clean_desc) if clean_desc else ""
                
                await async_update_memory(video_id, {
                    "source_url": target_url,
                    "original_title": video["title"],
                    "original_description": clean_desc,
                    "local_video_path": output_path,
                    "start_time": datetime.now(timezone.utc).isoformat(),
                    "github_repository": os.environ.get("GITHUB_REPOSITORY"),
                    "github_run_id": os.environ.get("GITHUB_RUN_ID"),
                    "github_run_url": f"{os.environ.get('GITHUB_SERVER_URL', 'https://github.com')}/{os.environ.get('GITHUB_REPOSITORY')}/actions/runs/{os.environ.get('GITHUB_RUN_ID')}" if os.environ.get("GITHUB_RUN_ID") else None
                })
                return
            else:
                logger.warning(f"yt-dlp failed for {target_url}. Trying next video...")
        except Exception as e:
            logger.warning(f"Error executing yt-dlp: {e}")
            
    logger.error("Failed to download ANY video from Twitter.")
    sys.exit(1)

if __name__ == "__main__":
    # We no longer strictly need the NITTER_RSS_URL from env, 
    # but we will accept it as an optional argument to prevent workflow errors
    nitter_rss = os.environ.get("NITTER_RSS_URL", "")
    asyncio.run(download_video(nitter_rss))
