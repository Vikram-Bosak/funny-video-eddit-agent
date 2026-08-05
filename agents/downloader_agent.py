import os
import sys
import uuid
import json
import asyncio
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
import re
from datetime import datetime, timezone, timedelta
from email.utils import parsedate_to_datetime
from loguru import logger
from openai import OpenAI
from memory_agent import async_update_memory

# Track downloaded videos locally and persist to git history.txt
HISTORY_FILE = "history.txt"

def load_history():
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            return set(line.strip() for line in f if line.strip())
    return set()

def save_history(url: str):
    with open(HISTORY_FILE, "a", encoding="utf-8") as f:
        f.write(url.strip() + "\n")
    # Commit and push back to repository to persist state across workflow runs
    try:
        import subprocess
        # Configure temporary git identity if needed
        subprocess.run(["git", "config", "user.name", "AI Video Agent"], check=True)
        subprocess.run(["git", "config", "user.email", "agent@ai.com"], check=True)
        subprocess.run(["git", "add", HISTORY_FILE], check=True)
        subprocess.run(["git", "commit", "-m", f"Track processed video: {url}"], check=True)
        subprocess.run(["git", "push", "origin", "main"], check=True)
        logger.info(f"Successfully committed and pushed {HISTORY_FILE} updates.")
    except Exception as e:
        logger.warning(f"Git commit/push for history tracking failed: {e}")

async def is_video_funny(title: str, description: str) -> bool:
    api_key = os.environ.get("NVIDIA_API_KEY", "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")
    client = OpenAI(
        base_url="https://integrate.api.nvidia.com/v1",
        api_key=api_key
    )
    prompt = f"""
    You are an expert comedy analyst. Analyze the following video title and description.
    Determine if this video content is likely a funny fail, funny moment, comedy prank, try not to laugh, or comedic meme video.
    
    Video Title: {title}
    Video Description: {description}
    
    Reply with ONLY the word "YES" if it is funny/humorous, or "NO" if it is serious, political, educational, news-related, or general spam. Do not explain your reasoning.
    """
    try:
        def query():
            completion = client.chat.completions.create(
              model="nvidia/nemotron-3-ultra-550b-a55b",
              messages=[{"role":"user","content": prompt}],
              temperature=0.1,
              max_tokens=5,
              stream=False
            )
            return completion.choices[0].message.content.strip().upper()
        res = await asyncio.to_thread(query)
        logger.info(f"Humor Check result for '{title}': {res}")
        return "YES" in res
    except Exception as e:
        logger.warning(f"Humor check failed, defaulting to True: {e}")
        return True

async def download_video(rss_url_arg: str = None):
    logger.info("Starting Multi-Keyword Nitter Search Downloader...")
    
    # Track duplicates
    processed_urls = load_history()
    
    keywords = [
        "funny fails", "epic fails", "comedy moments", 
        "hilarious prank", "try not to laugh meme", "unexpected funny video"
    ]
    
    nitter_instances = [
        "https://nitter.poast.org",
        "https://nitter.net",
        "https://nitter.privacydev.net",
        "https://nitter.perennialte.ch"
    ]
    
    time_limit = datetime.now(timezone.utc) - timedelta(days=7) # Look back 7 days for more options
    valid_videos = []
    
    for keyword in keywords:
        logger.info(f"Searching keyword: {keyword}")
        rss_fetched = False
        items = []
        
        for instance in nitter_instances:
            # Query Nitter search with media filter
            encoded_query = urllib.parse.quote(f"{keyword} filter:media")
            url = f"{instance}/search/rss?f=tweets&q={encoded_query}"
            try:
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
            except Exception:
                continue
                
        if not rss_fetched:
            logger.warning(f"Could not fetch search RSS for query: {keyword}")
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
                continue
                
            # Extract user from link if available
            username = "TwitterUser"
            try:
                username = link.split(".org/")[1].split("/status/")[0]
            except:
                try:
                    username = link.split(".net/")[1].split("/status/")[0]
                except:
                    pass
                    
            original_tweet_url = f"https://twitter.com/{username}/status/{tweet_id}"
            
            # Skip duplicates
            if original_tweet_url in processed_urls:
                continue
                
            # HTML tag cleaning for description check
            clean_desc = re.sub('<[^<]+?>', '', desc) if desc else ""
            
            valid_videos.append({
                "url": original_tweet_url,
                "title": title,
                "description": clean_desc
            })
            
    if not valid_videos:
        logger.error("No valid recent and un-processed videos found.")
        sys.exit(1)
        
    os.makedirs("downloads", exist_ok=True)
    video_id = str(uuid.uuid4())
    output_path = f"downloads/{video_id}.mp4"
    
    # Try downloading videos until one succeeds and passes humor check
    for video in valid_videos:
        target_url = video["url"]
        
        # Verify humor before downloading
        is_funny = await is_video_funny(video["title"], video["description"])
        if not is_funny:
            logger.info(f"Skipping non-funny video: {target_url}")
            continue
            
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
                
                # Save to history to prevent duplicates in future runs
                save_history(target_url)
                
                await async_update_memory(video_id, {
                    "source_url": target_url,
                    "original_title": video["title"],
                    "original_description": video["description"],
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
            
    logger.error("Failed to download any funny and unprocessed video.")
    sys.exit(1)

if __name__ == "__main__":
    nitter_rss = os.environ.get("NITTER_RSS_URL", "")
    asyncio.run(download_video(nitter_rss))
