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
        
        pat = os.environ.get("GH_TOKEN")
        if pat:
            repo = os.environ.get("GITHUB_REPOSITORY", "Vikram-Bosak/funny-video-eddit-agent")
            push_url = f"https://{pat}@github.com/{repo}.git"
            subprocess.run(["git", "push", push_url, "main"], check=True)
        else:
            subprocess.run(["git", "push", "origin", "main"], check=True)
            
        logger.info(f"Successfully committed and pushed {HISTORY_FILE} updates.")
    except Exception as e:
        logger.warning(f"Git commit/push for history tracking failed: {e}")

async def is_video_funny(title: str, description: str) -> bool:
    content_lower = (title + " " + description).lower()
    funny_keywords = ["funny", "fail", "comedy", "hilarious", "laugh", "prank", "meme", "lmao", "lol", "unexpected", "joke", "crazy"]
    
    if any(kw in content_lower for kw in funny_keywords):
        logger.info(f"Funniness verified by local keyword match: '{title}'")
        return True
        
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
    
    Is this video funny/humorous? Answer YES or NO.
    """
    try:
        def query():
            completion = client.chat.completions.create(
              model="nvidia/nemotron-3-ultra-550b-a55b",
              messages=[{"role":"user","content": prompt}],
              temperature=0.1,
              max_tokens=20,
              stream=False
            )
            return completion.choices[0].message.content.strip().upper()
        res = await asyncio.to_thread(query)
        logger.info(f"LLM humor check result for '{title}': {res}")
        # Return True unless the LLM explicitly says NO
        if "NO" in res and "YES" not in res:
            return False
        return True
    except Exception as e:
        logger.warning(f"Humor check failed, defaulting to True: {e}")
        return True

async def analyze_reels_metadata(title: str, description: str, username: str) -> dict:
    api_key = os.environ.get("NVIDIA_API_KEY", "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")
    client = OpenAI(
        base_url="https://integrate.api.nvidia.com/v1",
        api_key=api_key
    )
    prompt = f"""
    You are an expert social media analyst. Analyze the following tweet sharing an Instagram Reel:
    Username: {username}
    Tweet Title: {title}
    Tweet Description/Content: {description}
    
    Tasks:
    1. Extract or estimate the following Instagram metrics from the text if mentioned (use "N/A" if not specified):
       - Instagram Account Name (username)
       - Views count
       - Likes count
       - Comments count
       - Shares count
       - When the video was posted
    2. State why this video was selected (selection_reason) and what makes it viral (virality_reason).
    3. Calculate a Trending/Virality Score between 1.0 and 100.0 based on engagement speed and potential (growth rate, views vs time). If the video got massive views in a short time, give it a very high score (above 85.0). Any niche or category (sports, emotional, news, funny, lifestyle) is acceptable as long as it has high virality.
    
    Return ONLY a valid JSON object with the following keys (all values must be strings except trending_score):
    "instagram_account": "string",
    "views_count": "string",
    "likes_count": "string",
    "comments_count": "string",
    "shares_count": "string",
    "post_time": "string",
    "selection_reason": "string",
    "virality_reason": "string",
    "trending_score": float
    
    Do not output any explanation or extra text outside the JSON.
    """
    try:
        def query():
            completion = client.chat.completions.create(
              model="meta/llama-3.1-70b-instruct",
              messages=[{"role":"user","content": prompt}],
              temperature=0.1,
              max_tokens=500,
              stream=False
            )
            return completion.choices[0].message.content.strip()
        res = await asyncio.to_thread(query)
        logger.info(f"LLM reels analysis: {res}")
        clean_json = res.replace("```json", "").replace("```", "").strip()
        return json.loads(clean_json)
    except Exception as e:
        logger.warning(f"Failed to analyze reels metadata: {e}")
        return {
            "country_of_origin": "Unknown",
            "instagram_account": "Unknown",
            "views_count": "N/A",
            "likes_count": "N/A",
            "comments_count": "N/A",
            "shares_count": "N/A",
            "post_time": "N/A",
            "selection_reason": "Fallback due to error",
            "virality_reason": "Fallback due to error",
            "trending_score": 10.0,
            "is_funny": True
        }

async def download_video(rss_url_arg: str = None):
    logger.info("Starting Instagram Reels Trending Downloader...")
    
    # Track duplicates
    processed_urls = load_history()
    
    # Queries targeted to find instagram reels
    keywords = [
        "instagram.com/reel", "instagram.com/reels", "instagram.com/p",
        "#trendingreels", "#viralreels"
    ]
    
    nitter_instances = [
        "https://nitter.perennialte.ch",
        "https://xcancel.com",
        "https://nitter.poast.org",
        "https://nitter.net",
        "https://nitter.privacydev.net"
    ]
    
    # Dynamically query health monitor for working Nitter RSS mirrors
    try:
        import urllib.request
        import xml.etree.ElementTree as ET
        req = urllib.request.Request("https://status.d420.de/", headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=8) as response:
            html = response.read().decode('utf-8')
            
        rows = re.findall(r'<tr>(.*?)</tr>', html, re.DOTALL)
        dynamic_instances = []
        for row in rows:
            if 'healthy">✅' in row:
                rss_td = re.search(r'<td data-name="rss">(.*?)</td>', row, re.DOTALL)
                if rss_td and '✅' in rss_td.group(1):
                    match = re.search(r'href="https:&#x2F;&#x2F;([^"]+)"', row)
                    if match:
                        instance_domain = match.group(1).replace('&#x2F;', '/')
                        dynamic_instances.append(f"https://{instance_domain}")
                        
        logger.info(f"Dynamically discovered healthy Nitter RSS mirrors: {dynamic_instances}")
        if dynamic_instances:
            # Put healthy dynamic ones first, keeping hardcoded ones as fallback
            nitter_instances = dynamic_instances + [inst for inst in nitter_instances if inst not in dynamic_instances]
    except Exception as e:
        logger.warning(f"Failed to dynamically fetch healthy Nitter mirrors: {e}")
    
    time_limit = datetime.now(timezone.utc) - timedelta(days=7) # Look back 7 days
    valid_videos = []
    
    for keyword in keywords:
        logger.info(f"Searching keyword: {keyword}")
        rss_fetched = False
        items = []
        
        for instance in nitter_instances:
            encoded_query = urllib.parse.quote(f"{keyword}")
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
            title = item.find('title').text if item.find('title') is not None else "Instagram Reel"
            
            if not link or not pubDate_str:
                continue
                
            # Extract any instagram reel link from the content
            ig_urls = re.findall(r'https?://(?:www\.)?instagram\.com/(?:reel|reels|p)/[a-zA-Z0-9_\-]+', desc)
            if not ig_urls:
                continue
            
            ig_url = ig_urls[0]
            
            # Skip duplicates
            if ig_url in processed_urls:
                continue
                
            try:
                post_time_parsed = parsedate_to_datetime(pubDate_str)
                if post_time_parsed.tzinfo is None:
                    post_time_parsed = post_time_parsed.replace(tzinfo=timezone.utc)
            except:
                continue
                
            if post_time_parsed < time_limit:
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
            
            # HTML tag cleaning for description check
            clean_desc = re.sub('<[^<]+?>', '', desc) if desc else ""
            
            # Perform LLM analysis on the post to calculate Trending Score and Country of Origin
            analysis = await analyze_reels_metadata(title, clean_desc, username)
            
            valid_videos.append({
                "url": ig_url,
                "tweet_url": link,
                "title": title,
                "description": clean_desc,
                "country_of_origin": "Unknown",
                "instagram_account": analysis.get("instagram_account", "Unknown"),
                "views_count": analysis.get("views_count", "N/A"),
                "likes_count": analysis.get("likes_count", "N/A"),
                "comments_count": analysis.get("comments_count", "N/A"),
                "shares_count": analysis.get("shares_count", "N/A"),
                "post_time": analysis.get("post_time", "N/A"),
                "selection_reason": analysis.get("selection_reason", "N/A"),
                "virality_reason": analysis.get("virality_reason", "N/A"),
                "trending_score": float(analysis.get("trending_score", 10.0))
            })
            
    if not valid_videos:
        logger.error("No valid recent and un-processed Instagram videos found.")
        sys.exit(1)
        
    # Sort videos by trending score in descending order
    valid_videos.sort(key=lambda x: x["trending_score"], reverse=True)
    
    os.makedirs("downloads", exist_ok=True)
    video_id = str(uuid.uuid4())
    output_path = f"downloads/{video_id}.mp4"
    
    # Try downloading the best videos by trending score
    for video in valid_videos:
        target_url = video["url"]
        logger.info(f"Selected Top Video (Origin: {video['country_of_origin']}, Score: {video['trending_score']}): {target_url}")
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
            
            # Fallback to download from tweet itself if Instagram direct link download fails
            if proc.returncode != 0 or not os.path.exists(output_path):
                logger.warning(f"Direct Instagram download failed. Trying tweet backup download: {video['tweet_url']}")
                command_backup = [
                    "yt-dlp",
                    "--output", output_path,
                    "--quiet",
                    video["tweet_url"]
                ]
                proc_backup = await asyncio.create_subprocess_exec(
                    *command_backup,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                await proc_backup.communicate()
                
            if os.path.exists(output_path):
                logger.success(f"Download successful! Video ID: {video_id}")
                
                # Save to history to prevent duplicates in future runs
                save_history(target_url)
                
                await async_update_memory(video_id, {
                    "source_url": target_url,
                    "original_title": video["title"],
                    "original_description": video["description"],
                    "local_video_path": output_path,
                    "country_of_origin": video["country_of_origin"],
                    "trending_score": video["trending_score"],
                    "instagram_account": video["instagram_account"],
                    "views_count": video["views_count"],
                    "likes_count": video["likes_count"],
                    "comments_count": video["comments_count"],
                    "shares_count": video["shares_count"],
                    "post_time": video["post_time"],
                    "selection_reason": video["selection_reason"],
                    "virality_reason": video["virality_reason"],
                    "download_success": 1,
                    "edit_success": 0,
                    "start_time": datetime.now(timezone.utc).isoformat(),
                    "github_repository": os.environ.get("GITHUB_REPOSITORY"),
                    "github_run_id": os.environ.get("GITHUB_RUN_ID"),
                    "github_run_url": f"{os.environ.get('GITHUB_SERVER_URL', 'https://github.com')}/{os.environ.get('GITHUB_REPOSITORY')}/actions/runs/{os.environ.get('GITHUB_RUN_ID')}" if os.environ.get("GITHUB_RUN_ID") else None
                })
                return
            else:
                logger.warning(f"Download failed for {target_url}. Trying next highest scored video...")
        except Exception as e:
            logger.warning(f"Error executing download: {e}")
            
    logger.error("Failed to download any funny and unprocessed video.")
    sys.exit(1)

if __name__ == "__main__":
    nitter_rss = os.environ.get("NITTER_RSS_URL", "")
    asyncio.run(download_video(nitter_rss))
