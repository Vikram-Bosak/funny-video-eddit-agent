import os
import sys
import json
import asyncio
from loguru import logger
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from memory_agent import async_get_latest_video_id, async_get_memory, async_update_memory
from dotenv import load_dotenv

load_dotenv()

def get_oauth_credentials():
    scopes = ['https://www.googleapis.com/auth/drive']
    token_str = os.environ.get("GDRIVE_OAUTH_TOKEN")
    
    if token_str:
        logger.info("Loading Google Drive OAuth credentials from environment variable...")
        try:
            token_data = json.loads(token_str)
            creds = Credentials.from_authorized_user_info(token_data, scopes)
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            return creds
        except Exception as e:
            logger.error(f"Failed to parse GDRIVE_OAUTH_TOKEN: {e}")
            
    # Local fallback
    if os.path.exists("token.json"):
        logger.info("Loading Google Drive OAuth credentials from token.json...")
        try:
            creds = Credentials.from_authorized_user_file("token.json", scopes)
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            return creds
        except Exception as e:
            logger.error(f"Failed to load token.json: {e}")
            
    return None

async def upload_video():
    video_id = await async_get_latest_video_id()
    if not video_id:
        logger.error("No video_id found in memory.")
        sys.exit(1)
        
    memory = await async_get_memory(video_id)
    final_video = memory.final_video_path
    
    if not final_video:
        logger.error("Final video path not found in memory.")
        sys.exit(1)
        
    logger.info(f"Uploading video {final_video} to Google Drive...")
    
    folder_id = os.environ.get("GDRIVE_FOLDER_ID")
    creds = get_oauth_credentials()
    
    if not creds:
        logger.warning("Google Drive OAuth credentials not found. Skipping upload.")
        await async_update_memory(video_id, {"google_drive_public_url": "https://drive.google.com/local-test-no-creds"})
        return
        
    try:
        def do_upload():
            service = build('drive', 'v3', credentials=creds)
            file_metadata = {'name': f'{video_id}_final.mp4'}
            if folder_id:
                file_metadata['parents'] = [folder_id]
                
            media = MediaFileUpload(final_video, mimetype='video/mp4', resumable=True)
            
            file = service.files().create(body=file_metadata, media_body=media, fields='id').execute()
            file_id = file.get('id')
            
            permission = {'type': 'anyone', 'role': 'reader'}
            service.permissions().create(fileId=file_id, body=permission).execute()
            
            file_metadata = service.files().get(fileId=file_id, fields='webViewLink').execute()
            return file_metadata.get('webViewLink')
            
        drive_url = await asyncio.to_thread(do_upload)
        
        await async_update_memory(video_id, {"google_drive_public_url": drive_url})
        logger.success(f"Video upload complete. URL: {drive_url}")
        
    except Exception as e:
        logger.error(f"Error during Google Drive upload: {e}")
        await async_update_memory(video_id, {
            "error": str(e),
            "google_drive_public_url": f"https://drive.google.com/error-fallback-local-{video_id}"
        })
        logger.warning("Continuing pipeline despite upload failure.")

if __name__ == "__main__":
    asyncio.run(upload_video())
