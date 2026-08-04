import os
import sys
import json
import asyncio
from loguru import logger
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from memory_agent import async_get_latest_video_id, async_get_memory, async_update_memory

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
    
    creds_json_str = os.environ.get("GDRIVE_SERVICE_ACCOUNT_JSON")
    if not creds_json_str:
        logger.warning("GDRIVE_SERVICE_ACCOUNT_JSON not set. Skipping upload.")
        await async_update_memory(video_id, {"google_drive_public_url": "https://drive.google.com/local-test-no-creds"})
        return
        
    try:
        def do_upload():
            creds_dict = json.loads(creds_json_str)
            creds = service_account.Credentials.from_service_account_info(
                creds_dict, scopes=['https://www.googleapis.com/auth/drive.file']
            )
            
            service = build('drive', 'v3', credentials=creds)
            file_metadata = {'name': f'{video_id}_final.mp4'}
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
        await async_update_memory(video_id, {"error": str(e)})
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(upload_video())
