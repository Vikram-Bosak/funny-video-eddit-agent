import os
import sys
import asyncio
import subprocess
from loguru import logger
import ffmpeg
from memory_agent import async_get_latest_video_id, async_get_memory, async_update_memory

async def edit_video():
    video_id = await async_get_latest_video_id()
    if not video_id:
        logger.error("No video_id found in memory.")
        sys.exit(1)
        
    memory = await async_get_memory(video_id)
    video_path = memory.local_video_path
    voiceover_path = memory.voiceover_file
    
    if not video_path or not voiceover_path:
        logger.error("Required paths not found in memory.")
        sys.exit(1)
        
    logger.info("Editing video using FFmpeg (crop to 9:16 and add audio)...")
    
    os.makedirs("exports", exist_ok=True)
    temp_video = f"exports/{video_id}_temp.mp4"
    final_video_path = f"exports/{video_id}_final.mp4"
    
    try:
        # 1. Crop video to 9:16 using ffmpeg-python
        logger.info("Cropping video...")
        stream = ffmpeg.input(video_path)
        # Crop center to 9:16 aspect ratio
        stream = ffmpeg.crop(stream, 'iw/2-ih*(9/32)', 0, 'ih*(9/16)', 'ih')
        stream = ffmpeg.output(stream, temp_video, vcodec='libx264', acodec='copy')
        ffmpeg.run(stream, overwrite_output=True, quiet=True)
        
        # 2. Add Voiceover and loop/trim video to match audio length
        logger.info("Syncing voiceover...")
        command = [
            "ffmpeg", "-y",
            "-stream_loop", "-1", "-i", temp_video,  # Loop video if shorter than audio
            "-i", voiceover_path,
            "-c:v", "libx264",
            "-c:a", "aac",
            "-pix_fmt", "yuv420p",
            "-map", "0:v:0",
            "-map", "1:a:0",
            "-shortest",  # Stop encoding when the shortest stream ends (audio)
            final_video_path
        ]
        
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        # Cleanup temp
        if os.path.exists(temp_video):
            os.remove(temp_video)
            
        await async_update_memory(video_id, {"final_video_path": final_video_path})
        logger.success("Video editing complete.")
        
    except Exception as e:
        logger.error(f"Error during video editing: {e}")
        await async_update_memory(video_id, {"error": str(e)})
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(edit_video())
