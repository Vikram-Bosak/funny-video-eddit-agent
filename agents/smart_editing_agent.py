import os
import sys
import asyncio
import subprocess
from loguru import logger
from datetime import datetime, timezone
import ffmpeg
from memory_agent import async_get_latest_video_id, async_get_memory, async_update_memory

def format_time(seconds: float) -> str:
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    centiseconds = int(round((secs - int(secs)) * 100))
    if centiseconds == 100:
        centiseconds = 99
    return f"{hours}:{minutes:02d}:{int(secs):02d}.{centiseconds:02d}"

def generate_ass_subtitles(voiceover_path: str, ass_path: str):
    from faster_whisper import WhisperModel
    logger.info("Transcribing voiceover with word-level timestamps using Whisper (tiny/cpu)...")
    model = WhisperModel("tiny", device="cpu", compute_type="int8")
    segments, info = model.transcribe(voiceover_path, word_timestamps=True)
    
    words = []
    for segment in segments:
        if segment.words:
            for w in segment.words:
                words.append({
                    "start": w.start,
                    "end": w.end,
                    "text": w.word.strip().upper()  # Replicate ALL CAPS style
                })
                
    if not words:
        logger.warning("No words detected in voiceover to generate subtitles.")
        return False
        
    # Group words into short 2-word phrases or 1.0 second max duration
    phrases = []
    current_phrase = []
    phrase_start = 0.0
    
    for w in words:
        if not current_phrase:
            phrase_start = w["start"]
        current_phrase.append(w)
        
        duration = w["end"] - phrase_start
        if len(current_phrase) >= 2 or duration >= 1.0:
            phrases.append(current_phrase)
            current_phrase = []
            
    if current_phrase:
        phrases.append(current_phrase)
        
    # Write ASS file with YouTube Shorts formatting
    with open(ass_path, "w", encoding="utf-8") as f:
        f.write("[Script Info]\n")
        f.write("ScriptType: v4.00+\n")
        f.write("PlayResX: 720\n")
        f.write("PlayResY: 1280\n\n")
        
        f.write("[V4+ Styles]\n")
        f.write("Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\n")
        # Solid White primary color (&H00FFFFFF), thick black outline (Outline=4), lower-middle vertical positioning (MarginV=580)
        f.write("Style: Default,Arial Black,64,&H00FFFFFF,&H00000000,&H00000000,&H00000000,-1,0,0,0,100,100,0,0,1,4,1,2,10,10,580,1\n\n")
        
        f.write("[Events]\n")
        f.write("Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n")
        
        for phrase in phrases:
            start_str = format_time(phrase[0]["start"])
            end_str = format_time(phrase[-1]["end"])
            phrase_text = " ".join([w["text"] for w in phrase])
            f.write(f"Dialogue: 0,{start_str},{end_str},Default,,0,0,0,,{phrase_text}\n")
                
    logger.info(f"ASS subtitles generated at: {ass_path}")
    return True

def draw_hook_circle(video_path: str, output_path: str) -> bool:
    import cv2
    from ultralytics import YOLO
    
    logger.info("Detecting subject head to draw red hook circle...")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error("Could not open video to draw circle.")
        return False
        
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    
    # We must use 'mp4v' or another standard codec for writing
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Reset video capture to start
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    circle_duration_frames = int(fps * 1.5)  # 1.5 seconds duration
    
    # Pre-load YOLO model
    try:
        yolo_model = YOLO('yolov8n.pt')
    except Exception as e:
        logger.warning(f"Could not load YOLO for tracking: {e}")
        yolo_model = None
        
    last_known_circle = None  # (cx, cy, r)
    frame_idx = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        if frame_idx < circle_duration_frames:
            cx, cy, r = None, None, None
            if yolo_model:
                try:
                    # Run detection on current frame
                    results = yolo_model(frame, verbose=False)
                    best_person = None
                    max_area = 0
                    
                    for r_item in results:
                        boxes = r_item.boxes
                        for box in boxes:
                            if int(box.cls[0]) == 0:  # Class 0 is person
                                x1, y1, x2, y2 = map(int, box.xyxy[0])
                                area = (x2 - x1) * (y2 - y1)
                                if area > max_area:
                                    max_area = area
                                    best_person = (x1, y1, x2, y2)
                                    
                    if best_person:
                        x1, y1, x2, y2 = best_person
                        cx = int((x1 + x2) / 2)
                        cy = int(y1 + (y2 - y1) * 0.15)
                        r = int((x2 - x1) * 0.28)
                        last_known_circle = (cx, cy, r)
                except Exception as e:
                    pass
            
            # Fallback to last known circle if detection failed in this frame
            if cx is None and last_known_circle is not None:
                cx, cy, r = last_known_circle
                
            # Full fallback to upper center if no detection at all
            if cx is None:
                cx = int(width / 2)
                cy = int(height * 0.35)
                r = int(width * 0.18)
                
            # Draw red circle (BGR: 0, 0, 255), thickness=4
            cv2.circle(frame, (cx, cy), r, (0, 0, 255), 4)
            
        out.write(frame)
        frame_idx += 1
        
    cap.release()
    out.release()
    logger.info("Successfully added red hook circle to video start.")
    return True

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
        # 1. Crop video to 9:16 using FFmpeg CLI directly
        logger.info("Cropping video...")
        crop_command = [
            "ffmpeg", "-y",
            "-i", video_path,
            "-vf", "crop=ih*(9/16):ih:(iw-ih*(9/16))/2:0",
            "-c:v", "libx264",
            "-an",
            temp_video
        ]
        subprocess.run(crop_command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        # 1.5. Draw Red Hook Circle at the start
        temp_circle_video = f"exports/{video_id}_temp_circle.mp4"
        if draw_hook_circle(temp_video, temp_circle_video):
            if os.path.exists(temp_video):
                os.remove(temp_video)
            os.rename(temp_circle_video, temp_video)
            
        # 2. Generate subtitles
        ass_path = f"exports/{video_id}_subs.ass"
        subs_success = generate_ass_subtitles(voiceover_path, ass_path)
        
        # 3. Add Voiceover and burn in subtitles
        logger.info("Syncing voiceover and burning subtitles...")
        command = [
            "ffmpeg", "-y",
            "-stream_loop", "-1", "-i", temp_video,
            "-i", voiceover_path,
        ]
        
        if subs_success:
            escaped_ass_path = ass_path.replace("\\", "/")
            command.extend(["-vf", f"subtitles={escaped_ass_path}"])
            
        command.extend([
            "-c:v", "libx264",
            "-c:a", "aac",
            "-pix_fmt", "yuv420p",
            "-map", "0:v:0",
            "-map", "1:a:0",
            "-shortest",
            final_video_path
        ])
        
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        # Cleanup temp
        if os.path.exists(temp_video):
            os.remove(temp_video)
            
        await async_update_memory(video_id, {
            "final_video_path": final_video_path,
            "end_time": datetime.now(timezone.utc).isoformat()
        })
        logger.success("Video editing and subtitle burn-in complete.")
        
    except ffmpeg.Error as e:
        logger.error(f"FFmpeg error: {e.stderr.decode() if e.stderr else e}")
        await async_update_memory(video_id, {"error": str(e)})
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error during video editing: {e}")
        await async_update_memory(video_id, {"error": str(e)})
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(edit_video())
