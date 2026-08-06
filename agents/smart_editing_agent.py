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
    import subprocess
    import numpy as np
    import ffmpeg
    from ultralytics import YOLO
    
    logger.info("Detecting subject head to draw red hook circle...")
    
    # Probe video metadata using FFmpeg
    try:
        probe = ffmpeg.probe(video_path)
        video_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
        if not video_stream:
            logger.error("No video stream found in temp video.")
            return False
        width = int(video_stream['width'])
        height = int(video_stream['height'])
        # Calculate FPS
        r_frame_rate = video_stream.get('r_frame_rate', '30/1')
        num, den = map(int, r_frame_rate.split('/'))
        fps = num / den if den != 0 else 30.0
    except Exception as e:
        logger.error(f"Failed to probe temp video: {e}")
        return False
        
    # Start FFmpeg process to read video frames as raw BGR24 bytes
    ffmpeg_read_cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-f", "image2pipe",
        "-pix_fmt", "bgr24",
        "-vcodec", "rawvideo",
        "-"
    ]
    read_process = subprocess.Popen(ffmpeg_read_cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    
    # Start FFmpeg process to write video via pipe
    ffmpeg_cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-s", f"{width}x{height}",
        "-pix_fmt", "bgr24",
        "-r", str(fps),
        "-i", "-",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        output_path
    ]
    process = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    circle_duration_frames = int(fps * 1.5)  # 1.5 seconds duration
    
    # Pre-load YOLO model
    try:
        yolo_model = YOLO('yolov8n.pt')
    except Exception as e:
        logger.warning(f"Could not load YOLO for tracking: {e}")
        yolo_model = None
        
    last_known_circle = None  # (cx, cy, r)
    frame_idx = 0
    frame_size = width * height * 3
    
    while True:
        in_bytes = read_process.stdout.read(frame_size)
        if not in_bytes or len(in_bytes) < frame_size:
            break
            
        frame = np.frombuffer(in_bytes, np.uint8).reshape((height, width, 3))
        # Copy to allow modifications on writable memory
        frame = frame.copy()
            
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
            
        process.stdin.write(frame.tobytes())
        frame_idx += 1
        
    read_process.stdout.close()
    read_process.wait()
    process.stdin.close()
    process.wait()
    
    logger.info(f"Successfully processed {frame_idx} frames and added red hook circle to video start.")
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
        
    logger.info("Normalizing downloaded video timestamps...")
    
    os.makedirs("exports", exist_ok=True)
    temp_clean_input = f"exports/{video_id}_clean.mp4"
    temp_video = f"exports/{video_id}_temp.mp4"
    final_video_path = f"exports/{video_id}_final.mp4"
    
    try:
        # Normalize input timestamps using fast stream copy
        subprocess.run([
            "ffmpeg", "-y",
            "-i", video_path,
            "-c", "copy",
            "-start_at_zero",
            temp_clean_input
        ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        video_path = temp_clean_input

        # 1. Crop video to 9:16, seek to crop_start, set crop_duration
        crop_start = memory.crop_start if memory.crop_start is not None else 0.0
        crop_duration = memory.crop_duration if memory.crop_duration is not None else 59.0
        
        logger.info(f"Cropping video starting at {crop_start:.2f}s for {crop_duration:.2f}s...")
        crop_command = [
            "ffmpeg", "-y",
            "-ss", str(crop_start),
            "-t", str(crop_duration),
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
        
        # Check if original video has audio
        has_audio = False
        actual_duration = crop_duration
        try:
            probe = ffmpeg.probe(video_path)
            for stream in probe.get("streams", []):
                if stream.get("codec_type") == "audio":
                    has_audio = True
            if "format" in probe and "duration" in probe["format"]:
                actual_duration = float(probe["format"]["duration"])
        except Exception as e:
            logger.warning(f"Failed to probe audio in original video: {e}")
            
        # 3. Add Voiceover and burn in subtitles
        logger.info("Syncing voiceover with original audio integration (Rule 114) and burning subtitles...")
        
        T = min(crop_duration, actual_duration)
        
        command = ["ffmpeg", "-y"]
        if has_audio:
            command.extend([
                "-stream_loop", "-1", "-i", temp_video,
                "-i", video_path,
                "-i", voiceover_path
            ])
        else:
            command.extend([
                "-stream_loop", "-1", "-i", temp_video,
                "-i", voiceover_path
            ])

        filter_complex_parts = []
        video_output_label = "0:v:0"
        if subs_success:
            escaped_ass = ass_path.replace("\\", "/").replace(":", "\\:")
            filter_complex_parts.append(f"[0:v:0]subtitles='{escaped_ass}'[v_subbed]")
            video_output_label = "[v_subbed]"
            
        if has_audio:
            # Splicing parameters for original audio integration
            t1 = T * 0.3
            t2 = T * 0.7
            d1 = 2.5
            d2 = 2.5
            
            if t1 + d1 >= t2:
                t2 = t1 + d1 + 2.0
            if t2 + d2 >= T:
                t2 = T - d2 - 1.0
                t1 = t2 - d1 - 2.0
                if t1 < 0:
                    t1 = 1.0
                    t2 = t1 + d1 + 1.0
                    
            t1_ms = int(t1 * 1000)
            t2_ms = int(t2 * 1000)
            t1_plus_d1_ms = int((t1 + d1) * 1000)
            t2_plus_d2_ms = int((t2 + d2) * 1000)
            
            filter_complex_parts.extend([
                f"[2:a]apad,asplit=3[vo_p1][vo_p2][vo_p3]",
                f"[1:a]atrim=start={crop_start + t1}:end={crop_start + t1 + d1},asetpts=PTS-STARTPTS,adelay={t1_ms}|{t1_ms}[orig_seg1]",
                f"[1:a]atrim=start={crop_start + t2}:end={crop_start + t2 + d2},asetpts=PTS-STARTPTS,adelay={t2_ms}|{t2_ms}[orig_seg2]",
                f"[vo_p1]atrim=start=0:end={t1},asetpts=PTS-STARTPTS[v_piece1]",
                f"[vo_p2]atrim=start={t1}:end={t2 - d1},asetpts=PTS-STARTPTS,adelay={t1_plus_d1_ms}|{t1_plus_d1_ms}[v_piece2]",
                f"[vo_p3]atrim=start={t2 - d1}:end={T},asetpts=PTS-STARTPTS,adelay={t2_plus_d2_ms}|{t2_plus_d2_ms}[v_piece3]",
                f"[orig_seg1][orig_seg2][v_piece1][v_piece2][v_piece3]amix=inputs=5:normalize=0[final_audio]"
            ])
            audio_output_label = "[final_audio]"
        else:
            audio_output_label = "1:a:0"
            
        if filter_complex_parts:
            command.extend(["-filter_complex", ";".join(filter_complex_parts)])
            
        command.extend([
            "-map", video_output_label,
            "-map", audio_output_label,
            "-c:v", "libx264",
            "-c:a", "aac",
            "-pix_fmt", "yuv420p",
            "-shortest",
            "-t", str(T),
            final_video_path
        ])
        
        res = subprocess.run(command, capture_output=True, text=True)
        if res.returncode != 0:
            logger.error(f"FFmpeg failed with exit code {res.returncode}")
            logger.error(f"FFmpeg stderr: {res.stderr}")
            raise Exception(f"FFmpeg error: {res.stderr}")
        
        # Cleanup temp
        if os.path.exists(temp_video):
            os.remove(temp_video)
        if os.path.exists(temp_clean_input):
            os.remove(temp_clean_input)
            
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
