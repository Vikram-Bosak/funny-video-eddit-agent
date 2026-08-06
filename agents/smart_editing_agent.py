import os
import sys
import asyncio
import subprocess
import math
import wave
import struct
import json
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

def generate_sfx(sfx_type, filepath):
    # 44.1 kHz, 16-bit mono
    sample_rate = 44100
    
    if sfx_type == "ding":
        # Bright sine wave at 1000Hz decaying exponentially over 0.5s
        duration = 0.5
        num_samples = int(duration * sample_rate)
        data = bytearray()
        for i in range(num_samples):
            t = i / sample_rate
            freq = 1000.0
            val = math.sin(2 * math.pi * freq * t) * math.exp(-6 * t)
            val = int(val * 32767)
            data.extend(struct.pack('<h', val))
            
    elif sfx_type == "boing":
        # Spring boing: sine wave sweeping from 200Hz to 500Hz and back
        duration = 0.8
        num_samples = int(duration * sample_rate)
        data = bytearray()
        for i in range(num_samples):
            t = i / sample_rate
            # Sweeping frequency
            freq = 200 + 300 * abs(math.sin(2 * math.pi * 3 * t))
            val = math.sin(2 * math.pi * freq * t) * (1.0 - t/duration)
            val = int(val * 32767)
            data.extend(struct.pack('<h', val))
            
    elif sfx_type == "whoosh":
        # Bandpassed noise sweeping frequency
        duration = 0.6
        num_samples = int(duration * sample_rate)
        data = bytearray()
        import random
        random.seed(42)
        for i in range(num_samples):
            t = i / sample_rate
            # Noise modulated by sine wave envelope
            env = math.sin(math.pi * t / duration)
            val = (random.random() * 2.0 - 1.0) * env * 0.5
            val = int(val * 32767)
            data.extend(struct.pack('<h', val))
            
    elif sfx_type == "alert":
        # High pitched double beep
        duration = 0.4
        num_samples = int(duration * sample_rate)
        data = bytearray()
        for i in range(num_samples):
            t = i / sample_rate
            # Double beep: beep 0.1s, silent 0.05s, beep 0.1s
            if t < 0.15 or (t > 0.22 and t < 0.37):
                freq = 1200.0
                val = math.sin(2 * math.pi * freq * t)
            else:
                val = 0
            val = int(val * 32767 * 0.7)
            data.extend(struct.pack('<h', val))
            
    elif sfx_type == "fail":
        # Sad descending trombone note
        duration = 1.0
        num_samples = int(duration * sample_rate)
        data = bytearray()
        for i in range(num_samples):
            t = i / sample_rate
            # Descending frequency from 300Hz to 150Hz
            freq = 300.0 - 150.0 * (t / duration)
            val = math.sin(2 * math.pi * freq * t) * (1.0 - t/duration)
            val = int(val * 32767 * 0.8)
            data.extend(struct.pack('<h', val))
            
    else:  # "laugh"
        # Chuckle: modulated low-frequency wave with chuckle envelop
        duration = 1.2
        num_samples = int(duration * sample_rate)
        data = bytearray()
        for i in range(num_samples):
            t = i / sample_rate
            # Chuckle pulsation
            pulse = abs(math.sin(2 * math.pi * 5 * t))
            freq = 180.0 + 40.0 * pulse
            val = math.sin(2 * math.pi * freq * t) * pulse * (1.0 - t/duration)
            val = int(val * 32767 * 0.6)
            data.extend(struct.pack('<h', val))

    with wave.open(filepath, 'wb') as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(data)

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
        # 1. Crop video to 9:16, seek to crop_start, set crop_duration
        crop_start = memory.crop_start if memory.crop_start is not None else 0.0
        crop_duration = memory.crop_duration if memory.crop_duration is not None else 59.0
        
        logger.info(f"Cropping video starting at {crop_start:.2f}s for {crop_duration:.2f}s...")
        # Use output seeking (-ss after -i) and setpts=PTS-STARTPTS to safely reset video PTS to 0
        crop_command = [
            "ffmpeg", "-y",
            "-i", video_path,
            "-ss", str(crop_start),
            "-t", str(crop_duration),
            "-vf", "crop=ih*(9/16):ih:(iw-ih*(9/16))/2:0,setpts=PTS-STARTPTS",
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
        
        # 3. Add Voiceover, plan sound effects (Rule 114 completely removed) and burn in subtitles
        logger.info("Mixing voiceover and planned sound effects (original audio completely removed)...")
        
        # Retrieve planned sound effects from memory
        sound_effects_str = memory.sound_effects if hasattr(memory, "sound_effects") and memory.sound_effects else "[]"
        try:
            sound_effects = json.loads(sound_effects_str)
        except Exception:
            sound_effects = []
            
        logger.info(f"Retrieved planned sound effects: {sound_effects}")
        
        # Generate SFX files on demand
        sfx_paths = []
        for i, sfx in enumerate(sound_effects):
            sfx_path = f"exports/{video_id}_sfx_{i}.wav"
            try:
                generate_sfx(sfx["type"], sfx_path)
                sfx_paths.append((sfx_path, float(sfx["time_offset"])))
            except Exception as e:
                logger.error(f"Failed to generate SFX {sfx['type']}: {e}")

        T = crop_duration
        
        # FFmpeg command inputs
        command = ["ffmpeg", "-y"]
        command.extend([
            "-stream_loop", "-1", "-i", temp_video,
            "-i", voiceover_path
        ])
        for sfx_path, _ in sfx_paths:
            command.extend(["-i", sfx_path])

        filter_complex_parts = []
        video_output_label = "0:v:0"
        if subs_success:
            escaped_ass = ass_path.replace("\\", "/").replace(":", "\\:")
            filter_complex_parts.append(f"[0:v:0]subtitles='{escaped_ass}'[v_subbed]")
            video_output_label = "[v_subbed]"
            
        # Build audio filter graph
        filter_parts = []
        mix_inputs = ["[1:a]"]
        
        # Delay each sound effect to its planned time_offset
        for idx, (_, time_offset) in enumerate(sfx_paths):
            sfx_input_idx = 2 + idx
            sfx_time_ms = int(time_offset * 1000)
            out_label = f"[sfx_delayed_{idx}]"
            filter_parts.append(f"[{sfx_input_idx}:a]asetpts=PTS-STARTPTS,adelay={sfx_time_ms}|{sfx_time_ms}{out_label}")
            mix_inputs.append(out_label)
            
        # Mix voiceover with sound effects and reset final audio PTS to 0
        filter_parts.append("".join(mix_inputs) + f"amix=inputs={len(mix_inputs)}:normalize=0,asetpts=PTS-STARTPTS[final_audio]")
        filter_complex_parts.extend(filter_parts)
        audio_output_label = "[final_audio]"
            
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
        
        # Cleanup temp files
        if os.path.exists(temp_video):
            os.remove(temp_video)
        for sfx_path, _ in sfx_paths:
            if os.path.exists(sfx_path):
                os.remove(sfx_path)
            
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
