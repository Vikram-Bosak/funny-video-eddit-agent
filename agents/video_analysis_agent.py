import os
import sys
import asyncio
import cv2
from loguru import logger
from scenedetect import detect, ContentDetector
from faster_whisper import WhisperModel
import easyocr
from ultralytics import YOLO
from openai import OpenAI

from memory_agent import async_get_latest_video_id, async_get_memory, async_update_memory

async def analyze_video():
    video_id = await async_get_latest_video_id()
    if not video_id:
        logger.error("No video_id found in memory.")
        sys.exit(1)
        
    memory = await async_get_memory(video_id)
    video_path = memory.local_video_path
    
    if not video_path or not os.path.exists(video_path):
        logger.error(f"Video path {video_path} not found.")
        sys.exit(1)

    logger.info(f"Starting Local AI Video Analysis for {video_path}...")
    
    scene_analysis = []
    transcript = ""
    ocr_results = []
    
    try:
        # 1. Scene Detection (PySceneDetect)
        logger.info("Running PySceneDetect...")
        scene_list = await asyncio.to_thread(detect, video_path, ContentDetector())
        for i, scene in enumerate(scene_list):
            scene_analysis.append({
                "scene_num": i + 1,
                "start_time": scene[0].get_seconds(),
                "end_time": scene[1].get_seconds()
            })
            
        # 2. Transcription (faster-whisper)
        logger.info("Running faster-whisper (tiny/cpu)...")
        # CPU config to prevent OOM in GH Actions
        model = WhisperModel("tiny", device="cpu", compute_type="int8")
        segments, info = await asyncio.to_thread(model.transcribe, video_path, beam_size=5)
        for segment in segments:
            transcript += f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}\n"

        # 3. OCR and Object Detection (EasyOCR + YOLO)
        logger.info("Running EasyOCR and YOLOv8n on key frames...")
        yolo_model = YOLO('yolov8n.pt')
        reader = easyocr.Reader(['en'], gpu=False)
        
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Analyze 1 frame per scene to save memory/time
        for scene in scene_analysis:
            frame_num = int(scene["start_time"] * fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = cap.read()
            if ret:
                # OCR
                text_results = reader.readtext(frame, detail=0)
                if text_results:
                    ocr_results.extend(text_results)
                    
                # YOLO
                results = yolo_model(frame, verbose=False)
                for r in results:
                    classes = r.boxes.cls
                    for c in classes:
                        class_name = yolo_model.names[int(c)]
                        scene["objects"] = scene.get("objects", [])
                        if class_name not in scene["objects"]:
                            scene["objects"].append(class_name)
        cap.release()
        
        # AI Translation and Summary
        logger.info("Using NVIDIA LLM to translate and summarize...")
        api_key = os.environ.get("NVIDIA_API_KEY", "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")
        client = OpenAI(
            base_url="https://integrate.api.nvidia.com/v1",
            api_key=api_key
        )
        
        objects_detected = set()
        for scene in scene_analysis:
            if "objects" in scene:
                objects_detected.update(scene["objects"])
        objects_str = ", ".join(list(objects_detected)) if objects_detected else "None"
        ocr_str = " | ".join(list(set(ocr_results))) if ocr_results else "None"
        raw_transcript = transcript.strip() or "No dialogue detected."
        
        translate_prompt = f"""
        You are an expert multilingual translator. 
        Analyze the following transcript from a video. If it is in a language other than English, translate it to English. 
        If it is already in English, output the exact same transcript.
        Do not add any explanations, introductory text, or notes. ONLY output the translated/original transcript text.
        
        Transcript:
        {raw_transcript}
        """
        
        summary_prompt = f"""
        You are an AI video summarizer.
        Analyze this video content and write a short, clear, and comprehensive summary (2-3 sentences) in English describing the actual events, actions, and settings shown in the video.
        
        Dialogue Transcript: {raw_transcript}
        OCR Text (visible on screen): {ocr_str}
        Detected Objects: {objects_str}
        """
        
        def run_llm(prompt):
            completion = client.chat.completions.create(
              model="nvidia/nemotron-3-ultra-550b-a55b",
              messages=[{"role":"user","content": prompt}],
              temperature=0.5,
              top_p=0.95,
              max_tokens=1024,
              stream=False
            )
            return completion.choices[0].message.content.strip()

        try:
            translation_text = await asyncio.to_thread(run_llm, translate_prompt)
        except Exception as e:
            logger.warning(f"Translation failed, using original transcript: {e}")
            translation_text = raw_transcript

        try:
            summary_text = await asyncio.to_thread(run_llm, summary_prompt)
        except Exception as e:
            logger.warning(f"Summarization failed: {e}")
            summary_text = "Summary generation failed."
        
        # Get video duration
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_duration = total_frames / fps if fps > 0 else 0
        cap.release()
        logger.info(f"Video duration: {video_duration:.2f} seconds")
        
        # Determine crop timestamps (max 59 seconds)
        crop_start = 0.0
        crop_duration = min(59.0, video_duration)
        
        if video_duration > 59.0:
            logger.info("Video is longer than 59s. Requesting AI to find the most funny portion...")
            select_prompt = f"""
            You are an expert social media editor. Analyze this video timeline data and select the single most funny, engaging, or viral continuous portion of the video.
            The selected portion MUST be at most 59 seconds long.
            
            Total Video Duration: {video_duration:.2f} seconds
            
            Timeline and Scene Analysis:
            {json.dumps(scene_analysis[:20], indent=2)}
            
            Transcript with Timestamps:
            {raw_transcript[:2000]}
            
            OCR Text (visible on screen): {ocr_str[:1000]}
            
            Identify the start time and duration of the best funny segment to crop.
            Return ONLY a valid JSON object with keys "start_time" and "duration" (in seconds as floats/integers). Example response:
            {{"start_time": 15.2, "duration": 45.0}}
            Do not output any explanation or extra text.
            """
            try:
                llm_response = await asyncio.to_thread(run_llm, select_prompt)
                logger.info(f"AI selection response: {llm_response}")
                import json
                clean_json = llm_response.replace("```json", "").replace("```", "").strip()
                data = json.loads(clean_json)
                crop_start = float(data.get("start_time", 0.0))
                crop_duration = float(data.get("duration", 59.0))
                # Validate bounds
                if crop_start < 0 or crop_start >= video_duration:
                    crop_start = 0.0
                if crop_duration <= 0 or crop_duration > 59.0 or (crop_start + crop_duration) > video_duration:
                    crop_duration = min(59.0, video_duration - crop_start)
            except Exception as e:
                logger.warning(f"Failed to parse AI selection, defaulting to first 59s: {e}")
                crop_start = 0.0
                crop_duration = min(59.0, video_duration)

        logger.info(f"Selected crop window: start={crop_start:.2f}s, duration={crop_duration:.2f}s")
        
        # Save results
        await async_update_memory(video_id, {
            "scene_analysis": scene_analysis,
            "transcript": raw_transcript,
            "translation": translation_text,
            "summary": summary_text,
            "ocr_text": ocr_str,
            "crop_start": crop_start,
            "crop_duration": crop_duration
        })
        
        logger.success("Video analysis complete.")

    except Exception as e:
        logger.error(f"Error during video analysis: {e}")
        await async_update_memory(video_id, {"error": str(e)})
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(analyze_video())
