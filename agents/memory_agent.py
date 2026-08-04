import sqlite3
import json
import asyncio
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field
from loguru import logger
from dotenv import load_dotenv

load_dotenv()

DB_FILE = "memory.db"

class MemoryModel(BaseModel):
    video_id: str
    source_url: Optional[str] = None
    original_title: Optional[str] = None
    original_description: Optional[str] = None
    local_video_path: Optional[str] = None
    transcript: Optional[str] = None
    scene_analysis: Optional[str] = None  # JSON string
    ocr_text: Optional[str] = None
    generated_script: Optional[str] = None
    voiceover_file: Optional[str] = None
    final_video_path: Optional[str] = None
    google_drive_public_url: Optional[str] = None
    error: Optional[str] = None

def init_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS memory (
            video_id TEXT PRIMARY KEY,
            source_url TEXT,
            original_title TEXT,
            original_description TEXT,
            local_video_path TEXT,
            transcript TEXT,
            scene_analysis TEXT,
            ocr_text TEXT,
            generated_script TEXT,
            voiceover_file TEXT,
            final_video_path TEXT,
            google_drive_public_url TEXT,
            error TEXT
        )
    ''')
    conn.commit()
    conn.close()
    logger.info("Database initialized.")

async def async_get_memory(video_id: str) -> Optional[MemoryModel]:
    def fetch():
        conn = sqlite3.connect(DB_FILE)
        conn.row_factory = sqlite3.Row
        c = conn.cursor()
        c.execute("SELECT * FROM memory WHERE video_id=?", (video_id,))
        row = c.fetchone()
        conn.close()
        if row:
            return MemoryModel(**dict(row))
        return None
    return await asyncio.to_thread(fetch)

async def async_update_memory(video_id: str, updates: Dict[str, Any]):
    def update():
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        
        # Check if exists
        c.execute("SELECT video_id FROM memory WHERE video_id=?", (video_id,))
        exists = c.fetchone()
        
        if not exists:
            c.execute("INSERT INTO memory (video_id) VALUES (?)", (video_id,))
            
        for key, value in updates.items():
            if isinstance(value, (list, dict)):
                value = json.dumps(value)
            c.execute(f"UPDATE memory SET {key} = ? WHERE video_id = ?", (value, video_id))
            
        conn.commit()
        conn.close()
        logger.info(f"Memory updated for {video_id}: {list(updates.keys())}")
        
    await asyncio.to_thread(update)

async def async_get_latest_video_id() -> Optional[str]:
    def fetch():
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        c.execute("SELECT video_id FROM memory LIMIT 1")
        row = c.fetchone()
        conn.close()
        if row:
            return row[0]
        return None
    return await asyncio.to_thread(fetch)

# Initialize DB on import
init_db()
