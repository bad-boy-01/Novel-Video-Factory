import os
import sqlite3
import logging
from typing import List, Dict

logger = logging.getLogger(__name__)

class RenderQueue:
    """
    Phase 0: Render Queue.
    Tracks rendering status of scenes to allow instant crash recovery on Kaggle.
    """
    def __init__(self, project_dir: str):
        output_dir = os.path.join(project_dir, 'output')
        os.makedirs(output_dir, exist_ok=True)
        self.db_path = os.path.join(output_dir, 'render_queue.db')
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS queue (
                    scene_id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    clip_id INTEGER,
                    retries INTEGER DEFAULT 0
                )
            ''')
            conn.commit()

    def enqueue(self, scene_id: str, clip_id: int):
        """Add a scene to the render queue if it doesn't exist."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT OR IGNORE INTO queue (scene_id, status, clip_id)
                VALUES (?, 'pending', ?)
            ''', (scene_id, clip_id))
            conn.commit()

    def update_status(self, scene_id: str, status: str):
        """Update the status of a scene (pending, done, failed, retry)."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                UPDATE queue SET status = ? WHERE scene_id = ?
            ''', (status, scene_id))
            conn.commit()

    def increment_retry(self, scene_id: str):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                UPDATE queue SET retries = retries + 1, status = 'retry' WHERE scene_id = ?
            ''', (scene_id,))
            conn.commit()

    def get_pending_scenes(self) -> List[Dict]:
        """Fetch all scenes that need rendering."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT scene_id, status, clip_id, retries FROM queue WHERE status IN ('pending', 'retry')")
            rows = cursor.fetchall()
            return [{"scene_id": r[0], "status": r[1], "clip_id": r[2], "retries": r[3]} for r in rows]
            
    def get_status(self, scene_id: str) -> str:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT status FROM queue WHERE scene_id = ?", (scene_id,))
            row = cursor.fetchone()
            return row[0] if row else "unknown"
