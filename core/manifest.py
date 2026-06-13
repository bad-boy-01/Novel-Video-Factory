import os
import json
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

class ManifestManager:
    """
    Phase 0: Manifest System.
    Single source of truth for the project's data flow.
    """
    def __init__(self, project_dir: str):
        self.manifest_path = os.path.join(project_dir, 'manifest.json')
        self.data = self._load()

    def _load(self) -> Dict[str, Any]:
        if os.path.exists(self.manifest_path):
            with open(self.manifest_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {
            "project_name": os.path.basename(os.path.normpath(self.manifest_path).split(os.sep)[-2]),
            "chapter_range": [1, 1],
            "clip_count": 0,
            "images_generated": 0,
            "storyboard_version": 1,
            "memory_version": 1,
            "prompt_version": 1,
            "last_completed_clip": None,
            "current_scene": None
        }

    def save(self):
        with open(self.manifest_path, 'w', encoding='utf-8') as f:
            json.dump(self.data, f, indent=4)
        logger.debug("Manifest updated.")

    def get(self, key: str, default: Any = None) -> Any:
        return self.data.get(key, default)

    def update(self, key: str, value: Any):
        self.data[key] = value
        self.save()
        
    def increment(self, key: str, amount: int = 1):
        current = self.data.get(key, 0)
        self.data[key] = current + amount
        self.save()

class ProjectStateManager:
    """
    Phase 0: Project State Manager.
    Execution state for the pipeline (e.g., tracking which clips are done/running).
    """
    def __init__(self, project_dir: str):
        self.state_path = os.path.join(project_dir, 'project_state.json')
        self.data = self._load()

    def _load(self) -> Dict[str, Any]:
        if os.path.exists(self.state_path):
            with open(self.state_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {
            "translation": "pending",
            "memory": "pending",
            "storyboard": "pending",
            "prompts": "pending",
            "character_sheets": "pending"
        }

    def save(self):
        with open(self.state_path, 'w', encoding='utf-8') as f:
            json.dump(self.data, f, indent=4)

    def set_status(self, stage_or_clip: str, status: str):
        """status: 'pending', 'running', 'done', 'failed'"""
        self.data[stage_or_clip] = status
        self.save()

    def get_status(self, stage_or_clip: str) -> str:
        return self.data.get(stage_or_clip, "pending")
