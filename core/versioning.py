import os
import json
import logging
from typing import List

logger = logging.getLogger(__name__)

class ArtifactVersionManager:
    """
    Phase 0: Metadata Artifact Versioning.
    Tracks 'v1', 'v2' etc for a specific artifact category (like storyboard or memory)
    by updating a metadata.json, avoiding duplicate large directories.
    """
    def __init__(self, base_dir: str, category: str):
        """
        base_dir: e.g. projects/martial_god/output/
        category: e.g. 'storyboard' or 'prompts'
        """
        self.category_dir = os.path.join(base_dir, category)
        os.makedirs(self.category_dir, exist_ok=True)
        self.metadata_path = os.path.join(self.category_dir, 'metadata.json')
        self._ensure_metadata()

    def _ensure_metadata(self):
        if not os.path.exists(self.metadata_path):
            initial_data = {
                "active": "v1",
                "latest": "v1",
                "history": ["v1"]
            }
            with open(self.metadata_path, 'w', encoding='utf-8') as f:
                json.dump(initial_data, f, indent=4)

    def _load_metadata(self) -> dict:
        with open(self.metadata_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _save_metadata(self, data: dict):
        with open(self.metadata_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4)

    def get_active_version(self) -> str:
        data = self._load_metadata()
        return data.get("active", "v1")

    def get_active_dir(self) -> str:
        """Returns the absolute path to the active version directory."""
        v = self.get_active_version()
        d = os.path.join(self.category_dir, v)
        os.makedirs(d, exist_ok=True)
        return d

    def create_new_version(self) -> str:
        """Bumps the version, adds to history, and sets as active."""
        data = self._load_metadata()
        current_latest = data.get("latest", "v1")
        
        # Parse 'vX' to X+1
        try:
            num = int(current_latest.replace('v', ''))
            new_v = f"v{num + 1}"
        except ValueError:
            new_v = "v2"
            
        data["latest"] = new_v
        data["active"] = new_v
        if "history" not in data:
            data["history"] = []
        data["history"].append(new_v)
        
        self._save_metadata(data)
        
        d = os.path.join(self.category_dir, new_v)
        os.makedirs(d, exist_ok=True)
        logger.info(f"Created new version {new_v} for {self.category_dir}")
        return new_v

    def rollback_to(self, version: str):
        """Sets an older version as the active version."""
        data = self._load_metadata()
        if version in data.get("history", []):
            data["active"] = version
            self._save_metadata(data)
            logger.info(f"Rolled back {self.category_dir} to {version}")
        else:
            raise ValueError(f"Version {version} not found in history.")
