from typing import Dict, List, Any, Optional
import datetime
import hashlib
import json

class CanonicalScene:
    """
    Phase 0: Canonical Scene Object.
    A unified schema that represents the state of a single scene throughout the entire pipeline.
    Every stage should read and update this object rather than rebuilding context.
    """
    def __init__(self, scene_id: str, chapter: int, clip_id: str, scene_index: int):
        self.scene_id = scene_id
        self.chapter = chapter
        self.clip = clip_id
        self.scene_index = scene_index
        
        self.duration: float = 8.0
        self.status: str = "pending" # pending, done, failed, rendering, retry
        
        # Generation Config
        self.seed: Optional[int] = None
        self.scheduler: str = "dpmpp_2m"
        self.steps: int = 20
        self.cfg: float = 7.0
        self.resolution: List[int] = [832, 480]
        
        # State
        self.characters: List[str] = []
        self.resolved_visual_state: Dict[str, Any] = {}
        self.location: Dict[str, Any] = {}
        self.camera: Dict[str, Any] = {}
        self.motion: Dict[str, Any] = {}
        
        # Prompts & Assets
        self.prompt: Optional[str] = None
        self.negative_prompt: Optional[str] = None
        self.prompt_hash: Optional[str] = None
        self.reference_images: List[str] = []
        self.background_hash: Optional[str] = None
        
        # Output Tracking
        self.quality_score: Optional[float] = None
        self.image_path: Optional[str] = None
        self.video_segment: Optional[str] = None
        self.retry_count: int = 0
        
        self.created_at: str = datetime.datetime.utcnow().isoformat()
        self.updated_at: str = self.created_at

    def generate_fingerprint(self) -> str:
        """
        Creates a SHA-256 hash of the visual dependencies.
        If this hash doesn't change, regeneration can be safely skipped.
        """
        dependency_state = {
            "visual_state": self.resolved_visual_state,
            "location": self.location,
            "camera": self.camera,
            "seed": self.seed
        }
        state_str = json.dumps(dependency_state, sort_keys=True)
        return hashlib.sha256(state_str.encode('utf-8')).hexdigest()

    def update(self):
        self.updated_at = datetime.datetime.utcnow().isoformat()

    def to_dict(self) -> dict:
        return {
            "scene_id": self.scene_id,
            "chapter": self.chapter,
            "clip": self.clip,
            "scene_index": self.scene_index,
            "duration": self.duration,
            "status": self.status,
            "seed": self.seed,
            "scheduler": self.scheduler,
            "steps": self.steps,
            "cfg": self.cfg,
            "resolution": self.resolution,
            "characters": self.characters,
            "resolved_visual_state": self.resolved_visual_state,
            "location": self.location,
            "camera": self.camera,
            "motion": self.motion,
            "prompt": self.prompt,
            "negative_prompt": self.negative_prompt,
            "prompt_hash": self.prompt_hash,
            "reference_images": self.reference_images,
            "background_hash": self.background_hash,
            "quality_score": self.quality_score,
            "image_path": self.image_path,
            "video_segment": self.video_segment,
            "retry_count": self.retry_count,
            "created_at": self.created_at,
            "updated_at": self.updated_at
        }
        
    @classmethod
    def from_dict(cls, data: dict) -> 'CanonicalScene':
        scene = cls(
            data.get("scene_id", ""), 
            data.get("chapter", 1), 
            data.get("clip", "clip01"),
            data.get("scene_index", 1)
        )
        for key, value in data.items():
            if hasattr(scene, key):
                setattr(scene, key, value)
        return scene
