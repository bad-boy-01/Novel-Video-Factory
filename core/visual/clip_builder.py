import logging
from typing import List, Dict

logger = logging.getLogger(__name__)

class ClipBuilder:
    """
    Layer 5c: Rendering Orchestration.
    Groups cinematic shots into balanced 10-minute clips for rendering.
    """
    def __init__(self, target_duration: float = 600.0):
        self.target_duration = target_duration

    def build_clips(self, shots: List[Dict]) -> List[Dict]:
        """
        Iterates through shots and groups them into clips based on accumulated duration.
        """
        clips = []
        current_clip_id = 1
        current_shots = []
        current_duration = 0.0
        
        for shot in shots:
            shot_duration = float(shot.get('duration_estimate', 5.0))
            
            # If adding this shot exceeds target, close current clip
            if current_duration + shot_duration > self.target_duration and current_shots:
                clips.append({
                    "clip_id": f"clip{current_clip_id:02d}",
                    "duration": current_duration,
                    "shots": current_shots,
                    "status": "pending"
                })
                current_clip_id += 1
                current_shots = []
                current_duration = 0.0
            
            current_shots.append(shot)
            current_duration += shot_duration
            
        # Add final clip
        if current_shots:
            clips.append({
                "clip_id": f"clip{current_clip_id:02d}",
                "duration": current_duration,
                "shots": current_shots,
                "status": "pending"
            })
            
        logger.info(f"Balanced {len(shots)} shots into {len(clips)} cinematic clips.")
        return clips
