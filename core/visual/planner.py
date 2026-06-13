import json
import logging
import re
from typing import List, Dict

logger = logging.getLogger(__name__)

class ScenePlanner:
    """
    Layer 5a: Narrative Scene Planning (The Showrunner).
    Identifies continuous narrative beats and groups them into Scenes.
    """
    def __init__(self, llm_adapter):
        self.llm = llm_adapter

    def extract_narrative_scenes(self, text_chunk: str, start_index: int = 0) -> List[Dict]:
        """
        Groups text into narrative scenes with complexity scores.
        """
        system_prompt = (
            "You are a professional film showrunner. Analyze the following story text and divide it into Narrative Scenes.\n"
            "A NEW SCENE starts when:\n"
            "1. The location changes.\n"
            "2. There is a significant time skip.\n"
            "3. A character enters or leaves a conversation.\n"
            "4. The tone shifts dramatically (e.g., from quiet talk to a sudden fight).\n\n"
            "For each Scene, provide:\n"
            "- scene_id: A unique ID (e.g., SC001).\n"
            "- location: The name of the setting.\n"
            "- characters: List of character names present.\n"
            "- emotion: The primary mood (e.g., tense, joyful).\n"
            "- action: A short summary of what happens.\n"
            "- complexity: A score from 1 to 10 (1=simple dialogue/panoramic, 5=walking/multiple speakers, 10=intense action/large battle).\n"
            "- source_sentences: The indices of the sentences covered (relative to this chunk).\n"
            f"The first sentence of this text has global index: {start_index}.\n\n"
            "Return STRICTLY as a valid JSON array of objects. Example:\n"
            "[{\"scene_id\": \"SC001\", \"location\": \"Village Square\", \"characters\": [\"Xu\", \"Li\"], \"emotion\": \"tense\", \"action\": \"Xu confronts Li about the rice\", \"complexity\": 4, \"source_sentences\": [0, 1, 2, 3]}]"
        )
        
        response = self.llm.generate(text_chunk, system_prompt=system_prompt, temperature=0.1)
        
        try:
            match = re.search(r'(\[.*\])', response, re.DOTALL)
            if match:
                response = match.group(1)
                
            data = json.loads(response)
            if isinstance(data, list):
                # Coverage validation (Showrunner sanity check)
                import re
                sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text_chunk) if s.strip()]
                covered = set()
                for scene in data:
                    covered.update(scene.get('source_sentences', []))
                
                missing = [i for i in range(len(sentences)) if i not in covered]
                if missing:
                    logger.warning(f"Narrative Gap in Chunk: Sentences {missing} are not covered by any scene.")
                
                return data
            return []
        except Exception as e:
            logger.error(f"Failed to parse Narrative Scenes: {e}\nResponse: {response}")
            return []

class ShotDirector:
    """
    Layer 5b: Cinematic Shot Planning (The Director).
    Breaks a Narrative Scene into multiple visual shots based on complexity.
    """
    def __init__(self, llm_adapter):
        self.llm = llm_adapter

    def direct_shots(self, scene: Dict, scene_text: str) -> List[Dict]:
        """
        Breaks a single narrative scene into N cinematic shots.
        """
        complexity = scene.get('complexity', 1)
        # Rule: 1 complexity = ~1 shot, but at least 1 and max 12.
        target_shot_count = max(1, min(complexity, 12))
        
        system_prompt = (
            "You are a professional film director. Break this narrative SCENE into multiple cinematic SHOTS.\n"
            f"TARGET SHOT COUNT: {target_shot_count}.\n\n"
            "For each Shot, provide:\n"
            "- shot_id: Unique ID (e.g., SH001_A).\n"
            "- camera: Angle and movement (e.g., 'close-up', 'wide shot', 'panning left', 'low angle').\n"
            "- visual_prompt_tags: 3-8 Danbooru tags describing the visual composition.\n"
            "- narration_text: The EXACT piece of dialogue or narration for this shot.\n"
            "- duration_estimate: Predicted seconds (e.g., 4.5).\n\n"
            "VARIETY IS KEY: Use a mix of establishing shots, medium shots, and close-ups to make the scene feel professional.\n"
            "Return STRICTLY as a JSON array of objects."
        )
        
        scene_summary = f"LOCATION: {scene['location']}\nCHARACTERS: {scene['characters']}\nACTION: {scene['action']}\nTEXT: {scene_text}"
        response = self.llm.generate(scene_summary, system_prompt=system_prompt, temperature=0.3)
        
        try:
            match = re.search(r'(\[.*\])', response, re.DOTALL)
            if match:
                response = match.group(1)
            return json.loads(response)
        except Exception as e:
            logger.error(f"Director failed to split scene {scene['scene_id']}: {e}")
            # Fallback: Create 1 single shot for the whole scene
            return [{
                "shot_id": f"{scene['scene_id']}_A",
                "camera": "medium shot",
                "visual_prompt_tags": f"{scene['location']}, {scene['action']}",
                "narration_text": scene_text,
                "duration_estimate": len(scene_text.split()) * 0.5 # Rough estimate
            }]
