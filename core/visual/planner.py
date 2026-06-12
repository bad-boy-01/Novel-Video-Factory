import json
import logging
from typing import List, Dict

logger = logging.getLogger(__name__)

class ScenePlanner:
    """
    Layer 5: Scene Planning.
    Splits chunks of text into individual scenes with camera and lighting metadata.
    """
    def __init__(self, llm_adapter):
        self.llm = llm_adapter

    def plan_scenes(self, text_chunk: str) -> List[Dict]:
        """
        Uses the LLM to slice a text chunk into visual scenes.
        """
        system_prompt = (
            "You are an expert anime director. Break the following story text into cinematic scenes. "
            "For each scene, provide a 'scene_id', 'description' (what is happening), 'characters_present' (list of names), "
            "'camera_angle', and 'lighting'. "
            "Return STRICTLY as a valid JSON array of objects. Example: "
            "[{\"scene_id\": \"SC001\", \"description\": \"A boy holding a sword.\", \"characters_present\": [\"Xu Changshou\"], \"camera_angle\": \"Close up\", \"lighting\": \"Bright daylight\"}]"
        )
        
        response = self.llm.generate(text_chunk, system_prompt=system_prompt, temperature=0.3)
        
        try:
            response = response.strip()
            if response.startswith("```json"):
                response = response[7:]
            if response.endswith("```"):
                response = response[:-3]
                
            data = json.loads(response)
            if isinstance(data, list):
                return data
            else:
                logger.error("ScenePlanner LLM did not return a JSON array.")
                return []
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM scene planning response: {e}\nResponse: {response}")
            return []
