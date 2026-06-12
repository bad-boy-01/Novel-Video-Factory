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
            "You are an expert storyboard artist for an anime/manhwa adaptation. "
            "Break the following story text into a highly detailed sequence of cinematic scenes.\n"
            "CRITICAL RULES:\n"
            "1. DO NOT SUMMARIZE. Create one scene for every 1 to 2 sentences of the text. A 1000 word chapter should yield at least 30-40 scenes.\n"
            "2. For 'description', 'camera_angle', and 'lighting', use short, comma-separated Danbooru tags (e.g. '1boy, waking up, confused'), NOT full English sentences.\n"
            "3. The 'description' MUST be under 15 words.\n"
            "For each scene, provide a 'scene_id', 'description' (comma-separated tags), 'characters_present' (list of names), "
            "'camera_angle' (tags), and 'lighting' (tags).\n"
            "Return STRICTLY as a valid JSON array of objects. Example: "
            "[{\"scene_id\": \"SC001\", \"description\": \"1boy, waking up, confused\", \"characters_present\": [\"Xu Changshou\"], \"camera_angle\": \"close up\", \"lighting\": \"morning sunlight\"}]"
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
