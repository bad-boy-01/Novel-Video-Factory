import json
import re
import logging
from typing import Dict, List

logger = logging.getLogger(__name__)

class MemoryExtractor:
    """
    Extracts characters, locations, and their visual DNA from translated text.
    """
    def __init__(self, llm_adapter):
        self.llm = llm_adapter

    def extract_all(self, text_chunk: str, existing_characters: List[Dict] = None) -> Dict[str, List[Dict]]:
        """
        Extracts characters, locations, and world concepts in a single LLM call to save API rate limits.
        Utilizes existing_characters to prevent duplicates.
        """
        existing_char_list = ", ".join([c['canonical_name'] for c in existing_characters]) if existing_characters else "None"
        
        system_prompt = (
            "You are an expert lore master and character designer. Read the story text and extract ALL characters, locations, world concepts, and character relationships.\n"
            "Return the data STRICTLY as a JSON object with keys: 'characters', 'locations', 'world_concepts', 'relationships'.\n\n"
            "EXISTING CHARACTERS (Reuse these names exactly if they appear): " + existing_char_list + "\n\n"
            "CRITICAL RULES FOR RELATIONSHIPS:\n"
            "1. Identify connections between characters (e.g., 'A is B's master', 'A and B are enemies').\n"
            "2. relationship_type should be one of: 'master-disciple', 'enemy', 'romantic', 'family', 'friend', 'neutral'.\n"
            "3. Include 'staging' notes (e.g., 'protective', 'distant', 'aggressive').\n\n"
            "EXAMPLE OUTPUT FORMAT:\n"
            "{\n"
            "  \"characters\": [...],\n"
            "  \"locations\": [...],\n"
            "  \"world_concepts\": [...],\n"
            "  \"relationships\": [{\"char1\": \"Xu Changshou\", \"char2\": \"Master Lin\", \"type\": \"master-disciple\", \"staging\": \"respectful distance\"}]\n"
            "}"
        )
        
        response = self.llm.generate(text_chunk, system_prompt=system_prompt, temperature=0.1)
        
        try:
            # Robust JSON extraction from markdown wrappers
            import re
            match = re.search(r'(\{.*\})', response, re.DOTALL)
            if match:
                response = match.group(1)
                
            response = re.sub(r',\s*([\]}])', r'\1', response)
                
            data = json.loads(response)
            if isinstance(data, dict):
                return {
                    "characters": data.get("characters", []),
                    "locations": data.get("locations", []),
                    "world_concepts": data.get("world_concepts", [])
                }
            return {"characters": [], "locations": [], "world_concepts": []}
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM combined extraction response: {e}\nResponse: {response}")
            return {"characters": [], "locations": [], "world_concepts": []}

    def extract_world_style(self, text_chunk: str) -> str:
        """
        Deduces the overall visual atmosphere and setting of the story for image generation.
        """
        system_prompt = (
            "You are an expert storyboard artist. Read the following text and determine the visual setting and atmosphere. "
            "For example: 'ancient china, historical, wuxia, poor village, rustic', OR 'sci-fi, futuristic, cyberpunk, neon lights', OR 'modern day, urban, city'. "
            "Return ONLY a comma-separated list of 3-6 Danbooru-style setting tags. Do not write anything else."
        )
        response = self.llm.generate(text_chunk, system_prompt=system_prompt, temperature=0.1)
        return response.strip().strip('"').strip("'")
