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

    def extract_all(self, text_chunk: str) -> Dict[str, List[Dict]]:
        """
        Extracts characters, locations, and world concepts in a single LLM call to save API rate limits.
        """
        system_prompt = (
            "You are an expert lore master. Read the following story text and extract ALL characters, locations, and world concepts mentioned.\n"
            "Return the data STRICTLY as a valid JSON object with three keys: 'characters', 'locations', and 'world_concepts'.\n\n"
            "CRITICAL RULES FOR CHARACTERS:\n"
            "1. You MUST include specific Danbooru-style tags for 'age', 'body_type', 'face_info', 'hair', 'eyes', and 'clothing'. DO NOT use full sentences.\n"
            "2. If a visual attribute is missing, you MUST invent a highly plausible, consistent anime/manhwa design and stick to it!\n\n"
            "EXAMPLE OUTPUT FORMAT:\n"
            "{\n"
            "  \"characters\": [{\"canonical_name\": \"John Doe\", \"aliases\": [\"Johnny\"], \"visual_dna\": {\"age\": \"20s\", \"body_type\": \"muscular\", \"face_info\": \"sharp jaw, scar on cheek\", \"hair\": \"black hair\", \"eyes\": \"blue eyes\", \"clothing\": \"red robe\"}}],\n"
            "  \"locations\": [{\"canonical_name\": \"Cloud Peak\", \"description\": \"A high mountain shrouded in mist.\"}],\n"
            "  \"world_concepts\": [{\"concept_type\": \"sect\", \"name\": \"Heavenly Sword Sect\", \"description\": \"A powerful sect\"}]\n"
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
