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
            "You are an expert lore master and character designer. Read the story text and extract ALL characters, locations, and world concepts.\n"
            "Return the data STRICTLY as a JSON object with keys: 'characters', 'locations', 'world_concepts'.\n\n"
            "EXISTING CHARACTERS (Reuse these names exactly if they appear): " + existing_char_list + "\n\n"
            "CRITICAL RULES FOR CHARACTERS:\n"
            "1. You MUST use Danbooru-style tags for 'age', 'body_type', 'face_info', 'hair', 'eyes', and 'clothing'.\n"
            "2. For children, use tags like 'young boy', 'little girl', 'toddler', 'child'.\n"
            "3. For 'age', be specific (e.g., '6 years old', 'late 30s').\n"
            "4. If a character is already in the EXISTING list, only provide visual_dna if new information is found; otherwise keep it consistent.\n"
            "5. Character DNA must be specific enough for high-quality image generation.\n"
            "6. DO NOT extract locations or objects as characters (e.g., 'Village', 'Sword' are NOT characters).\n\n"
            "EXAMPLE OUTPUT FORMAT:\n"
            "{\n"
            "  \"characters\": [{\"canonical_name\": \"Xu Changshou\", \"visual_dna\": {\"age\": \"6 years old\", \"body_type\": \"young boy, small, weak\", \"face_info\": \"pale skin, black eyes\", \"hair\": \"short black hair\", \"clothing\": \"tattered ancient peasant clothes\"}}],\n"

            "  \"locations\": [{\"canonical_name\": \"Xiaoyan Village\", \"description\": \"A poor, famine-stricken mountain village.\"}],\n"
            "  \"world_concepts\": [{\"concept_type\": \"item\", \"name\": \"Amber Egg\", \"description\": \"A preserved century egg, translucent and golden.\"}]\n"
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
