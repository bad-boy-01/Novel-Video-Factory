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

            # V3 Upgrade: Aggressive JSON Cleaning
            # 1. Remove trailing commas before closing braces/brackets
            response = re.sub(r',\s*([\]}])', r'\1', response)
            # 2. Fix unescaped double quotes inside values (common LLM error)
            # This is a heuristic: replaces " with ' if it's not a JSON key/value delimiter
            # But safer to just try and catch the specific error

            data = json.loads(response)
            if isinstance(data, dict):
                # V3 Upgrade: Robust Data Normalization
                # Ensure every entry is a dictionary to prevent AttributeError
                def normalize(items, keys):
                    results = []
                    for item in items:
                        if isinstance(item, dict):
                            results.append(item)
                        elif isinstance(item, str):
                            # Convert string name to basic object
                            results.append({keys[0]: item})
                    return results

                return {
                    "characters": normalize(data.get("characters", []), ["canonical_name"]),
                    "locations": normalize(data.get("locations", []), ["canonical_name", "description"]),
                    "world_concepts": normalize(data.get("world_concepts", []), ["name", "description"]),
                    "relationships": normalize(data.get("relationships", []), ["char1", "char2"])
                }
            return {"characters": [], "locations": [], "world_concepts": [], "relationships": []}
        except json.JSONDecodeError as e:
            # Secondary recovery: try to fix common quote issues
            try:
                # Replace internal double quotes that aren't delimiters
                # This is a bit risky but can save a failed run
                cleaned_response = re.sub(r'(?<!:)\s*"(?![:,\]}])', "'", response)
                cleaned_response = re.sub(r'(?<![\[{,])\s*"(?!:)', "'", cleaned_response)
                data = json.loads(cleaned_response)
                logger.info("Successfully repaired malformed JSON from LLM.")
                # (Normalization logic repeat...)
                return {
                    "characters": [{"canonical_name": c} if isinstance(c, str) else c for c in data.get("characters", [])],
                    "locations": [{"canonical_name": l} if isinstance(l, str) else l for l in data.get("locations", [])],
                    "world_concepts": [{"name": w} if isinstance(w, str) else w for w in data.get("world_concepts", [])],
                    "relationships": data.get("relationships", [])
                }
            except Exception:
                logger.error(f"Failed to parse LLM combined extraction response: {e}\nResponse: {response}")
                return {"characters": [], "locations": [], "world_concepts": [], "relationships": []}

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
