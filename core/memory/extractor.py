import json
import logging
from typing import Dict, List

logger = logging.getLogger(__name__)

class MemoryExtractor:
    """
    Extracts characters, locations, and their visual DNA from translated text.
    """
    def __init__(self, llm_adapter):
        self.llm = llm_adapter

    def extract_characters(self, text_chunk: str) -> List[Dict]:
        """
        Uses the LLM to identify characters and extract their visual DNA.
        """
        system_prompt = (
            "You are an expert lore master. Read the following story text and extract all characters mentioned. "
            "For each character, extract their canonical name, any aliases, and their 'Visual DNA' (hair color, eye color, clothing, etc.).\n"
            "CRITICAL RULES:\n"
            "1. Use ONLY short, comma-separated Danbooru tags for the visual DNA (e.g., 'black hair, brown eyes, simple village clothes, tattered tunic'). DO NOT use full sentences or subjective descriptions like 'weak and listless'.\n"
            "2. DO NOT write 'not specified' or 'unknown'. If a visual attribute is missing from the text, you MUST invent a highly plausible, consistent anime/manhwa design (e.g., 'black hair', 'brown eyes') and stick to it so the character has a concrete visual identity!\n"
            "Return the data STRICTLY as a valid JSON array of objects. Example: "
            "[{\"canonical_name\": \"John Doe\", \"aliases\": [\"Johnny\"], \"visual_dna\": {\"hair\": \"black hair\", \"eyes\": \"blue eyes\", \"clothing\": \"red robe, loose clothing\"}}]"
        )
        
        response = self.llm.generate(text_chunk, system_prompt=system_prompt, temperature=0.1)
        
        try:
            # Basic cleanup if the LLM adds markdown formatting
            response = response.strip()
            if response.startswith("```json"):
                response = response[7:]
            if response.endswith("```"):
                response = response[:-3]
                
            data = json.loads(response)
            if isinstance(data, list):
                return data
            else:
                logger.error("LLM did not return a JSON array.")
                return []
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM character extraction response: {e}\nResponse: {response}")
            return []

    def extract_locations(self, text_chunk: str) -> List[Dict]:
        system_prompt = (
            "You are an expert lore master. Extract all locations and settings mentioned. "
            "Return STRICTLY as a valid JSON array of objects. Example: "
            "[{\"canonical_name\": \"Cloud Peak\", \"description\": \"A high mountain shrouded in mist.\"}]"
        )
        response = self.llm.generate(text_chunk, system_prompt=system_prompt, temperature=0.1)
        try:
            response = response.strip()
            if response.startswith("```json"): response = response[7:]
            if response.endswith("```"): response = response[:-3]
            data = json.loads(response)
            return data if isinstance(data, list) else []
        except json.JSONDecodeError:
            return []

    def extract_world_concepts(self, text_chunk: str) -> List[Dict]:
        system_prompt = (
            "You are an expert lore master. Extract all world concepts (currencies, sects, magic skills) mentioned. "
            "Return STRICTLY as a valid JSON array of objects. Example: "
            "[{\"concept_type\": \"sect\", \"name\": \"Heavenly Sword Sect\", \"description\": \"A powerful sect\"}]"
        )
        response = self.llm.generate(text_chunk, system_prompt=system_prompt, temperature=0.1)
        try:
            response = response.strip()
            if response.startswith("```json"): response = response[7:]
            if response.endswith("```"): response = response[:-3]
            data = json.loads(response)
            return data if isinstance(data, list) else []
        except json.JSONDecodeError:
            return []

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
