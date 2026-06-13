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

    def plan_scenes(self, text_chunk: str, start_index: int = 0) -> List[Dict]:
        """
        Uses the LLM to slice a text chunk into visual scenes with sentence tracking.
        """
        system_prompt = (
            "You are an expert storyboard artist for an anime/manhwa adaptation. "
            "Break the following story text into a highly detailed sequence of cinematic scenes.\n"
            "CRITICAL RULES:\n"
            "1. DO NOT SUMMARIZE. Create one scene for every 1 to 2 sentences of the text. A 1000 word chapter should yield at least 30-40 scenes.\n"
            "2. For 'visual_prompt_tags', 'camera_angle', and 'lighting', use short, comma-separated Danbooru tags, NOT full English sentences.\n"
            "3. The 'visual_prompt_tags' MUST be under 15 words.\n"
            "4. The 'narration_text' MUST contain the exact, uncut dialogue or narration from the original text.\n"
            "5. OUTPUT STRICTLY IN ENGLISH.\n"
            "6. AVOID JSON ERRORS: Use single quotes (') for dialogue inside narration_text.\n"
            "7. SENTENCE TRACKING: For each scene, include a 'source_sentences' list containing the indices of the sentences covered (e.g. [0, 1]).\n"
            f"The first sentence of this text has index: {start_index}.\n"
            "Return STRICTLY as a valid JSON array of objects. Example: "
            "[{\"scene_id\": \"SC001\", \"source_sentences\": [0, 1], \"visual_prompt_tags\": \"1boy, waking up, confused\", \"narration_text\": \"John woke up and said, 'Where am I?'\", \"characters_present\": [\"John\"], \"camera_angle\": \"close up\", \"lighting\": \"morning sunlight\"}]"
        )
        
        response = self.llm.generate(text_chunk, system_prompt=system_prompt, temperature=0.3)
        
        try:
            import re
            match = re.search(r'(\[.*\])', response, re.DOTALL)
            if match:
                response = match.group(1)
                
            data = json.loads(response)
            if isinstance(data, list):
                # Verify coverage (Zero Information Loss check)
                import re
                sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text_chunk) if s.strip()]
                covered_indices = set()
                for scene in data:
                    covered_indices.update(scene.get('source_sentences', []))
                
                missing = [i + start_index for i in range(len(sentences)) if (i + start_index) not in covered_indices]
                if missing:
                    logger.warning(f"Information Loss Warning: Sentences {missing} are not covered by any scene.")
                
                return data
            else:
                logger.error("ScenePlanner LLM did not return a JSON array.")
                return []
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM scene planning response: {e}\nResponse: {response}")
            return []
