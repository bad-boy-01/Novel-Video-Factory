import logging
from typing import Dict, List

logger = logging.getLogger(__name__)

class PromptGenerator:
    """
    Layer 6: Visual Generation.
    Takes planned scenes and Memory Database information to assemble high-quality image generation prompts.
    """
    def __init__(self, memory_engine, base_style: str = "Cinematic, high quality Korean Manhwa style, detailed line art, masterpiece, best quality"):
        self.memory_engine = memory_engine
        self.base_style = base_style
        
    def generate_prompt_for_scene(self, scene: Dict) -> Dict:
        """
        Creates a Stable Diffusion/FLUX prompt for a single scene, injecting Character DNA.
        """
        characters_present = scene.get("characters_present", [])
        dna_descriptions = []
        
        for char_name in characters_present:
            char_data = self.memory_engine.get_character_by_name(char_name)
            if char_data:
                dna = char_data.get("visual_dna", {})
                # Format DNA into a string
                dna_str = ", ".join([f"{k} {v}" if not isinstance(v, dict) else "" for k, v in dna.items()]).strip()
                if dna_str:
                    dna_descriptions.append(f"({char_name}: {dna_str})")
                else:
                    dna_descriptions.append(char_name)
            else:
                dna_descriptions.append(char_name)
                
        # Build prompt
        action_desc = scene.get('visual_prompt_tags', '')
        camera = scene.get('camera_angle', 'medium shot')
        lighting = scene.get('lighting', 'cinematic lighting')
        
        character_prompt = ", ".join(dna_descriptions)
        
        full_prompt = f"{self.base_style}. {camera}. {action_desc}. {character_prompt}. {lighting}."
        
        return {
            "scene_id": scene.get("scene_id"),
            "prompt": full_prompt,
            "negative_prompt": "lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry",
            "metadata": scene
        }
