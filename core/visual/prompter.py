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
        
        # Inject Dynamic World Style if it exists
        import os
        project_dir = self.memory_engine.project_dir if hasattr(self.memory_engine, 'project_dir') else ""
        if project_dir:
            world_style_path = os.path.join(project_dir, 'memory', 'world_style.txt')
            if os.path.exists(world_style_path):
                with open(world_style_path, 'r', encoding='utf-8') as f:
                    world_tags = f.read().strip()
                if world_tags:
                    self.base_style = f"{world_tags}, {self.base_style}"
                    logger.info(f"Loaded Dynamic World Style: {world_tags}")
        
    def generate_prompt_for_scene(self, scene: Dict) -> Dict:
        """
        Creates a Stable Diffusion/FLUX prompt for a single scene, injecting Character DNA.
        """
        characters_present = scene.get("characters_present", [])
        dna_descriptions = []
        
        import os
        project_dir = self.memory_engine.project_dir if hasattr(self.memory_engine, 'project_dir') else ""
        ref_images = []
        
        for char_name in characters_present:
            char_data = self.memory_engine.get_character_by_name(char_name)
            if char_data:
                dna = char_data.get("visual_dna", {})
                # Format DNA into a string
                dna_str = ", ".join([f"{v}" if not isinstance(v, dict) else "" for k, v in dna.items()]).strip()
                
                # Deduce gender for Danbooru-based models
                dna_lower = dna_str.lower()
                name_lower = char_name.lower()
                if any(w in dna_lower or w in name_lower for w in ["girl", "woman", "female", "sister", "mother", "wife", "chunni", "xiue", "mei", "her ", "she ", "madam", "dress", "aunt", "lady"]):
                    gender_tag = "1girl"
                else:
                    gender_tag = "1boy"
                
                if dna_str:
                    dna_descriptions.append(f"{gender_tag}, {char_name}, {dna_str}")
                else:
                    dna_descriptions.append(f"{gender_tag}, {char_name}")
                    
                # Look for reference image
                if project_dir and len(characters_present) == 1:
                    import re
                    safe_name = re.sub(r'[\\/*?:"<>|]', "", char_name).strip().replace(" ", "_")
                    img_path = os.path.join(project_dir, 'memory', 'character_sheets', f"{safe_name}.png")
                    
                    # Fallback to ID-based naming for backward compatibility
                    if not os.path.exists(img_path):
                        img_path = os.path.join(project_dir, 'memory', 'character_sheets', f"{char_data.get('id')}.png")
                        
                    if os.path.exists(img_path):
                        ref_images.append(img_path)
            else:
                dna_descriptions.append(char_name)
                
        # Build prompt
        action_desc = scene.get('visual_prompt_tags', '')
        camera = scene.get('camera_angle', 'medium shot')
        lighting = scene.get('lighting', 'cinematic lighting')
        
        character_prompt = ", ".join(dna_descriptions)
        
        # Quality and Style Tags for Animagine XL 4.0 Manhwa style
        quality_tags = "masterpiece, high score, great score, absurdres"
        wuxia_details = "long hair, traditional chinese clothing, ancient chinese architecture"
        
        # Build Structured Prompt: Subject -> Details -> Style -> Quality
        full_prompt = f"{character_prompt}, {camera}, {action_desc}, {lighting}, {wuxia_details}, rating_safe, {self.base_style}, {quality_tags}"
        
        return {
            "scene_id": scene.get("scene_id"),
            "prompt": full_prompt,
            "negative_prompt": "lowres, bad anatomy, bad hands, text, error, missing finger, extra digits, fewer digits, cropped, worst quality, low quality, low score, bad score, average score, signature, watermark, username, blurry",
            "metadata": scene,
            "reference_images": ref_images
        }
