import logging
import os
import re
import random
import hashlib
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
        Creates a Stable Diffusion/FLUX prompt for a single scene, injecting Character DNA and World Context.
        """
        characters_present = scene.get("characters_present", [])
        dna_descriptions = []
        
        project_dir = self.memory_engine.project_dir if hasattr(self.memory_engine, 'project_dir') else ""
        ref_images = []
        
        # 1. Character DNA Injection
        for char_name in characters_present:
            char_data = self.memory_engine.get_character_by_name(char_name)
            if char_data:
                dna = char_data.get("visual_dna", {})
                # Format DNA into a detailed string of tags
                dna_tags = []
                for k, v in dna.items():
                    if v and str(v).lower() not in ['none', 'unknown', 'not specified']:
                        dna_tags.append(str(v))
                
                dna_str = ", ".join(dna_tags)
                
                # Deduce gender for Danbooru-based models
                dna_lower = dna_str.lower()
                name_lower = char_name.lower()
                if any(w in dna_lower or w in name_lower for w in ["girl", "woman", "female", "sister", "mother", "wife", "chunni", "xiue", "mei", "her ", "she ", "madam", "dress", "aunt", "lady"]):
                    gender_tag = "1girl"
                else:
                    gender_tag = "1boy"
                
                if dna_str:
                    dna_descriptions.append(f"{gender_tag}, {dna_str}")
                else:
                    dna_descriptions.append(f"{gender_tag}, {char_name}")
                    
                # Look for reference image
                if project_dir:
                    safe_name = re.sub(r'[\\/*?:"<>|]', "", char_name).strip().replace(" ", "_")
                    img_path = os.path.join(project_dir, 'memory', 'character_sheets', f"{safe_name}.png")
                    
                    if os.path.exists(img_path):
                        ref_images.append(img_path)
            else:
                dna_descriptions.append(char_name)
                
        # 2. World Concept & Location Injection
        location_tags = ""
        world_tags = ""
        staging_tags = self.memory_engine.get_relationship_staging(characters_present)
        narration = scene.get('narration_text', '').lower()
        
        # Check for locations in the narration/scene metadata
        with self.memory_engine.Session() as session:
            from core.memory.database import Location, WorldConcept
            locations = session.query(Location).all()
            for loc in locations:
                if loc.canonical_name.lower() in narration:
                    location_tags += f"{loc.description}, "
                    # V3 Upgrade: Background Reference
                    if loc.background_path and os.path.exists(loc.background_path):
                        ref_images.append(loc.background_path)
            
            concepts = session.query(WorldConcept).all()
            for concept in concepts:
                if concept.name.lower() in narration:
                    world_tags += f"{concept.name}, {concept.description}, "

        # Build prompt components
        action_desc = scene.get('visual_prompt_tags', '')
        camera = scene.get('camera_angle', 'medium shot')
        lighting = scene.get('lighting', 'cinematic lighting')
        
        character_prompt = ", ".join(dna_descriptions)
        
        # Quality and Style Tags for Animagine XL 4.0 Manhwa style
        quality_tags = "masterpiece, high score, great score, absurdres"
        manhwa_core = "manhwa, webtoon, korean style, thick outlines, vibrant colors"
        cinematic_tags = "cinematic lighting, atmospheric, moody, soft focus, high resolution"
        year_tag = "year 2024"
        
        # Build Structured Prompt: Subject -> General -> Style -> Quality
        # We prioritize character features and scene-specific items to keep them connected to the story
        full_prompt = f"{character_prompt}, {staging_tags}, {action_desc}, {world_tags}{location_tags}{camera}, {manhwa_core}, {lighting}, {cinematic_tags}, {year_tag}, {quality_tags}, rating_safe"
        negative_prompt = "lowres, bad anatomy, bad hands, text, error, missing finger, extra digits, fewer digits, cropped, worst quality, low quality, low score, bad score, average score, signature, watermark, username, blurry"
        
        # V3 Upgrade: Persistent Seeds and Prompt Hashing
        seed = scene.get('seed', random.randint(0, 2147483647))
        prompt_hash = hashlib.sha256((full_prompt + negative_prompt).encode('utf-8')).hexdigest()
        
        return {
            "scene_id": scene.get("scene_id"),
            "prompt": full_prompt,
            "negative_prompt": negative_prompt,
            "metadata": scene,
            "reference_images": ref_images,
            "generation_params": {
                "seed": seed,
                "steps": 28,
                "cfg": 5.0,
                "width": 1280,
                "height": 720
            },
            "prompt_hash": prompt_hash
        }
