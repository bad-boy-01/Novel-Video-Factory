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
        
    def generate_prompt_for_shot(self, shot: Dict, scene: Dict) -> Dict:
        """
        Creates a Stable Diffusion/FLUX prompt for a single cinematic SHOT, 
        injecting context from the parent Narrative Scene.
        """
        characters_present = scene.get("characters", [])
        dna_descriptions = []
        
        project_dir = self.memory_engine.project_dir if hasattr(self.memory_engine, 'project_dir') else ""
        ref_images = []
        
        # 1. Character DNA Injection
        for char_name in characters_present:
            char_data = self.memory_engine.get_character_by_name(char_name)
            if char_data:
                dna = char_data.get("visual_dna", {})
                dna_tags = [str(v) for k, v in dna.items() if v and str(v).lower() not in ['none', 'unknown', 'not specified']]
                dna_str = ", ".join(dna_tags)
                
                # Deduce gender
                dna_lower = dna_str.lower()
                name_lower = char_name.lower()
                if any(w in dna_lower or w in name_lower for w in ["girl", "woman", "female", "sister", "mother", "wife", "chunni", "xiue", "mei", "her ", "she ", "madam", "dress", "aunt", "lady"]):
                    gender_tag = "1girl"
                else:
                    gender_tag = "1boy"
                
                dna_descriptions.append(f"{gender_tag}, {dna_str}" if dna_str else f"{gender_tag}, {char_name}")
                    
                # Reference image
                if project_dir:
                    safe_name = re.sub(r'[\\/*?:"<>|]', "", char_name).strip().replace(" ", "_")
                    img_path = os.path.join(project_dir, 'memory', 'character_sheets', f"{safe_name}.png")
                    if os.path.exists(img_path):
                        ref_images.append(img_path)
            else:
                dna_descriptions.append(char_name)
                
        # 2. World Context (from Scene)
        location_tags = ""
        world_tags = ""
        staging_tags = self.memory_engine.get_relationship_staging(characters_present)
        location_name = scene.get('location', '').lower()
        
        with self.memory_engine.Session() as session:
            from core.memory.database import Location, WorldConcept
            locations = session.query(Location).all()
            for loc in locations:
                if loc.canonical_name.lower() in location_name or loc.canonical_name.lower() in shot.get('visual_prompt_tags', '').lower():
                    location_tags += f"{loc.description}, "
                    if loc.background_path and os.path.exists(loc.background_path):
                        ref_images.append(loc.background_path)
            
            concepts = session.query(WorldConcept).all()
            for concept in concepts:
                if concept.name.lower() in shot.get('narration_text', '').lower():
                    world_tags += f"{concept.name}, {concept.description}, "

        # 3. Shot-Specific Visuals
        action_tags = shot.get('visual_prompt_tags', '')
        camera = shot.get('camera', 'medium shot')
        lighting = scene.get('lighting', 'cinematic lighting')
        
        # Character Description (The Subject)
        character_prompt = ", ".join(dna_descriptions)
        
        # Art Style Definition (Modern Digital Webtoon)
        # Using specific tags from 'The Otherworldly Nutritionist' analysis
        art_style = "manhwa style, webtoon style, digital media, clean lineart, flat color, cel shading, high contrast, vibrant colors, official art"
        
        # Cinematic Effects (The Lighting/Atmosphere)
        cinematic = f"{lighting}, bloom, backlighting, atmospheric, {camera}"
        
        # Quality Enhancement (Official 4.0 sequence)
        quality = "masterpiece, high score, great score, absurdres"
        year_tag = "year 2024" # Or 2025 for extremely modern look
        
        # Build the final prompt in the OFFICIAL model sequence:
        # [Subject] -> [Rating] -> [Content] -> [Art Style] -> [Quality]
        full_prompt = f"{character_prompt}, rating_safe, {action_tags}, {world_tags}{location_tags}{cinematic}, {staging_tags}, {art_style}, {year_tag}, {quality}"
        
        # Aggressive Negative Prompt to strip away 'Painterly/Watercolor' look
        negative_prompt = "watercolor, oil painting, traditional media, painterly, brush strokes, textured paper, canvas, sketch, pencil, charcoal, blurry, lowres, bad anatomy, bad hands, text, error, worst quality, low quality, signature, watermark, username"
        
        # Deterministic Seed
        import random, hashlib
        seed = shot.get('seed', random.randint(0, 2147483647))
        prompt_hash = hashlib.sha256((full_prompt + negative_prompt).encode('utf-8')).hexdigest()
        
        return {
            "shot_id": shot.get("shot_id"),
            "prompt": full_prompt,
            "negative_prompt": negative_prompt,
            "metadata": {**scene, **shot}, # Merge metadata
            "reference_images": list(set(ref_images)),
            "generation_params": {
                "seed": seed,
                "steps": 28,
                "cfg": 5.0,
                "width": 1280,
                "height": 720
            },
            "prompt_hash": prompt_hash
        }

    def generate_prompt_for_scene(self, scene: Dict) -> Dict:
        """Legacy support for Scene-based prompting (will be removed)."""
        # (Internal logic redirecting to generate_prompt_for_shot with a mock shot)
        mock_shot = {
            "shot_id": f"{scene.get('scene_id')}_A",
            "camera": scene.get('camera_angle', 'medium shot'),
            "visual_prompt_tags": scene.get('visual_prompt_tags', ''),
            "narration_text": scene.get('narration_text', ''),
            "duration_estimate": 5.0
        }
        return self.generate_prompt_for_shot(mock_shot, scene)
