import os
import json
import logging
import shutil

logger = logging.getLogger(__name__)

class PublishingGenerator:
    """
    Layer 9: Publishing Engine.
    Generates SEO metadata, Youtube Title/Description, and extracts the thumbnail.
    """
    def __init__(self, llm_adapter, project_dir: str):
        self.llm = llm_adapter
        self.output_dir = os.path.join(project_dir, 'output')
        self.images_dir = os.path.join(self.output_dir, 'images')
        
    def generate_seo_metadata(self, translated_text: str):
        """Generates title, description, and tags using LLM."""
        system_prompt = (
            "You are a YouTube SEO expert specializing in anime and web novel recaps. "
            "Based on the following chapter text, generate a viral YouTube title, an engaging description, "
            "and a list of 10 relevant SEO tags. "
            "Return STRICTLY as valid JSON. Example: "
            "{\"title\": \"He Woke Up In Another World!\", \"description\": \"In this episode...\", \"tags\": [\"manhwa recap\", \"anime\"]}"
        )
        
        response = self.llm.generate(translated_text, system_prompt=system_prompt, temperature=0.5)
        
        try:
            response = response.strip()
            if response.startswith("```json"): response = response[7:]
            if response.endswith("```"): response = response[:-3]
            data = json.loads(response)
        except json.JSONDecodeError:
            # Fallback
            data = {
                "title": "Epic Web Novel Recap",
                "description": "Watch the latest episode of this amazing web novel.",
                "tags": ["webnovel", "recap", "anime", "manhwa"]
            }
            
        out_path = os.path.join(self.output_dir, 'seo_metadata.json')
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
            
        logger.info(f"Saved SEO metadata to {out_path}")
        return data

    def select_thumbnail(self):
        """Picks the first generated image to be the thumbnail."""
        if not os.path.exists(self.images_dir):
            logger.warning("Images dir not found, cannot select thumbnail.")
            return
            
        images = [f for f in os.listdir(self.images_dir) if f.endswith('.png')]
        if not images:
            logger.warning("No images found for thumbnail.")
            return
            
        # Just grab the first scene image
        source_img = os.path.join(self.images_dir, images[0])
        thumb_out = os.path.join(self.output_dir, 'thumbnail.png')
        
        shutil.copy2(source_img, thumb_out)
        logger.info(f"Selected {images[0]} as the YouTube Thumbnail at {thumb_out}")
