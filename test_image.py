import sys
import os

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import logging
logging.basicConfig(level=logging.INFO)

from models.image_adapter import LocalImageAdapter

def test_gen():
    try:
        adapter = LocalImageAdapter()
        print("Adapter initialized.")
        
        # Style Calibration Prompt: Testing the new Professional Manhwa V5 settings
        prompt = "manhwa, webtoon style, digital media, official art, sharp lineart, uniform lines, flat color, cel shading, high contrast, vibrant colors, 1boy, solo, short black hair, piercing eyes, ancient eastern royal robes, cinematic lighting, bloom, year 2024, masterpiece, best quality, official art, rating_safe"
        negative = "watercolor, oil painting, traditional media, sketch, pencil, graphite, charcoal, canvas, textured paper, brush strokes, painterly, blurry, lowres, bad anatomy, bad hands, text, error"
        
        params = {
            "seed": 12345,
            "steps": 30,
            "cfg": 7.0,
            "width": 1024,
            "height": 1024
        }
        
        print(f"Generating calibration image...")
        adapter.generate_image(prompt, "style_calibration.png", negative_prompt=negative, generation_params=params)
        print("Style calibration image saved to style_calibration.png.")
    except Exception as e:
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_gen()
