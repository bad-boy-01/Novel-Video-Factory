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
        
        # Style Calibration V5 (Official 4.0 Sequence)
        # 1. Subject: 1boy, character traits
        # 2. Rating: rating_safe
        # 3. Content: scene, clothing, lighting, camera
        # 4. Art Style: manhwa/webtoon tags
        # 5. Quality: official 4.0 quality tags
        prompt = "1boy, solo, short black hair, piercing eyes, rating_safe, ancient eastern royal robes, cinematic lighting, bloom, backlighting, close up, manhwa style, webtoon style, digital media, official art, sharp lineart, flat color, cel shading, high contrast, vibrant colors, year 2024, masterpiece, high score, great score, absurdres"
        negative = "watercolor, oil painting, traditional media, painterly, brush strokes, textured paper, canvas, sketch, pencil, charcoal, blurry, lowres, bad anatomy, bad hands, text, error"
        
        params = {
            "seed": 12345,
            "steps": 30,
            "cfg": 7.0,
            "width": 1024,
            "height": 1024
        }
        
        print(f"Generating calibration image (Official 4.0 Structure)...")
        adapter.generate_image(prompt, "style_calibration.png", negative_prompt=negative, generation_params=params)
        print("Style calibration image saved to style_calibration.png.")
    except Exception as e:
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_gen()
