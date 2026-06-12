import os
import logging

os.environ["SDL_AUDIODRIVER"] = "dummy"
os.environ["SDL_VIDEODRIVER"] = "dummy"

import json
from moviepy.editor import ImageClip, AudioFileClip, concatenate_videoclips

logger = logging.getLogger(__name__)

class VideoRenderer:
    """
    Layer 8: Video Production.
    Assembles generated images and audio into a final mp4 video.
    """
    def __init__(self, project_dir: str):
        self.output_dir = os.path.join(project_dir, 'output')
        self.prompts_path = os.path.join(self.output_dir, 'prompts.json')
        self.images_dir = os.path.join(self.output_dir, 'images')
        self.audio_dir = os.path.join(self.output_dir, 'audio')
        self.final_video_path = os.path.join(self.output_dir, 'final_video.mp4')
        
    def render(self):
        if not os.path.exists(self.prompts_path):
            logger.error("prompts.json not found. Cannot render video.")
            return

        with open(self.prompts_path, 'r', encoding='utf-8') as f:
            prompts_data = json.load(f)

        clips = []
        
        for p in prompts_data:
            scene_id = p.get('scene_id')
            img_path = os.path.join(self.images_dir, f"{scene_id}.png")
            aud_path = os.path.join(self.audio_dir, f"{scene_id}.wav")
            
            if not os.path.exists(img_path) or not os.path.exists(aud_path):
                logger.warning(f"Missing media for {scene_id}, skipping.")
                continue
                
            try:
                # Load Audio to determine duration
                audio_clip = AudioFileClip(aud_path)
                duration = audio_clip.duration
                
                # Minimum duration safeguard (in case audio is too short or mock audio is 0s)
                if duration < 1.0:
                    duration = 3.0
                    
                # Load Image and set duration to match audio
                img_clip = ImageClip(img_path).set_duration(duration)
                
                try:
                    # Generate Subtitle
                    from moviepy.editor import TextClip, CompositeVideoClip
                    from core.config_manager import ConfigManager
                    config = ConfigManager()
                    
                    subtitle_text = p.get('metadata', {}).get('description', '')
                    
                    if subtitle_text:
                        txt_clip = TextClip(
                            subtitle_text,
                            font=config.get('video.font', 'Arial'),
                            fontsize=config.get('video.font_size', 40),
                            color='white',
                            bg_color='black'
                        ).set_position(('center', 'bottom')).set_duration(duration)
                        
                        # Composite the text over the image
                        img_clip = CompositeVideoClip([img_clip, txt_clip])
                except Exception as text_e:
                    logger.warning(f"Failed to generate TextClip (ImageMagick might be missing): {text_e}")

                img_clip = img_clip.set_audio(audio_clip)
                
                # Basic fade transition (fadein for half a second)
                if len(clips) > 0:
                    import moviepy.video.fx.all as vfx
                    from core.config_manager import ConfigManager
                    config = ConfigManager()
                    crossfade = config.get('video.crossfade_duration', 0.5)
                    
                    # Apply standard fadein directly via the fx method
                    img_clip = img_clip.fx(vfx.fadein, crossfade)
                    
                clips.append(img_clip)
                logger.info(f"Assembled {scene_id} - Duration: {duration}s")
            except Exception as e:
                logger.error(f"Failed to process {scene_id}: {e}")
                
        if not clips:
            logger.error("No valid clips assembled.")
            return
            
        logger.info("Concatenating clips and rendering final video... This may take a while.")
        
        # Concatenate using compose method to support crossfades
        final_video = concatenate_videoclips(clips, method="compose")
        
        # Write file
        try:
            final_video.write_videofile(
                self.final_video_path,
                fps=24,
                codec="libx264",
                audio_codec="aac",
                logger=None # Suppress moviepy internal spam
            )
            logger.info(f"Final video rendered successfully: {self.final_video_path}")
        except Exception as e:
            logger.error(f"Error rendering final video: {e}")
