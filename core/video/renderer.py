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

        batch_size = 50
        batches = [prompts_data[i:i + batch_size] for i in range(0, len(prompts_data), batch_size)]
        
        logger.info(f"Total scenes: {len(prompts_data)}. Rendering in {len(batches)} batches to prevent OOM.")
        
        for batch_idx, batch_prompts in enumerate(batches):
            clips = []
            logger.info(f"--- Starting Render Batch {batch_idx + 1}/{len(batches)} ---")
            
            for p in batch_prompts:
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
                    
                    if duration < 1.0:
                        duration = 3.0
                        
                    # Load Image and set duration
                    img_clip = ImageClip(img_path).set_duration(duration)
                    
                    try:
                        # Generate Subtitle
                        from moviepy.editor import TextClip, CompositeVideoClip
                        from core.config_manager import ConfigManager
                        config = ConfigManager()
                        
                        subtitle_text = p.get('metadata', {}).get('narration_text', '')
                        
                        if subtitle_text:
                            txt_clip = TextClip(
                                subtitle_text,
                                font=config.get('video.font', 'Arial'),
                                fontsize=config.get('video.font_size', 40),
                                color='white',
                                bg_color='black',
                                method='caption',
                                size=(img_clip.w - 100, None)
                            ).set_position(('center', 'bottom')).set_duration(duration)
                            
                            img_clip = CompositeVideoClip([img_clip, txt_clip])
                    except Exception as text_e:
                        logger.warning(f"Failed to generate TextClip: {text_e}")

                    img_clip = img_clip.set_audio(audio_clip)
                    
                    if len(clips) > 0:
                        import moviepy.video.fx.all as vfx
                        from core.config_manager import ConfigManager
                        config = ConfigManager()
                        crossfade = config.get('video.crossfade_duration', 0.5)
                        img_clip = img_clip.fx(vfx.fadein, crossfade)
                        
                    clips.append(img_clip)
                    logger.info(f"Assembled {scene_id} - Duration: {duration}s")
                except Exception as e:
                    logger.error(f"Failed to process {scene_id}: {e}")
                    
            if not clips:
                logger.error(f"No valid clips assembled for batch {batch_idx + 1}.")
                continue
                
            # Dynamic output naming
            part_suffix = f"_part{batch_idx + 1}" if len(batches) > 1 else ""
            batch_video_path = self.final_video_path.replace('.mp4', f'{part_suffix}.mp4')
            
            logger.info(f"Concatenating {len(clips)} clips for {batch_video_path}... This may take a while.")
            
            # Concatenate using compose method to support crossfades
            final_video = concatenate_videoclips(clips, method="compose")
            
            try:
                final_video.write_videofile(
                    batch_video_path,
                    fps=24,
                    codec="libx264",
                    audio_codec="aac",
                    logger=None 
                )
                logger.info(f"Batch {batch_idx + 1} rendered successfully: {batch_video_path}")
            except Exception as e:
                logger.error(f"Error rendering batch {batch_idx + 1}: {e}")
                
            # CRITICAL MEMORY CLEANUP
            logger.info(f"Clearing RAM for batch {batch_idx + 1}...")
            try:
                final_video.close()
                for c in clips:
                    c.close()
            except Exception as e:
                logger.warning(f"Failed to close all clips: {e}")
