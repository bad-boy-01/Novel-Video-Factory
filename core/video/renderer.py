import os
import logging
import json
import random
import re
import gc
import torch
import subprocess

os.environ["SDL_AUDIODRIVER"] = "dummy"
os.environ["SDL_VIDEODRIVER"] = "dummy"

from moviepy.editor import ImageClip, AudioFileClip, concatenate_videoclips

logger = logging.getLogger(__name__)

class VideoRenderer:
    """
    Layer 8: Video Production (Cinematic V4).
    Assembles cinematic shots into Clips and finally a Master Movie.
    """
    def __init__(self, project_dir: str):
        self.project_dir = project_dir
        self.output_dir = os.path.join(project_dir, 'output')
        self.videos_dir = os.path.join(self.output_dir, 'videos')
        os.makedirs(self.videos_dir, exist_ok=True)
        
        self.clips_path = os.path.join(self.output_dir, 'clips.json')
        self.images_dir = os.path.join(self.output_dir, 'images')
        self.audio_dir = os.path.join(self.output_dir, 'audio')
        self.final_video_path = os.path.join(self.videos_dir, 'final_video.mp4')
        
    def render(self):
        if not os.path.exists(self.clips_path):
            logger.error("clips.json not found. Cannot render video.")
            return

        with open(self.clips_path, 'r', encoding='utf-8') as f:
            clips_data = json.load(f)

        logger.info(f"Rendering {len(clips_data)} balanced cinematic clips...")
        
        rendered_clips = []
        for clip in clips_data:
            clip_id = clip['clip_id']
            shots = clip['shots']
            logger.info(f"--- Starting Render: {clip_id} ({len(shots)} shots) ---")
            
            shot_clips = []
            for shot in shots:
                shot_id = shot.get('shot_id')
                img_path = os.path.join(self.images_dir, f"{shot_id}.png")
                aud_path = os.path.join(self.audio_dir, f"{shot_id}.wav")
                
                if not os.path.exists(img_path) or not os.path.exists(aud_path):
                    logger.warning(f"Missing media for {shot_id}, skipping.")
                    continue
                    
                try:
                    # Audio and Duration
                    audio_clip = AudioFileClip(aud_path)
                    duration = audio_clip.duration
                    if duration < 0.5: duration = 2.0
                    
                    # Image
                    img_clip = ImageClip(img_path).set_duration(duration)
                    
                    # Layer 8.1: Ken Burns (Dynamic Motion)
                    zoom_direction = random.choice(['in', 'out'])
                    zoom_speed = 0.04
                    def zoom_effect(t):
                        if zoom_direction == 'in':
                            return 1 + (zoom_speed * t / duration)
                        else:
                            return 1 + zoom_speed - (zoom_speed * t / duration)
                    img_clip = img_clip.resize(zoom_effect).set_position('center')
                    
                    # Subtitles
                    try:
                        from moviepy.editor import TextClip, CompositeVideoClip, ColorClip
                        from core.config_manager import ConfigManager
                        config = ConfigManager()
                        
                        subtitle_text = shot.get('narration_text', '')
                        if subtitle_text:
                            # Split into sentences for readability (max 2)
                            sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', subtitle_text) if s.strip()]
                            groups = []
                            i = 0
                            while i < len(sentences):
                                s1 = sentences[i]
                                if i+1 < len(sentences) and len(s1) < 80 and (len(s1)+len(sentences[i+1])) < 140:
                                    groups.append(s1 + " " + sentences[i+1])
                                    i += 2
                                else:
                                    groups.append(s1)
                                    i += 1
                                    
                            num_groups = len(groups)
                            group_dur = duration / num_groups
                            subtitle_clips = []
                            for idx, g_text in enumerate(groups):
                                t_start = idx * group_dur
                                txt = TextClip(
                                    g_text, font=config.get('video.font', 'Arial-Bold'),
                                    fontsize=config.get('video.font_size', 44), color='white',
                                    method='caption', size=(img_clip.w * 0.8, None), align='center'
                                ).set_duration(group_dur).set_start(t_start)
                                
                                bg = ColorClip(size=(img_clip.w, txt.h + 40), color=(0,0,0)).set_opacity(0.4).set_duration(group_dur).set_start(t_start)
                                bg = bg.set_position(('center', 'bottom'))
                                txt = txt.set_position(('center', (img_clip.h - (txt.h + 40)) + 20))
                                subtitle_clips.extend([bg, txt])
                                
                            img_clip = CompositeVideoClip([img_clip] + subtitle_clips)
                    except Exception as te:
                        logger.warning(f"TextClip failed: {te}")

                    img_clip = img_clip.set_audio(audio_clip)
                    shot_clips.append(img_clip)
                except Exception as e:
                    logger.error(f"Failed to process shot {shot_id}: {e}")
                    
            if not shot_clips: continue
            
            # Render individual clip
            clip_video_path = os.path.join(self.videos_dir, f"{clip_id}.mp4")
            final_clip = concatenate_videoclips(shot_clips, method="compose")
            
            try:
                final_clip.write_videofile(
                    clip_video_path,
                    fps=24, codec="libx264", audio_codec="aac", logger=None 
                )
                rendered_clips.append(f"{clip_id}.mp4")
            except Exception as e:
                logger.error(f"Error rendering {clip_id}: {e}")
                
            # Cleanup RAM
            final_clip.close()
            for sc in shot_clips: sc.close()
            gc.collect()
            if torch.cuda.is_available(): torch.cuda.empty_cache()

        # Final Stitching
        if len(rendered_clips) > 1:
            logger.info("Stitching clips into Master Movie...")
            list_path = os.path.join(self.videos_dir, 'concat_list.txt')
            with open(list_path, 'w', encoding='utf-8') as f:
                for rc in rendered_clips: f.write(f"file '{rc}'\n")
            
            try:
                subprocess.run([
                    'ffmpeg', '-y', '-f', 'concat', '-safe', '0', 
                    '-i', 'concat_list.txt', '-c', 'copy', 'final_video_master.mp4'
                ], check=True, cwd=self.videos_dir, capture_output=True)
                
                os.remove(list_path)
                # Cleanup clips
                for rc in rendered_clips: os.remove(os.path.join(self.videos_dir, rc))
                os.rename(os.path.join(self.videos_dir, 'final_video_master.mp4'), self.final_video_path)
                logger.info("Master Video ready!")
            except Exception as e:
                logger.error(f"FFmpeg stitch failed: {e}")
        elif rendered_clips:
            os.rename(os.path.join(self.videos_dir, rendered_clips[0]), self.final_video_path)
