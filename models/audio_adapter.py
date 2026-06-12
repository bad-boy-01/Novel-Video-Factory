import logging
import os
import time

logger = logging.getLogger(__name__)

class LocalAudioAdapter:
    """
    Adapter for local text-to-speech generation using Coqui-TTS or similar.
    """
    def __init__(self):
        from core.config_manager import ConfigManager
        config = ConfigManager()
        self.model_name = config.get('models.audio.model', "tts_models/en/vctk/vits")
        self.speaker = config.get('models.audio.voice', "p267")
        self.tts = None
        self._init_tts()

    def _init_tts(self):
        """Attempts to load the TTS model. Falls back to mock on failure."""
        try:
            from TTS.api import TTS
            self.tts = TTS(self.model_name, progress_bar=False, gpu=True)
            logger.info(f"Initialized TTS model {self.model_name}")
        except ImportError:
            logger.warning("Coqui-TTS not installed properly. Running Audio Adapter in MOCK mode.")
            self.tts = None

    def generate_audio(self, text: str, output_path: str):
        """
        Generates an audio file from text and saves it.
        """
        if self.tts is not None:
            logger.info("Generating audio (Real)...")
            self.tts.tts_to_file(text=text, speaker=self.speaker, file_path=output_path)
            logger.info(f"Saved real generated audio to {output_path}")
        else:
            logger.info(f"Generating mock audio for text: {text[:30]}...")
            time.sleep(0.5) # Simulate generation time
            
            # Create a blank dummy wav file (1 second of silence)
            import wave
            import struct
            with wave.open(output_path, 'wb') as f:
                f.setnchannels(1)
                f.setsampwidth(2)
                f.setframerate(22050)
                # 1 second = 22050 frames
                frames = struct.pack('<h', 0) * 22050
                f.writeframes(frames)
            
            logger.info(f"Saved mock audio to {output_path}")
