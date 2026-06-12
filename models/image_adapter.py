import logging
import os
import time

logger = logging.getLogger(__name__)

class LocalImageAdapter:
    """
    Adapter for local image generation using diffusers (e.g. FLUX, SDXL).
    """
    def __init__(self):
        from core.config_manager import ConfigManager
        config = ConfigManager()
        self.model_name = config.get('models.image_generation.model', "cagliostrolab/animagine-xl-3.1")
        self.pipeline = None
        self.width = config.get('models.image.width', 1280)
        self.height = config.get('models.image.height', 720)
        self._init_pipeline()

    def _init_pipeline(self):
        """Attempts to load the diffusers pipeline. Falls back to mock on failure."""
        try:
            import torch
            from diffusers import AutoPipelineForText2Image
            
            self.pipeline = AutoPipelineForText2Image.from_pretrained(
                self.model_name, torch_dtype=torch.float16
            )
            # IP Adapter support for SDXL
            try:
                self.pipeline.load_ip_adapter("h94/IP-Adapter", subfolder="sdxl_models", weight_name="ip-adapter_sdxl.bin")
                self.pipeline.set_ip_adapter_scale(0.35)
                logger.info("IP-Adapter loaded successfully for Character Reference injection.")
            except Exception as e:
                logger.warning(f"Could not load IP-Adapter. Falling back to text-only generation. {e}")
                
            try:
                self.pipeline.enable_xformers_memory_efficient_attention()
                logger.info("xformers enabled for faster generation.")
            except Exception:
                pass
                
            self.pipeline.to("cuda")
            logger.info(f"Initialized Diffusers pipeline for {self.model_name} on CUDA")
        except ImportError:
            logger.warning("Diffusers/Torch not installed properly. Running Image Adapter in MOCK mode.")
            self.pipeline = None

    def generate_image(self, prompt: str, output_path: str, negative_prompt: str = "", reference_image_paths: list = None):
        """
        Generates an image from a prompt and saves it.
        """
        if self.pipeline is not None:
            # Real generation
            logger.info(f"Generating image (Real)...")
            
            kwargs = {
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "width": 1280,
                "height": 720,
                "num_inference_steps": 20,
                "guidance_scale": 7.0
            }
            
            # Inject IP Adapter if reference images are provided
            if reference_image_paths:
                try:
                    from PIL import Image
                    ref_images = [Image.open(p).convert("RGB") for p in reference_image_paths if os.path.exists(p)]
                    if ref_images:
                        kwargs["ip_adapter_image"] = ref_images
                        logger.info(f"Injecting {len(ref_images)} character reference images via IP-Adapter.")
                except Exception as e:
                    logger.warning(f"Failed to load reference images: {e}")
            
            image = self.pipeline(**kwargs).images[0]
            image.save(output_path)
            logger.info(f"Saved real generated image to {output_path}")
        else:
            # Mock generation
            logger.info(f"Generating mock image for prompt: {prompt[:50]}...")
            time.sleep(1) # Simulate generation time
            
            # Create a blank dummy image
            try:
                from PIL import Image
                img = Image.new('RGB', (1024, 1024), color = (73, 109, 137))
                img.save(output_path)
            except ImportError:
                # If PIL isn't available, just create an empty file
                with open(output_path, 'wb') as f:
                    f.write(b"MOCK IMAGE DATA")
            
            logger.info(f"Saved mock image to {output_path}")
