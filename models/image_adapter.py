import logging
import os
import time
import gc
import torch

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
            from diffusers import AutoPipelineForText2Image
            
            self.pipeline = AutoPipelineForText2Image.from_pretrained(
                self.model_name, torch_dtype=torch.float16
            )
            # We will load IP-Adapter dynamically when needed
            self.ip_adapter_loaded = False
            try:
                self.pipeline.enable_xformers_memory_efficient_attention()
                logger.info("xformers enabled for faster generation.")
            except Exception:
                pass
                
            self.pipeline.enable_model_cpu_offload()
            logger.info(f"Initialized Diffusers pipeline for {self.model_name} with CPU offloading")
        except ImportError:
            logger.warning("Diffusers/Torch not installed properly. Running Image Adapter in MOCK mode.")
            self.pipeline = None

    def _encode_prompt(self, prompt, negative_prompt):
        """
        Encodes the prompt into embeddings, supporting long prompts (> 77 tokens)
        by chunking them and concatenating the embeddings.
        """
        # SDXL has two text encoders
        # text_encoder (CLIP-L) and text_encoder_2 (OpenCLIP-G)
        device = self.pipeline.device
        
        def get_embeds(p, encoder, tokenizer):
            input_ids = tokenizer(
                p, padding="max_length", max_length=tokenizer.model_max_length, truncation=False, return_tensors="pt"
            ).input_ids.to(device)
            
            # If prompt is longer than 77 tokens, we chunk it
            if input_ids.shape[1] > tokenizer.model_max_length:
                # Remove BOS/EOS and chunk
                # tokenizer.model_max_length is usually 77
                max_length = tokenizer.model_max_length
                # Simple chunking logic:
                chunks = []
                for i in range(0, input_ids.shape[1], max_length - 2):
                    chunk = input_ids[:, i:i + max_length - 2]
                    # Add BOS and EOS
                    bos = torch.tensor([[tokenizer.bos_token_id]], device=device)
                    eos = torch.tensor([[tokenizer.eos_token_id]], device=device)
                    chunk = torch.cat([bos, chunk, eos], dim=1)
                    # Pad if needed
                    if chunk.shape[1] < max_length:
                        padding = torch.full((1, max_length - chunk.shape[1]), tokenizer.pad_token_id, device=device)
                        chunk = torch.cat([chunk, padding], dim=1)
                    chunks.append(chunk[:, :max_length])
                
                # Encode chunks
                all_embeds = []
                all_pooled = []
                for chunk in chunks:
                    output = encoder(chunk, output_hidden_states=True)
                    all_embeds.append(output.hidden_states[-2]) # Penultimate layer
                    if hasattr(output, 'text_embeds'):
                        all_pooled.append(output.text_embeds)
                
                return torch.cat(all_embeds, dim=1), (torch.mean(torch.stack(all_pooled), dim=0) if all_pooled else None)
            else:
                # Normal short prompt
                output = encoder(input_ids, output_hidden_states=True)
                return output.hidden_states[-2], (output.text_embeds if hasattr(output, 'text_embeds') else None)

        # Encode Positive Prompt
        prompt_embeds, pooled_prompt_embeds = get_embeds(prompt, self.pipeline.text_encoder, self.pipeline.tokenizer)
        prompt_embeds_2, pooled_prompt_embeds_2 = get_embeds(prompt, self.pipeline.text_encoder_2, self.pipeline.tokenizer_2)
        
        # SDXL concatenates the embeddings from the two encoders
        prompt_embeds = torch.cat([prompt_embeds, prompt_embeds_2], dim=-1)
        
        # Encode Negative Prompt
        neg_embeds, neg_pooled = get_embeds(negative_prompt, self.pipeline.text_encoder, self.pipeline.tokenizer)
        neg_embeds_2, neg_pooled_2 = get_embeds(negative_prompt, self.pipeline.text_encoder_2, self.pipeline.tokenizer_2)
        neg_embeds = torch.cat([neg_embeds, neg_embeds_2], dim=-1)
        
        # Ensure negative prompt length matches positive
        if prompt_embeds.shape[1] > neg_embeds.shape[1]:
            padding = torch.zeros((1, prompt_embeds.shape[1] - neg_embeds.shape[1], neg_embeds.shape[-1]), device=device, dtype=neg_embeds.dtype)
            neg_embeds = torch.cat([neg_embeds, padding], dim=1)
        elif neg_embeds.shape[1] > prompt_embeds.shape[1]:
            neg_embeds = neg_embeds[:, :prompt_embeds.shape[1], :]

        return prompt_embeds, pooled_prompt_embeds_2, neg_embeds, neg_pooled_2

    def generate_image(self, prompt: str, output_path: str, negative_prompt: str = "", reference_image_paths: list = None, generation_params: dict = None):
        """
        Generates an image from a prompt and saves it. Uses generation_params for deterministic output.
        """
        if self.pipeline is not None:
            # Real generation
            logger.info(f"Generating image (Real)...")
            
            # Layer 8.3: Long Prompt Support
            prompt_embeds, pooled_prompt_embeds, neg_embeds, neg_pooled = self._encode_prompt(prompt, negative_prompt)
            
            # V3 Upgrade: Persistent Seeds and Configurable Params
            if generation_params is None:
                generation_params = {}
                
            seed = generation_params.get("seed", 42)
            steps = generation_params.get("steps", 30)
            cfg = generation_params.get("cfg", 5.0)
            width = generation_params.get("width", 1280)
            height = generation_params.get("height", 720)
            
            generator = torch.Generator(self.pipeline.device).manual_seed(seed)
            
            kwargs = {
                "prompt_embeds": prompt_embeds,
                "pooled_prompt_embeds": pooled_prompt_embeds,
                "negative_prompt_embeds": neg_embeds,
                "negative_pooled_prompt_embeds": neg_pooled,
                "width": width,
                "height": height,
                "num_inference_steps": steps,
                "guidance_scale": cfg,
                "generator": generator
            }
            
            # Inject IP Adapter if reference images are provided
            if reference_image_paths:
                try:
                    from PIL import Image
                    ref_images = [Image.open(p).convert("RGB") for p in reference_image_paths if os.path.exists(p)]
                    if ref_images:
                        if not self.ip_adapter_loaded:
                            logger.info("Loading IP-Adapter into pipeline...")
                            self.pipeline.load_ip_adapter("h94/IP-Adapter", subfolder="sdxl_models", weight_name="ip-adapter_sdxl.bin")
                            self.ip_adapter_loaded = True
                        self.pipeline.set_ip_adapter_scale(0.35)
                        kwargs["ip_adapter_image"] = ref_images
                        logger.info(f"Injecting {len(ref_images)} character reference images via IP-Adapter.")
                except Exception as e:
                    logger.warning(f"Failed to load reference images: {e}")
            else:
                if self.ip_adapter_loaded:
                    logger.info("Unloading IP-Adapter for pure text generation...")
                    self.pipeline.unload_ip_adapter()
                    self.ip_adapter_loaded = False
            
            image = self.pipeline(**kwargs).images[0]
            image.save(output_path)
            logger.info(f"Saved real generated image to {output_path}")
            
            # V3 Upgrade: Kaggle Optimization (Step 16)
            # Aggressive memory cleanup to prevent OOM on long runs
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        else:
            # Mock generation
            logger.info(f"Generating mock image for prompt: {prompt}")
            time.sleep(0.1) 
            
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
