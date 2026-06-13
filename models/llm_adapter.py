import json
import requests
import logging

logger = logging.getLogger(__name__)

class LocalLLMAdapter:
    """
    Adapter for communicating with a local Ollama instance.
    """
    def __init__(self, model_name: str, host: str = "http://localhost:11434"):
        self.model_name = model_name
        self.host = host
        self.api_url = f"{self.host}/api/generate"

    def check_health(self) -> bool:
        """Check if the Ollama service is running and the model is available."""
        try:
            response = requests.get(f"{self.host}/api/tags")
            if response.status_code == 200:
                models = [m['name'] for m in response.json().get('models', [])]
                if self.model_name in models or f"{self.model_name}:latest" in models:
                    return True
                logger.warning(f"Model {self.model_name} not found in local Ollama.")
                return False
            return False
        except requests.ConnectionError:
            logger.error(f"Cannot connect to Ollama host at {self.host}")
            return False

    def generate(self, prompt: str, system_prompt: str = None, temperature: float = 0.7) -> str:
        """
        Generate text using the local LLM.
        """
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature
            }
        }
        if system_prompt:
            payload["system"] = system_prompt

        try:
            response = requests.post(self.api_url, json=payload, timeout=120)
            response.raise_for_status()
            data = response.json()
            return data.get("response", "")
        except Exception as e:
            logger.warning(f"Error generating text with model {self.model_name}: {e}. Using mock response.")
            if "extract all characters" in (system_prompt or ""):
                return '[{"canonical_name": "Xu Changshou", "aliases": ["Little Xu"], "visual_dna": {"clothing": "blue patched robe"}}, {"canonical_name": "Elder Lin", "aliases": ["Master Lin"], "visual_dna": {"hair": "long white beard", "expression": "stern"}}]'
            if "extract all locations" in (system_prompt or ""):
                return '[{"canonical_name": "Cloud Peak", "description": "A high mountain shrouded in mist."}]'
            if "extract all world concepts" in (system_prompt or ""):
                return '[{"concept_type": "sect", "name": "Heavenly Sword Sect", "description": "A powerful sect"}]'
            if "YouTube SEO expert" in (system_prompt or ""):
                return '{"title": "He Woke Up In Another World! [EP 1]", "description": "Xu Changshou wakes up with a mysterious system...", "tags": ["manhwa recap", "anime", "system"]}'
            if "cinematic scenes" in (system_prompt or ""):
                return '[{"scene_id": "SC001", "description": "A young man wearing a blue robe wakes up, looking at his hands in disbelief.", "characters_present": ["Xu Changshou"], "camera_angle": "Medium shot", "lighting": "Bright blinding light"}, {"scene_id": "SC002", "description": "An old man with a long white beard walks into the room.", "characters_present": ["Elder Lin"], "camera_angle": "Wide shot", "lighting": "Soft morning light"}]'
            return f"[MOCK TRANSLATION OF]: {prompt}"

class OnlineLLMAdapter:
    """
    Adapter for communicating with cloud LLM providers like Groq or OpenAI via REST API.
    """
    def __init__(self, provider: str = "groq", model_name: str = "llama-3.3-70b-versatile", api_key: str = None):
        import os
        self.provider = provider.lower()
        self.model_name = model_name
        self.api_key = api_key or os.environ.get(f"{self.provider.upper()}_API_KEY")
        
        # Kaggle Secrets Fallback
        if not self.api_key:
            try:
                from kaggle_secrets import UserSecretsClient
                user_secrets = UserSecretsClient()
                self.api_key = user_secrets.get_secret(f"{self.provider.upper()}_API_KEY")
            except ImportError:
                pass
            except Exception as e:
                logger.warning(f"Failed to read from Kaggle Secrets: {e}")
        
        if self.provider == "groq":
            self.api_url = "https://api.groq.com/openai/v1/chat/completions"
        elif self.provider == "openai":
            self.api_url = "https://api.openai.com/v1/chat/completions"
        elif self.provider == "gemini":
            self.api_url = "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
        else:
            raise ValueError(f"Unsupported provider: {provider}")

    def check_health(self) -> bool:
        if not self.api_key:
            logger.warning(f"No API key found for {self.provider}. Please set {self.provider.upper()}_API_KEY in environment.")
            return False
        return True

    def generate(self, prompt: str, system_prompt: str = None, temperature: float = 0.7) -> str:
        if not self.api_key:
            logger.error(f"Missing API key for {self.provider}")
            raise ValueError(f"Missing API key for {self.provider}")
            
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temperature
        }
        
        import time
        max_retries = 5
        base_delay = 3

        for attempt in range(max_retries):
            try:
                response = requests.post(self.api_url, headers=headers, json=payload, timeout=120)
                
                # Handle rate limits
                if response.status_code == 429:
                    retry_after = response.headers.get("Retry-After")
                    sleep_time = float(retry_after) if retry_after else (base_delay * (2 ** attempt))
                    logger.warning(f"Rate limit hit for {self.provider}. Retrying in {sleep_time:.2f} seconds... (Attempt {attempt+1}/{max_retries})")
                    time.sleep(sleep_time)
                    continue
                    
                response.raise_for_status()
                data = response.json()
                return data["choices"][0]["message"]["content"]
                
            except requests.exceptions.RequestException as e:
                # If it's a 429 caught by raise_for_status, we handle it above.
                # Other errors we break or raise.
                if hasattr(e, 'response') and e.response is not None and e.response.status_code == 429:
                    sleep_time = base_delay * (2 ** attempt)
                    logger.warning(f"Rate limit hit for {self.provider}. Retrying in {sleep_time} seconds... (Attempt {attempt+1}/{max_retries})")
                    time.sleep(sleep_time)
                    continue
                else:
                    logger.error(f"Error generating text with {self.provider} model {self.model_name}: {e}")
                    if hasattr(e, 'response') and e.response is not None:
                        logger.error(f"API Response: {e.response.text}")
                    raise
                    
        raise Exception(f"Max retries exceeded for {self.provider} API.")
