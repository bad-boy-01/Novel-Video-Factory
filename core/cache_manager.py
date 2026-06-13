import os
import json
import hashlib
import logging

logger = logging.getLogger(__name__)

class CacheManager:
    """
    Implements the Asset Cache and Dependency Graph from Chapter 8.
    Hashes the inputs to determine if a stage needs to be run.
    """
    def __init__(self, project_dir: str):
        self.project_dir = project_dir
        self.cache_manifest = os.path.join(project_dir, 'cache_manifest.json')
        self.cache = self._load_cache()

    def _load_cache(self):
        if os.path.exists(self.cache_manifest):
            with open(self.cache_manifest, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}

    def _save_cache(self):
        with open(self.cache_manifest, 'w', encoding='utf-8') as f:
            json.dump(self.cache, f, indent=2)

    def hash_file(self, file_path: str):
        if not os.path.exists(file_path):
            return None
        hasher = hashlib.md5()
        with open(file_path, 'rb') as f:
            buf = f.read()
            hasher.update(buf)
        return hasher.hexdigest()

    def hash_directory(self, dir_path: str):
        if not os.path.exists(dir_path):
            return None
        hasher = hashlib.md5()
        for root, dirs, files in os.walk(dir_path):
            for file in sorted(files):
                file_path = os.path.join(root, file)
                hasher.update(self.hash_file(file_path).encode('utf-8'))
        return hasher.hexdigest()

    def should_run_stage(self, stage_name: str, input_paths: list) -> bool:
        from core.config_manager import ConfigManager
        if not ConfigManager().get('system.cache_enabled', True):
            return True
            
        current_hash = ""
        for path in input_paths:
            if os.path.isdir(path):
                current_hash += self.hash_directory(path) or ""
            else:
                current_hash += self.hash_file(path) or ""
                
        final_hash = hashlib.md5(current_hash.encode('utf-8')).hexdigest()
        
        cache_val = self.cache.get(stage_name)
        if cache_val is True or str(cache_val).lower() == "true" or cache_val == final_hash:
            logger.info(f"[{stage_name}] Cache hit or bypassed! Skipping execution.")
            return False
            
        return True

    def mark_stage_complete(self, stage_name: str, input_paths: list):
        current_hash = ""
        for path in input_paths:
            if os.path.isdir(path):
                current_hash += self.hash_directory(path) or ""
            else:
                current_hash += self.hash_file(path) or ""
                
        final_hash = hashlib.md5(current_hash.encode('utf-8')).hexdigest()
        self.cache[stage_name] = final_hash
        self._save_cache()
