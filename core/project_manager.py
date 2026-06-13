import os
import json
import logging
import atexit

logger = logging.getLogger(__name__)

class ProjectManager:
    """Manages the workspace, I/O, and checkpoints for a specific novel project."""
    
    def __init__(self, base_dir: str, project_name: str):
        self.base_dir = base_dir
        self.project_name = project_name
        self.project_dir = os.path.join(self.base_dir, 'projects', self.project_name)
        
        # Project Locking (Phase 0)
        self.lock_file = os.path.join(self.project_dir, '.lock')
        if os.path.exists(self.lock_file):
            raise RuntimeError(f"Project '{self.project_name}' is locked. Another process may be running. Delete the .lock file manually if this is a mistake.")
        
        os.makedirs(self.project_dir, exist_ok=True)
        with open(self.lock_file, 'w') as f:
            f.write("locked")
            
        # Ensure lock is cleaned up if script crashes
        atexit.register(self.unlock)
            
        # Define standard directories
        self.dirs = {
            'input': os.path.join(self.project_dir, 'input'),
            'memory': os.path.join(self.project_dir, 'memory'),
            'output': os.path.join(self.project_dir, 'output'),
            'checkpoints': os.path.join(self.project_dir, 'checkpoints')
        }
        self._ensure_directories()

    def _ensure_directories(self):
        for dir_path in self.dirs.values():
            os.makedirs(dir_path, exist_ok=True)

    def unlock(self):
        """Release the project lock."""
        if os.path.exists(self.lock_file):
            os.remove(self.lock_file)

    def get_versioned_dir(self, base_category: str, version: int = 1) -> str:
        """
        Returns a versioned directory path, e.g., projects/<name>/output/storyboard/v1
        base_category could be 'storyboard', 'prompts', etc.
        """
        path = os.path.join(self.dirs['output'], base_category, f"v{version}")
        os.makedirs(path, exist_ok=True)
        return path

    def get_input_files(self):
        """Return a list of .txt files in the input directory."""
        input_dir = self.dirs['input']
        files = [f for f in os.listdir(input_dir) if f.endswith('.txt')]
        return [os.path.join(input_dir, f) for f in files]

    def read_input(self, filename: str) -> str:
        with open(filename, 'r', encoding='utf-8') as f:
            return f.read()

    def save_output(self, filename: str, content: str):
        path = os.path.join(self.dirs['output'], filename)
        with open(path, 'w', encoding='utf-8') as f:
            f.write(content)
        logger.info(f"Saved output to {path}")

    def save_checkpoint(self, stage: str, data: dict):
        path = os.path.join(self.dirs['checkpoints'], f"{stage}.json")
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        logger.info(f"Checkpoint saved for stage: {stage}")

    def load_checkpoint(self, stage: str) -> dict:
        path = os.path.join(self.dirs['checkpoints'], f"{stage}.json")
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return None
