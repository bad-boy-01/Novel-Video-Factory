import os
import sys
import subprocess
import time
import argparse

def main():
    parser = argparse.ArgumentParser(description="Novel Video Factory Universal Orchestrator")
    parser.add_argument("project", nargs="?", default="novel", help="The name of the project folder (e.g. dummy_novel, chapter_1)")
    parser.add_argument("--resume", action="store_true", help="Resume a crashed generation by skipping cache/directory wipes")
    args = parser.parse_args()
    project_name = args.project
    
    print(f"=== Novel Video Factory Orchestrator ===")
    print(f"Project Target: {project_name}")
    print("="*40)
    
    # 1. Directory Initialization
    print("\n[1/5] Initializing Project Structures...")
    project_input_dir = f"projects/{project_name}/input"
    project_output_dir = f"projects/{project_name}/output"
    os.makedirs(project_input_dir, exist_ok=True)
    os.makedirs(project_output_dir, exist_ok=True)
    print(f"Verified directories: {project_input_dir}, {project_output_dir}")

    # 2. System Dependency Management (Only runs if running on Linux/Kaggle and not already installed)
    print("\n[2/5] Checking System Dependencies...")
    import shutil
    if os.path.exists("/usr/bin/apt-get"):
        if not shutil.which("ollama"):
            print("Installing Ollama and dependencies...")
            try:
                subprocess.run("apt-get update && apt-get install -y espeak-ng imagemagick zstd", shell=True, check=True)
                subprocess.run("curl -fsSL https://ollama.com/install.sh | sh", shell=True, check=True)
                print("System dependencies and Ollama installed.")
            except subprocess.CalledProcessError as e:
                print(f"Warning: Failed to install system dependencies (are you root?): {e}")
        else:
            print("Ollama is already installed. Skipping dependency installation.")
    else:
        print("Skipping apt-get (not on a Debian/Ubuntu system).")

    # 3. Local Server Boot (Only if using local model)
    print("\n[3/5] Checking LLM Provider...")
    import yaml
    config_path = "config/default.yaml"
    use_local_llm = True
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            config_data = yaml.safe_load(f)
            provider = config_data.get("models", {}).get("translation", {}).get("primary", {}).get("provider", "local")
            if provider != "local" and provider != "ollama":
                use_local_llm = False
                
    if use_local_llm:
        print("Starting Ollama Backend for local generation...")
        try:
            # Check if Ollama is already running
            subprocess.run(["curl", "-s", "-f", "http://localhost:11434/"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            print("Ollama server is already running.")
        except subprocess.CalledProcessError:
            print("Booting detached Ollama server...")
            subprocess.Popen(
                ["ollama", "serve"], 
                stdout=subprocess.DEVNULL, 
                stderr=subprocess.DEVNULL,
                start_new_session=True
            )
            time.sleep(5) # Wait for startup
            
        print("Ensuring qwen2.5:7b is pulled...")
        subprocess.run("ollama pull qwen2.5:7b", shell=True)
    else:
        print(f"Online LLM provider ({provider}) detected. Skipping Ollama boot.")

    import shutil

    # 4. Storage & Cache Reset
    print("\n[4/5] Preparing Caches & Output Directories...")
    cache_file = f"projects/{project_name}/cache_manifest.json"
    db_file = "core/memory/characters.db"
    output_dir = f"projects/{project_name}/output"
    
    if args.resume:
        print("RESUME MODE ACTIVE: Preserving existing outputs and cache manifest.")
        os.makedirs(output_dir, exist_ok=True)
    else:
        print("Clean Start: Resetting caches and outputs...")
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
            print(f"Cleared output directory: {output_dir}")
        os.makedirs(output_dir, exist_ok=True)
        
        if os.path.exists(cache_file):
            os.remove(cache_file)
            print(f"Cleared cache: {cache_file}")
            
        # Force full English translation bypass
        chapter_txt = os.path.join(project_input_dir, "chapter1.txt")
        if os.path.exists(chapter_txt):
            shutil.copy(chapter_txt, os.path.join(output_dir, "translated_chapter1.txt"))
            with open(cache_file, "w") as f: 
                f.write('{"translate": true}')
            print("Full English script mapped successfully! Bypassing the translation summarizer.")
        else:
            print(f"WARNING: No chapter1.txt found in {project_input_dir}. Please ensure your story file is uploaded.")

    # 5. Pipeline Execution
    print("\n[5/5] Launching Main Pipeline...")
    print("="*40)
    process = subprocess.Popen(
        f"python main.py {project_name} --stage all", 
        shell=True, 
        stdout=subprocess.PIPE, 
        stderr=subprocess.STDOUT, 
        text=True,
        bufsize=1
    )

    # Stream real-time output
    for line in iter(process.stdout.readline, ''):
        sys.stdout.write(line)
        sys.stdout.flush()

    process.wait()
    print("="*40)
    if process.returncode == 0:
        print("\n✅ Processing Loop Successfully Concluded.")
    else:
        print(f"\n❌ Pipeline terminated with errors (Exit code {process.returncode}).")

if __name__ == "__main__":
    main()
