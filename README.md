# Novel Video Factory

A fully automated AI-powered production pipeline designed to transform long-form novels into professional YouTube videos.

## Quick Start Guide

### 1. Creating a Project & Importing a Script
You can now initialize a project and import a script in a single command:
```bash
python start_pipeline.py my_epic_novel --input "path/to/your/script.txt"
```
This will automatically:
- Create the project folder structure in `projects/my_epic_novel/`.
- Import your script into the `input/` directory.
- Start the entire pipeline (Translation -> Memory -> Visual -> Generation -> Video).

### 2. Running Locally (Windows/Mac/Linux)
You will need a powerful GPU to run the models locally.
1. Install [Ollama](https://ollama.com/) and start the application.
2. The orchestrator will automatically pull the required model (`qwen2.5:7b`) on first run.
3. Ensure you have the required Python dependencies: `pip install -r requirements.txt`.
4. Run the orchestrator:
   ```bash
   python start_pipeline.py my_epic_novel
   ```

### 3. Character Consistency & Reference Sheets
The pipeline now features a "Character Sheets" stage that extracts detailed visual DNA (age, clothing, hair, etc.) and generates reference portraits for every character. These are stored in `projects/<name>/memory/character_sheets/`.
- The system uses these reference images via **IP-Adapter** to ensure character consistency across every scene in your video.
- To regenerate sheets or fix character DNA, you can edit the database directly or use the `--stage character_sheets` flag.

### 3. Running on Kaggle (Free Cloud GPUs)
If you do not have a powerful local GPU, you can run this entirely for free on Kaggle!
1. Go to Kaggle.com and create a new Notebook.
2. In the right sidebar, change the **Accelerator** to **GPU P100** or **GPU T4x2**.
3. Upload the `Kaggle_Run.ipynb` file from this repository to your Kaggle notebook.
4. Upload your project folder (e.g. `projects/my_epic_novel/input/chapter1.txt`) to the Kaggle workspace.
5. Click **"Run All"** in the notebook.
   - *Note: The notebook will automatically execute `kaggle_setup.sh`, which installs Ollama, downloads the LLM, and installs all required Python packages.*

### 4. Viewing the Output
Once the pipeline finishes, navigate to `projects/my_epic_novel/output/`.
You will find:
- `translated_chapter1.txt`: The revised translation.
- `prompts.json`: The cinematic camera plans.
- `images/`: The generated Stable Diffusion / FLUX images.
- `audio/`: The generated voiceover files.
- `final_video.mp4`: The fully assembled YouTube video with crossfades and subtitles!
- `seo_metadata.json`: Auto-generated YouTube titles and tags.
- `thumbnail.png`: The auto-selected YouTube thumbnail.

## The CLI Pipeline Stages
If you want to run the factory step-by-step to manually review the outputs, use the `--stage` flag:

1. `python main.py my_epic_novel --stage translate`
2. `python main.py my_epic_novel --stage memory`
3. `python main.py my_epic_novel --stage visual`
4. `python main.py my_epic_novel --stage generation`
5. `python main.py my_epic_novel --stage audio`
6. `python main.py my_epic_novel --stage video`
7. `python main.py my_epic_novel --stage publishing`
