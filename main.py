import argparse
import sys
import logging
import os
import re
import json
import shutil
import uuid
import subprocess
import datetime
import hashlib

from core.project_manager import ProjectManager
from core.translation.pipeline import TranslationPipeline
from core.qa.validator import ZeroInformationLossValidator
from core.memory.database import MemoryEngine
from core.memory.extractor import MemoryExtractor
from models.llm_adapter import LocalLLMAdapter
from core.config_manager import ConfigManager

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('NovelVideoFactory')

def main():
    parser = argparse.ArgumentParser(description="Novel Video Factory - Automated AI Novel to Video Pipeline")
    parser.add_argument('project', help="Name of the project/novel")
    parser.add_argument('--stage', type=str, default='all', 
                        choices=['all', 'translate', 'memory', 'character_sheets', 'visual', 'generation', 'audio', 'video', 'publishing', 'export'],
                        help='Which stage of the pipeline to run')
    parser.add_argument('--config', default='config/default.yaml', help="Path to configuration file")
    parser.add_argument('--input', help="Path to a novel script (.txt) to import into the project")

    args = parser.parse_args()

    logger.info(f"Starting Novel Video Factory for project: {args.project}")
    
    config_manager = ConfigManager(args.config)
    base_dir = os.path.dirname(os.path.abspath(__file__))
    pm = ProjectManager(base_dir, args.project)

    # Handle script import
    if args.input:
        if not os.path.exists(args.input):
            logger.error(f"Input file not found: {args.input}")
            sys.exit(1)
        dest = os.path.join(pm.dirs['input'], os.path.basename(args.input))
        shutil.copy(args.input, dest)
        logger.info(f"Imported script {args.input} to {dest}")
    
    # Initialize Adapters with Config values
    llm_provider = config_manager.get('models.translation.primary.provider', 'local')
    llm_model = config_manager.get('models.translation.primary.model', 'qwen2.5:7b')
    
    if llm_provider in ['local', 'ollama']:
        llm_adapter = LocalLLMAdapter(model_name=llm_model)
    else:
        from models.llm_adapter import OnlineLLMAdapter
        llm_adapter = OnlineLLMAdapter(provider=llm_provider, model_name=llm_model)
    
    if args.stage in ['all', 'translate']:
        logger.info("Running Translation Engine...")
        from core.cache_manager import CacheManager
        from core.qa.pipeline import QAPipeline
        
        cache = CacheManager(pm.project_dir)
        qa = QAPipeline(llm_adapter)
        
        input_files = pm.get_input_files()
        if not input_files:
            logger.warning("No input files found in project input directory.")
        
        if cache.should_run_stage('translate', input_files):
            pipeline = TranslationPipeline(config={}, llm_adapter=llm_adapter)
            validator = ZeroInformationLossValidator()
            
            for file in input_files:
                logger.info(f"Processing file: {file}")
                text = pm.read_input(file)
                
                # This handles chunking and translation
                translated_text = pipeline.process_chapter(text)
                
                # Run QA
                qa.verify_translation(text, translated_text)
                
                # Save Output
                filename = os.path.basename(file)
                pm.save_output(f"translated_{filename}", translated_text)
                
            cache.mark_stage_complete('translate', input_files)

    if args.stage in ['all', 'memory']:
        logger.info("Running Memory Engine...")
        
        extractor = MemoryExtractor(llm_adapter)
        memory_db = MemoryEngine(pm.project_dir)
        text_chunker = TranslationPipeline(config={}, llm_adapter=None)
        
        translated_files = [f for f in os.listdir(pm.dirs['output']) if f.startswith('translated_')]
        for file in translated_files:
            text = pm.read_input(os.path.join(pm.dirs['output'], file))
            chunks = text_chunker.chunk_text(text, max_words=500)
            
            for chunk_idx, chunk_data in enumerate(chunks):
                chunk_marker = os.path.join(pm.project_dir, 'memory', f"chunk_{file}_{chunk_idx}.done")
                if os.path.exists(chunk_marker):
                    logger.info(f"Skipping Memory Extraction for Chunk {chunk_idx + 1}/{len(chunks)} (Already Extracted)")
                    continue
                    
                chunk_text = " ".join([s["text"] for s in chunk_data["sentences"]])
                logger.info(f"Extracting Memory from Chunk {chunk_idx + 1}/{len(chunks)}")
                
                # Batch extract all memory to save LLM API rate limits
                existing_chars = memory_db.get_all_characters()
                existing_rels = memory_db.get_all_relationships()
                memory_data = extractor.extract_all(chunk_text, existing_characters=existing_chars, existing_relationships=existing_rels)
                
                # Characters
                for c in memory_data.get("characters", []):
                    c_id = str(uuid.uuid4())[:8]
                    if memory_db.add_character(c_id, c.get('canonical_name', 'Unknown'), c.get('visual_dna', {})):
                        logger.info(f"Saved character to DB: {c.get('canonical_name')}")
                    
                # Locations
                for loc in memory_data.get("locations", []):
                    if memory_db.add_location(loc.get('canonical_name', 'Unknown'), loc.get('description', '')):
                        logger.info(f"Saved location to DB: {loc.get('canonical_name')}")
                    
                # World Concepts
                for concept in memory_data.get("world_concepts", []):
                    if memory_db.add_world_concept(concept.get('concept_type', 'misc'), concept.get('name', 'Unknown'), concept.get('description', '')):
                        logger.info(f"Saved world concept to DB: {concept.get('name')}")
                
                # V3 Upgrade: Relationships
                for rel in memory_data.get("relationships", []):
                    if memory_db.add_relationship(rel.get('char1'), rel.get('char2'), rel.get('type'), rel.get('staging', '')):
                        logger.info(f"Saved relationship: {rel.get('char1')} <-> {rel.get('char2')} ({rel.get('type')})")
                    
                # Mark chunk as complete
                with open(chunk_marker, 'w') as f:
                    f.write("done")
            
            # V3 Upgrade: Background Cache (Step 14)
            logger.info("Generating Background Cache for all locations...")
            from models.image_adapter import LocalImageAdapter
            image_adapter = LocalImageAdapter()
            loc_dir = os.path.join(pm.project_dir, 'memory', 'locations')
            os.makedirs(loc_dir, exist_ok=True)
            
            locations = memory_db.get_all_locations()
            for loc in locations:
                name = loc['canonical_name']
                desc = loc['description']
                safe_name = re.sub(r'[\\/*?:"<>|]', "", name).strip().replace(" ", "_")
                bg_path = os.path.join(loc_dir, f"{safe_name}.png")
                
                if not loc.get('background_path') or not os.path.exists(bg_path):
                    logger.info(f"Generating Base Background for: {name}")
                    
                    # V3 Upgrade: Unique Seed for Location
                    loc_seed = int(hashlib.sha256(name.encode('utf-8')).hexdigest(), 16) % 2147483647
                    
                    manhwa_core = "manhwa, webtoon, korean style, thick outlines, vibrant colors"
                    quality_tags = "masterpiece, high score, great score, absurdres"
                    
                    # Prioritize location description at the FRONT
                    bg_prompt = f"{desc}, {manhwa_core}, landscape, detailed background, cinematic lighting, year 2024, {quality_tags}, rating_safe"
                    
                    bg_params = {
                        "seed": loc_seed,
                        "steps": 30,
                        "cfg": 6.5, # High adherence for master backgrounds
                        "width": 1280,
                        "height": 720
                    }
                    
                    image_adapter.generate_image(bg_prompt, bg_path, generation_params=bg_params)
                    memory_db.update_location_background(name, bg_path)
                else:
                    logger.info(f"Background for {name} already exists.")
                    
                # World Style extraction (only needed once per chapter/file)
                if chunk_idx == 0:
                    style_tags = extractor.extract_world_style(chunk_text)
                    style_file = os.path.join(pm.project_dir, 'memory', 'world_style.txt')
                    os.makedirs(os.path.dirname(style_file), exist_ok=True)
                    with open(style_file, 'w', encoding='utf-8') as f:
                        f.write(style_tags)
                    logger.info(f"Extracted World Atmosphere Setting: {style_tags}")

    if args.stage in ['all', 'character_sheets']:
        logger.info("Running Character Sheets Generation...")
        memory_db = MemoryEngine(pm.project_dir)
        from models.image_adapter import LocalImageAdapter
        image_adapter = LocalImageAdapter()
        
        chars_dir = os.path.join(pm.project_dir, 'memory', 'character_sheets')
        os.makedirs(chars_dir, exist_ok=True)
        
        style_modifier = config_manager.get('prompts.style_modifier', 'Cinematic, high quality')
        
        session = memory_db.Session()
        try:
            from core.memory.database import Character
            characters = session.query(Character).all()
            for char in characters:
                # Sanitize name for filename to make it human readable
                safe_name = re.sub(r'[\\/*?:"<>|]', "", char.canonical_name).strip().replace(" ", "_")
                img_path = os.path.join(chars_dir, f"{safe_name}.png")
                
                if not os.path.exists(img_path):
                    logger.info(f"Generating Character Reference Sheet for {char.canonical_name}...")
                    dna_tags = []
                    for k, v in char.visual_dna.items():
                        if isinstance(v, str) and v.lower() not in ['not specified', 'unknown', 'none']:
                            dna_tags.append(v)
                    dna_str = ", ".join(dna_tags)
                    
                    # Deduce gender for the model
                    dna_lower = dna_str.lower()
                    name_lower = char.canonical_name.lower()
                    if any(w in dna_lower or w in name_lower for w in ["girl", "woman", "female", "sister", "mother", "wife", "chunni", "xiue", "mei", "her ", "she ", "madam", "dress", "aunt", "lady"]):
                        gender_tag = "1girl"
                    else:
                        gender_tag = "1boy"
                    
                    # Add Age and Quality tags for Animagine XL 3.1
                    age_tags = ""
                    if "years old" in dna_lower or "age" in dna_lower:
                        # Extract age if possible or just use the DNA string
                        pass 
                    
                    # Moving style tags to the front to avoid 77-token truncation
                    quality_tags = "masterpiece, high score, great score, absurdres"
                    manhwa_core = "manhwa, webtoon, korean style, thick outlines, vibrant colors"
                    year_tag = "year 2024"
                    
                    # Generate a unique but deterministic seed for this character
                    char_seed = int(hashlib.sha256(char.canonical_name.encode('utf-8')).hexdigest(), 16) % 2147483647
                    
                    # Prioritize unique features at the FRONT of the prompt
                    prompt = f"{gender_tag}, solo, {dna_str}, {manhwa_core}, traditional eastern clothing, cinematic portrait, {year_tag}, {quality_tags}, rating_safe"
                    negative = "lowres, bad anatomy, bad hands, text, error, missing finger, extra digits, fewer digits, cropped, worst quality, low quality, low score, bad score, average score, signature, watermark, username, blurry"
                    
                    char_params = {
                        "seed": char_seed,
                        "steps": 30,
                        "cfg": 6.0,
                        "width": 1024,
                        "height": 1024
                    }
                    image_adapter.generate_image(prompt, img_path, negative_prompt=negative, generation_params=char_params)
                else:
                    logger.info(f"Reference Sheet already exists for {char.canonical_name}, skipping.")
        finally:
            session.close()

    if args.stage in ['all', 'visual']:
        logger.info("Running Visual Planning (Cinematic V4)...")
        from core.visual.planner import ScenePlanner, ShotDirector
        from core.visual.prompter import PromptGenerator
        from core.visual.clip_builder import ClipBuilder
        
        memory_db = MemoryEngine(pm.project_dir)
        showrunner = ScenePlanner(llm_adapter)
        director = ShotDirector(llm_adapter)
        prompter = PromptGenerator(memory_db)
        clip_builder = ClipBuilder(target_duration=600.0) # 10-minute clips
        
        # Read translated output
        translated_files = [f for f in os.listdir(pm.dirs['output']) if f.startswith('translated_')]
        all_cinematic_shots = []
        
        # We instantiate a dummy pipeline just to use its chunk_text utility
        text_chunker = TranslationPipeline(config={}, llm_adapter=None)
        
        for file in translated_files:
            text = pm.read_input(os.path.join(pm.dirs['output'], file))
            chunks = text_chunker.chunk_text(text, max_words=500)
            
            file_id = os.path.basename(file).replace('translated_', '').split('.')[0]
            
            current_sentence_index = 0
            for chunk_idx, chunk_data in enumerate(chunks):
                visual_marker = os.path.join(pm.dirs['output'], f"v4_visual_{file_id}_{chunk_idx}.json")
                
                if os.path.exists(visual_marker):
                    logger.info(f"Loading cached Cinematic Plan for Chunk {chunk_idx + 1}")
                    with open(visual_marker, 'r', encoding='utf-8') as f:
                        chunk_shots = json.load(f)
                    all_cinematic_shots.extend(chunk_shots)
                    current_sentence_index += len(chunk_data.get("sentences", []))
                    continue

                chunk_text = " ".join([s["text"] for s in chunk_data["sentences"]])
                logger.info(f"[Showrunner] Extracting Narrative Scenes from Chunk {chunk_idx + 1}...")
                
                narrative_scenes = showrunner.extract_narrative_scenes(chunk_text, start_index=current_sentence_index)
                
                chunk_shots = []
                for scene in narrative_scenes:
                    # Extract text belonging to this scene
                    scene_sentences = scene.get('source_sentences', [])
                    chunk_relative_sentences = [idx for idx in scene_sentences]
                    
                    scene_text_list = []
                    for s_idx in chunk_relative_sentences:
                        if 0 <= s_idx < len(chunk_data["sentences"]):
                            scene_text_list.append(chunk_data["sentences"][s_idx]["text"])
                    
                    scene_text = " ".join(scene_text_list)
                    
                    logger.info(f"[Director] Breaking Scene {scene['scene_id']} into Shots (Complexity: {scene['complexity']})...")
                    raw_shots = director.direct_shots(scene, scene_text)
                    
                    for shot in raw_shots:
                        # Enrich shot with prompt and global unique ID
                        shot['shot_id'] = f"{file_id}_C{chunk_idx}_{shot['shot_id']}"
                        enriched_shot = prompter.generate_prompt_for_shot(shot, scene)
                        chunk_shots.append(enriched_shot)
                        all_cinematic_shots.append(enriched_shot)

                current_sentence_index += len(chunk_data["sentences"])
                with open(visual_marker, 'w', encoding='utf-8') as f:
                    json.dump(chunk_shots, f, indent=2)
                
        # Build Clips (Rendering Playlist)
        final_clips = clip_builder.build_clips(all_cinematic_shots)
        pm.save_output("clips.json", json.dumps(final_clips, indent=2))
        pm.save_output("prompts.json", json.dumps(all_cinematic_shots, indent=2)) # Legacy fallback
        logger.info("V4 Cinematic Storyboard successfully generated and saved to clips.json")

    if args.stage in ['all', 'generation']:
        logger.info("Running Image Generation (Cinematic V4)...")
        from models.image_adapter import LocalImageAdapter
        
        image_adapter = LocalImageAdapter()
        clips_path = os.path.join(pm.dirs['output'], 'clips.json')
        images_dir = os.path.join(pm.dirs['output'], 'images')
        os.makedirs(images_dir, exist_ok=True)
        
        if os.path.exists(clips_path):
            with open(clips_path, 'r', encoding='utf-8') as f:
                clips_data = json.load(f)
                
            for clip in clips_data:
                for p in clip['shots']:
                    shot_id = p.get('shot_id')
                    prompt = p.get('prompt')
                    negative_prompt = p.get('negative_prompt', '')
                    ref_images = p.get('reference_images', [])
                    gen_params = p.get('generation_params', {})
                    prompt_hash = p.get('prompt_hash')
                    
                    output_path = os.path.join(images_dir, f"{shot_id}.png")
                    
                    # Check Prompt Cache (V3 Upgrade)
                    cached_hash = pm.get_checkpoint_value('prompt_cache', shot_id)
                    if os.path.exists(output_path) and cached_hash == prompt_hash:
                        logger.info(f"Shot {shot_id} exists and hash matches. Skipping.")
                        continue
                    
                    image_adapter.generate_image(prompt, output_path, negative_prompt, reference_image_paths=ref_images, generation_params=gen_params)
                    pm.save_checkpoint('prompt_cache', prompt_hash, sub_key=shot_id)
        else:
            logger.warning("No clips.json found. Run visual stage first.")
            
    if args.stage in ['all', 'audio']:
        logger.info("Running Audio Generation (Cinematic V4)...")
        from models.audio_adapter import LocalAudioAdapter
        
        audio_adapter = LocalAudioAdapter()
        clips_path = os.path.join(pm.dirs['output'], 'clips.json')
        audio_dir = os.path.join(pm.dirs['output'], 'audio')
        os.makedirs(audio_dir, exist_ok=True)
        
        if os.path.exists(clips_path):
            with open(clips_path, 'r', encoding='utf-8') as f:
                clips_data = json.load(f)
                
            for clip in clips_data:
                for p in clip['shots']:
                    shot_id = p.get('shot_id')
                    narration_text = p.get('narration_text', 'Silence.')
                    output_path = os.path.join(audio_dir, f"{shot_id}.wav")
                    
                    if os.path.exists(output_path):
                        continue
                        
                    audio_adapter.generate_audio(narration_text, output_path)
        else:
            logger.warning("No clips.json found.")

    if args.stage in ['all', 'video']:
        logger.info("Running Video Production...")
        from core.video.renderer import VideoRenderer
        renderer = VideoRenderer(pm.project_dir)
        renderer.render()

    if args.stage in ['all', 'publishing']:
        logger.info("Running Publishing Engine...")
        from core.publishing.generator import PublishingGenerator
        pub_gen = PublishingGenerator(llm_adapter, pm.project_dir)
        
        translated_files = [f for f in os.listdir(pm.dirs['output']) if f.startswith('translated_')]
        full_text = ""
        for file in translated_files:
            full_text += pm.read_input(os.path.join(pm.dirs['output'], file)) + "\n"
            
        pub_gen.generate_seo_metadata(full_text[:3000])
        pub_gen.select_thumbnail()

    if args.stage in ['all', 'export']:
        logger.info("Running Export Engine...")
        
        # 1. Read SEO metadata to get the clickbait title
        title = "Export"
        seo_path = os.path.join(pm.dirs['output'], 'seo_metadata.json')
        if os.path.exists(seo_path):
            try:
                with open(seo_path, 'r', encoding='utf-8') as f:
                    seo_data = json.load(f)
                    raw_title = seo_data.get('title', 'Export')
                    # Clean title for folder name (remove illegal characters and spaces)
                    title = re.sub(r'[\\/*?:"<>|]', "", raw_title).strip().replace(" ", "_")
            except Exception as e:
                logger.warning(f"Could not read SEO title: {e}")
                
        export_dir = os.path.join(pm.project_dir, 'export', title)
        os.makedirs(export_dir, exist_ok=True)
        
        # Copy memory db
        db_path = os.path.join(pm.project_dir, 'memory', 'novel_memory.db')
        if os.path.exists(db_path):
            shutil.copy(db_path, os.path.join(export_dir, 'novel_memory.db'))
            logger.info("Packaged novel_memory.db for persistence.")
            
            # Create human readable dump
            try:
                memory_db = MemoryEngine(pm.project_dir)
                session = memory_db.Session()
                try:
                    from core.memory.database import Character
                    chars = session.query(Character).all()
                    data = [{"name": c.canonical_name, "visual_dna": c.visual_dna} for c in chars]
                    with open(os.path.join(export_dir, 'characters_dump.json'), 'w', encoding='utf-8') as f:
                        json.dump(data, f, indent=2)
                    logger.info("Generated human-readable characters_dump.json.")
                finally:
                    session.close()
            except Exception as e:
                logger.warning(f"Could not generate character dump: {e}")
                    
        # Copy videos
        videos_dir = os.path.join(pm.project_dir, 'output', 'videos')
        if os.path.exists(videos_dir):
            for f in os.listdir(videos_dir):
                if f.endswith('.mp4'):
                    video_path = os.path.join(videos_dir, f)
                    dest_path = os.path.join(export_dir, f)
                    shutil.copy(video_path, dest_path)
                    logger.info(f"Packaged video: {f}")
                    
                    # Kaggle specific: copy to root for easy download
                    try:
                        root_video_path = os.path.join(base_dir, f)
                        shutil.copy(video_path, root_video_path)
                        logger.info(f"Kaggle shortcut created: {root_video_path}")
                        
                        # Generate HTML Download Link
                        print(f"\n--- DOWNLOAD LINK ---")
                        print(f"File ready for download in Kaggle Output: {f}")
                        print(f"----------------------\n")
                    except Exception as k_e:
                        logger.warning(f"Could not create Kaggle shortcut: {k_e}")
                    
                    # Layer 9: Automated Google Drive Backup
                    try:
                        from core.publishing.drive_uploader import DriveUploader
                        uploader = DriveUploader(config_manager)
                        uploader.upload_file(video_path)
                    except Exception as drive_e:
                        logger.warning(f"Google Drive upload failed or skipped: {drive_e}")
                    
        # Copy configuration file
        config_path = os.path.abspath(args.config)
        if os.path.exists(config_path):
            shutil.copy(config_path, os.path.join(export_dir, 'pipeline_config.yaml'))
            logger.info("Packaged pipeline configuration.")
                
        logger.info(f"Exported all final assets to {export_dir}")
        
        # Git push automatically
        logger.info("Pushing packaged folder to GitHub...")
        try:
            # Add only the specific export folder
            export_rel_path = f"projects/{args.project}/export/{title}"
            subprocess.run(['git', 'add', export_rel_path], check=True, cwd=base_dir)
            subprocess.run(['git', 'commit', '-m', f"Auto-publish: {title}"], check=True, cwd=base_dir)
            subprocess.run(['git', 'push'], check=True, cwd=base_dir)
            logger.info("Successfully pushed video package to GitHub!")
        except subprocess.CalledProcessError as e:
            logger.error(f"Git push failed. You may need to authenticate. Error: {e}")

    logger.info("Pipeline completed successfully.")

if __name__ == '__main__':
    main()
