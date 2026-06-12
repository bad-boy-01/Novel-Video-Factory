import argparse
import sys
import logging
import os

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
                        choices=['all', 'translate', 'memory', 'visual', 'generation', 'audio', 'video', 'publishing', 'export'],
                        help='Which stage of the pipeline to run')
    parser.add_argument('--config', default='config/default.yaml', help="Path to configuration file")

    args = parser.parse_args()

    logger.info(f"Starting Novel Video Factory for project: {args.project}")
    
    config_manager = ConfigManager(args.config)
    base_dir = os.path.dirname(os.path.abspath(__file__))
    pm = ProjectManager(base_dir, args.project)
    
    # Initialize Adapters with Config values
    llm_model = config_manager.get('models.translation.primary.model', 'qwen2.5:7b')
    llm_adapter = LocalLLMAdapter(model_name=llm_model)
    
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
        from core.memory.extractor import MemoryExtractor
        from core.translation.pipeline import TranslationPipeline
        import uuid
        
        extractor = MemoryExtractor(llm_adapter)
        memory_db = MemoryEngine(pm.project_dir)
        text_chunker = TranslationPipeline(config={}, llm_adapter=None)
        
        translated_files = [f for f in os.listdir(pm.dirs['output']) if f.startswith('translated_')]
        for file in translated_files:
            text = pm.read_input(os.path.join(pm.dirs['output'], file))
            chunks = text_chunker.chunk_text(text, max_words=500)
            
            for chunk_idx, chunk_data in enumerate(chunks):
                chunk_text = " ".join([s["text"] for s in chunk_data["sentences"]])
                logger.info(f"Extracting Memory from Chunk {chunk_idx + 1}/{len(chunks)}")
                
                # Character extraction
                chars = extractor.extract_characters(chunk_text)
                for c in chars:
                    c_id = str(uuid.uuid4())[:8]
                    memory_db.add_character(c_id, c.get('canonical_name', 'Unknown'), c.get('visual_dna', {}))
                    logger.info(f"Saved character to DB: {c.get('canonical_name')}")
                    
                # Location extraction
                locs = extractor.extract_locations(chunk_text)
                for loc in locs:
                    memory_db.add_location(loc.get('canonical_name', 'Unknown'), loc.get('description', ''))
                    logger.info(f"Saved location to DB: {loc.get('canonical_name')}")
                    
                # World Concept extraction
                concepts = extractor.extract_world_concepts(chunk_text)
                for concept in concepts:
                    memory_db.add_world_concept(concept.get('concept_type', 'misc'), concept.get('name', 'Unknown'), concept.get('description', ''))
                    logger.info(f"Saved world concept to DB: {concept.get('name')}")

    if args.stage in ['all', 'character_sheets']:
        logger.info("Running Character Sheets Generation...")
        memory_db = MemoryEngine(pm.project_dir)
        from models.image_adapter import LocalImageAdapter
        image_adapter = LocalImageAdapter()
        
        chars_dir = os.path.join(pm.project_dir, 'memory', 'characters')
        os.makedirs(chars_dir, exist_ok=True)
        
        style_modifier = config_manager.get('prompts.style_modifier', 'Cinematic, high quality')
        
        with memory_db.Session() as session:
            from core.memory.database import Character
            characters = session.query(Character).all()
            for char in characters:
                img_path = os.path.join(chars_dir, f"{char.id}.png")
                if not os.path.exists(img_path):
                    logger.info(f"Generating Character Reference Sheet for {char.canonical_name}...")
                    dna_tags = []
                    for k, v in char.visual_dna.items():
                        if isinstance(v, str) and v.lower() not in ['not specified', 'unknown', 'none']:
                            dna_tags.append(v)
                    dna_str = ", ".join(dna_tags)
                    
                    prompt = f"1boy/1girl, {dna_str}, cinematic portrait, detailed background, looking at viewer, masterpiece, {style_modifier}"
                    negative = config_manager.get('prompts.negative_prompt', 'ugly, bad anatomy')
                    image_adapter.generate_image(prompt, img_path, negative_prompt=negative)
                else:
                    logger.info(f"Reference Sheet already exists for {char.canonical_name}, skipping.")

    if args.stage in ['all', 'visual']:
        logger.info("Running Visual Planning...")
        from core.visual.planner import ScenePlanner
        from core.visual.prompter import PromptGenerator
        
        memory_db = MemoryEngine(pm.project_dir)
        planner = ScenePlanner(llm_adapter)
        style_modifier = config_manager.get('prompts.style_modifier', 'Cinematic, high quality Korean Manhwa style, detailed line art, masterpiece, best quality')
        prompter = PromptGenerator(memory_db, base_style=style_modifier)
        
        # Read translated output
        translated_files = [f for f in os.listdir(pm.dirs['output']) if f.startswith('translated_')]
        all_prompts = []
        
        # We instantiate a dummy pipeline just to use its chunk_text utility
        text_chunker = TranslationPipeline(config={}, llm_adapter=None)
        
        for file in translated_files:
            text = pm.read_input(os.path.join(pm.dirs['output'], file))
            chunks = text_chunker.chunk_text(text, max_words=500)
            
            for chunk_idx, chunk_data in enumerate(chunks):
                chunk_text = " ".join([s["text"] for s in chunk_data["sentences"]])
                logger.info(f"Visual Planning for Chunk {chunk_idx + 1}/{len(chunks)} ({chunk_data['word_count']} words)")
                
                scenes = planner.plan_scenes(chunk_text)
                
                for scene in scenes:
                    prompt_data = prompter.generate_prompt_for_scene(scene)
                    all_prompts.append(prompt_data)
                    logger.info(f"Generated prompt for {scene.get('scene_id')}: {prompt_data['prompt']}")
                
        # Save prompts
        import json
        pm.save_output("prompts.json", json.dumps(all_prompts, indent=2))
        logger.info("Saved all visual prompts to prompts.json")

    if args.stage in ['all', 'generation']:
        logger.info("Running Image Generation...")
        from models.image_adapter import LocalImageAdapter
        import json
        
        image_adapter = LocalImageAdapter()
        prompts_path = os.path.join(pm.dirs['output'], 'prompts.json')
        images_dir = os.path.join(pm.dirs['output'], 'images')
        os.makedirs(images_dir, exist_ok=True)
        
        if os.path.exists(prompts_path):
            with open(prompts_path, 'r', encoding='utf-8') as f:
                prompts_data = json.load(f)
                
            for p in prompts_data:
                scene_id = p.get('scene_id')
                prompt = p.get('prompt')
                negative_prompt = p.get('negative_prompt', '')
                ref_images = p.get('reference_images', [])
                output_path = os.path.join(images_dir, f"{scene_id}.png")
                
                image_adapter.generate_image(prompt, output_path, negative_prompt, reference_image_paths=ref_images)
        else:
            logger.warning("No prompts.json found. Run the visual stage first.")
            
    if args.stage in ['all', 'audio']:
        logger.info("Running Audio Generation...")
        from models.audio_adapter import LocalAudioAdapter
        import json
        
        audio_adapter = LocalAudioAdapter()
        prompts_path = os.path.join(pm.dirs['output'], 'prompts.json')
        audio_dir = os.path.join(pm.dirs['output'], 'audio')
        os.makedirs(audio_dir, exist_ok=True)
        
        if os.path.exists(prompts_path):
            with open(prompts_path, 'r', encoding='utf-8') as f:
                prompts_data = json.load(f)
                
            for p in prompts_data:
                scene_id = p.get('scene_id')
                # Read the exact script dialogue for subtitles/audio
                narration_text = p.get('metadata', {}).get('narration_text', 'Silence.')
                output_path = os.path.join(audio_dir, f"{scene_id}.wav")
                
                audio_adapter.generate_audio(narration_text, output_path)
        else:
            logger.warning("No prompts.json found. Run the visual stage first.")

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
        import shutil
        import json
        import re
        import subprocess
        
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
                with memory_db.Session() as session:
                    from core.memory.database import Character
                    chars = session.query(Character).all()
                    data = [{"name": c.canonical_name, "visual_dna": c.visual_dna} for c in chars]
                    with open(os.path.join(export_dir, 'characters_dump.json'), 'w', encoding='utf-8') as f:
                        json.dump(data, f, indent=2)
                logger.info("Generated human-readable characters_dump.json.")
            except Exception as e:
                logger.warning(f"Could not generate character dump: {e}")
                    
        # Copy videos
        output_dir = os.path.join(pm.project_dir, 'output')
        if os.path.exists(output_dir):
            for f in os.listdir(output_dir):
                if f.endswith('.mp4'):
                    shutil.copy(os.path.join(output_dir, f), os.path.join(export_dir, f))
                    logger.info(f"Packaged video: {f}")
                    
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
