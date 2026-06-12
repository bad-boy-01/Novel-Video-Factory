import logging
import re
from typing import List, Dict

logger = logging.getLogger(__name__)

class TranslationPipeline:
    """
    Layer 2: Translation, Revision, and Canonical Script.
    Ensures zero information loss during translation using local LLMs.
    """
    def __init__(self, config: dict, llm_adapter):
        self.config = config
        self.llm = llm_adapter
        self.sentence_counter = 0

    def clean_text(self, text: str) -> str:
        """Remove invalid artifacts without skipping dialogue or poems."""
        # Basic cleanup of multiple spaces and newlines
        text = re.sub(r'[ \t]+', ' ', text)
        text = re.sub(r'\n{3,}', '\n\n', text)
        return text.strip()

    def _split_into_sentences(self, text: str) -> List[str]:
        """Naively split text into sentences respecting punctuation."""
        # Matches common sentence endings (including Chinese/Japanese ones)
        sentences = re.split(r'(?<=[.!?。！？])\s+', text)
        return [s.strip() for s in sentences if s.strip()]

    def chunk_text(self, text: str, max_words: int = 1000) -> List[Dict]:
        """
        Split text into chunks at sentence boundaries.
        Returns a list of dictionaries containing chunk metadata.
        """
        paragraphs = text.split('\n\n')
        chunks = []
        current_chunk_sentences = []
        current_word_count = 0
        
        chunk_id = 1
        
        for paragraph in paragraphs:
            sentences = self._split_into_sentences(paragraph)
            for sentence in sentences:
                self.sentence_counter += 1
                s_id = f"S{self.sentence_counter:08d}"
                word_count = len(sentence.split()) # Approx for English, simplistic for CJK
                
                if current_word_count + word_count > max_words and current_chunk_sentences:
                    # Finalize current chunk
                    chunks.append({
                        "chunk_id": f"C{chunk_id:04d}",
                        "sentences": current_chunk_sentences,
                        "word_count": current_word_count
                    })
                    chunk_id += 1
                    current_chunk_sentences = []
                    current_word_count = 0
                
                current_chunk_sentences.append({"id": s_id, "text": sentence})
                current_word_count += word_count
                
        # Add remaining
        if current_chunk_sentences:
            chunks.append({
                "chunk_id": f"C{chunk_id:04d}",
                "sentences": current_chunk_sentences,
                "word_count": current_word_count
            })
            
        return chunks

    def translate_chunk(self, chunk: Dict) -> str:
        """Translate a single chunk with zero information loss rules."""
        text_to_translate = " ".join([s["text"] for s in chunk["sentences"]])
        system_prompt = (
            "Translate the following text. You must follow these rules strictly:\n"
            "1. Translate every sentence and paragraph.\n"
            "2. Never summarize, skip, or compress.\n"
            "3. Preserve dialogue and formatting.\n"
            "Return complete translated text only."
        )
        return self.llm.generate(text_to_translate, system_prompt=system_prompt)

    def process_chapter(self, chapter_text: str) -> str:
        """Run the full translation pipeline for a single chapter."""
        self.sentence_counter = 0 # Reset per chapter for now (should be per part/project long term)
        cleaned = self.clean_text(chapter_text)
        chunks = self.chunk_text(cleaned)
        
        translated_chunks = []
        for i, chunk in enumerate(chunks):
            logger.info(f"Translating chunk {chunk['chunk_id']} ({i+1}/{len(chunks)}) - Words: {chunk['word_count']}")
            translated = self.translate_chunk(chunk)
            
            # TODO: Implement Zero Information Loss Validator here
            
            translated_chunks.append(translated)
            
        return "\n\n".join(translated_chunks)
