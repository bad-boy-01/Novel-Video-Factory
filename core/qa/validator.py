import logging
import re
from typing import Dict, List

logger = logging.getLogger(__name__)

class ZeroInformationLossValidator:
    """
    Validates that no information was lost during translation or revision.
    """
    
    def validate_translation(self, source_chunk: Dict, translated_text: str) -> bool:
        """
        Compare the source chunk against the translated text.
        Returns True if validation passes, False otherwise.
        """
        # Simplistic checks: sentence count and dialogue count
        # A more advanced version would use an LLM or embedding similarity.
        
        source_sentences = len(source_chunk['sentences'])
        
        # Count sentences in translated text
        translated_sentences = len(re.split(r'(?<=[.!?。！？])\s+', translated_text))
        
        # Count dialogue markers (quotes)
        source_text = " ".join([s["text"] for s in source_chunk["sentences"]])
        source_quotes = len(re.findall(r'["\']', source_text))
        translated_quotes = len(re.findall(r'["\']', translated_text))
        
        # Validate sentence count (allow 20% margin for translation differences)
        if translated_sentences < (source_sentences * 0.8):
            logger.warning(f"Validation failed: Translated sentence count ({translated_sentences}) is much lower than source ({source_sentences}).")
            return False
            
        # Validate dialogue
        if source_quotes > 0 and translated_quotes == 0:
            logger.warning("Validation failed: Dialogue was lost during translation.")
            return False
            
        logger.info(f"Validation passed for chunk {source_chunk['chunk_id']}.")
        return True
