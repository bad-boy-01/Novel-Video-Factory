import logging

logger = logging.getLogger(__name__)

class QAPipeline:
    """
    Chapter 10: QA Pipeline.
    Implements autonomous review of the AI outputs.
    """
    def __init__(self, llm_adapter):
        self.llm = llm_adapter
        from core.config_manager import ConfigManager
        self.enabled = ConfigManager().get('system.qa_enabled', True)

    def verify_translation(self, source_text: str, translated_text: str) -> bool:
        if not self.enabled:
            return True
            
        system_prompt = (
            "You are a strict QA reviewer. Compare the source text with the translation. "
            "Did the translation drop any important character names, locations, or key plot details? "
            "Respond ONLY with 'PASS' or 'FAIL'."
        )
        
        prompt = f"SOURCE:\n{source_text}\n\nTRANSLATION:\n{translated_text}"
        response = self.llm.generate(prompt, system_prompt=system_prompt, temperature=0.1)
        
        if "FAIL" in response.upper():
            logger.error("QA Pipeline rejected the translation for dropping information!")
            return False
        
        logger.info("QA Pipeline approved the translation.")
        return True
