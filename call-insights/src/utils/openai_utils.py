import logging
from dotenv import load_dotenv
import os
load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Truncation function removed - no longer needed with optimized clean message format

def safe_model_invoke(model, messages, max_retries=3):
    """Safely invoke model - optimized for clean message format"""
    for attempt in range(max_retries):
        try:
            # Direct invocation - our clean messages are already optimized
            response = model.invoke(messages)
            return response
            
        except Exception as e:
            if "context_length_exceeded" in str(e) and attempt < max_retries - 1:
                logger.warning(f"Context length exceeded, retrying with shorter context (attempt {attempt + 1})")
                # Simple fallback: keep only essential messages
                if len(messages) > 3:
                    messages = [msg for msg in messages if msg.get('role') == 'system'] + messages[-2:]
                continue
            else:
                logger.error(f"Model invocation failed: {e}")
                raise