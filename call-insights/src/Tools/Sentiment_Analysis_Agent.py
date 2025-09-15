from langchain_core.tools import tool
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import logging
from typing import Optional, Annotated
from langchain_core.tools import tool, InjectedToolCallId
from langchain_core.messages import ToolMessage
from langgraph.types import Command
from dotenv import load_dotenv
import os
from utils.openai_utils import safe_model_invoke
from langchain_core.messages import SystemMessage
from utils.prompt_templates import Sentiment_Analysis_Model_Template
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_openai import ChatOpenAI

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def sentiment_analysis(text: str):
    """Analyze the sentiment of the text"""
    try:
        model_name = os.getenv("SENTIMENT_ANALYSIS_MODEL_NAME")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        
        # Try GPU first, fallback to CPU if not available
        try:
            sentiment_analyzer = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer, device=0)
        except Exception:
            logger.warning("GPU not available, falling back to CPU")
            sentiment_analyzer = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer, device=-1)
        
        result = sentiment_analyzer(text)
        logger.info(f"Sentiment analysis result: {result}")
        return result
    except Exception as e:
        logger.error(f"Error in sentiment analysis: {e}")
        return {"error": str(e)}

# @tool
# def update_state_Sentiment_Analysis_Agent(
#     processing_status: Optional[str] = None,
#     processing_complete: Optional[bool] = None,
#     sentiment_label: Optional[str] = None,
#     sentiment_scores: Optional[float] = None,
#     tool_call_id: Annotated[Optional[str], InjectedToolCallId] = None
# ) -> str:
#     """
#     Update the state of the agent
#     Args:
#         processing_status: Status of the processing
#         processing_complete: Whether processing is complete
#         sentiment_label: The sentiment label of the text
#         sentiment_scores: The sentiment scores of the text
#     Returns:
#         Confirmation message that the state has been updated.
#     """
#     update_dict = {}
#     if processing_status is not None:
#         update_dict['processing_status'] = processing_status
#     if processing_complete is not None:
#         update_dict['processing_complete'] = processing_complete
#     if sentiment_label is not None:
#         update_dict['sentiment_label'] = sentiment_label
#     if sentiment_scores is not None:
#         update_dict['sentiment_scores'] = sentiment_scores

#     update_dict["messages"] = [
#         ToolMessage(
#             content="State updated successfully.",
#             tool_call_id=tool_call_id,
#         )
#     ]
#     # Return a Command to update the state
#     return Command(update=update_dict)

