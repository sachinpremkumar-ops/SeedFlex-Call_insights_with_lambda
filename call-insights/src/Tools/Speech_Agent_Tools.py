from typing import Literal, TypedDict, Sequence, Annotated, Optional, Union, List, Dict, Any
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.tools import tool, InjectedToolCallId
from langgraph.prebuilt import create_react_agent, ToolNode
from langgraph.types import Command  # Added missing import
from pydantic import BaseModel 
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import boto3
import json
import psycopg2
from psycopg2.extras import RealDictCursor
from botocore.exceptions import ClientError
import os
import io
import logging
<<<<<<< HEAD
from openai import OpenAI
from utils.s3_utils import s3_get_audio_file
=======
from openai import AzureOpenAI
from src.utils.s3_utils import s3_get_audio_file        
>>>>>>> bff7368 (Lambda integration, with filtered messages)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize Azure OpenAI client
client = AzureOpenAI(
    api_key=os.getenv("AZURE_API_KEY"),
    azure_endpoint=os.getenv("AZURE_ENDPOINT"),
    api_version=os.getenv("AZURE_API_VERSION", "2024-12-01-preview")
)

# Constants
<<<<<<< HEAD
BUCKET_NAME = "experiment2407"
PROCESSING_PREFIX = "processing/"
PROCESSED_PREFIX = "processed/"
=======
BUCKET_NAME = os.getenv("S3_BUCKET_NAME", "experimentbucket1234")
PROCESSING_PREFIX = os.getenv("S3_PROCESSING_PREFIX", "processing/")
PROCESSED_PREFIX = os.getenv("S3_PROCESSED_PREFIX", "processed_latest/")
>>>>>>> bff7368 (Lambda integration, with filtered messages)

@tool
def transcribe_audio(file_name: str):
    """Transcribe audio file"""
    try:
        file_name = file_name.split('/')[-1]
        
        # Try multiple possible locations for the file
        possible_keys = [
            f'processing/{file_name}',  # Expected location
            f'processed_latest/{file_name}',
            file_name,  # Direct filename
            f'audio/{file_name}',
            f'uploads/{file_name}'
        ]
        
        audio_bytes = None
        found_key = None
        
        for key in possible_keys:
            audio_bytes = s3_get_audio_file(key)
            if audio_bytes is not None:
                found_key = key
                logger.info(f"Found audio file at: {key}")
                break
        
        if audio_bytes is None:
<<<<<<< HEAD
            logger.error(f"Failed to fetch audio file: {processing_key}")
            return None
=======
            logger.error(f"Failed to fetch audio file: {file_name}. Tried locations: {possible_keys}")
            return "Error: Audio file not found in any expected location"
>>>>>>> bff7368 (Lambda integration, with filtered messages)
        
        # Create a file-like object with proper filename for OpenAI API
        
        audio_file = io.BytesIO(audio_bytes)
        audio_file.name = file_name  # Set the filename for format detection
        
        transcription = client.audio.transcriptions.create(
            model=os.getenv("WHISPER_MODEL_NAME", "whisper"),
            file=audio_file
        )
        logger.info(f"Successfully transcribed {file_name}")
        print(transcription.text)
        return transcription.text
    except Exception as e:
        logger.error(f"Error transcribing {file_name}: {e}")
        print(f"Error transcribing {file_name}: {e}")
        return None

@tool
def translate_audio(file_name:str):
    """ Translate Audio files into English if the original is not in English"""
    try:
        file_name = file_name.split('/')[-1]
        
        # Try multiple possible locations for the file
        possible_keys = [
            f'processing/{file_name}',  # Expected location
            f'processed_latest/{file_name}',
            file_name,  # Direct filename
            f'audio/{file_name}',
            f'uploads/{file_name}'
        ]
        
        audio_bytes = None
        found_key = None
        
        for key in possible_keys:
            audio_bytes = s3_get_audio_file(key)
            if audio_bytes is not None:
                found_key = key
                logger.info(f"Found audio file for translation at: {key}")
                break
        
        if audio_bytes is None:
            logger.error(f"Failed to fetch audio file for translation: {file_name}. Tried locations: {possible_keys}")
            return None

        audio_file= io.BytesIO(audio_bytes)
        audio_file.name = file_name

        translatedtranscript = client.audio.translations.create(
            model="whisper",
            file=audio_file
        )
        logger.info(f"Successfully translated {file_name}")
        print(translatedtranscript.text)
        return translatedtranscript.text
    except Exception as e:
        logger.error(f"Error translating audio file: {e}")
        return None

@tool
def update_state_Speech_Agent(
    processing_status: Optional[str] = None,
    processing_complete: Optional[bool] = None,
    transcription: Optional[str] = None,
    translation: Optional[str] = None,
    tool_call_id: Annotated[Optional[str], InjectedToolCallId] = None
) -> str:
    """
    Update the state of the agent
    Args:
        processing_status: Status of the processing
        processing_complete: Whether processing is complete
    Returns:
        Confirmation message that the state has been updated.
    """
    update_dict = {}
    if processing_status is not None:
        update_dict['processing_status'] = processing_status
    if processing_complete is not None:
        update_dict['processing_complete'] = processing_complete
    if transcription is not None:
        update_dict['transcription'] = transcription
    if translation is not None:
        update_dict['translation'] = translation

    update_dict["messages"] = [
        ToolMessage(
            content="State updated successfully.",
            tool_call_id=tool_call_id,
        )
    ]
    # Return a Command to update the state
    return Command(update=update_dict)