from typing import Literal, TypedDict, Sequence, Annotated, Optional, Union, List, Dict, Any
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.tools import tool, InjectedToolCallId
from langgraph.prebuilt import ToolNode
from langgraph.types import Command
from pydantic import BaseModel 
from langchain_openai import ChatOpenAI,OpenAIEmbeddings,AzureChatOpenAI
from dotenv import load_dotenv
import boto3
import json
import psycopg2
from psycopg2.extras import RealDictCursor
from botocore.exceptions import ClientError
import os
import io
from src.utils.prompt_templates_optimized import Ingestion_Model_Template, Speech_Model_Template, Summarization_Model_Template, Storage_Model_Template, Sentiment_Analysis_Model_Template, Insights_Agent_Template
import logging
from src.Tools.Ingestion_Agent_Tools import get_single_audio_file_from_s3, move_file_to_processing, roll_back_file_from_processing
from openai import OpenAI
from src.Tools.Speech_Agent_Tools import transcribe_audio, translate_audio, update_state_Speech_Agent
from langgraph.graph.state import LastValue, LastValueAfterFinish, BinaryOperatorAggregate
from src.Tools.Summarization_Agent_Tools import update_state_Summarization_Agent
from src.Tools.Storage_Agent_Tools import insert_data_all, move_file_to_processed
from src.Tools.Insights_Agent_Tools import update_state_Insights_Agent
from src.utils.rds_utils import connect_to_rds
from transformers import pipeline
from src.Tools.Sentiment_Analysis_Agent import sentiment_analysis
from src.utils.openai_utils import safe_model_invoke
import uuid
from datetime import datetime, timedelta
from functools import wraps
from src.sql.tables_sql import create_tables

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Validate required environment variables
# required_env_vars = ["OPENAI_API_KEY"]  # Only OpenAI is required
# missing_vars = [var for var in required_env_vars if not os.getenv(var)]
# if missing_vars:
#     raise ValueError(f"Missing required environment variables: {missing_vars}")
    
BUCKET_NAME = os.getenv("S3_BUCKET_NAME")
PROCESSING_PREFIX = os.getenv("S3_PROCESSING_PREFIX", "processing/")
PROCESSED_PREFIX = os.getenv("S3_PROCESSED_PREFIX", "processed_latest/")

try:
    model_gpt4 = AzureChatOpenAI(
        model="gpt-4",
        api_key=os.getenv("AZURE_API_KEY"),
        azure_endpoint=os.getenv("AZURE_ENDPOINT"),
        api_version=os.getenv("AZURE_API_VERSION", "2024-12-01-preview"),
        temperature=0,
        max_tokens=2000,
    )
    
    model_gpt35 = AzureChatOpenAI(
        model="gpt-35-turbo",
        api_key=os.getenv("AZURE_API_KEY"),
        azure_endpoint=os.getenv("AZURE_ENDPOINT"),
        api_version=os.getenv("AZURE_API_VERSION", "2024-12-01-preview"),
        temperature=0,
        max_tokens=1000,
    )
    
    model = model_gpt35

    # ollama_model= ChatOllama(model="llama3.2:1b", temperature=0, base_url="http://localhost:11434")
    # ollama_model= ChatOllama(model="llama3.2", temperature=0, base_url="http://localhost:11434")
    embeddings_model = OpenAIEmbeddings(model="text-embedding-ada-002", api_key=os.getenv("OPENAI_API_KEY"))
    # embeddings_model = GoogleGenerativeAIEmbeddings(model="embedding-001", google_api_key=os.getenv("GEMINI_API_KEY"))
    client = OpenAI()
except Exception as e:
    logger.error(f"Failed to initialize OpenAI models: {e}")
    raise e

try:
    s3_client = boto3.client('s3')  
except Exception as e:
    logger.error(f"Failed to initialize S3 client: {e}")
    raise e

class AudioFile(BaseModel):
    key: str
    size: int
    bucket: str = os.getenv("S3_BUCKET_NAME", "experimentbucket1234")
    last_modified: str
    original_key: str = ""

class AgentState(TypedDict, total=False):
    workflow_id: Annotated[str, LastValue]
    messages: Annotated[Sequence[Union[HumanMessage, AIMessage, SystemMessage]], add_messages]
    audio_files: Annotated[Optional[AudioFile], LastValue]
    audio_file_key: Annotated[str, LastValue]
    transcription: Annotated[str, LastValue]
    translation: Annotated[str, LastValue]
    summary: Annotated[str, LastValue]
    action_items: Annotated[str, LastValue]
    key_points: Annotated[str, LastValue]
    topic: Annotated[str, LastValue]
    sentiment: Annotated[str, LastValue]
    processing_status: Annotated[str, LastValue]
    processing_complete: Annotated[bool, LastValue]
    error_message: Annotated[Optional[str], LastValue]
    embeddings: Annotated[List[float], LastValue]
    sentiment_label: Annotated[str, LastValue]
    sentiment_scores: Annotated[float, LastValue]

def log_agent_execution(workflow_id: str, agent_name: str, status: str, execution_time_seconds: float, error_message: str = None):
    """Simple function to log agent execution to database"""
    logger.info(f"🔄 Logging agent execution: {agent_name} - {status} - {execution_time_seconds:.2f}s")
    try:
        connection = connect_to_rds()
        if not connection:
            logger.error("❌ Failed to connect to database for agent execution logging")
            return
            
        with connection.cursor() as cursor:
            create_tables(connection)
            cursor.execute("""  
                INSERT INTO agent_executions(workflow_id, agent_name, execution_time_ms, status, error_message)
                VALUES (%s, %s, %s, %s, %s)
                """, (
                workflow_id, 
                agent_name,
                int(execution_time_seconds * 1000), 
                status,
                error_message
            ))
            connection.commit()
            logger.info(f"✅ Successfully logged agent execution: {agent_name}")
    except Exception as e:
        logger.error(f"❌ Failed to log agent execution to database: {e}")
    finally:
        if connection:
            connection.close()


def Ingestion_Model(state: AgentState) -> AgentState:
    """Create an agent for audio file ingestion workflow"""
    start_time = datetime.now()

    audio_file_key = state.get('audio_file_key')
    if not audio_file_key:
        messages = state.get('messages', [])
        if messages and hasattr(messages[0], 'content'):
            audio_file_key = messages[0].content
    
    if not audio_file_key:
        logger.error("❌ No audio_file_key found in state or messages")
        return {
            **state,
            "messages": state.get("messages", []) + [AIMessage(content="Error: No audio file key provided")]
        }

    workflow_id = str(uuid.uuid4())

    try:
        logger.info(f"🚀 Starting ingestion for: {audio_file_key}")
        logger.info(f"🔧 Generated workflow_id: {workflow_id}")

        audio_file = get_single_audio_file_from_s3(audio_file_key)
        if not audio_file:
            raise Exception(f"Failed to get audio file: {audio_file_key}")

        move_file_to_processing(audio_file_key)
        logger.info(f"📁 Moved {audio_file_key} to processing folder")

        new_state = {
            **state,
            "workflow_id": workflow_id,
            "audio_file_key": audio_file_key,
            "audio_files": audio_file,
            "messages": state.get("messages", []) + [AIMessage(content=f"Ingestion completed for {audio_file_key}")]
        }

        end_time = datetime.now()
        execution_time = (end_time - start_time).total_seconds()
        log_agent_execution(workflow_id, "Ingestion_Agent", "completed", execution_time)
        logger.info(f"✅ Ingestion_Agent completed in {execution_time:.2f}s - workflow_id: {workflow_id}")
        
        return new_state

    except Exception as e:

        end_time = datetime.now()
        execution_time = (end_time - start_time).total_seconds()
        
        log_agent_execution(workflow_id, "Ingestion_Agent", "failed", execution_time, str(e))
        logger.error(f"❌ Ingestion_Agent failed after {execution_time:.2f}s - workflow_id: {workflow_id} - Error: {str(e)}")
        
        if audio_file_key:
            roll_back_file_from_processing(audio_file_key)
            logger.info(f"🔄 Rolled back file {audio_file_key}")
        
        return {
            **state,
            "messages": state.get("messages", []) + [AIMessage(content=f"Error: {str(e)}")]
        }
        


Ingestion_Agent = Ingestion_Model
            
Speech_Model_Tools = [transcribe_audio, update_state_Speech_Agent, translate_audio]

def Speech_Model(state:AgentState) -> AgentState:
    start_time = datetime.now()

    workflow_id = state.get('workflow_id', None)
    audio_file_key = state.get('audio_file_key', None)

    
    try:
        messages = state.get('messages', [])
        # Keep recent messages but limit to avoid token bloat - filter out orphaned tool messages
        messages = messages[-8:] if len(messages) > 8 else messages
        
        # Filter out orphaned tool messages to prevent API errors
        filtered_messages = []
        for i, msg in enumerate(messages):
            if hasattr(msg, 'type') and msg.type == 'tool':
                # Only keep tool messages if the previous message has tool_calls
                if i > 0 and hasattr(messages[i-1], 'tool_calls') and messages[i-1].tool_calls:
                    filtered_messages.append(msg)
                # Skip orphaned tool messages
            else:
                filtered_messages.append(msg)
        
        messages = filtered_messages + [SystemMessage(content=Speech_Model_Template)]
        logger.info(f"🔍 Workflow_id: {workflow_id}")
        logger.info(f"🔍 Audio_file_key: {audio_file_key}")
        Speech_Model_With_Tools=model_gpt4.bind_tools(Speech_Model_Tools)   
        response=safe_model_invoke(Speech_Model_With_Tools, messages)
        end_time = datetime.now()
        if isinstance(response, Command):
            # Start with the original messages plus the AI response
            new_messages = messages + [response]
            # Add tool response messages if any
            if "messages" in response.update:
                new_messages = new_messages + response.update["messages"]
            
            # Create new state with updated messages and other fields
            new_state = {**state, "messages": new_messages}
            for key, value in response.update.items():
                if key != "messages":  
                    new_state[key] = value 
            execution_time = (end_time - start_time).total_seconds()
            if workflow_id:
                log_agent_execution(workflow_id, "Speech_Model", "completed", execution_time)
                logger.info(f"✅ Speech_Model completed in {execution_time:.2f}s - workflow_id: {workflow_id}")
            return new_state
        else:
            execution_time = (end_time - start_time).total_seconds()
            if workflow_id:
                log_agent_execution(workflow_id, "Speech_Model", "completed", execution_time)
                logger.info(f"✅ Speech_Model completed in {execution_time:.2f}s - workflow_id: {workflow_id}")
            return {**state, "messages" : messages + [response]}
    except Exception as e:
        end_time = datetime.now()
        execution_time = (end_time - start_time).total_seconds()
        if workflow_id:
            log_agent_execution(workflow_id, "Speech_Model", "failed", execution_time, str(e))
            logger.error(f"❌ Speech_Model failed after {execution_time:.2f}s - workflow_id: {workflow_id} - Error: {str(e)}")
        logger.error(f"Error in Speech_Model: {e}")
        
        if audio_file_key:
            roll_back_file_from_processing(audio_file_key)
            logger.info(f"Rolled back file {audio_file_key}")
        else:
            audio_file_key = state.get('audio_file_key') or (state.get('audio_files', {}).get('key') if state.get('audio_files') else None)
            if audio_file_key:
                roll_back_file_from_processing(audio_file_key)
                logger.info(f"Rolled back file {audio_file_key}")
            else:
                logger.warning("Could not determine audio_file_key for rollback")
        return {
            **state,
            "messages": state.get("messages", []) + [AIMessage(content=f"Error: {str(e)}")]
        }
    
def Speech_model_Should_Continue(state:AgentState):
    """check if the last message contains tool calls"""
    messages=state['messages']
    last_message=messages[-1]
    if isinstance(last_message, AIMessage) and getattr(last_message, "tool_calls", None):
        return 'Speech_Model_Tools'
    return END

Speech_Agent=(StateGraph(AgentState)
            .add_node("Speech_Model", Speech_Model)
            .add_node("Speech_Model_Tools", ToolNode(Speech_Model_Tools))
            .add_conditional_edges(
                'Speech_Model',
                Speech_model_Should_Continue,
                {
                    'Speech_Model_Tools': 'Speech_Model_Tools',
                    END: END
                }
            )
            .add_edge(START, "Speech_Model")
            .add_edge("Speech_Model_Tools", "Speech_Model")
            ).compile()

Summarization_Model_Tools = [update_state_Summarization_Agent]

def Summarization_Model(state:AgentState):
    logger.info(f"🚀 SUMMARIZATION_AGENT STARTING")
    start_time = datetime.now()
    workflow_id = state.get('workflow_id', None)
    audio_file_key = state.get('audio_file_key', None)
    logger.info(f"🔍 SUMMARIZATION_AGENT - Workflow_id: {workflow_id}")
    logger.info(f"🔍 SUMMARIZATION_AGENT - Audio_file_key: {audio_file_key}")
    
    try:
        messages = state.get('messages', [])
        messages = messages[-5:] if len(messages) > 5 else messages
        
        filtered_messages = []
        for i, msg in enumerate(messages):
            if hasattr(msg, 'type') and msg.type == 'tool':
                if i > 0 and hasattr(messages[i-1], 'tool_calls') and messages[i-1].tool_calls:
                    filtered_messages.append(msg)
            else:
                filtered_messages.append(msg)
        
        messages = filtered_messages + [SystemMessage(content=Summarization_Model_Template)]
        Summarization_Model_With_Tools = model_gpt35.bind_tools(Summarization_Model_Tools)  
        response = safe_model_invoke(Summarization_Model_With_Tools, messages)
        end_time = datetime.now()
        if isinstance(response, Command):
            new_messages = messages + [response]
            if "messages" in response.update:
                new_messages = new_messages + response.update["messages"]
            
            new_state = {**state, "messages": new_messages}
            for key, value in response.update.items():
                if key != "messages":  
                    new_state[key] = value  
            
            execution_time = (end_time - start_time).total_seconds()
            if workflow_id:
                log_agent_execution(workflow_id, "Summarization_Model", "completed", execution_time)
                logger.info(f"✅ Summarization_Model completed in {execution_time:.2f}s - workflow_id: {workflow_id}")
            return new_state
        else:
            execution_time = (end_time - start_time).total_seconds()
            if workflow_id:
                log_agent_execution(workflow_id, "Summarization_Model", "completed", execution_time)
                logger.info(f"✅ SUMMARIZATION_AGENT COMPLETED in {execution_time:.2f}s - workflow_id: {workflow_id}")
            return {**state, "messages": messages + [response]}
    except Exception as e:
        end_time = datetime.now()
        execution_time = (end_time - start_time).total_seconds()
        if workflow_id:
            log_agent_execution(workflow_id, "Summarization_Model", "failed", execution_time, str(e))
            logger.error(f"❌ Summarization_Model failed after {execution_time:.2f}s - workflow_id: {workflow_id} - Error: {str(e)}")
        logger.error(f"Error in Summarization_Model: {e}")
        audio_file_key = state.get('audio_file_key')
        if audio_file_key:
            roll_back_file_from_processing(audio_file_key)
            logger.info(f"Rolled back file {audio_file_key}")
        else:
            audio_file_key = state.get('audio_file_key') or (state.get('audio_files', {}).get('key') if state.get('audio_files') else None)
            if audio_file_key:
                roll_back_file_from_processing(audio_file_key)
                logger.info(f"Rolled back file {audio_file_key}")
            else:
                logger.warning("Could not determine audio_file_key for rollback")
        return {
            **state,
            "messages": state.get("messages", []) + [AIMessage(content=f"Error: {str(e)}")]
        }

def Summarization_Model_Should_Continue(state:AgentState):
    """check if the last message contains tool calls"""
    messages = state.get('messages', [])
    if not messages:
        return END
    last_message = messages[-1]
    if isinstance(last_message, AIMessage) and getattr(last_message, "tool_calls", None):
        return 'Summarization_Model_Tools'
    return END

Summarization_Agent=(StateGraph(AgentState)
            .add_node("Summarization_Model", Summarization_Model)
            .add_node("Summarization_Model_Tools", ToolNode(Summarization_Model_Tools))
            .add_conditional_edges(
                'Summarization_Model',
                Summarization_Model_Should_Continue,
                {
                    'Summarization_Model_Tools': 'Summarization_Model_Tools',
                    END: END
                }
            )
            .add_edge(START, "Summarization_Model")
            .add_edge("Summarization_Model_Tools", "Summarization_Model")
            ).compile()


Insights_Model_Tools = [update_state_Insights_Agent]

def Insights_Model(state:AgentState):
    try:
        start_time = datetime.now()
        workflow_id = state.get('workflow_id', None)
        audio_file_key = state.get('audio_file_key', None)
        
        # Get transcription from state to check for unavailable calls
        transcription = state.get('transcription', '')
        
        # Check if this is an unavailable call and provide appropriate insights
        if transcription and 'unavailable' in transcription.lower():
            logger.info("🔍 Detected unavailable call - providing default insights")
            
            # Provide appropriate insights for unavailable calls
            default_insights = {
                **state,
                "topic": "Other: Unavailable Call",
                "key_points": "• Person called was unavailable\n• No conversation occurred\n• Call attempt made",
                "action_items": "• Follow up with customer at a later time\n• Try alternative contact methods",
                "messages": [{"role": "assistant", "content": "Unavailable call processed"}]
            }
            
            if workflow_id:
                log_agent_execution(workflow_id, "Insights_Model", "completed", 0.1)
                logger.info(f"✅ Insights_Model completed for unavailable call - workflow_id: {workflow_id}")
            
            return default_insights
        
        # For regular calls, use LLM to extract insights
        messages = state.get('messages', [])
        messages = messages + [SystemMessage(content=Insights_Agent_Template)]
        messages = messages[-8:] if len(messages) > 8 else messages
        logger.info(f"🔍 INSIGHTS_AGENT - Workflow_id: {workflow_id}")
        logger.info(f"🔍 INSIGHTS_AGENT - Audio_file_key: {audio_file_key}")

        
        filtered_messages = []
        for i, msg in enumerate(messages):
            if hasattr(msg, 'type') and msg.type == 'tool':
                if i > 0 and hasattr(messages[i-1], 'tool_calls') and messages[i-1].tool_calls:
                    filtered_messages.append(msg)
            else:
                filtered_messages.append(msg)
        
        messages = filtered_messages 
        Insights_Model_With_Tools = model_gpt4.bind_tools(Insights_Model_Tools) 
        response = safe_model_invoke(Insights_Model_With_Tools, messages)
        end_time = datetime.now()
        if isinstance(response, Command):   
            new_messages = messages + [response]
            if "messages" in response.update:
                new_messages = new_messages + response.update["messages"]
            
            new_state = {**state, "messages": new_messages}
            for key, value in response.update.items():
                if key != "messages":  
                    new_state[key] = value  
            
            execution_time = (end_time - start_time).total_seconds()
            if workflow_id:
                log_agent_execution(workflow_id, "Insights_Model", "completed", execution_time)
                logger.info(f"✅ Insights_Model completed in {execution_time:.2f}s - workflow_id: {workflow_id}")
            return new_state
        else:
            execution_time = (end_time - start_time).total_seconds()
            logger.info(f"🔍 Insights_Model - No tool calls made, response type: {type(response)}")
            logger.info(f"🔍 Insights_Model - Response content: {getattr(response, 'content', 'No content')}")
            
            if not state.get('topic') and not state.get('key_points') and not state.get('action_items'):
                logger.info("🔍 Insights_Model - No insights extracted, providing defaults")
                
                transcription = state.get('transcription', '')
                if 'unavailable' in transcription.lower():
                    topic = "Other: Unavailable Call"
                    key_points = "• Person called was unavailable\n• No conversation occurred\n• Call attempt made"
                    action_items = "• Follow up with customer at a later time\n• Try alternative contact methods"
                else:
                    topic = "Other: General Call"
                    key_points = f"• Call content: {transcription}\n• Conversation occurred\n• Information exchanged"
                    action_items = "• Review call details\n• Follow up as needed"
                
                default_state = {
                    **state,
                    "topic": topic,
                    "key_points": key_points,
                    "action_items": action_items,
                    "messages": messages + [response]
                }
            else:
                default_state = {**state, "messages": messages + [response]}
            
            if workflow_id:
                log_agent_execution(workflow_id, "Insights_Model", "completed", execution_time)
                logger.info(f"✅ Insights_Model completed in {execution_time:.2f}s - workflow_id: {workflow_id}")
            return default_state
    except Exception as e:
        end_time = datetime.now()
        execution_time = (end_time - start_time).total_seconds()
        if workflow_id:
            log_agent_execution(workflow_id, "Insights_Model", "failed", execution_time, str(e))
            logger.error(f"❌ Insights_Model failed after {execution_time:.2f}s - workflow_id: {workflow_id} - Error: {str(e)}")
        logger.error(f"Error in Insights_Model: {e}")
        audio_file_key = state.get('audio_file_key')
        if audio_file_key:
            roll_back_file_from_processing(audio_file_key)
            logger.info(f"Rolled back file {audio_file_key}")
        else:
            audio_file_key = state.get('audio_file_key')
            if not audio_file_key:
                audio_files = state.get('audio_files')
                if audio_files and hasattr(audio_files, 'key'):
                    audio_file_key = audio_files.key
            if audio_file_key:
                roll_back_file_from_processing(audio_file_key)
                logger.info(f"Rolled back file {audio_file_key}")
            else:
                logger.warning("Could not determine audio_file_key for rollback")
        return {
            **state,
            "messages": state.get("messages", []) + [AIMessage(content=f"Error: {str(e)}")]
        }

def Insights_Model_Should_Continue(state:AgentState):
    """check if the last message contains tool calls"""
    messages = state.get('messages', [])
    if not messages:
        return END
    last_message = messages[-1]
    if isinstance(last_message, AIMessage) and getattr(last_message, "tool_calls", None):
        return 'Insights_Model_Tools'
    return END

Insights_Agent=(StateGraph(AgentState)
            .add_node("Insights_Model", Insights_Model)
            .add_node("Insights_Model_Tools", ToolNode(Insights_Model_Tools))
            .add_conditional_edges(
                'Insights_Model',
                Insights_Model_Should_Continue,
                {
                    'Insights_Model_Tools': 'Insights_Model_Tools',
                    END: END
                }
            )
            .add_edge(START, "Insights_Model")
            .add_edge("Insights_Model_Tools", "Insights_Model")
            ).compile()


def Sentiment_Analysis_Model(state: AgentState):
    """
    Direct sentiment analysis function (agenticai style) - no LLM interaction
    Analyzes sentiment of transcription and summary text using pre-trained models
    """
    start_time = datetime.now()
    workflow_id = state.get('workflow_id', None)
    audio_file_key = state.get('audio_file_key', None)
    
    try:
        logger.info(f"🔍 Starting Sentiment_Analysis_Model - workflow_id: {workflow_id}")
        
        # Get transcription and summary from state for sentiment analysis
        transcription = state.get('transcription', '')
        summary = state.get('summary', '')
        
        # Combine transcription and summary for analysis
        text_to_analyze = f"Transcription: {transcription}\nSummary: {summary}"
        
        if not text_to_analyze.strip() or text_to_analyze.strip() in ['Transcription: \nSummary:', 'Transcription: \nSummary: ']:
            logger.warning("⚠️ No transcription or summary available for sentiment analysis")
            sentiment_label = "neutral"
            sentiment_scores = 0.5
        else:
            sentiment_result = sentiment_analysis(text_to_analyze)
            
            if isinstance(sentiment_result, list) and len(sentiment_result) > 0:
                result = sentiment_result[0]
                sentiment_label = result.get('label', 'neutral')
                sentiment_scores = result.get('score', 0.5)
            elif isinstance(sentiment_result, dict):
                if 'error' in sentiment_result:
                    logger.error(f"❌ Sentiment analysis error: {sentiment_result['error']}")
                    sentiment_label = "neutral"
                    sentiment_scores = 0.5
                else:
                    sentiment_label = sentiment_result.get('label', 'neutral')
                    sentiment_scores = sentiment_result.get('score', 0.5)
            else:
                logger.warning("⚠️ Unexpected sentiment analysis result format")
                sentiment_label = "neutral"
                sentiment_scores = 0.5
        
        logger.info(f"✅ Sentiment analysis completed - Label: {sentiment_label}, Score: {sentiment_scores}")
        
        new_state = {
            **state,
            "sentiment_label": sentiment_label,
            "sentiment_scores": sentiment_scores,
            "processing_status": "sentiment_analysis_complete"
        }
        
        end_time = datetime.now()
        execution_time = (end_time - start_time).total_seconds()
        
        if workflow_id:
            log_agent_execution(workflow_id, "Sentiment_Analysis_Model", "completed", execution_time)
            logger.info(f"✅ Sentiment_Analysis_Model completed in {execution_time:.2f}s - workflow_id: {workflow_id}")
        
        return new_state
        
    except Exception as e:
        end_time = datetime.now()
        execution_time = (end_time - start_time).total_seconds()
        
        if workflow_id:
            log_agent_execution(workflow_id, "Sentiment_Analysis_Model", "failed", execution_time, str(e))
            logger.error(f"❌ Sentiment_Analysis_Model failed after {execution_time:.2f}s - workflow_id: {workflow_id} - Error: {str(e)}")
        
        return {
            **state,
            "sentiment_label": "neutral",
            "sentiment_scores": 0.5,
            "processing_status": "sentiment_analysis_failed"
        }
    
Sentiment_Analysis_Agent = Sentiment_Analysis_Model


def Storage_Agent_Model(state:AgentState):
    """Storage agent - agenticai style (direct function calls)"""
    start_time = datetime.now()
    workflow_id = state.get('workflow_id', None)
    audio_file_key = state.get('audio_file_key', None)
    
    try:
        logger.info(f"🚀 Starting storage for: {audio_file_key}")
        logger.info(f"🔧 Workflow_id: {workflow_id}")
        
        audio_files = state.get('audio_files')
        transcription = state.get('transcription')
        translation = state.get('translation')
        summary = state.get('summary')
        topic = state.get('topic')
        key_points = state.get('key_points')
        action_items = state.get('action_items')
        sentiment_label = state.get('sentiment_label')
        sentiment_scores = state.get('sentiment_scores')
        
        
        if isinstance(audio_files, str):
            audio_files_data = json.loads(audio_files)
            file_size = audio_files_data.get('file', {}).get('size', 0)
            last_modified = audio_files_data.get('file', {}).get('last_modified', '')
        else:
            file_size = 0
            last_modified = ''
            
        try:
            insert_result = insert_data_all(
                workflow_id=workflow_id,
                file_key=audio_file_key,
                file_size=file_size,
                uploaded_at=last_modified,
                transcription=transcription,
                translation=translation,
                topic=topic,
                summary=summary,
                key_points=key_points,
                action_items=action_items,
                sentiment_label=sentiment_label,
                sentiment_scores=sentiment_scores
            )
            
            if not insert_result:
                raise Exception("Failed to insert data to database")
            logger.info(f"✅ Data inserted successfully for {audio_file_key}")
        except Exception as e:
            logger.error(f"❌ Failed to insert data to database: {e}")
        
        move_result = move_file_to_processed(file_key=audio_file_key)
        if move_result:
            logger.info(f"✅ File moved to processed folder: {audio_file_key}")
        
        new_state = {
            **state,
            "processing_complete": True,
            "processing_status": "completed",
            "messages": state.get("messages", []) + [AIMessage(content=f"Storage completed for {audio_file_key}")]
        }

        end_time = datetime.now()
        execution_time = (end_time - start_time).total_seconds()
        log_agent_execution(workflow_id, "Storage_Agent", "completed", execution_time)
        logger.info(f"✅ Storage_Agent completed in {execution_time:.2f}s - workflow_id: {workflow_id}")
        
        return new_state

    except Exception as e:
        end_time = datetime.now()
        execution_time = (end_time - start_time).total_seconds()
        
        log_agent_execution(workflow_id, "Storage_Agent", "failed", execution_time, str(e))
        logger.error(f"❌ Storage_Agent failed after {execution_time:.2f}s - workflow_id: {workflow_id} - Error: {str(e)}")
        
        if audio_file_key:
            roll_back_file_from_processing(audio_file_key)
            logger.info(f"🔄 Rolled back file {audio_file_key}")
        
        return {
            **state,
            "processing_status": "failed",
            "error_message": str(e),
            "messages": state.get("messages", []) + [AIMessage(content=f"Error: {str(e)}")]
        }
Storage_Agent = Storage_Agent_Model

workflow=StateGraph(AgentState)
workflow.add_node("Ingestion_Agent", Ingestion_Agent)
workflow.add_node("Speech_Agent", Speech_Agent)
workflow.add_node("Summarization_Agent", Summarization_Agent)
workflow.add_node("Insights_Agent", Insights_Agent)
workflow.add_node("Sentiment_Analysis_Agent", Sentiment_Analysis_Agent)
workflow.add_node("Storage_Agent", Storage_Agent)

workflow.add_edge(START, "Ingestion_Agent")
workflow.add_edge("Ingestion_Agent", "Speech_Agent")
workflow.add_edge('Speech_Agent', "Summarization_Agent")    
workflow.add_edge("Summarization_Agent", "Insights_Agent")
workflow.add_edge("Insights_Agent", "Sentiment_Analysis_Agent")
workflow.add_edge("Sentiment_Analysis_Agent", "Storage_Agent")
workflow.add_edge("Storage_Agent", END)

graph=workflow.compile()

 