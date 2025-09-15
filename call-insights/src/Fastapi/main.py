from fastapi import FastAPI
from pydantic import BaseModel
from langchain_core.messages import HumanMessage
from src.utils.s3_utils import check_if_file_is_processed
import asyncio
from src.graph import graph
from datetime import datetime

max_concurrent = 1  
processing_semaphore = asyncio.Semaphore(max_concurrent)

app= FastAPI()

class Process(BaseModel):
    audio_file_key: str

@app.get("/")
def index():
    return {"message": "Call Insights API", "version": "1.0", "endpoints": ["/process", "/process/batch", "/status"]}

@app.post("/process")
async def process(request: Process):

    async with processing_semaphore:
        try:
            # Check if file is already processed to prevent duplicate processing
            if check_if_file_is_processed(request.audio_file_key):
                print(f"⚠️ FastAPI: File {request.audio_file_key} is already processed, skipping")
                return {
                    "status": "already_processed",
                    "message": f"File {request.audio_file_key} has already been processed",
                    "audio_file_key": request.audio_file_key
                }
            
            initial_state = {
                "messages": [HumanMessage(content=request.audio_file_key)],
                "audio_file_key": request.audio_file_key 
            }
            
            print(f"🔧 FastAPI: Setting initial messages: {request.audio_file_key}")
            print(f"🔧 FastAPI: Setting audio_file_key: {request.audio_file_key}")
            print(f"🔍 FastAPI: Initial state: {initial_state}")

            response = await graph.ainvoke(initial_state)
            return response
            
        except Exception as e:
            print(f"❌ FastAPI: Error processing {request.audio_file_key}: {str(e)}")
            return {
                "status": "error",
                "message": f"Error processing file: {str(e)}",
                "audio_file_key": request.audio_file_key
            }




    