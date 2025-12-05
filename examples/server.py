from fastapi import FastAPI, WebSocket, UploadFile, File, HTTPException, Request, Body
from fastapi.middleware.cors import CORSMiddleware

import numpy as np
import soundfile as sf
import base64
import os
from typing import Dict, List, Optional
import signal
import sys

from thestage_speechkit.streaming import StreamingPipeline

app = FastAPI()
streaming_manager = None


class StreamingManager:
    def __init__(self):
        self.active_sessions: Dict[str, dict] = {}
        self.streaming_pipe = None
        self.initialization_time = None
        self.model_status = "not_initialized"
        self.model_error = None
    
    async def create_session(self, session_id: str) -> None:
        if self.model_status != "ready":
            if not self.init_streaming_backend():
                if self.model_error:
                    raise HTTPException(status_code=500, detail=f"Failed to initialize model: {self.model_error}")
                else:
                    raise HTTPException(status_code=401, detail="Invalid token or model initialization failed")
        
        self.active_sessions[session_id] = {'is_active': True}

    def init_streaming_backend(self) -> bool:
        self.model_status = "initializing"
        self.model_error = None
        self.streaming_pipe = StreamingPipeline(
            model='TheStageAI/thewhisper-large-v3-turbo',
            chunk_length_s=15,
            platform='apple',
            agreement_history_size=5,
            agreement_majority_threshold=2,
        )
        print("Streaming backend initialized")
        return True

    async def end_session(self, session_id: str) -> None:
        if session_id in self.active_sessions:
            session = self.active_sessions[session_id]
            session['is_active'] = False
            del self.active_sessions[session_id]
            # self.streaming_pipe.clear()
    
    async def add_audio_chunk(self, session_id: str, audio_data: np.ndarray) -> None:
        if session_id not in self.active_sessions:
            raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
        
        self.streaming_pipe.add_new_chunk(audio_data)
    
    async def process_chunk(self, session_id: str) -> List[dict]:
        if session_id not in self.active_sessions:
            raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
            
        return self.streaming_pipe.process_new_chunk()
    
    async def clear_session(self, session_id: str) -> None:
        if session_id not in self.active_sessions:
            raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
        # self.streaming_pipe.clear()
    
    def cleanup(self):
        """Cleanup resources before shutdown"""
        print("Cleaning up streaming manager resources...")
        # Clean up active sessions
        for session_id in list(self.active_sessions.keys()):
            self.active_sessions[session_id]['is_active'] = False
        self.active_sessions.clear()
        # if self.streaming_pipe:
            # self.streaming_pipe.clear()


@app.post("/session/create/")
async def create_session():
    """Create a new streaming session"""
    global streaming_manager
    session_id = base64.urlsafe_b64encode(os.urandom(16)).decode('ascii')
    await streaming_manager.create_session(session_id)
    return {"session_id": session_id}


@app.post("/session/{session_id}/end")
async def end_session(session_id: str):
    """End a streaming session"""
    global streaming_manager
    await streaming_manager.end_session(session_id)
    return {"status": "success"}


@app.post("/session/{session_id}/add_chunk")
async def add_chunk(session_id: str, audio_data: str):
    """Add a new audio chunk to the session"""
    global streaming_manager
    try:
        audio_np = np.frombuffer(base64.b64decode(audio_data), dtype=np.float32)
        await streaming_manager.add_audio_chunk(session_id, audio_np)
        return {"status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/session/{session_id}/process")
async def process_chunk(session_id: str):
    """Process the current audio chunks and return new words"""
    global streaming_manager
    try:
        words, uncommited_words = await streaming_manager.process_chunk(session_id)
        return {"words": words, "uncommited_words": uncommited_words}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/session/{session_id}/clear")
async def clear_session(session_id: str):
    """Clear the current audio chunks"""
    global streaming_manager
    await streaming_manager.clear_session(session_id)
    return {"status": "success"}


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

server = None

def signal_handler(sig, frame):
    """Handle termination signals"""
    print(f"Received signal {sig}, shutting down...")
    if streaming_manager:
        streaming_manager.cleanup()
    if os.path.exists("/tmp/asr_streaming_server.pid"):
        os.remove("/tmp/asr_streaming_server.pid")
    if server:
        server.should_exit = True
    sys.exit(0)


def main():
    global streaming_manager
    global server
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    with open("/tmp/asr_streaming_server.pid", "w") as f:
        f.write(str(os.getpid()))
    
    import uvicorn

    host = os.getenv("ASR_STREAMING_HOST", "127.0.0.1")
    port = os.getenv("ASR_STREAMING_PORT", 8000)
    
    streaming_manager = StreamingManager()

    print("Server started")
    
    config = uvicorn.Config(app, host=host, port=port, log_level="error", access_log=False)
    server = uvicorn.Server(config)
    server.run()


if __name__ == "__main__":
    main()
