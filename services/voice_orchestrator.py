#!/usr/bin/env python3
"""
Enhanced Voice Orchestrator for Turkish LLM
Comprehensive voice processing pipeline with WebSocket support, STT, TTS, and LLM integration.
"""

import asyncio
import json
import logging
import websockets
import websockets.server
import numpy as np
import wave
import threading
import queue
import time
from typing import Dict, List, Optional, Any, Callable, Union
from pathlib import Path
import tempfile
import os
import re
from dataclasses import dataclass, asdict
import aiohttp
import aiofiles
from datetime import datetime

try:
    import whisper
except ImportError:
    whisper = None

try:
    import pyaudio
except ImportError:
    pyaudio = None

try:
    import pyttsx3
except ImportError:
    pyttsx3 = None

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class AudioConfig:
    """Audio configuration for voice processing."""

    sample_rate: int = 16000
    channels: int = 1
    chunk_size: int = 1024
    format: int = pyaudio.paInt16 if pyaudio else None
    max_seconds: int = 30
    silence_threshold: float = 0.01
    silence_duration: float = 2.0


@dataclass
class VoiceConfig:
    """Voice processing configuration."""

    stt_model: str = "base"  # Whisper model size
    tts_voice: str = "tr_TR"  # Turkish voice
    llm_endpoint: str = "http://localhost:8000/inference"
    websocket_host: str = "localhost"
    websocket_port: int = 8765
    enable_ssml: bool = True
    enable_emotion: bool = True
    max_audio_duration: int = 30
    audio_output_dir: str = "audio_output"
    supported_formats: List[str] = None

    def __post_init__(self):
        if self.supported_formats is None:
            self.supported_formats = ["wav", "mp3", "ogg"]


class AudioProcessor:
    """Enhanced audio processing with better error handling."""

    def __init__(self, config: AudioConfig):
        self.config = config
        self.audio = None
        self.recording = False
        self.audio_queue = queue.Queue()
        self.recording_thread = None

        if pyaudio:
            try:
                self.audio = pyaudio.PyAudio()
                logger.info(" PyAudio initialized successfully")
            except Exception as e:
                logger.error(f" Failed to initialize PyAudio: {e}")
                self.audio = None
        else:
            logger.warning(" PyAudio not available, audio recording disabled")

    def start_recording(self) -> bool:
        """Start audio recording."""
        if not self.audio:
            logger.error(" Audio system not available")
            return False

        try:
            self.recording = True
            self.recording_thread = threading.Thread(target=self._record_audio)
            self.recording_thread.start()
            logger.info(" Started audio recording")
            return True
        except Exception as e:
            logger.error(f" Failed to start recording: {e}")
            return False

    def stop_recording(self) -> Optional[bytes]:
        """Stop recording and return audio data."""
        if not self.recording:
            return None

        self.recording = False
        if self.recording_thread:
            self.recording_thread.join(timeout=5.0)

        # Collect audio data
        audio_data = []
        while not self.audio_queue.empty():
            try:
                audio_data.append(self.audio_queue.get_nowait())
            except queue.Empty:
                break

        if audio_data:
            return b"".join(audio_data)
        return None

    def _record_audio(self):
        """Internal recording method."""
        if not self.audio:
            return

        try:
            stream = self.audio.open(
                format=self.config.format,
                channels=self.config.channels,
                rate=self.config.sample_rate,
                input=True,
                frames_per_buffer=self.config.chunk_size,
            )

            while self.recording:
                try:
                    data = stream.read(
                        self.config.chunk_size, exception_on_overflow=False
                    )
                    self.audio_queue.put(data)
                except Exception as e:
                    logger.error(f" Recording error: {e}")
                    break

            stream.stop_stream()
            stream.close()

        except Exception as e:
            logger.error(f" Audio stream error: {e}")

    def save_audio(self, audio_data: bytes, filename: str) -> bool:
        """Save audio data to file."""
        try:
            with wave.open(filename, "wb") as wf:
                wf.setnchannels(self.config.channels)
                wf.setsampwidth(
                    self.audio.get_sample_size(self.config.format) if self.audio else 2
                )
                wf.setframerate(self.config.sample_rate)
                wf.writeframes(audio_data)
            return True
        except Exception as e:
            logger.error(f" Failed to save audio: {e}")
            return False

    def cleanup(self):
        """Cleanup audio resources."""
        self.recording = False
        if self.recording_thread and self.recording_thread.is_alive():
            self.recording_thread.join(timeout=2.0)

        if self.audio:
            try:
                self.audio.terminate()
            except Exception as e:
                logger.error(f" Audio cleanup error: {e}")


class SpeechToText:
    """Enhanced Speech-to-Text with Whisper integration."""

    def __init__(self, model_name: str = "base"):
        self.model = None
        self.model_name = model_name
        self._load_model()

    def _load_model(self):
        """Load Whisper model."""
        if not whisper:
            logger.error(" Whisper not available")
            return

        try:
            self.model = whisper.load_model(self.model_name)
            logger.info(f" Loaded Whisper model: {self.model_name}")
        except Exception as e:
            logger.error(f" Failed to load Whisper model: {e}")

    async def transcribe(self, audio_file: str, language: str = "tr") -> Optional[str]:
        """Transcribe audio file to text."""
        if not self.model:
            logger.error(" STT model not available")
            return None

        try:
            # Run transcription in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None, lambda: self.model.transcribe(audio_file, language=language)
            )

            text = result.get("text", "").strip()
            logger.info(f" Transcribed: {text[:100]}...")
            return text

        except Exception as e:
            logger.error(f" Transcription failed: {e}")
            return None

    def transcribe_bytes(
        self, audio_data: bytes, temp_dir: str = "/tmp"
    ) -> Optional[str]:
        """Transcribe audio from bytes."""
        if not self.model:
            return None

        try:
            # Save to temporary file
            temp_file = os.path.join(temp_dir, f"temp_audio_{int(time.time())}.wav")

            with wave.open(temp_file, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(16000)
                wf.writeframes(audio_data)

            # Transcribe
            result = self.model.transcribe(temp_file, language="tr")
            text = result.get("text", "").strip()

            # Cleanup
            try:
                os.remove(temp_file)
            except:
                pass

            return text

        except Exception as e:
            logger.error(f" Bytes transcription failed: {e}")
            return None


class TextToSpeech:
    """Enhanced Text-to-Speech with multiple backend support."""

    def __init__(self, voice: str = "tr_TR"):
        self.voice = voice
        self.engine = None
        self._initialize_engine()

    def _initialize_engine(self):
        """Initialize TTS engine."""
        if pyttsx3:
            try:
                self.engine = pyttsx3.init()
                self.engine.setProperty("rate", 150)
                self.engine.setProperty("volume", 0.9)
                logger.info(" TTS engine initialized")
            except Exception as e:
                logger.error(f" Failed to initialize TTS: {e}")
        else:
            logger.warning(" pyttsx3 not available, TTS disabled")

    async def synthesize(self, text: str, output_file: str, ssml: bool = False) -> bool:
        """Synthesize text to speech."""
        if not self.engine:
            logger.error(" TTS engine not available")
            return False

        try:
            # Clean SSML tags if present
            if ssml:
                text = re.sub(r"<[^>]+>", "", text)

            # Run synthesis in thread pool
            loop = asyncio.get_event_loop()
            success = await loop.run_in_executor(
                None, self._synthesize_sync, text, output_file
            )

            return success

        except Exception as e:
            logger.error(f" TTS synthesis failed: {e}")
            return False

    def _synthesize_sync(self, text: str, output_file: str) -> bool:
        """Synchronous synthesis method."""
        try:
            self.engine.save_to_file(text, output_file)
            self.engine.runAndWait()
            return os.path.exists(output_file)
        except Exception as e:
            logger.error(f" Sync synthesis failed: {e}")
            return False


class SSMLProcessor:
    """SSML processing for enhanced speech synthesis."""

    def __init__(self, enable_emotion: bool = True):
        self.enable_emotion = enable_emotion
        self.emotion_patterns = {
            r"\b(mutlu|sevinçli|neşeli)\b": '<prosody rate="fast" pitch="+10%">\\1</prosody>',
            r"\b(üzgün|kederli|melankolik)\b": '<prosody rate="slow" pitch="-10%">\\1</prosody>',
            r"\b(öfkeli|sinirli|kızgın)\b": '<prosody rate="fast" volume="loud">\\1</prosody>',
            r"\b(sakin|huzurlu|dingin)\b": '<prosody rate="slow" volume="soft">\\1</prosody>',
        }

    def enhance_text(self, text: str) -> str:
        """Enhance text with SSML tags."""
        if not self.enable_emotion:
            return text

        enhanced = text

        try:
            # Apply emotion patterns
            for pattern, replacement in self.emotion_patterns.items():
                enhanced = re.sub(pattern, replacement, enhanced, flags=re.IGNORECASE)

            # Add pauses for punctuation
            enhanced = re.sub(r"[.!?]", r'\\g<0><break time="500ms"/>', enhanced)
            enhanced = re.sub(r"[,;:]", r'\\g<0><break time="200ms"/>', enhanced)

            # Wrap in SSML
            enhanced = f"<speak>{enhanced}</speak>"

        except Exception as e:
            logger.error(f" SSML enhancement failed: {e}")
            return text

        return enhanced


class LLMClient:
    """Enhanced LLM client with better error handling and retry logic."""

    def __init__(self, endpoint: str):
        self.endpoint = endpoint
        self.session = None
        self.retry_count = 3
        self.timeout = 30

    async def __aenter__(self):
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.timeout)
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    async def generate_response(
        self, text: str, sector_hint: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """Generate response from LLM with retry logic."""
        if not self.session:
            logger.error(" LLM client session not initialized")
            return None

        payload = {
            "text": text,
            "max_length": 512,
            "temperature": 0.7,
            "top_p": 0.9,
            "use_sector_routing": True,
        }

        if sector_hint:
            payload["sector_hint"] = sector_hint

        for attempt in range(self.retry_count):
            try:
                async with self.session.post(self.endpoint, json=payload) as response:
                    if response.status == 200:
                        result = await response.json()
                        logger.info(f" LLM response received (attempt {attempt + 1})")
                        return result
                    else:
                        logger.warning(
                            f" LLM request failed with status {response.status} (attempt {attempt + 1})"
                        )

            except asyncio.TimeoutError:
                logger.warning(f" LLM request timeout (attempt {attempt + 1})")
            except Exception as e:
                logger.error(f" LLM request error (attempt {attempt + 1}): {e}")

            if attempt < self.retry_count - 1:
                await asyncio.sleep(2**attempt)  # Exponential backoff

        logger.error(" All LLM request attempts failed")
        return None


class VoiceOrchestrator:
    """Enhanced Voice Orchestrator with comprehensive error handling."""

    def __init__(self, config: VoiceConfig):
        self.config = config
        self.audio_config = AudioConfig()

        # Initialize components
        self.audio_processor = AudioProcessor(self.audio_config)
        self.stt = SpeechToText(config.stt_model)
        self.tts = TextToSpeech(config.tts_voice)
        self.ssml_processor = SSMLProcessor(config.enable_emotion)

        # Create output directory
        os.makedirs(config.audio_output_dir, exist_ok=True)

        # Statistics
        self.stats = {
            "sessions": 0,
            "messages_processed": 0,
            "errors": 0,
            "start_time": datetime.now(),
        }

        logger.info(" Voice Orchestrator initialized")

    async def start_websocket_server(self):
        """Start WebSocket server for voice interactions."""

        async def handle_client(websocket, path):
            client_id = f"{websocket.remote_address[0]}:{websocket.remote_address[1]}"
            logger.info(f" Client connected: {client_id}")
            self.stats["sessions"] += 1

            try:
                await self._handle_voice_session(websocket, client_id)
            except websockets.exceptions.ConnectionClosed:
                logger.info(f" Client disconnected: {client_id}")
            except Exception as e:
                logger.error(f" Session error for {client_id}: {e}")
                self.stats["errors"] += 1

        logger.info(
            f" Starting WebSocket server on {self.config.websocket_host}:{self.config.websocket_port}"
        )

        return await websockets.serve(
            handle_client, self.config.websocket_host, self.config.websocket_port
        )

    async def _handle_voice_session(self, websocket, client_id: str):
        """Handle individual voice session."""
        async for message in websocket:
            try:
                data = json.loads(message)
                message_type = data.get("type")

                if message_type == "start_recording":
                    await self._handle_start_recording(websocket, data)
                elif message_type == "stop_recording":
                    await self._handle_stop_recording(websocket, data)
                elif message_type == "audio_data":
                    await self._handle_audio_data(websocket, data)
                elif message_type == "text_input":
                    await self._handle_text_input(websocket, data)
                elif message_type == "get_stats":
                    await self._handle_get_stats(websocket)
                else:
                    await websocket.send(
                        json.dumps(
                            {
                                "type": "error",
                                "message": f"Unknown message type: {message_type}",
                            }
                        )
                    )

                self.stats["messages_processed"] += 1

            except json.JSONDecodeError:
                await websocket.send(
                    json.dumps({"type": "error", "message": "Invalid JSON format"})
                )
            except Exception as e:
                logger.error(f" Message handling error: {e}")
                await websocket.send(json.dumps({"type": "error", "message": str(e)}))

    async def _handle_start_recording(self, websocket, data):
        """Handle start recording request."""
        success = self.audio_processor.start_recording()
        await websocket.send(
            json.dumps({"type": "recording_started", "success": success})
        )

    async def _handle_stop_recording(self, websocket, data):
        """Handle stop recording and process audio."""
        audio_data = self.audio_processor.stop_recording()

        if not audio_data:
            await websocket.send(
                json.dumps({"type": "error", "message": "No audio data received"})
            )
            return

        # Process the audio
        await self._process_audio_pipeline(websocket, audio_data)

    async def _handle_audio_data(self, websocket, data):
        """Handle direct audio data."""
        try:
            # Decode base64 audio data if present
            import base64

            audio_b64 = data.get("audio")
            if audio_b64:
                audio_data = base64.b64decode(audio_b64)
                await self._process_audio_pipeline(websocket, audio_data)
            else:
                await websocket.send(
                    json.dumps({"type": "error", "message": "No audio data in message"})
                )
        except Exception as e:
            await websocket.send(
                json.dumps(
                    {"type": "error", "message": f"Audio processing failed: {str(e)}"}
                )
            )

    async def _handle_text_input(self, websocket, data):
        """Handle direct text input (bypass STT)."""
        text = data.get("text", "").strip()
        if not text:
            await websocket.send(
                json.dumps({"type": "error", "message": "No text provided"})
            )
            return

        # Process through LLM and TTS
        await self._process_text_pipeline(websocket, text, data.get("sector_hint"))

    async def _handle_get_stats(self, websocket):
        """Handle statistics request."""
        stats = self.stats.copy()
        stats["uptime_seconds"] = (datetime.now() - stats["start_time"]).total_seconds()
        stats["start_time"] = stats["start_time"].isoformat()

        await websocket.send(json.dumps({"type": "stats", "data": stats}))

    async def _process_audio_pipeline(self, websocket, audio_data: bytes):
        """Complete audio processing pipeline: STT -> LLM -> TTS."""
        try:
            # Step 1: Speech to Text
            await websocket.send(
                json.dumps({"type": "status", "message": "Transcribing audio..."})
            )

            text = self.stt.transcribe_bytes(audio_data)
            if not text:
                await websocket.send(
                    json.dumps(
                        {"type": "error", "message": "Failed to transcribe audio"}
                    )
                )
                return

            await websocket.send(json.dumps({"type": "transcription", "text": text}))

            # Continue with text processing
            await self._process_text_pipeline(websocket, text)

        except Exception as e:
            logger.error(f" Audio pipeline error: {e}")
            await websocket.send(
                json.dumps(
                    {"type": "error", "message": f"Audio processing failed: {str(e)}"}
                )
            )

    async def _process_text_pipeline(
        self, websocket, text: str, sector_hint: Optional[str] = None
    ):
        """Text processing pipeline: LLM -> TTS."""
        try:
            # Step 2: LLM Processing
            await websocket.send(
                json.dumps({"type": "status", "message": "Generating response..."})
            )

            async with LLMClient(self.config.llm_endpoint) as llm_client:
                response = await llm_client.generate_response(text, sector_hint)

                if not response:
                    await websocket.send(
                        json.dumps(
                            {
                                "type": "error",
                                "message": "Failed to generate LLM response",
                            }
                        )
                    )
                    return

                generated_text = response.get("generated_text", "")
                if not generated_text:
                    await websocket.send(
                        json.dumps(
                            {"type": "error", "message": "Empty response from LLM"}
                        )
                    )
                    return

                await websocket.send(
                    json.dumps(
                        {
                            "type": "llm_response",
                            "text": generated_text,
                            "sector": response.get("sector", "general"),
                            "confidence": response.get("confidence", 0.0),
                        }
                    )
                )

                # Step 3: Text to Speech
                await websocket.send(
                    json.dumps({"type": "status", "message": "Synthesizing speech..."})
                )

                timestamp = int(time.time())
                audio_file = os.path.join(
                    self.config.audio_output_dir, f"response_{timestamp}.wav"
                )

                success = await self.tts.synthesize(
                    generated_text, audio_file, ssml=self.config.enable_ssml
                )

                if success:
                    # Send audio file path (in production, you might want to serve the file or encode it)
                    await websocket.send(
                        json.dumps(
                            {
                                "type": "audio_response",
                                "audio_file": audio_file,
                                "text": generated_text,
                            }
                        )
                    )
                else:
                    await websocket.send(
                        json.dumps(
                            {
                                "type": "response_complete",
                                "text": generated_text,
                                "audio_file": None,
                                "message": "Text response generated, but audio synthesis failed",
                            }
                        )
                    )

        except Exception as e:
            logger.error(f" Text pipeline error: {e}")
            await websocket.send(
                json.dumps(
                    {"type": "error", "message": f"Text processing failed: {str(e)}"}
                )
            )

    async def start(self):
        """Start the voice orchestrator."""
        logger.info(" Starting Voice Orchestrator...")

        # Start WebSocket server
        server = await self.start_websocket_server()
        logger.info(
            f" Voice Orchestrator running on ws://{self.config.websocket_host}:{self.config.websocket_port}"
        )

        # Keep the server running
        await server.wait_closed()

    def cleanup(self):
        """Cleanup resources."""
        self.audio_processor.cleanup()
        logger.info(" Voice Orchestrator cleaned up")


async def main():
    """Main function."""
    config = VoiceConfig()
    orchestrator = VoiceOrchestrator(config)

    try:
        await orchestrator.start()
    except KeyboardInterrupt:
        logger.info(" Shutting down...")
    finally:
        orchestrator.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
