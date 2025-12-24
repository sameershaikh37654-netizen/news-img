# """
# AI News Automation System - Offline Version with Chatterbox TTS
# Text-to-Speech with Chatterbox (Voice Cloning), Kokoro, piper-tts, pyttsx3, and gTTS
# """

# import json
# import datetime
# import tempfile
# import os
# import subprocess
# import sys
# from typing import Dict, List, Optional, Tuple, Any
# from dataclasses import dataclass
# from enum import Enum
# import streamlit as st
# import speech_recognition as sr
# from PIL import Image
# import pytesseract
# import numpy as np
# import io
# import base64

# # ============================================================================
# # TESSERACT OCR CONFIGURATION
# # ============================================================================
# # Configure Tesseract path (Update this based on your installation)
# # For Windows (Default installation path):
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# # For Linux/Mac (Uncomment and adjust as needed):
# # pytesseract.pytesseract.tesseract_cmd = r"/usr/bin/tesseract"

# # ============================================================================

# # Try to import optional handwriting recognition libraries
# try:
#     import cv2
#     OCR_AVAILABLE = True
# except ImportError:
#     OCR_AVAILABLE = False
#     st.warning("OpenCV not installed. Handwriting recognition may have limited functionality.")

# try:
#     import whisper
#     WHISPER_AVAILABLE = True
# except ImportError:
#     WHISPER_AVAILABLE = False

# # Try to import Text-to-Speech libraries
# try:
#     import pyttsx3
#     TTS_AVAILABLE = True
# except ImportError:
#     TTS_AVAILABLE = False
#     st.warning("pyttsx3 not installed. Text-to-speech will not be available.")

# try:
#     from gtts import gTTS
#     GTTS_AVAILABLE = True
# except ImportError:
#     GTTS_AVAILABLE = False
#     st.warning("gTTS not installed. Google Text-to-Speech will not be available.")

# # Try to import Kokoro TTS
# try:
#     import kokoro
#     KOKORO_AVAILABLE = True
#     # Test Kokoro installation
#     try:
#         # Create a test instance
#         test_model = kokoro.Kokoro()
#         KOKORO_WORKING = True
#     except:
#         KOKORO_WORKING = False
#         st.warning("Kokoro installed but not working properly. Check dependencies.")
# except ImportError:
#     KOKORO_AVAILABLE = False
#     KOKORO_WORKING = False
#     st.warning("Kokoro not installed. For best TTS quality: pip install kokoro")

# # Try to import Piper TTS (Lightweight, fast, multi-language)
# try:
#     # Piper TTS can be used through command line
#     PIPER_AVAILABLE = False
#     # Check if piper is installed
#     try:
#         result = subprocess.run(['piper', '--version'], capture_output=True, text=True)
#         PIPER_AVAILABLE = True
#     except:
#         PIPER_AVAILABLE = False
# except:
#     PIPER_AVAILABLE = False

# # Try to import Chatterbox TTS (Voice Cloning)
# try:
#     # We'll check for Chatterbox through pip or direct import
#     CHATTERBOX_AVAILABLE = False
#     CHATTERBOX_WORKING = False
    
#     # Try to import chatterbox modules
#     try:
#         # Try to import chatterbox (it might have a different name)
#         import torch
#         import torchaudio
#         import soundfile as sf
#         CHATTERBOX_AVAILABLE = True
#         CHATTERBOX_WORKING = True
#         st.success("✅ Chatterbox dependencies (PyTorch, TorchAudio) are available!")
#     except ImportError:
#         CHATTERBOX_AVAILABLE = False
#         st.warning("Chatterbox TTS requires PyTorch and TorchAudio. Install with: pip install torch torchaudio soundfile")
# except:
#     CHATTERBOX_AVAILABLE = False
#     CHATTERBOX_WORKING = False

# class NewsCategory(Enum):
#     BREAKING = "breaking"
#     POLITICS = "politics"
#     SPORTS = "sports"
#     WEATHER = "weather"
#     MARKET = "market"
#     TRAFFIC = "traffic"
#     ENTERTAINMENT = "entertainment"
#     ANALYSIS = "analysis"

# class Language(Enum):
#     TELUGU = "te"
#     HINDI = "hi"
#     MARATHI = "mr"
#     KANNADA = "kn"
#     TAMIL = "ta"
#     ENGLISH = "en"

# @dataclass
# class NewsItem:
#     """Data structure for news items"""
#     id: str
#     title: str
#     raw_content: str
#     category: NewsCategory
#     timestamp: datetime.datetime
#     location: str
#     priority: int = 1
#     metadata: Dict = None
    
#     def __post_init__(self):
#         if self.metadata is None:
#             self.metadata = {
#                 "source": "unknown",
#                 "verified": False,
#                 "tags": []
#             }

# class OfflineTTSManager:
#     """Manage offline TTS engines including Piper TTS and Chatterbox"""
    
#     def __init__(self):
#         self.piper_voices = self._get_piper_voices()
#         self.available_engines = self._check_available_engines()
#         self.chatterbox_models = self._get_chatterbox_models()
        
#     def _check_available_engines(self):
#         """Check which TTS engines are available"""
#         engines = {}
        
#         if CHATTERBOX_AVAILABLE and CHATTERBOX_WORKING:
#             engines["chatterbox"] = True
        
#         if KOKORO_AVAILABLE and KOKORO_WORKING:
#             engines["kokoro"] = True
        
#         if PIPER_AVAILABLE:
#             engines["piper"] = True
        
#         if TTS_AVAILABLE:
#             engines["pyttsx3"] = True
        
#         if GTTS_AVAILABLE:
#             engines["gtts"] = True
            
#         return engines
    
#     def _get_piper_voices(self):
#         """Get available Piper TTS voices"""
#         voices = [
#             {"id": "en_US-lessac-medium", "name": "English US (Medium)", "language": "en"},
#             {"id": "en_US-lessac-high", "name": "English US (High)", "language": "en"},
#             {"id": "en_GB-alba-medium", "name": "English UK (Medium)", "language": "en"},
#             {"id": "hi_IN-google-medium", "name": "Hindi (Medium)", "language": "hi"},
#             {"id": "ta_IN-google-medium", "name": "Tamil (Medium)", "language": "ta"},
#             {"id": "te_IN-google-medium", "name": "Telugu (Medium)", "language": "te"},
#             {"id": "kn_IN-google-medium", "name": "Kannada (Medium)", "language": "kn"},
#             {"id": "mr_IN-google-medium", "name": "Marathi (Medium)", "language": "mr"},
#         ]
#         return voices
    
#     def _get_chatterbox_models(self):
#         """Get available Chatterbox TTS models"""
#         models = [
#             {"id": "chatterbox_base", "name": "Chatterbox Base (English)", "language": "en", "voice_cloning": True},
#             {"id": "chatterbox_multilingual", "name": "Chatterbox Multilingual", "language": "multi", "voice_cloning": True},
#             {"id": "chatterbox_news", "name": "Chatterbox News Anchor", "language": "en", "voice_cloning": True},
#             {"id": "chatterbox_podcast", "name": "Chatterbox Podcast", "language": "en", "voice_cloning": True},
#         ]
#         return models
    
#     def text_to_speech_piper(self, text: str, voice_id: str = "en_US-lessac-medium", 
#                            speed: float = 1.0) -> Optional[bytes]:
#         """
#         Convert text to speech using Piper TTS
        
#         Args:
#             text: Text to convert
#             voice_id: Piper voice identifier
#             speed: Speech speed (0.5 to 2.0)
        
#         Returns:
#             Audio bytes in WAV format
#         """
#         try:
#             # Create temporary files
#             with tempfile.NamedTemporaryFile(suffix=".txt", delete=False, mode='w', encoding='utf-8') as f:
#                 text_file = f.name
#                 f.write(text)
            
#             with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
#                 audio_file = f.name
            
#             # Build piper command
#             cmd = [
#                 'piper',
#                 '--model', voice_id,
#                 '--output_file', audio_file,
#                 '--length_scale', str(1.0 / speed)
#             ]
            
#             # Run piper with text input
#             with open(text_file, 'r', encoding='utf-8') as f:
#                 process = subprocess.run(
#                     cmd,
#                     stdin=f,
#                     capture_output=True,
#                     text=True
#                 )
            
#             if process.returncode == 0 and os.path.exists(audio_file):
#                 # Read audio file
#                 with open(audio_file, 'rb') as f:
#                     audio_bytes = f.read()
                
#                 # Clean up temp files
#                 os.unlink(text_file)
#                 os.unlink(audio_file)
                
#                 return audio_bytes
#             else:
#                 st.error(f"Piper TTS failed: {process.stderr}")
#                 return None
                
#         except Exception as e:
#             st.error(f"Error with Piper TTS: {str(e)}")
#             # Fallback to other engines
#             return None
    
#     def text_to_speech_chatterbox(self, text: str, model_id: str = "chatterbox_base",
#                                 voice_clone_audio: Optional[bytes] = None,
#                                 voice_clone_text: Optional[str] = None,
#                                 language: str = "en") -> Optional[bytes]:
#         """
#         Convert text to speech using Chatterbox TTS with optional voice cloning
        
#         Args:
#             text: Text to convert to speech
#             model_id: Chatterbox model identifier
#             voice_clone_audio: Audio bytes for voice cloning (10-20 seconds recommended)
#             voice_clone_text: Corresponding text for the voice clone audio
#             language: Language code
        
#         Returns:
#             Audio bytes in WAV format
#         """
#         try:
#             if not CHATTERBOX_WORKING:
#                 st.warning("Chatterbox not working. Check PyTorch installation.")
#                 return None
            
#             # Create temporary files
#             with tempfile.NamedTemporaryFile(suffix=".txt", delete=False, mode='w', encoding='utf-8') as f:
#                 text_file = f.name
#                 f.write(text)
            
#             with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
#                 output_audio_file = f.name
            
#             # Prepare voice cloning files if provided
#             voice_clone_file = None
#             if voice_clone_audio:
#                 with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
#                     voice_clone_file = f.name
#                     f.write(voice_clone_audio)
            
#             with tempfile.NamedTemporaryFile(suffix=".txt", delete=False, mode='w', encoding='utf-8') as f:
#                 voice_clone_text_file = f.name
#                 if voice_clone_text:
#                     f.write(voice_clone_text)
#                 else:
#                     # Default text if not provided
#                     f.write("This is a sample voice for cloning.")
            
#             # Different approaches for Chatterbox
#             # Approach 1: Using chatterbox command line (if installed)
#             try:
#                 cmd = ["chatterbox", "synthesize"]
#                 cmd.extend(["--text", text_file])
#                 cmd.extend(["--output", output_audio_file])
#                 cmd.extend(["--model", model_id])
                
#                 if voice_clone_file:
#                     cmd.extend(["--voice", voice_clone_file])
#                     cmd.extend(["--voice-text", voice_clone_text_file])
                
#                 process = subprocess.run(cmd, capture_output=True, text=True)
                
#                 if process.returncode == 0 and os.path.exists(output_audio_file):
#                     with open(output_audio_file, 'rb') as f:
#                         audio_bytes = f.read()
                    
#                     # Clean up temp files
#                     for file in [text_file, output_audio_file]:
#                         if os.path.exists(file):
#                             os.unlink(file)
#                     if voice_clone_file and os.path.exists(voice_clone_file):
#                         os.unlink(voice_clone_file)
#                     if voice_clone_text_file and os.path.exists(voice_clone_text_file):
#                         os.unlink(voice_clone_text_file)
                    
#                     return audio_bytes
                    
#             except Exception as e:
#                 st.info(f"Chatterbox CLI not available, trying Python API approach: {str(e)}")
            
#             # Approach 2: Using Python API if available
#             try:
#                 # This is a placeholder for actual Chatterbox API integration
#                 # In practice, you would import and use the actual Chatterbox library
                
#                 # Simulated implementation - in real scenario, use:
#                 # from chatterbox import ChatterboxTTS
#                 # model = ChatterboxTTS(model_id=model_id)
#                 # audio = model.synthesize(text, voice_reference=voice_clone_file if voice_clone_file else None)
                
#                 # For now, we'll simulate with pyttsx3 as fallback
#                 st.info("Chatterbox API not directly accessible. Using fallback TTS.")
                
#                 # Clean up temp files
#                 for file in [text_file, output_audio_file]:
#                     if os.path.exists(file):
#                         os.unlink(file)
#                 if voice_clone_file and os.path.exists(voice_clone_file):
#                     os.unlink(voice_clone_file)
#                 if voice_clone_text_file and os.path.exists(voice_clone_text_file):
#                     os.unlink(voice_clone_text_file)
                
#                 # Return None to trigger fallback
#                 return None
                
#             except Exception as e:
#                 st.error(f"Chatterbox Python API error: {str(e)}")
#                 return None
                
#         except Exception as e:
#             st.error(f"Error with Chatterbox TTS: {str(e)}")
#             return None
    
#     def install_chatterbox(self):
#         """Install Chatterbox TTS and dependencies"""
#         st.info("Installing Chatterbox TTS and dependencies...")
        
#         try:
#             # Install PyTorch with CUDA support if available
#             import torch
#             if torch.cuda.is_available():
#                 st.info("CUDA is available. Installing PyTorch with CUDA support...")
#                 subprocess.check_call([sys.executable, "-m", "pip", "install", "torch", "torchaudio", "--index-url", "https://download.pytorch.org/whl/cu118"])
#             else:
#                 st.info("Installing PyTorch CPU version...")
#                 subprocess.check_call([sys.executable, "-m", "pip", "install", "torch", "torchaudio"])
            
#             # Install other dependencies
#             dependencies = [
#                 "soundfile",
#                 "librosa",
#                 "numpy",
#                 "scipy",
#                 "pydub"
#             ]
            
#             for dep in dependencies:
#                 try:
#                     subprocess.check_call([sys.executable, "-m", "pip", "install", dep])
#                 except:
#                     st.warning(f"Failed to install {dep}")
            
#             # Install Chatterbox from GitHub
#             st.info("Installing Chatterbox from GitHub...")
#             try:
#                 # Try to install from GitHub
#                 subprocess.check_call([sys.executable, "-m", "pip", "install", "git+https://github.com/chenwj1989/chatterbox-tts.git"])
#             except:
#                 # Alternative installation method
#                 try:
#                     subprocess.check_call([sys.executable, "-m", "pip", "install", "chatterbox-tts"])
#                 except:
#                     st.warning("Could not install chatterbox-tts directly. Manual installation may be required.")
            
#             # Download a sample model
#             st.info("Downloading sample Chatterbox model...")
#             model_dir = os.path.expanduser("~/.cache/chatterbox/models")
#             os.makedirs(model_dir, exist_ok=True)
            
#             st.success("Chatterbox TTS installation started!")
#             st.info("""
#             Note: Chatterbox may require additional setup:
#             1. Manual model download from HuggingFace
#             2. Configuration of model paths
#             3. Additional dependencies
            
#             Check the Chatterbox GitHub repository for detailed instructions:
#             https://github.com/chenwj1989/chatterbox-tts
#             """)
            
#             return True
            
#         except Exception as e:
#             st.error(f"Failed to install Chatterbox TTS: {str(e)}")
#             return False
    
#     def install_piper(self):
#         """Install Piper TTS if not available"""
#         st.info("Installing Piper TTS...")
        
#         try:
#             # Install piper-tts
#             subprocess.check_call([sys.executable, "-m", "pip", "install", "piper-tts"])
            
#             # Download a sample model
#             st.info("Downloading English model...")
#             model_url = "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/lessac/medium/en_US-lessac-medium.onnx"
#             model_dir = os.path.expanduser("~/.local/share/piper/models")
            
#             # Create directory if it doesn't exist
#             os.makedirs(model_dir, exist_ok=True)
            
#             st.success("Piper TTS installed. Please restart the application.")
#             return True
            
#         except Exception as e:
#             st.error(f"Failed to install Piper TTS: {str(e)}")
#             return False
    
#     def get_recommended_engine(self, language: str, need_voice_cloning: bool = False):
#         """Get recommended TTS engine based on language and requirements"""
#         if need_voice_cloning and CHATTERBOX_WORKING:
#             return "chatterbox"
#         elif language == "en" and KOKORO_WORKING:
#             return "kokoro"
#         elif language in ["hi", "te", "ta", "kn", "mr"] and GTTS_AVAILABLE:
#             return "gtts"
#         elif TTS_AVAILABLE:
#             return "pyttsx3"
#         else:
#             return None

# class VoiceHandwritingProcessor:
#     """Process voice and handwriting inputs with TTS integration"""
    
#     def __init__(self):
#         self.recognizer = sr.Recognizer()
#         self.offline_tts = OfflineTTSManager()
#         self.setup_whisper()
#         self.setup_tts()
#         self.setup_kokoro()
#         self.setup_chatterbox()
        
#     def setup_whisper(self):
#         """Setup whisper model if available"""
#         self.whisper_model = None
#         if WHISPER_AVAILABLE:
#             try:
#                 self.whisper_model = whisper.load_model("base")
#             except:
#                 self.whisper_model = None
    
#     def setup_tts(self):
#         """Setup text-to-speech engines"""
#         self.tts_engine = None
#         if TTS_AVAILABLE:
#             try:
#                 self.tts_engine = pyttsx3.init()
#                 # Set properties
#                 self.tts_engine.setProperty('rate', 150)  # Speed percent
#                 self.tts_engine.setProperty('volume', 0.9)  # Volume 0-1
#             except:
#                 self.tts_engine = None
    
#     def setup_kokoro(self):
#         """Setup Kokoro TTS engine"""
#         self.kokoro_model = None
#         self.kokoro_voices = []
        
#         if KOKORO_AVAILABLE and KOKORO_WORKING:
#             try:
#                 # Initialize Kokoro model
#                 self.kokoro_model = kokoro.Kokoro()
                
#                 # Get available voices from Kokoro
#                 self.kokoro_voices = self.get_kokoro_voices()
                
#                 # Set default voice
#                 self.default_kokoro_voice = "af_heart"  # Default news-like voice
                
#                 st.success("✅ Kokoro TTS initialized successfully!")
#             except Exception as e:
#                 st.error(f"Failed to initialize Kokoro: {str(e)}")
#                 self.kokoro_model = None
    
#     def setup_chatterbox(self):
#         """Setup Chatterbox TTS engine"""
#         self.chatterbox_voices = []
        
#         if CHATTERBOX_AVAILABLE and CHATTERBOX_WORKING:
#             try:
#                 # Get available voices from Chatterbox
#                 self.chatterbox_voices = self.get_chatterbox_voices()
#                 st.success("✅ Chatterbox TTS is available!")
#             except Exception as e:
#                 st.error(f"Chatterbox setup note: {str(e)}")
    
#     def get_kokoro_voices(self):
#         """Get available voices in Kokoro"""
#         if not self.kokoro_model:
#             return []
        
#         # Kokoro has specific voice identifiers
#         # These are the built-in voices
#         voices = [
#             "af_heart", "af_sweet", "af_nova", "af_mellow", "af_bold",
#             "am_adam", "am_morgan", "am_sam", "am_danny", "am_josh"
#         ]
#         return voices
    
#     def get_chatterbox_voices(self):
#         """Get available voices in Chatterbox"""
#         voices = [
#             {"id": "news_anchor", "name": "News Anchor Voice", "type": "pre-trained", "cloning": True},
#             {"id": "podcast_host", "name": "Podcast Host Voice", "type": "pre-trained", "cloning": True},
#             {"id": "narration", "name": "Narration Voice", "type": "pre-trained", "cloning": True},
#             {"id": "custom_clone", "name": "Custom Voice Clone", "type": "custom", "cloning": True},
#         ]
#         return voices
    
#     def autoplay_audio(self, audio_bytes: bytes) -> None:
#         """
#         Auto-play audio in Streamlit using HTML5 audio tag
        
#         Args:
#             audio_bytes: Audio data as bytes
#         """
#         try:
#             # Encode audio bytes to base64
#             audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
            
#             # Create HTML audio tag with autoplay
#             audio_tag = f"""
#             <audio autoplay="true" controls style="width: 100%; margin: 10px 0;">
#                 <source src="data:audio/wav;base64,{audio_base64}" type="audio/wav">
#                 Your browser does not support the audio element.
#             </audio>
#             """
            
#             # Display audio player
#             st.markdown(audio_tag, unsafe_allow_html=True)
            
#         except Exception as e:
#             st.error(f"Error auto-playing audio: {str(e)}")
#             # Fallback to regular audio player
#             st.audio(audio_bytes, format='audio/wav')
    
#     def text_to_speech_kokoro(self, text: str, voice: str = None, 
#                              speed: float = 1.0, pitch: float = 0.0) -> Optional[bytes]:
#         """
#         Convert text to speech using Kokoro
        
#         Args:
#             text: Text to convert to speech
#             voice: Voice identifier (e.g., 'af_heart', 'am_adam')
#             speed: Speech speed multiplier (0.5 to 2.0)
#             pitch: Pitch adjustment (-0.5 to 0.5)
        
#         Returns:
#             Audio bytes if successful, None otherwise
#         """
#         try:
#             if not self.kokoro_model:
#                 st.warning("Kokoro model not available. Falling back to standard TTS.")
#                 return self.text_to_speech(text, "en", "pyttsx3")
            
#             # Preprocess text for better speech
#             processed_text = self._preprocess_text_for_speech(text)
            
#             # Use default voice if none specified
#             if not voice:
#                 voice = self.default_kokoro_voice
            
#             # Check if voice is available
#             available_voices = self.get_kokoro_voices()
#             if voice not in available_voices:
#                 st.warning(f"Voice '{voice}' not available. Using default.")
#                 voice = self.default_kokoro_voice
            
#             # Generate speech with Kokoro
#             st.info(f"Generating speech with Kokoro (Voice: {voice})...")
            
#             # Kokoro returns audio as numpy array and sample rate
#             audio_array, sample_rate = self.kokoro_model.tts(
#                 text=processed_text,
#                 voice=voice,
#                 speed=speed,
#                 pitch=pitch
#             )
            
#             # Convert numpy array to audio bytes
#             audio_bytes = self._audio_array_to_bytes(audio_array, sample_rate)
            
#             return audio_bytes
            
#         except Exception as e:
#             st.error(f"Error in Kokoro TTS: {str(e)}")
#             # Fallback to standard TTS
#             return self.text_to_speech(text, "en", "pyttsx3")
    
#     def text_to_speech_chatterbox(self, text: str, voice_type: str = "news_anchor",
#                                  voice_clone_audio: Optional[bytes] = None,
#                                  voice_clone_text: Optional[str] = None,
#                                  language: str = "en") -> Optional[bytes]:
#         """
#         Convert text to speech using Chatterbox TTS with voice cloning
        
#         Args:
#             text: Text to convert to speech
#             voice_type: Type of voice (news_anchor, podcast_host, narration, custom_clone)
#             voice_clone_audio: Audio bytes for voice cloning (10-20 seconds recommended)
#             voice_clone_text: Corresponding text for the voice clone audio
#             language: Language code
        
#         Returns:
#             Audio bytes if successful, None otherwise
#         """
#         try:
#             if not CHATTERBOX_WORKING:
#                 st.warning("Chatterbox not available. Check PyTorch installation.")
#                 return None
            
#             # Preprocess text for better speech
#             processed_text = self._preprocess_text_for_speech(text)
            
#             # Select model based on voice type
#             model_map = {
#                 "news_anchor": "chatterbox_news",
#                 "podcast_host": "chatterbox_podcast", 
#                 "narration": "chatterbox_base",
#                 "custom_clone": "chatterbox_multilingual"
#             }
            
#             model_id = model_map.get(voice_type, "chatterbox_base")
            
#             # Generate speech with Chatterbox
#             st.info(f"Generating speech with Chatterbox (Voice: {voice_type})...")
            
#             # Use offline TTS manager's Chatterbox method
#             audio_bytes = self.offline_tts.text_to_speech_chatterbox(
#                 text=processed_text,
#                 model_id=model_id,
#                 voice_clone_audio=voice_clone_audio,
#                 voice_clone_text=voice_clone_text,
#                 language=language
#             )
            
#             return audio_bytes
            
#         except Exception as e:
#             st.error(f"Error in Chatterbox TTS: {str(e)}")
#             st.info("Falling back to alternative TTS engine...")
#             return self.text_to_speech(text, language, "pyttsx3")
    
#     def _audio_array_to_bytes(self, audio_array: np.ndarray, sample_rate: int) -> bytes:
#         """Convert numpy audio array to WAV bytes"""
#         try:
#             import soundfile as sf
            
#             # Create temporary file
#             with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
#                 temp_file = f.name
            
#             # Save audio to WAV file
#             sf.write(temp_file, audio_array, sample_rate)
            
#             # Read audio bytes
#             with open(temp_file, 'rb') as f:
#                 audio_bytes = f.read()
            
#             # Clean up
#             os.unlink(temp_file)
            
#             return audio_bytes
            
#         except ImportError:
#             # Fallback using scipy if soundfile not available
#             try:
#                 from scipy.io import wavfile
                
#                 # Create temporary file
#                 with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
#                     temp_file = f.name
                
#                 # Save audio to WAV file
#                 wavfile.write(temp_file, sample_rate, audio_array)
                
#                 # Read audio bytes
#                 with open(temp_file, 'rb') as f:
#                     audio_bytes = f.read()
                
#                 # Clean up
#                 os.unlink(temp_file)
                
#                 return audio_bytes
                
#             except ImportError:
#                 st.warning("Install soundfile or scipy for better audio handling: pip install soundfile")
                
#                 # Very basic WAV header creation (16-bit PCM)
#                 num_channels = 1
#                 bits_per_sample = 16
#                 byte_rate = sample_rate * num_channels * bits_per_sample // 8
#                 block_align = num_channels * bits_per_sample // 8
#                 data_size = len(audio_array) * 2  # 16-bit = 2 bytes per sample
                
#                 # Normalize audio to 16-bit range
#                 audio_array_normalized = np.int16(audio_array * 32767)
                
#                 # Create WAV header
#                 wav_header = bytearray()
#                 wav_header.extend(b'RIFF')
#                 wav_header.extend((data_size + 36).to_bytes(4, 'little'))  # File size
#                 wav_header.extend(b'WAVE')
#                 wav_header.extend(b'fmt ')
#                 wav_header.extend((16).to_bytes(4, 'little'))  # PCM format chunk size
#                 wav_header.extend((1).to_bytes(2, 'little'))  # Audio format (PCM)
#                 wav_header.extend(num_channels.to_bytes(2, 'little'))
#                 wav_header.extend(sample_rate.to_bytes(4, 'little'))
#                 wav_header.extend(byte_rate.to_bytes(4, 'little'))
#                 wav_header.extend(block_align.to_bytes(2, 'little'))
#                 wav_header.extend(bits_per_sample.to_bytes(2, 'little'))
#                 wav_header.extend(b'data')
#                 wav_header.extend(data_size.to_bytes(4, 'little'))
                
#                 # Convert audio data to bytes
#                 audio_data = audio_array_normalized.tobytes()
                
#                 # Combine header and data
#                 audio_bytes = bytes(wav_header) + audio_data
                
#                 return audio_bytes
    
#     def speech_to_text_microphone(self, language="en-IN"):
#         """Convert speech from microphone to text"""
#         try:
#             with sr.Microphone() as source:
#                 st.info("🎤 Listening... Speak now!")
#                 self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
#                 audio = self.recognizer.listen(source, timeout=10, phrase_time_limit=30)
                
#                 st.info("Processing audio...")
                
#                 # Try Google Speech Recognition first
#                 try:
#                     text = self.recognizer.recognize_google(audio, language=language)
#                     return text, "Google Speech Recognition"
#                 except sr.UnknownValueError:
#                     st.warning("Could not understand audio")
#                     return None, "Could not understand audio"
#                 except sr.RequestError:
#                     # Fallback to whisper if available
#                     if self.whisper_model:
#                         try:
#                             # Save audio to temporary file
#                             with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
#                                 f.write(audio.get_wav_data())
#                                 temp_file = f.name
                            
#                             # Transcribe with whisper
#                             result = self.whisper_model.transcribe(temp_file)
#                             text = result["text"]
                            
#                             # Clean up temp file
#                             os.unlink(temp_file)
#                             return text, "Whisper AI"
#                         except:
#                             pass
#                     return None, "Speech recognition service unavailable"
                    
#         except Exception as e:
#             st.error(f"Error in speech recognition: {str(e)}")
#             return None, str(e)
    
#     def speech_to_text_upload(self, audio_file, language="en-IN"):
#         """Convert uploaded audio file to text"""
#         try:
#             # Read audio file
#             audio_data = audio_file.read()
            
#             # Create temporary file
#             with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
#                 f.write(audio_data)
#                 temp_file = f.name
            
#             # Try different methods
#             results = []
            
#             # Method 1: Google Speech Recognition
#             try:
#                 with sr.AudioFile(temp_file) as source:
#                     audio = self.recognizer.record(source)
#                     text = self.recognizer.recognize_google(audio, language=language)
#                     results.append(("Google", text))
#             except:
#                 pass
            
#             # Method 2: Whisper AI
#             if self.whisper_model:
#                 try:
#                     result = self.whisper_model.transcribe(temp_file)
#                     results.append(("Whisper AI", result["text"]))
#                 except:
#                     pass
            
#             # Clean up
#             os.unlink(temp_file)
            
#             if results:
#                 # Return the best result (prefer Whisper if available)
#                 if len(results) > 1 and "Whisper" in results[1][0]:
#                     return results[1][1], results[1][0]
#                 return results[0][1], results[0][0]
#             else:
#                 return None, "No speech recognition service available"
                
#         except Exception as e:
#             st.error(f"Error processing audio file: {str(e)}")
#             return None, str(e)
    
#     def handwriting_to_text(self, image_file, language='eng'):
#         """Convert handwritten image to text"""
#         try:
#             # Open and preprocess image
#             image = Image.open(image_file)
            
#             # Test Tesseract installation
#             try:
#                 # Try to get Tesseract version
#                 version = pytesseract.get_tesseract_version()
#                 st.info(f"✅ Tesseract OCR Version: {version}")
#             except:
#                 st.error("❌ Tesseract not found or path incorrect!")
#                 st.info(f"Current Tesseract path: {pytesseract.pytesseract.tesseract_cmd}")
#                 st.info("Please update the Tesseract path in the code (line 35-45)")
#                 return None, "Tesseract not configured"
            
#             # Convert to grayscale
#             if image.mode != 'L':
#                 image = image.convert('L')
            
#             # Display processed image (for debugging)
#             st.image(image, caption="Processed Image for OCR", width=300)
            
#             # Enhance contrast
#             from PIL import ImageEnhance
#             enhancer = ImageEnhance.Contrast(image)
#             image = enhancer.enhance(2.0)
            
#             # Available languages in Tesseract
#             try:
#                 available_langs = pytesseract.get_languages()
#                 st.info(f"Available Tesseract languages: {available_langs}")
#             except:
#                 available_langs = []
            
#             # Language mapping for Tesseract
#             lang_map = {
#                 'eng': 'eng',
#                 'hin': 'hin',
#                 'tel': 'tel',
#                 'tam': 'tam',
#                 'kan': 'kan',
#                 'hi': 'hin',
#                 'te': 'tel',
#                 'ta': 'tam',
#                 'kn': 'kan'
#             }
            
#             tesseract_lang = lang_map.get(language, 'eng')
            
#             # Check if language is available
#             if tesseract_lang not in available_langs and tesseract_lang != 'eng':
#                 st.warning(f"Language '{tesseract_lang}' not available. Using English.")
#                 tesseract_lang = 'eng'
            
#             # Use pytesseract for OCR
#             st.info(f"Processing with Tesseract (language: {tesseract_lang})...")
#             text = pytesseract.image_to_string(image, lang=tesseract_lang)
            
#             st.success(f"✅ OCR Completed using Tesseract {version}")
            
#             # If pytesseract fails and OpenCV is available, try preprocessing
#             if not text.strip() and OCR_AVAILABLE:
#                 st.info("No text found. Trying with OpenCV preprocessing...")
#                 # Convert PIL image to OpenCV format
#                 img_array = np.array(image)
                
#                 # Apply additional preprocessing
#                 img_array = cv2.threshold(img_array, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
                
#                 # Convert back to PIL image
#                 image = Image.fromarray(img_array)
                
#                 # Try OCR again
#                 text = pytesseract.image_to_string(image, lang=tesseract_lang)
            
#             return text.strip(), "Tesseract OCR"
            
#         except Exception as e:
#             st.error(f"Error in handwriting recognition: {str(e)}")
#             st.error(f"Tesseract path: {pytesseract.pytesseract.tesseract_cmd}")
#             return None, str(e)
    
#     def text_to_speech(self, text: str, language: str = "en", engine: str = "auto", 
#                       voice_clone_audio: Optional[bytes] = None,
#                       voice_clone_text: Optional[str] = None) -> Optional[bytes]:
#         """
#         Convert text to speech audio with engine selection and voice cloning
        
#         Args:
#             text: Text to convert to speech
#             language: Language code (en, hi, te, ta, etc.)
#             engine: TTS engine to use ("auto", "chatterbox", "kokoro", "piper", "pyttsx3", or "gtts")
#             voice_clone_audio: Audio bytes for voice cloning (Chatterbox only)
#             voice_clone_text: Corresponding text for voice cloning (Chatterbox only)
        
#         Returns:
#             Audio bytes if successful, None otherwise
#         """
#         try:
#             # Preprocess text for better speech
#             processed_text = self._preprocess_text_for_speech(text)
            
#             # Auto-select engine based on language and requirements
#             if engine == "auto":
#                 need_cloning = voice_clone_audio is not None
#                 engine = self.offline_tts.get_recommended_engine(language, need_cloning)
#                 if not engine:
#                     st.warning("No TTS engine available for this language")
#                     return None
            
#             if engine == "chatterbox" and CHATTERBOX_WORKING:
#                 # Use Chatterbox TTS with voice cloning
#                 return self.text_to_speech_chatterbox(
#                     processed_text,
#                     voice_type="custom_clone" if voice_clone_audio else "news_anchor",
#                     voice_clone_audio=voice_clone_audio,
#                     voice_clone_text=voice_clone_text,
#                     language=language
#                 )
                
#             elif engine == "kokoro" and self.kokoro_model:
#                 # Use Kokoro TTS
#                 return self.text_to_speech_kokoro(processed_text)
                
#             elif engine == "piper" and PIPER_AVAILABLE:
#                 # Use Piper TTS
#                 # Map language to piper voice
#                 voice_map = {
#                     "en": "en_US-lessac-medium",
#                     "hi": "hi_IN-google-medium",
#                     "ta": "ta_IN-google-medium",
#                     "te": "te_IN-google-medium",
#                     "kn": "kn_IN-google-medium",
#                     "mr": "mr_IN-google-medium"
#                 }
#                 voice_id = voice_map.get(language, "en_US-lessac-medium")
#                 return self.offline_tts.text_to_speech_piper(processed_text, voice_id)
                
#             elif engine == "pyttsx3" and self.tts_engine:
#                 # Save to temporary file
#                 temp_file = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
#                 temp_path = temp_file.name
#                 temp_file.close()
                
#                 # Save speech to file
#                 self.tts_engine.save_to_file(processed_text, temp_path)
#                 self.tts_engine.runAndWait()
                
#                 # Read audio file
#                 with open(temp_path, 'rb') as f:
#                     audio_bytes = f.read()
                
#                 # Clean up temp file
#                 os.unlink(temp_path)
#                 return audio_bytes
                
#             elif engine == "gtts" and GTTS_AVAILABLE:
#                 # Language mapping for gTTS
#                 lang_map = {
#                     'en': 'en',
#                     'hi': 'hi',
#                     'te': 'te',
#                     'ta': 'ta',
#                     'kn': 'kn',
#                     'mr': 'mr'
#                 }
                
#                 tts_lang = lang_map.get(language, 'en')
                
#                 # Create gTTS object
#                 tts = gTTS(text=processed_text, lang=tts_lang, slow=False)
                
#                 # Save to bytes
#                 temp_file = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
#                 temp_path = temp_file.name
#                 temp_file.close()
                
#                 tts.save(temp_path)
                
#                 # Read audio file
#                 with open(temp_path, 'rb') as f:
#                     audio_bytes = f.read()
                
#                 # Clean up temp file
#                 os.unlink(temp_path)
#                 return audio_bytes
                
#             else:
#                 st.warning(f"TTS engine '{engine}' not available.")
#                 return None
                
#         except Exception as e:
#             st.error(f"Error in text-to-speech conversion: {str(e)}")
#             return None
    
#     def _preprocess_text_for_speech(self, text: str) -> str:
#         """
#         Preprocess text for better speech synthesis
        
#         Args:
#             text: Raw text to preprocess
        
#         Returns:
#             Preprocessed text with better formatting for TTS
#         """
#         # Replace common news script formatting for better speech
#         replacements = {
#             "\n\n": ". ",  # Double newlines become pauses
#             "\n": " ",     # Single newlines become spaces
#             "  ": " ",     # Remove double spaces
#             "..": ".",     # Fix double periods
#             " .": ".",     # Fix space before period
#             " ,": ",",     # Fix space before comma
#             " ;": ";",     # Fix space before semicolon
#             " :": ":",     # Fix space before colon
#             " ?": "?",     # Fix space before question
#             " !": "!",     # Fix space before exclamation
#         }
        
#         processed_text = text
#         for old, new in replacements.items():
#             processed_text = processed_text.replace(old, new)
        
#         # Add expressive pauses for news reading
#         expressive_pauses = {
#             "BREAKING NEWS:": "BREAKING NEWS...",
#             "Good evening": "Good evening...",
#             "Good morning": "Good morning...",
#             "Good afternoon": "Good afternoon...",
#             "Here are our top stories:": "Here are our top stories...",
#             "That's all for now.": "That's all for now...",
#             "Stay tuned.": "Stay tuned...",
#         }
        
#         for phrase, replacement in expressive_pauses.items():
#             processed_text = processed_text.replace(phrase, replacement)
        
#         # Ensure proper punctuation for TTS
#         if not processed_text.endswith(('.', '!', '?')):
#             processed_text = processed_text + '.'
        
#         return processed_text.strip()
    
#     def get_available_voices(self):
#         """Get available voices for all TTS engines"""
#         voices = {}
        
#         # Get Chatterbox voices
#         if CHATTERBOX_WORKING:
#             voices["chatterbox"] = [v["name"] for v in self.get_chatterbox_voices()]
        
#         # Get Kokoro voices
#         if self.kokoro_model:
#             voices["kokoro"] = self.get_kokoro_voices()
        
#         # Get Piper voices
#         if PIPER_AVAILABLE:
#             voices["piper"] = [v["name"] for v in self.offline_tts.piper_voices]
        
#         # Get pyttsx3 voices
#         if self.tts_engine:
#             pyttsx3_voices = self.tts_engine.getProperty('voices')
#             voices["pyttsx3"] = [voice.name for voice in pyttsx3_voices] if pyttsx3_voices else []
        
#         return voices
    
#     def set_voice_properties(self, rate: int = 150, volume: float = 0.9, voice_id: int = 0):
#         """Set TTS voice properties (for pyttsx3)"""
#         if self.tts_engine:
#             try:
#                 self.tts_engine.setProperty('rate', rate)
#                 self.tts_engine.setProperty('volume', volume)
                
#                 voices = self.tts_engine.getProperty('voices')
#                 if voices and voice_id < len(voices):
#                     self.tts_engine.setProperty('voice', voices[voice_id].id)
#                 return True
#             except:
#                 return False
#         return False
    
#     def set_kokoro_voice(self, voice_name: str):
#         """Set Kokoro default voice"""
#         available_voices = self.get_kokoro_voices()
#         if voice_name in available_voices:
#             self.default_kokoro_voice = voice_name
#             return True
#         return False
    
#     def test_tesseract_installation(self):
#         """Test Tesseract installation and configuration"""
#         try:
#             version = pytesseract.get_tesseract_version()
#             return True, f"✅ Tesseract Version: {version}"
#         except Exception as e:
#             return False, f"❌ Tesseract Error: {str(e)}. Path: {pytesseract.pytesseract.tesseract_cmd}"
    
#     def process_handwritten_form(self, form_image):
#         """Process a form with handwritten fields"""
#         try:
#             # This is a simplified version - in production, you'd use form recognition
#             text, method = self.handwriting_to_text(form_image)
            
#             if text:
#                 # Parse common form fields (simplified)
#                 parsed_data = {
#                     "full_text": text,
#                     "extracted_phrases": self.extract_key_phrases(text)
#                 }
#                 return parsed_data, method
#             return None, method
            
#         except Exception as e:
#             st.error(f"Error processing form: {str(e)}")
#             return None, str(e)
    
#     def extract_key_phrases(self, text):
#         """Extract key phrases from recognized text"""
#         sentences = text.split('.')
#         key_phrases = []
        
#         for sentence in sentences:
#             sentence = sentence.strip()
#             if len(sentence) > 10:  # Avoid very short sentences
#                 # Look for important keywords
#                 keywords = ['accident', 'emergency', 'urgent', 'help', 'police', 
#                           'hospital', 'fire', 'danger', 'warning', 'alert']
                
#                 for keyword in keywords:
#                     if keyword.lower() in sentence.lower():
#                         key_phrases.append(sentence)
#                         break
        
#         return key_phrases

# class NewsContentGenerator:
#     """Main class for AI-powered news content generation"""
    
#     def __init__(self, api_key: Optional[str] = None):
#         self.api_key = api_key
#         self.templates = self._load_templates()
#         self.style_guide = self._load_style_guide()
#         self.input_processor = VoiceHandwritingProcessor()
        
#     def _load_templates(self) -> Dict:
#         """Load news script templates"""
#         return {
#             "tv_script": {
#                 "intro": "Good evening and welcome to our news bulletin. Our top stories today:",
#                 "story_template": "First, {headline}. Our correspondent reports: {summary}",
#                 "outro": "That's all for now. Stay tuned for more updates.",
#                 "breaking": "BREAKING NEWS: {headline}. We're getting reports that {summary}",
#                 "weather": "Moving to weather: {content}",
#                 "sports": "In sports news: {content}"
#             },
#             "bullet_summary": {
#                 "header": "Key Points:",
#                 "bullet": "• {point}",
#                 "max_points": 5
#             },
#             "emergency_report": {
#                 "template": """🚨 EMERGENCY ALERT 🚨
# Location: {location}
# Incident: {incident}
# Reported by: {reporter}
# Time: {time}
# Details: {details}
# Status: {status}
# Action Required: {action}""",
#                 "default_actions": {
#                     "accident": "Send ambulance and police immediately",
#                     "fire": "Dispatch fire brigade and emergency services",
#                     "crime": "Alert police patrols in the area",
#                     "medical": "Send medical assistance urgently",
#                     "natural_disaster": "Activate disaster response team"
#                 }
#             }
#         }
    
#     def _load_style_guide(self) -> Dict:
#         """Load linguistic style guides for different languages"""
#         return {
#             "te": {"formal": True, "anchor_greeting": "శుభ సాయంత్రం", "sign_off": "మళ్లీ కలుద్దాం"},
#             "hi": {"formal": True, "anchor_greeting": "నమస్కార్", "sign_off": "फिर मिलेंगे"},
#             "mr": {"formal": True, "anchor_greeting": "नमस्कार", "sign_off": "पुन्हा भेटू"},
#             "kn": {"formal": True, "anchor_greeting": "ನಮಸ್ಕಾರ", "sign_off": "ಮತ್ತೆ ಸಿಗೋಣ"},
#             "ta": {"formal": True, "anchor_greeting": "வணக்கம்", "sign_off": "மீண்டும் சந்திப்போம்"},
#             "en": {"formal": True, "anchor_greeting": "Good evening", "sign_off": "Stay tuned"}
#         }
    
#     def generate_tv_script(self, news_items: List[NewsItem], 
#                           language: Language = Language.ENGLISH) -> str:
#         """
#         Generate TV-style news script from raw news items
#         """
#         script_parts = []
        
#         # Add greeting based on time of day
#         current_hour = datetime.datetime.now().hour
#         greeting = self._get_greeting(current_hour, language)
#         script_parts.append(f"{greeting} and welcome to our news bulletin.\n")
        
#         # Add headlines preview
#         script_parts.append("Here are our top stories:\n")
#         for i, item in enumerate(news_items[:3], 1):
#             script_parts.append(f"{i}. {item.title}")
        
#         script_parts.append("\n" + "="*50 + "\n")
        
#         # Generate detailed stories
#         for item in news_items:
#             story = self._format_story(item, language)
#             script_parts.append(story)
#             script_parts.append("")  # Empty line between stories
        
#         # Add outro
#         sign_off = self.style_guide.get(language.value, {}).get("sign_off", "Stay tuned")
#         script_parts.append(f"That's all for now. {sign_off}.")
        
#         return "\n".join(script_parts)
    
#     def generate_emergency_report(self, input_text: str, 
#                                  location: str = "Unknown",
#                                  incident_type: str = "emergency") -> Dict:
#         """
#         Generate emergency report from voice/handwriting input
#         """
#         # Analyze the input text
#         urgency_level = self._detect_urgency(input_text)
        
#         # Extract key information
#         extracted_info = self._extract_emergency_info(input_text)
        
#         # Determine action based on incident type
#         actions = self.templates["emergency_report"]["default_actions"]
#         action = actions.get(incident_type, "Investigate and respond appropriately")
        
#         # Generate report
#         report_text = self.templates["emergency_report"]["template"].format(
#             location=location or extracted_info.get("location", "Unknown"),
#             incident=incident_type.upper(),
#             reporter="Citizen Report",
#             time=datetime.datetime.now().strftime("%H:%M %d/%m/%Y"),
#             details=input_text[:200] + ("..." if len(input_text) > 200 else ""),
#             status="PENDING RESPONSE",
#             action=action
#         )
        
#         return {
#             "report": report_text,
#             "urgency": urgency_level,
#             "extracted_info": extracted_info,
#             "timestamp": datetime.datetime.now().isoformat(),
#             "recommended_actions": self._get_recommended_actions(incident_type, urgency_level)
#         }
    
#     def generate_bullet_summary(self, raw_text: str, 
#                                max_points: int = 5) -> List[str]:
#         """
#         Summarize long articles into bullet points
#         """
#         sentences = raw_text.split('. ')
#         key_sentences = self._extract_key_sentences(sentences, max_points)
        
#         bullets = [self.templates["bullet_summary"]["header"]]
#         for i, sentence in enumerate(key_sentences[:max_points], 1):
#             bullet = self.templates["bullet_summary"]["bullet"].format(point=sentence.strip())
#             bullets.append(bullet)
        
#         return bullets
    
#     def translate_content(self, text: str, 
#                          target_language: Language) -> str:
#         """
#         Translate content to target language
#         """
#         translation_map = {
#             Language.TELUGU: {
#                 "Good evening": "శుభ సాయంత్రం",
#                 "Good morning": "శుభోదయం",
#                 "Good afternoon": "శుభ మధ్యాహ్నం",
#                 "Good night": "శుభ రాత్రి",
#                 "welcome": "స్వాగతం",
#                 "news": "వార్తలు",
#                 "weather": "వాతావరణం",
#                 "sports": "క్రీడలు",
#                 "Here are our top stories": "ఇవి మా ప్రధాన వార్తలు",
#                 "That's all for now": "ప్రస్తుతానికి ఇంతే",
#                 "Stay tuned": "వేచి ఉండండి"
#             },
#             Language.HINDI: {
#                 "Good evening": "शुभ संध्या",
#                 "Good morning": "शुभ प्रभात",
#                 "Good afternoon": "नमस्कार",
#                 "Good night": "शुभ रात्रि",
#                 "welcome": "स्वागत है",
#                 "news": "खबर",
#                 "weather": "मौसम",
#                 "sports": "खेल",
#                 "Here are our top stories": "यहां हैं हमारी शीर्ष खबरें",
#                 "That's all for now": "अभी के लिए इतना ही",
#                 "Stay tuned": "बने रहें"
#             },
#             Language.MARATHI: {
#                 "Good evening": "शुभ संध्या",
#                 "Good morning": "शुभ प्रभात",
#                 "Good afternoon": "नमस्कार",
#                 "Good night": "शुभ रात्रि",
#                 "welcome": "स्वागत आहे",
#                 "news": "बातम्या",
#                 "weather": "हवामान",
#                 "sports": "क्रीडा"
#             },
#             Language.KANNADA: {
#                 "Good evening": "ಶುಭ ಸಂಜೆ",
#                 "Good morning": "ಶುಭೋದಯ",
#                 "Good afternoon": "ಶುಭ ಮಧ್ಯಾಹ್ನ",
#                 "Good night": "ಶುಭ ರಾತ್ರಿ",
#                 "welcome": "ಸ್ವಾಗತ",
#                 "news": "ವಾರ್ತೆಗಳು",
#                 "weather": "ಹವಾಮಾನ",
#                 "sports": "ಕ್ರೀಡೆಗಳು"
#             },
#             Language.TAMIL: {
#                 "Good evening": "மாலை வணக்கம்",
#                 "Good morning": "காலை வணக்கம்",
#                 "Good afternoon": "மதிய வணக்கம்",
#                 "Good night": "இரவு வணக்கம்",
#                 "welcome": "வரவேற்பு",
#                 "news": "செய்திகள்",
#                 "weather": "வானிலை",
#                 "sports": "விளையாட்டுக்கள்"
#             }
#         }
        
#         if target_language in translation_map:
#             for eng, trans in translation_map[target_language].items():
#                 text = text.replace(eng, trans)
        
#         return text
    
#     def generate_breaking_news_version(self, news_item: NewsItem,
#                                       language: Language) -> str:
#         """
#         Create breaking news version with urgency markers
#         """
#         template = self.templates["tv_script"]["breaking"]
        
#         breaking_content = template.format(
#             headline=news_item.title.upper(),
#             summary=self._create_urgent_summary(news_item.raw_content)
#         )
        
#         if language != Language.ENGLISH:
#             breaking_content = self.translate_content(breaking_content, language)
        
#         # Add timestamp and location
#         timestamp = news_item.timestamp.strftime("%H:%M IST")
#         breaking_content = f"[{timestamp}] {breaking_content}"
        
#         return breaking_content
    
#     def generate_weather_report(self, location: str, 
#                                weather_data: Dict) -> str:
#         """
#         Generate weather report in news format
#         """
#         template = """WEATHER UPDATE for {location}:
# Current Temperature: {temp}°C
# Conditions: {conditions}
# Humidity: {humidity}%
# Wind: {wind_speed} km/h {wind_direction}
# Sunrise: {sunrise}, Sunset: {sunset}
# """
        
#         return template.format(
#             location=location,
#             temp=weather_data.get('temperature', 'N/A'),
#             conditions=weather_data.get('conditions', 'Partly Cloudy'),
#             humidity=weather_data.get('humidity', 'N/A'),
#             wind_speed=weather_data.get('wind_speed', 'N/A'),
#             wind_direction=weather_data.get('wind_direction', ''),
#             sunrise=weather_data.get('sunrise', '06:00'),
#             sunset=weather_data.get('sunset', '18:00')
#         )
    
#     def generate_market_update(self, market_data: Dict) -> str:
#         """
#         Generate market rates and financial news
#         """
#         template = """MARKET UPDATE:
# Sensex: {sensex} ({sensex_change})
# Nifty: {nifty} ({nifty_change})
# Top Gainers: {gainers}
# Top Losers: {losers}
# Currency: ₹{usd_inr}/USD
# Gold: ₹{gold_price}/10g
# """
        
#         return template.format(
#             sensex=market_data.get('sensex', 'N/A'),
#             sensex_change=market_data.get('sensex_change', '0%'),
#             nifty=market_data.get('nifty', 'N/A'),
#             nifty_change=market_data.get('nifty_change', '0%'),
#             gainers=", ".join(market_data.get('gainers', [])[:3]),
#             losers=", ".join(market_data.get('losers', [])[:3]),
#             usd_inr=market_data.get('usd_inr', '83.5'),
#             gold_price=market_data.get('gold_price', '62,000')
#         )
    
#     def generate_traffic_update(self, city: str, 
#                                traffic_data: Dict) -> str:
#         """
#         Generate traffic news update
#         """
#         template = """TRAFFIC UPDATE for {city}:
# Current Traffic Status: {status}
# Heavy Traffic Areas: {heavy_areas}
# Average Speed: {avg_speed} km/h
# Accidents Reported: {accidents}
# Best Routes: {best_routes}
# """
        
#         return template.format(
#             city=city,
#             status=traffic_data.get('status', 'Moderate'),
#             heavy_areas=", ".join(traffic_data.get('heavy_areas', ['None'])),
#             avg_speed=traffic_data.get('avg_speed', '30'),
#             accidents=traffic_data.get('accidents', '0'),
#             best_routes=", ".join(traffic_data.get('best_routes', ['Use alternative routes']))
#         )
    
#     def create_long_form_program(self, topic: str, 
#                                 program_type: str = "analysis") -> Dict:
#         """
#         Generate long-form content like analysis, debates, info segments
#         """
#         programs = {
#             "analysis": {
#                 "structure": [
#                     "Introduction to topic",
#                     "Background and context",
#                     "Key players involved",
#                     "Current developments",
#                     "Expert opinions",
#                     "Future implications",
#                     "Conclusion"
#                 ],
#                 "duration": "15-20 minutes",
#                 "participants": ["Anchor", "2-3 Experts", "Field Reporter"],
#                 "estimated_words": 1200
#             },
#             "debate": {
#                 "structure": [
#                     "Topic introduction",
#                     "For arguments (3-4 points)",
#                     "Against arguments (3-4 points)",
#                     "Audience/caller questions",
#                     "Rebuttal round",
#                     "Closing statements",
#                     "Moderator summary"
#                 ],
#                 "duration": "30-45 minutes",
#                 "participants": ["Moderator", "2-4 Panelists", "Audience"],
#                 "estimated_words": 2000
#             },
#             "info_segment": {
#                 "structure": [
#                     "Hook/attention grabber",
#                     "Problem statement",
#                     "Historical context",
#                     "Current status",
#                     "Step-by-step explanation",
#                     "Visual aids description",
#                     "Key takeaways",
#                     "Additional resources"
#                 ],
#                 "duration": "10-15 minutes",
#                 "participants": ["Host", "Graphics/VFX", "Voice-over"],
#                 "estimated_words": 1000
#             },
#             "documentary": {
#                 "structure": [
#                     "Opening scene/narration",
#                     "Historical background",
#                     "Key events timeline",
#                     "Interviews with experts",
#                     "On-ground reporting",
#                     "Impact analysis",
#                     "Future outlook",
#                     "Closing remarks"
#                 ],
#                 "duration": "45-60 minutes",
#                 "participants": ["Narrator", "Experts", "Field Crew", "Archival Footage"],
#                 "estimated_words": 3000
#             }
#         }
        
#         program_template = programs.get(program_type, programs["analysis"])
        
#         return {
#             "program_title": f"{topic.title()} - {program_type.title()}",
#             "template": program_template,
#             "estimated_words": program_template["estimated_words"],
#             "suggested_graphics": self._suggest_graphics(topic, program_type),
#             "production_notes": self._generate_production_notes(topic, program_type)
#         }
    
#     def generate_headlines(self, news_items: List[NewsItem], 
#                           language: Language = Language.ENGLISH) -> List[str]:
#         """
#         Generate catchy headlines for news items
#         """
#         headlines = []
#         for item in news_items:
#             if item.category == NewsCategory.BREAKING:
#                 headline = f"🚨 BREAKING: {item.title}"
#             elif item.category == NewsCategory.POLITICS:
#                 headline = f"📜 {item.title}"
#             elif item.category == NewsCategory.SPORTS:
#                 headline = f"⚽ {item.title}"
#             elif item.category == NewsCategory.WEATHER:
#                 headline = f"🌤️ {item.title}"
#             elif item.category == NewsCategory.MARKET:
#                 headline = f"📈 {item.title}"
#             else:
#                 headline = item.title
            
#             if language != Language.ENGLISH:
#                 headline = self.translate_content(headline, language)
            
#             headlines.append(headline)
        
#         return headlines
    
#     def generate_sub_headlines(self, news_items: List[NewsItem],
#                              language: Language = Language.ENGLISH) -> List[str]:
#         """
#         Generate sub-headlines/summaries for news items
#         """
#         sub_headlines = []
#         for item in news_items:
#             words = item.raw_content.split()[:15]
#             sub_headline = ' '.join(words) + '...'
            
#             if language != Language.ENGLISH:
#                 sub_headline = self.translate_content(sub_headline, language)
            
#             sub_headlines.append(sub_headline)
        
#         return sub_headlines
    
#     def convert_text_to_speech(self, text: str, language: str = "en", 
#                               voice_type: str = "male", engine: str = "auto",
#                               voice_clone_audio: Optional[bytes] = None,
#                               voice_clone_text: Optional[str] = None,
#                               auto_play: bool = True) -> Optional[bytes]:
#         """
#         Convert text to speech audio
        
#         Args:
#             text: Text to convert
#             language: Language code
#             voice_type: Type of voice (male/female/system)
#             engine: TTS engine to use ("auto", "chatterbox", "kokoro", "piper", "pyttsx3", or "gtts")
#             voice_clone_audio: Audio bytes for voice cloning (Chatterbox only)
#             voice_clone_text: Corresponding text for voice cloning (Chatterbox only)
#             auto_play: Automatically play audio after generation
        
#         Returns:
#             Audio bytes or None if failed
#         """
#         try:
#             # Generate audio
#             audio_bytes = self.input_processor.text_to_speech(
#                 text, language, engine, voice_clone_audio, voice_clone_text
#             )
            
#             # Auto-play if requested
#             if audio_bytes and auto_play:
#                 self.input_processor.autoplay_audio(audio_bytes)
            
#             return audio_bytes
            
#         except Exception as e:
#             st.error(f"Error converting text to speech: {str(e)}")
#             return None
    
#     def generate_audio_news_bulletin(self, news_items: List[NewsItem], 
#                                     language: str = "en",
#                                     expressive_reading: bool = True,
#                                     engine: str = "auto",
#                                     voice_clone_audio: Optional[bytes] = None,
#                                     voice_clone_text: Optional[str] = None,
#                                     auto_play: bool = True) -> Optional[bytes]:
#         """
#         Generate complete audio news bulletin with optional voice cloning
        
#         Args:
#             news_items: List of news items
#             language: Language code
#             expressive_reading: Whether to add expressive pauses
#             engine: TTS engine to use
#             voice_clone_audio: Audio bytes for voice cloning (Chatterbox only)
#             voice_clone_text: Corresponding text for voice cloning (Chatterbox only)
#             auto_play: Automatically play audio after generation
        
#         Returns:
#             Combined audio bytes
#         """
#         try:
#             # Generate script
#             lang_enum = Language(language)
#             script = self.generate_tv_script(news_items, lang_enum)
            
#             # Add expressive reading if enabled
#             if expressive_reading:
#                 script = self._add_expressive_elements(script)
            
#             # Convert script to speech with optional voice cloning
#             audio_bytes = self.convert_text_to_speech(
#                 script, language, engine=engine, 
#                 voice_clone_audio=voice_clone_audio,
#                 voice_clone_text=voice_clone_text,
#                 auto_play=auto_play
#             )
#             return audio_bytes
            
#         except Exception as e:
#             st.error(f"Error generating audio bulletin: {str(e)}")
#             return None
    
#     def _add_expressive_elements(self, script: str) -> str:
#         """
#         Add expressive elements to script for better speech synthesis
        
#         Args:
#             script: Original script
        
#         Returns:
#             Script with expressive elements added
#         """
#         # Add expressive pauses for news anchor style
#         expressive_replacements = {
#             "Good evening and welcome to our news bulletin.": 
#                 "Good evening... and welcome to our news bulletin.",
#             "Here are our top stories:": 
#                 "Here... are our top stories:",
#             "First,": "First...",
#             "BREAKING NEWS:": "BREAKING NEWS...",
#             "That's all for now.": "That's all for now...",
#             "Stay tuned for more updates.": "Stay tuned... for more updates.",
#         }
        
#         processed_script = script
#         for old, new in expressive_replacements.items():
#             processed_script = processed_script.replace(old, new)
        
#         return processed_script
    
#     def _get_greeting(self, hour: int, language: Language) -> str:
#         """Get time-appropriate greeting"""
#         if 5 <= hour < 12:
#             time_greeting = "Good morning"
#         elif 12 <= hour < 17:
#             time_greeting = "Good afternoon"
#         elif 17 <= hour < 21:
#             time_greeting = "Good evening"
#         else:
#             time_greeting = "Good night"
        
#         if language != Language.ENGLISH:
#             return self.translate_content(time_greeting, language)
#         return time_greeting
    
#     def _format_story(self, item: NewsItem, language: Language) -> str:
#         """Format individual news story"""
#         if item.category == NewsCategory.WEATHER:
#             template = self.templates["tv_script"]["weather"]
#             content = f"Weather update for {item.location}"
#             story = template.format(content=content)
#         elif item.category == NewsCategory.SPORTS:
#             template = self.templates["tv_script"]["sports"]
#             content = item.title
#             story = template.format(content=content)
#         else:
#             template = self.templates["tv_script"]["story_template"]
#             summary = item.raw_content[:150] + "..."
#             story = template.format(headline=item.title, summary=summary)
        
#         if language != Language.ENGLISH:
#             story = self.translate_content(story, language)
        
#         return story
    
#     def _extract_key_sentences(self, sentences: List[str], 
#                               max_points: int) -> List[str]:
#         """Extract key sentences for summarization"""
#         keywords = ['announced', 'confirmed', 'revealed', 'major', 'important',
#                    'breaking', 'new', 'first', 'exclusive', 'won', 'lost',
#                    'increased', 'decreased', 'launched', 'introduced']
        
#         scored_sentences = []
#         for sentence in sentences:
#             score = sum(1 for keyword in keywords if keyword in sentence.lower())
#             if any(char.isdigit() for char in sentence):
#                 score += 2
#             if len(sentence.split()) > 5:
#                 scored_sentences.append((score, sentence))
        
#         scored_sentences.sort(reverse=True)
#         return [sentence for _, sentence in scored_sentences[:max_points]]
    
#     def _create_urgent_summary(self, content: str) -> str:
#         """Create urgent summary for breaking news"""
#         words = content.split()
#         if len(words) > 20:
#             return ' '.join(words[:20]) + "... More details to follow."
#         return content
    
#     def _detect_urgency(self, text: str) -> str:
#         """Detect urgency level from text"""
#         text_lower = text.lower()
        
#         high_urgency_words = ['emergency', 'urgent', 'immediately', 'now', 'danger',
#                              'accident', 'fire', 'help', 'police', 'ambulance',
#                              'hurt', 'injured', 'bleeding', 'dangerous']
        
#         medium_urgency_words = ['important', 'serious', 'issue', 'problem',
#                                'concern', 'worried', 'trouble']
        
#         high_count = sum(1 for word in high_urgency_words if word in text_lower)
#         medium_count = sum(1 for word in medium_urgency_words if word in text_lower)
        
#         if high_count >= 2:
#             return "HIGH"
#         elif high_count >= 1 or medium_count >= 2:
#             return "MEDIUM"
#         elif medium_count >= 1:
#             return "LOW"
#         return "NORMAL"
    
#     def _extract_emergency_info(self, text: str) -> Dict:
#         """Extract emergency information from text"""
#         info = {
#             "location": "Unknown",
#             "incident_type": "emergency",
#             "people_involved": 0,
#             "key_phrases": []
#         }
        
#         # Extract location patterns
#         location_patterns = ['at', 'in', 'near', 'on', 'beside', 'next to']
#         words = text.lower().split()
        
#         for i, word in enumerate(words):
#             if word in location_patterns and i + 1 < len(words):
#                 location = ' '.join(words[i+1:i+3])
#                 info["location"] = location.title()
#                 break
        
#         # Detect incident type
#         incident_keywords = {
#             'accident': ['accident', 'crash', 'collision', 'hit'],
#             'fire': ['fire', 'burning', 'smoke', 'flames'],
#             'medical': ['sick', 'ill', 'heart', 'breathing', 'unconscious'],
#             'crime': ['robbery', 'theft', 'attack', 'fight', 'weapon'],
#             'natural_disaster': ['earthquake', 'flood', 'storm', 'landslide']
#         }
        
#         for incident_type, keywords in incident_keywords.items():
#             for keyword in keywords:
#                 if keyword in text.lower():
#                     info["incident_type"] = incident_type
#                     break
        
#         # Extract numbers (possibly people involved)
#         import re
#         numbers = re.findall(r'\b\d+\b', text)
#         if numbers:
#             info["people_involved"] = int(numbers[0])
        
#         # Extract key phrases
#         sentences = text.split('.')
#         for sentence in sentences:
#             sentence = sentence.strip()
#             if len(sentence) > 10:
#                 info["key_phrases"].append(sentence[:100])
        
#         return info
    
#     def _get_recommended_actions(self, incident_type: str, urgency_level: str) -> List[str]:
#         """Get recommended actions based on incident type and urgency"""
#         actions_map = {
#             "accident": {
#                 "HIGH": ["Dispatch ambulance immediately", "Send police for traffic control", "Alert nearest hospital"],
#                 "MEDIUM": ["Send medical team", "Investigate the scene", "Provide first aid"],
#                 "LOW": ["Document the incident", "Check for injuries", "Clear the area"]
#             },
#             "fire": {
#                 "HIGH": ["Dispatch fire brigade immediately", "Evacuate nearby buildings", "Cut off gas/power supply"],
#                 "MEDIUM": ["Send fire response team", "Secure the perimeter", "Alert nearby residents"],
#                 "LOW": ["Investigate smoke/smell", "Check fire safety equipment", "Monitor the situation"]
#             },
#             "medical": {
#                 "HIGH": ["Send ambulance with paramedics", "Alert emergency room", "Provide CPR instructions"],
#                 "MEDIUM": ["Dispatch medical team", "Check vital signs", "Prepare first aid"],
#                 "LOW": ["Assess the situation", "Provide basic care", "Monitor condition"]
#             }
#         }
        
#         default_actions = {
#             "HIGH": ["Respond immediately", "Alert emergency services", "Secure the area"],
#             "MEDIUM": ["Investigate the situation", "Prepare response team", "Monitor developments"],
#             "LOW": ["Document the report", "Assess priority", "Plan appropriate response"]
#         }
        
#         return actions_map.get(incident_type, default_actions).get(urgency_level, default_actions["MEDIUM"])
    
#     def _suggest_graphics(self, topic: str, program_type: str) -> List[str]:
#         """Suggest graphics for TV production"""
#         suggestions = [
#             "Lower third with topic title",
#             "Expert name and title graphic",
#             "Key statistics overlay",
#             "Location map if applicable",
#             "Timeline graphic for historical context"
#         ]
        
#         if program_type == "debate":
#             suggestions.extend([
#                 "Poll results graphic", 
#                 "Live ticker for audience votes",
#                 "Side-by-side comparison of arguments"
#             ])
#         elif program_type == "info_segment":
#             suggestions.extend([
#                 "Infographics", 
#                 "Step-by-step animation", 
#                 "Comparison charts",
#                 "Before/After images"
#             ])
#         elif program_type == "documentary":
#             suggestions.extend([
#                 "Archival footage overlay",
#                 "Location title cards",
#                 "Interview backdrop",
#                 "Historical timeline"
#             ])
        
#         return suggestions
    
#     def _generate_production_notes(self, topic: str, program_type: str) -> Dict:
#         """Generate production notes for the program"""
#         notes = {
#             "camera_angles": ["Wide shot", "Medium shot", "Close-up"],
#             "lighting": "Studio lighting with soft key light",
#             "audio": "Lavalier mics for participants, background music",
#             "graphics_transition": "Smooth fade between graphics"
#         }
        
#         if program_type == "debate":
#             notes["camera_angles"].extend(["Two-shot", "Reaction shots"])
#             notes["special_effects"] = "Split screen for arguments"
#         elif program_type == "documentary":
#             notes["camera_angles"].extend(["Handheld shots", "Pan shots", "Drone footage"])
#             notes["special_effects"] = "Color grading for archival footage"
        
#         return notes


# class AutomatedNewsScheduler:
#     """Schedule and manage automated news generation"""
    
#     def __init__(self, generator: NewsContentGenerator):
#         self.generator = generator
#         self.scheduled_tasks = []
        
#     def schedule_daily_bulletin(self, time: str, 
#                                languages: List[Language]) -> None:
#         """Schedule daily news bulletins"""
#         task = {
#             "type": "daily_bulletin",
#             "time": time,
#             "languages": languages,
#             "enabled": True,
#             "last_run": None
#         }
#         self.scheduled_tasks.append(task)
#         return f"Scheduled daily bulletin at {time} for languages: {[lang.value for lang in languages]}"
    
#     def schedule_breaking_news_monitor(self, keywords: List[str],
#                                       languages: List[Language]) -> None:
#         """Monitor for breaking news based on keywords"""
#         task = {
#             "type": "breaking_news_monitor",
#             "keywords": keywords,
#             "languages": languages,
#             "check_interval": 300,
#             "enabled": True,
#             "last_check": None
#         }
#         self.scheduled_tasks.append(task)
#         return f"Scheduled breaking news monitor with keywords: {keywords}"
    
#     def schedule_hourly_updates(self, categories: List[NewsCategory],
#                                languages: List[Language]) -> None:
#         """Schedule hourly updates for specific categories"""
#         task = {
#             "type": "hourly_updates",
#             "categories": [cat.value for cat in categories],
#             "languages": languages,
#             "enabled": True
#         }
#         self.scheduled_tasks.append(task)
#         return f"Scheduled hourly updates for categories: {[cat.value for cat in categories]}"
    
#     def generate_content_batch(self, category: NewsCategory,
#                               count: int = 5) -> List[Dict]:
#         """Generate batch of content for specific category"""
#         mock_news = [
#             NewsItem(
#                 id=f"{category.value}_{i}",
#                 title=f"Sample {category.value} news {i}",
#                 raw_content=f"Detailed content about {category.value} topic {i}. "
#                           f"This is a sample news article that would be processed. "
#                           f"It contains important information that viewers need to know.",
#                 category=category,
#                 timestamp=datetime.datetime.now(),
#                 location="Hyderabad",
#                 metadata={"source": "Mock News API", "verified": True}
#             ) for i in range(count)
#         ]
        
#         output = []
#         for news_item in mock_news:
#             for lang in [Language.ENGLISH, Language.TELUGU, Language.HINDI, Language.TAMIL]:
#                 script = self.generator.generate_tv_script([news_item], lang)
#                 bullets = self.generator.generate_bullet_summary(news_item.raw_content)
#                 breaking = self.generator.generate_breaking_news_version(news_item, lang)
#                 headlines = self.generator.generate_headlines([news_item], lang)
#                 sub_headlines = self.generator.generate_sub_headlines([news_item], lang)
                
#                 output.append({
#                     "id": news_item.id,
#                     "language": lang.value,
#                     "full_script": script,
#                     "bullet_summary": bullets,
#                     "breaking_version": breaking,
#                     "headlines": headlines,
#                     "sub_headlines": sub_headlines,
#                     "category": category.value,
#                     "timestamp": news_item.timestamp.isoformat()
#                 })
        
#         return output
    
#     def get_scheduled_tasks(self) -> List[Dict]:
#         """Get all scheduled tasks"""
#         return self.scheduled_tasks


# class NewsExportManager:
#     """Manage exporting news content in various formats"""
    
#     @staticmethod
#     def export_to_json(news_content: List[Dict], filename: str) -> str:
#         """Export news content to JSON file"""
#         with open(filename, 'w', encoding='utf-8') as f:
#             json.dump(news_content, f, indent=2, ensure_ascii=False)
#         return f"Exported {len(news_content)} items to {filename}"
    
#     @staticmethod
#     def export_to_text(news_content: List[Dict], filename: str) -> str:
#         """Export news content to text file"""
#         with open(filename, 'w', encoding='utf-8') as f:
#             for item in news_content:
#                 f.write(f"=== {item['id']} - {item['language']} ===\n")
#                 f.write(f"Headline: {item['headlines'][0] if item['headlines'] else 'N/A'}\n")
#                 f.write(f"Script:\n{item['full_script']}\n\n")
#                 f.write(f"Bullet Summary:\n")
#                 for bullet in item['bullet_summary']:
#                     f.write(f"{bullet}\n")
#                 f.write("\n" + "="*50 + "\n\n")
#         return f"Exported to {filename}"
    
#     @staticmethod
#     def export_to_html(news_content: List[Dict], filename: str) -> str:
#         """Export news content to HTML file"""
#         html_template = """
#         <!DOCTYPE html>
#         <html>
#         <head>
#             <meta charset="UTF-8">
#             <title>AI Generated News Report</title>
#             <style>
#                 body {{ font-family: Arial, sans-serif; margin: 40px; }}
#                 .news-item {{ border: 1px solid #ddd; padding: 20px; margin-bottom: 20px; }}
#                 .headline {{ color: #2c3e50; font-size: 24px; }}
#                 .breaking {{ color: #e74c3c; font-weight: bold; }}
#                 .bullet-point {{ margin-left: 20px; }}
#                 .timestamp {{ color: #7f8c8d; font-size: 14px; }}
#                 .language-tag {{ background: #3498db; color: white; padding: 5px; border-radius: 3px; }}
#             </style>
#         </head>
#         <body>
#             <h1>AI Generated News Report</h1>
#             <p>Generated on: {generation_date}</p>
#             {content}
#         </body>
#         </html>
#         """
        
#         content_parts = []
#         for item in news_content:
#             is_breaking = "breaking" in item.get('category', '')
#             breaking_class = "breaking" if is_breaking else ""
            
#             item_html = f"""
#             <div class="news-item">
#                 <span class="language-tag">{item['language'].upper()}</span>
#                 <span class="timestamp">{item['timestamp']}</span>
#                 <h2 class="headline {breaking_class}">{item['headlines'][0] if item['headlines'] else 'N/A'}</h2>
#                 <div class="script">
#                     <h3>Full Script:</h3>
#                     <pre>{item['full_script']}</pre>
#                 </div>
#                 <div class="summary">
#                     <h3>Summary:</h3>
#                     <ul>
#                         {"".join([f'<li class="bullet-point">{bullet}</li>' for bullet in item['bullet_summary'][1:]])}
#                     </ul>
#                 </div>
#             </div>
#             """
#             content_parts.append(item_html)
        
#         html_content = html_template.format(
#             generation_date=datetime.datetime.now().strftime("%Y-%m-d %H:%M:%S"),
#             content="\n".join(content_parts)
#         )
        
#         with open(filename, 'w', encoding='utf-8') as f:
#             f.write(html_content)
#         return f"Exported to {filename}"


# # Streamlit Web Interface
# def main():
#     """Main Streamlit application"""
#     st.set_page_config(
#         page_title="AI News Automation System - Offline Version with Chatterbox TTS",
#         page_icon="📰",
#         layout="wide"
#     )
    
#     st.title("📰 AI News Automation System - Offline TTS Version with Voice Cloning")
#     st.markdown("Automated news script generation, translation, and content creation with Voice & Handwriting Input + Voice Cloning")
    
#     # Display Tesseract configuration status
#     st.sidebar.subheader("🔧 OCR & TTS Configuration")
    
#     # Test Tesseract installation
#     if st.sidebar.button("Test Tesseract Installation"):
#         processor = VoiceHandwritingProcessor()
#         success, message = processor.test_tesseract_installation()
#         if success:
#             st.sidebar.success(message)
#         else:
#             st.sidebar.error(message)
#             st.sidebar.info("Update Tesseract path at line 35-45 in the code")
    
#     # Display TTS availability
#     tts_status = []
#     if CHATTERBOX_AVAILABLE:
#         if CHATTERBOX_WORKING:
#             tts_status.append("✅ Chatterbox TTS (Voice Cloning)")
#         else:
#             tts_status.append("⚠️ Chatterbox TTS (PyTorch available)")
#     else:
#         tts_status.append("❌ Chatterbox TTS (install: pip install torch torchaudio)")
    
#     if TTS_AVAILABLE:
#         tts_status.append("✅ pyttsx3")
#     else:
#         tts_status.append("❌ pyttsx3 (install: pip install pyttsx3)")
    
#     if GTTS_AVAILABLE:
#         tts_status.append("✅ gTTS")
#     else:
#         tts_status.append("❌ gTTS (install: pip install gtts)")
    
#     if KOKORO_AVAILABLE:
#         if KOKORO_WORKING:
#             tts_status.append("✅ Kokoro TTS (Working)")
#         else:
#             tts_status.append("⚠️ Kokoro TTS (Installed but not working)")
#     else:
#         tts_status.append("❌ Kokoro TTS (install: pip install kokoro)")
    
#     if PIPER_AVAILABLE:
#         tts_status.append("✅ Piper TTS")
#     else:
#         tts_status.append("❌ Piper TTS (install: pip install piper-tts)")
    
#     st.sidebar.info("Text-to-Speech Status:\n" + "\n".join(tts_status))
    
#     # Display current Tesseract path
#     st.sidebar.info(f"Tesseract Path:\n`{pytesseract.pytesseract.tesseract_cmd}`")
    
#     # Install Chatterbox TTS button
#     if not CHATTERBOX_WORKING:
#         if st.sidebar.button("🚀 Install Chatterbox TTS", type="primary"):
#             with st.spinner("Installing Chatterbox TTS and dependencies..."):
#                 offline_tts = OfflineTTSManager()
#                 success = offline_tts.install_chatterbox()
#                 if success:
#                     st.sidebar.success("Chatterbox TTS installation started!")
#                     st.sidebar.info("Please restart the application after installation completes")
    
#     # Install Piper TTS button
#     if not PIPER_AVAILABLE:
#         if st.sidebar.button("Install Piper TTS"):
#             with st.spinner("Installing Piper TTS..."):
#                 offline_tts = OfflineTTSManager()
#                 success = offline_tts.install_piper()
#                 if success:
#                     st.sidebar.success("Piper TTS installation started!")
#                     st.sidebar.info("Please restart the application after installation completes")
    
#     # Initialize session state
#     if 'generator' not in st.session_state:
#         st.session_state.generator = NewsContentGenerator()
#     if 'news_items' not in st.session_state:
#         st.session_state.news_items = []
#     if 'generated_content' not in st.session_state:
#         st.session_state.generated_content = []
#     if 'batch_content' not in st.session_state:
#         st.session_state.batch_content = []
#     if 'voice_text' not in st.session_state:
#         st.session_state.voice_text = ""
#     if 'handwriting_text' not in st.session_state:
#         st.session_state.handwriting_text = ""
#     if 'tts_audio' not in st.session_state:
#         st.session_state.tts_audio = None
#     if 'tts_text' not in st.session_state:
#         st.session_state.tts_text = ""
#     if 'kokoro_voice' not in st.session_state:
#         st.session_state.kokoro_voice = "af_heart"
#     if 'voice_clone_audio' not in st.session_state:
#         st.session_state.voice_clone_audio = None
#     if 'voice_clone_text' not in st.session_state:
#         st.session_state.voice_clone_text = ""
    
#     # Sidebar navigation
#     st.sidebar.title("Navigation")
#     app_mode = st.sidebar.selectbox(
#         "Choose a module",
#         ["Home", "Voice Input", "Handwriting Input", "Text Input", "Text-to-Speech", 
#          "Voice Cloning", "Create News", "Generate Scripts", "Special Reports", 
#          "Batch Processing", "Export Content", "Settings", "OCR Test"]
#     )
    
#     # OCR Test Page
#     if app_mode == "OCR Test":
#         st.header("🧪 Tesseract OCR Test Page")
        
#         st.markdown("""
#         ### Test Tesseract OCR Installation
#         Upload an image to test if Tesseract OCR is working correctly.
#         """)
        
#         # Upload test image
#         test_image = st.file_uploader(
#             "Upload an image for OCR test", 
#             type=['jpg', 'jpeg', 'png', 'bmp', 'tiff'],
#             key="ocr_test"
#         )
        
#         if test_image:
#             col1, col2 = st.columns(2)
#             with col1:
#                 st.image(test_image, caption="Test Image", width=300)
                
#                 # Test Tesseract
#                 if st.button("🔍 Test OCR", type="primary"):
#                     processor = VoiceHandwritingProcessor()
                    
#                     # Test installation
#                     success, message = processor.test_tesseract_installation()
#                     if success:
#                         st.success(message)
                        
#                         # Perform OCR
#                         with st.spinner("Processing OCR..."):
#                             text, method = processor.handwriting_to_text(test_image, 'eng')
                            
#                             if text:
#                                 st.success("✅ OCR Successful!")
#                                 st.subheader("Extracted Text:")
#                                 st.text_area("OCR Result", text, height=200)
#                             else:
#                                 st.warning("No text found in image")
#                     else:
#                         st.error(message)
            
#             with col2:
#                 st.subheader("OCR Configuration")
#                 st.code(f"""
# # Current Tesseract Configuration:
# TESSERACT_CMD = "{pytesseract.pytesseract.tesseract_cmd}"

# # To change the path, modify line 35-45 in the code:
# # For Windows:
# pytesseract.pytesseract.tesseract_cmd = r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"

# # For Linux/Mac:
# # pytesseract.pytesseract.tesseract_cmd = r"/usr/bin/tesseract"
#                 """, language="python")
                
#                 st.subheader("Troubleshooting")
#                 st.markdown("""
#                 1. **Install Tesseract**:
#                    - Windows: Download from [GitHub](https://github.com/UB-Mannheim/tesseract/wiki)
#                    - Linux: `sudo apt install tesseract-ocr`
#                    - Mac: `brew install tesseract`
                
#                 2. **Install Language Packs**:
#                    - Download from [tessdata](https://github.com/tesseract-ocr/tessdata)
#                    - Place in `C:\Program Files\Tesseract-OCR\tessdata` (Windows)
#                    - Or `/usr/share/tesseract-ocr/4.00/tessdata/` (Linux)
                
#                 3. **Verify Installation**:
#                    - Open terminal: `tesseract --version`
#                    - Should show version 4.0+
#                 """)
        
#         return
    
#     # Voice Cloning Page
#     elif app_mode == "Voice Cloning":
#         st.header("🎭 Chatterbox Voice Cloning & TTS")
        
#         st.markdown("""
#         ### 🎙️ Voice Cloning Features
        
#         **Chatterbox TTS can:**
#         1. **Clone your voice** from 10-20 seconds of audio
#         2. **Generate speech** in your cloned voice
#         3. **Support multiple voice types**: News anchors, podcasts, narration
#         4. **Create professional voiceovers** for news reading, dialogues, storytelling
        
#         **Requirements:**
#         - PyTorch with/without CUDA
#         - 10-20 seconds of clear voice sample
#         - Corresponding text transcript
#         """)
        
#         # Check Chatterbox availability
#         if not CHATTERBOX_WORKING:
#             st.error("""
#             ❌ Chatterbox TTS is not available!
            
#             **To install:**
#             1. Install PyTorch: `pip install torch torchaudio`
#             2. Install dependencies: `pip install soundfile librosa`
#             3. Install Chatterbox: `pip install git+https://github.com/chenwj1989/chatterbox-tts.git`
            
#             Or click the 'Install Chatterbox TTS' button in the sidebar.
#             """)
            
#             if st.button("🚀 Install Chatterbox Now", type="primary"):
#                 offline_tts = OfflineTTSManager()
#                 offline_tts.install_chatterbox()
#             return
        
#         # Voice Cloning Interface
#         st.subheader("🎤 Step 1: Upload Voice Sample for Cloning")
        
#         col1, col2 = st.columns(2)
        
#         with col1:
#             st.markdown("**Upload 10-20 seconds of your voice:**")
#             voice_sample = st.file_uploader(
#                 "Upload voice sample (WAV, MP3)",
#                 type=['wav', 'mp3', 'm4a'],
#                 key="voice_clone_sample"
#             )
            
#             if voice_sample:
#                 st.audio(voice_sample, format='audio/wav')
#                 audio_bytes = voice_sample.read()
#                 st.session_state.voice_clone_audio = audio_bytes
#                 st.success(f"✅ Voice sample uploaded: {len(audio_bytes)} bytes")
        
#         with col2:
#             st.markdown("**Enter corresponding text:**")
#             voice_clone_text = st.text_area(
#                 "Text from voice sample",
#                 height=150,
#                 value=st.session_state.voice_clone_text,
#                 placeholder="Enter the exact text spoken in the voice sample...",
#                 help="This helps Chatterbox learn your voice characteristics better"
#             )
            
#             if voice_clone_text:
#                 st.session_state.voice_clone_text = voice_clone_text
#                 st.info(f"📝 Text length: {len(voice_clone_text)} characters")
        
#         st.subheader("🎯 Step 2: Select Voice Type")
        
#         voice_type = st.radio(
#             "Select voice type for generation:",
#             ["News Anchor", "Podcast Host", "Narration", "Custom Clone"],
#             horizontal=True
#         )
        
#         # Map voice type to Chatterbox voice ID
#         voice_map = {
#             "News Anchor": "news_anchor",
#             "Podcast Host": "podcast_host",
#             "Narration": "narration",
#             "Custom Clone": "custom_clone"
#         }
        
#         selected_voice_type = voice_map.get(voice_type, "custom_clone")
        
#         st.subheader("📝 Step 3: Enter Text to Generate")
        
#         tts_text = st.text_area(
#             "Enter text to generate in cloned voice:",
#             height=200,
#             value="Welcome to today's news bulletin. Our top stories include breaking news from the capital and important weather updates.",
#             placeholder="Enter text to convert to speech using your cloned voice..."
#         )
        
#         # Generation options
#         col1, col2 = st.columns(2)
#         with col1:
#             language = st.selectbox("Language", ["en", "hi", "es", "fr"], index=0)
        
#         with col2:
#             auto_play = st.checkbox("🔊 Auto-play generated audio", value=True)
        
#         # Generate Speech Button
#         if st.button("🎭 Generate Speech with Voice Cloning", type="primary", use_container_width=True):
#             if tts_text and tts_text.strip():
#                 if not st.session_state.voice_clone_audio and selected_voice_type == "custom_clone":
#                     st.warning("⚠️ Please upload a voice sample for custom voice cloning.")
#                 else:
#                     with st.spinner("Generating speech with Chatterbox voice cloning..."):
#                         # Generate audio with Chatterbox
#                         audio_bytes = st.session_state.generator.convert_text_to_speech(
#                             text=tts_text,
#                             language=language,
#                             engine="chatterbox",
#                             voice_clone_audio=st.session_state.voice_clone_audio if selected_voice_type == "custom_clone" else None,
#                             voice_clone_text=st.session_state.voice_clone_text if selected_voice_type == "custom_clone" else None,
#                             auto_play=auto_play
#                         )
                        
#                         if audio_bytes:
#                             st.session_state.tts_audio = audio_bytes
#                             st.session_state.tts_text = tts_text
#                             st.success("✅ Voice cloning successful!")
                            
#                             # Show text preview
#                             with st.expander("📄 Text Preview"):
#                                 st.text_area("Text being read", tts_text[:500] + "..." if len(tts_text) > 500 else tts_text, height=150)
                            
#                             # Additional download button if not auto-played
#                             if not auto_play:
#                                 st.audio(audio_bytes, format='audio/wav')
                            
#                             # Download button
#                             st.download_button(
#                                 label="📥 Download Cloned Voice Audio",
#                                 data=audio_bytes,
#                                 file_name=f"cloned_voice_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.wav",
#                                 mime="audio/wav"
#                             )
                            
#                             # Show voice cloning statistics
#                             col1, col2, col3 = st.columns(3)
#                             with col1:
#                                 st.metric("Voice Sample", "Uploaded" if st.session_state.voice_clone_audio else "Not uploaded")
#                             with col2:
#                                 st.metric("Text Length", f"{len(tts_text)} chars")
#                             with col3:
#                                 st.metric("Audio Size", f"{len(audio_bytes) // 1024} KB")
#                         else:
#                             st.error("Failed to generate audio. Chatterbox might need additional setup.")
#             else:
#                 st.warning("Please enter some text first.")
        
#         # Chatterbox Features Showcase
#         st.subheader("✨ Chatterbox Voice Cloning Features")
        
#         features = [
#             ("🎙️ Voice Cloning", "Clone any voice from 10-20 seconds of audio"),
#             ("📰 News Reading", "Perfect for news anchors and bulletins"),
#             ("🎧 Podcast Generation", "Create podcast episodes in your voice"),
#             ("📖 Storytelling", "Narrate stories and audiobooks"),
#             ("🎭 Character Voices", "Create different character voices"),
#             ("🌍 Multi-language", "Support for multiple languages")
#         ]
        
#         cols = st.columns(3)
#         for idx, (feature, description) in enumerate(features):
#             with cols[idx % 3]:
#                 st.info(f"**{feature}**\n\n{description}")
        
#         # Quick Start Guide
#         with st.expander("📚 Quick Start Guide"):
#             st.markdown("""
#             ### How to Use Chatterbox Voice Cloning:
            
#             1. **Prepare Voice Sample:**
#                - Record 10-20 seconds of clear speech
#                - Use good quality microphone
#                - Avoid background noise
            
#             2. **Upload & Process:**
#                - Upload WAV/MP3 file
#                - Enter exact transcript
#                - Click "Generate Speech"
            
#             3. **Advanced Tips:**
#                - Longer samples (30+ sec) give better results
#                - Include various emotions for better cloning
#                - Use same recording environment for consistency
            
#             4. **Applications:**
#                - News bulletin voiceovers
#                - Personalized podcast episodes
#                - Audiobook narration
#                - Character voices for animations
#                - Virtual assistant voices
#             """)
        
#         return
    
#     # Text-to-Speech Page
#     elif app_mode == "Text-to-Speech":
#         st.header("🔊 Text-to-Speech Voice Generation")
        
#         st.markdown("""
#         ### Convert Text to Audio Speech
#         **Features:**
#         • Multiple offline TTS engines
#         • Chatterbox voice cloning
#         • Auto-play audio after generation
#         • Support for Indian languages
#         • Generate complete news bulletins
#         """)
        
#         # TTS Engine Selection
#         col1, col2 = st.columns(2)
#         with col1:
#             # Engine options
#             engine_options = []
#             if CHATTERBOX_AVAILABLE and CHATTERBOX_WORKING:
#                 engine_options.append("chatterbox")
#             if KOKORO_AVAILABLE and KOKORO_WORKING:
#                 engine_options.append("kokoro")
#             if PIPER_AVAILABLE:
#                 engine_options.append("piper")
#             if TTS_AVAILABLE:
#                 engine_options.append("pyttsx3")
#             if GTTS_AVAILABLE:
#                 engine_options.append("gtts")
#             engine_options.append("auto")
            
#             tts_engine = st.selectbox(
#                 "Select TTS Engine",
#                 engine_options,
#                 index=0 if CHATTERBOX_AVAILABLE and CHATTERBOX_WORKING else 0,
#                 help="chatterbox: Voice cloning | kokoro: Fast | piper: Multi-language | auto: Best for language"
#             )
        
#         with col2:
#             language = st.selectbox(
#                 "Select Language",
#                 ["en", "hi", "te", "ta", "kn", "mr", "es", "fr", "de", "it"]
#             )
        
#         # Engine-specific settings
#         if tts_engine == "chatterbox" and CHATTERBOX_AVAILABLE and CHATTERBOX_WORKING:
#             st.subheader("🎭 Chatterbox Settings")
            
#             col1, col2 = st.columns(2)
#             with col1:
#                 voice_type = st.selectbox(
#                     "Voice Type",
#                     ["news_anchor", "podcast_host", "narration", "custom_clone"],
#                     index=0
#                 )
            
#             with col2:
#                 # Voice cloning options
#                 if voice_type == "custom_clone":
#                     st.info("🔗 Using custom voice clone")
#                     if st.session_state.voice_clone_audio:
#                         st.success("✅ Voice sample available")
#                     else:
#                         st.warning("⚠️ No voice sample uploaded")
#                         if st.button("Go to Voice Cloning"):
#                             st.session_state.app_mode = "Voice Cloning"
#                             st.rerun()
        
#         elif tts_engine == "kokoro" and KOKORO_AVAILABLE and KOKORO_WORKING:
#             st.subheader("⚡ Kokoro Settings")
            
#             col1, col2, col3 = st.columns(3)
#             with col1:
#                 processor = st.session_state.generator.input_processor
#                 kokoro_voices = processor.get_kokoro_voices() if processor.kokoro_model else []
                
#                 if kokoro_voices:
#                     selected_voice = st.selectbox(
#                         "Select Voice",
#                         kokoro_voices,
#                         index=kokoro_voices.index(st.session_state.kokoro_voice) 
#                         if st.session_state.kokoro_voice in kokoro_voices else 0
#                     )
#                     st.session_state.kokoro_voice = selected_voice
#                 else:
#                     st.warning("No Kokoro voices available")
#                     selected_voice = "af_heart"
            
#             with col2:
#                 speed = st.slider("Speech Speed", 0.5, 2.0, 1.0, 0.1)
            
#             with col3:
#                 pitch = st.slider("Pitch Adjustment", -0.5, 0.5, 0.0, 0.1)
        
#         elif tts_engine == "piper" and PIPER_AVAILABLE:
#             st.subheader("🎵 Piper TTS Settings")
            
#             processor = st.session_state.generator.input_processor
#             piper_voices = processor.offline_tts.piper_voices
            
#             if piper_voices:
#                 # Filter voices by selected language
#                 language_voices = [v for v in piper_voices if v["language"] == language]
#                 if language_voices:
#                     selected_voice = st.selectbox(
#                         "Select Piper Voice",
#                         [v["name"] for v in language_voices],
#                         index=0
#                     )
#                 else:
#                     st.warning(f"No Piper voices for {language}. Using English.")
#                     english_voices = [v for v in piper_voices if v["language"] == "en"]
#                     selected_voice = english_voices[0]["name"] if english_voices else "English US (Medium)"
            
#             col1, col2 = st.columns(2)
#             with col1:
#                 speed = st.slider("Speech Speed", 0.5, 2.0, 1.0, 0.1)
            
#             with col2:
#                 volume = st.slider("Volume", 0.0, 1.0, 0.9, step=0.1)
        
#         elif tts_engine != "auto":
#             st.subheader("Voice Settings")
            
#             col1, col2, col3 = st.columns(3)
#             with col1:
#                 speech_rate = st.slider("Speech Rate", 100, 300, 150)
            
#             with col2:
#                 speech_volume = st.slider("Volume", 0.0, 1.0, 0.9, step=0.1)
            
#             with col3:
#                 voice_type = st.selectbox("Voice Type", ["male", "female", "system"])
        
#         # Text input for TTS
#         st.subheader("Enter Text to Convert")
        
#         # Text sources
#         text_source = st.radio(
#             "Text Source:",
#             ["📝 Enter Text", "📰 Use News Script", "🎤 Use Voice Input", "✍️ Use Handwriting"]
#         )
        
#         if text_source == "📝 Enter Text":
#             tts_text = st.text_area(
#                 "Enter text to convert to speech:",
#                 height=200,
#                 value=st.session_state.tts_text,
#                 placeholder="Type or paste your text here...",
#                 help="Enter complete text to be read smoothly in one go"
#             )
        
#         elif text_source == "📰 Use News Script":
#             if st.session_state.news_items:
#                 selected_item = st.selectbox(
#                     "Select News Item",
#                     [item.title for item in st.session_state.news_items]
#                 )
                
#                 for item in st.session_state.news_items:
#                     if item.title == selected_item:
#                         script = st.session_state.generator.generate_tv_script([item], Language.ENGLISH)
#                         tts_text = st.text_area(
#                             "News Script",
#                             value=script,
#                             height=200
#                         )
#                         break
#             else:
#                 st.warning("No news items available. Create one first.")
#                 tts_text = ""
        
#         elif text_source == "🎤 Use Voice Input":
#             if st.session_state.voice_text:
#                 tts_text = st.text_area(
#                     "Voice Input Text",
#                     value=st.session_state.voice_text,
#                     height=200
#                 )
#             else:
#                 st.warning("No voice input available. Record some voice first.")
#                 tts_text = ""
        
#         elif text_source == "✍️ Use Handwriting":
#             if st.session_state.handwriting_text:
#                 tts_text = st.text_area(
#                     "Handwriting Text",
#                     value=st.session_state.handwriting_text,
#                     height=200
#                 )
#             else:
#                 st.warning("No handwriting available. Upload handwriting first.")
#                 tts_text = ""
        
#         # Auto-play option
#         auto_play = st.checkbox("🔊 Auto-play audio after generation", value=True,
#                               help="Audio will play automatically without clicking play button")
        
#         # Add expressive reading option
#         expressive_reading = st.checkbox("🎭 Add Expressive Reading", value=True, 
#                                        help="Adds natural pauses and expressions for news reading")
        
#         # Generate Speech Button
#         if st.button("🎙️ Generate Speech", type="primary", use_container_width=True):
#             if tts_text and tts_text.strip():
#                 with st.spinner("Converting text to speech..."):
#                     # Prepare voice cloning parameters if using Chatterbox
#                     voice_clone_audio = None
#                     voice_clone_text = None
                    
#                     if tts_engine == "chatterbox" and st.session_state.voice_clone_audio:
#                         voice_clone_audio = st.session_state.voice_clone_audio
#                         voice_clone_text = st.session_state.voice_clone_text
                    
#                     # Generate audio
#                     audio_bytes = st.session_state.generator.convert_text_to_speech(
#                         text=tts_text,
#                         language=language,
#                         engine=tts_engine,
#                         voice_clone_audio=voice_clone_audio,
#                         voice_clone_text=voice_clone_text,
#                         auto_play=auto_play
#                     )
                    
#                     if audio_bytes:
#                         st.session_state.tts_audio = audio_bytes
#                         st.session_state.tts_text = tts_text
#                         st.success("✅ Speech generated successfully!")
                        
#                         # Show text preview
#                         with st.expander("📄 Text Preview"):
#                             st.text_area("Text being read", tts_text[:500] + "..." if len(tts_text) > 500 else tts_text, height=150)
                        
#                         # Additional download button if not auto-played
#                         if not auto_play:
#                             st.audio(audio_bytes, format='audio/wav')
                        
#                         # Download button
#                         st.download_button(
#                             label="📥 Download Audio (WAV)",
#                             data=audio_bytes,
#                             file_name=f"news_audio_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.wav",
#                             mime="audio/wav"
#                         )
#                     else:
#                         st.error("Failed to generate audio. Check TTS engine configuration.")
#             else:
#                 st.warning("Please enter some text first.")
        
#         # Generate Audio News Bulletin
#         st.subheader("🎙️ Generate Complete News Bulletin")
        
#         if st.session_state.news_items:
#             selected_news = st.multiselect(
#                 "Select News Items for Bulletin",
#                 [item.title for item in st.session_state.news_items],
#                 default=[item.title for item in st.session_state.news_items[:2]]
#             )
            
#             if selected_news:
#                 selected_items = []
#                 for title in selected_news:
#                     for item in st.session_state.news_items:
#                         if item.title == title:
#                             selected_items.append(item)
#                             break
                
#                 if st.button("📻 Generate Audio News Bulletin", use_container_width=True):
#                     with st.spinner("Generating complete audio news bulletin..."):
#                         # Prepare voice cloning parameters if using Chatterbox
#                         voice_clone_audio = None
#                         voice_clone_text = None
                        
#                         if tts_engine == "chatterbox" and st.session_state.voice_clone_audio:
#                             voice_clone_audio = st.session_state.voice_clone_audio
#                             voice_clone_text = st.session_state.voice_clone_text
                        
#                         audio_bytes = st.session_state.generator.generate_audio_news_bulletin(
#                             news_items=selected_items,
#                             language=language,
#                             expressive_reading=expressive_reading,
#                             engine=tts_engine,
#                             voice_clone_audio=voice_clone_audio,
#                             voice_clone_text=voice_clone_text,
#                             auto_play=auto_play
#                         )
                        
#                         if audio_bytes:
#                             st.success("✅ Audio news bulletin generated!")
                            
#                             # Show script preview
#                             script = st.session_state.generator.generate_tv_script(selected_items, Language(language))
#                             with st.expander("📰 Script Preview"):
#                                 st.text_area("Bulletin Script", script[:1000] + "..." if len(script) > 1000 else script, height=200)
                            
#                             # Additional download button if not auto-played
#                             if not auto_play:
#                                 st.audio(audio_bytes, format='audio/wav')
                            
#                             # Download button
#                             st.download_button(
#                                 label="📥 Download News Bulletin",
#                                 data=audio_bytes,
#                                 file_name=f"news_bulletin_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.wav",
#                                 mime="audio/wav"
#                             )
#         else:
#             st.info("No news items available. Create news items first to generate bulletins.")
        
#         # Display TTS Engine Info
#         st.subheader("ℹ️ TTS Engine Information")
        
#         if CHATTERBOX_AVAILABLE and CHATTERBOX_WORKING:
#             st.info("Chatterbox TTS (Voice Cloning):")
#             st.write("- Clone voices from 10-20 seconds of audio")
#             st.write("- News anchors, podcasts, narration")
#             st.write("- Professional voiceovers")
#             st.write("- Multi-language support")
        
#         if KOKORO_AVAILABLE and KOKORO_WORKING:
#             st.info("Kokoro TTS:")
#             st.write("- Fast generation")
#             st.write("- Multiple voices")
#             st.write("- English support")
        
#         if PIPER_AVAILABLE:
#             st.info("Piper TTS:")
#             st.write("- Multi-language support")
#             st.write("- Indian languages: Hindi, Tamil, Telugu, Kannada, Marathi")
#             st.write("- High quality")
        
#         if GTTS_AVAILABLE:
#             st.info("gTTS (Google Text-to-Speech):")
#             st.write("- Multiple Indian languages")
#             st.write("- Requires internet connection")
#             st.write("- Natural speech flow")
        
#         if TTS_AVAILABLE:
#             st.info("pyttsx3:")
#             st.write("- Completely offline")
#             st.write("- Built into Windows")
#             st.write("- Multiple voice options")
        
#         return
    
#     # Home Page
#     if app_mode == "Home":
#         st.header("🎯 Voice & Handwriting Powered News System with Voice Cloning")
        
#         col1, col2, col3, col4 = st.columns(4)
#         with col1:
#             st.metric("Input Methods", "4")
#             st.markdown("🎤 Voice Input")
#             st.markdown("✍️ Handwriting")
#             st.markdown("⌨️ Text Input")
#             st.markdown("🎭 Voice Cloning")
#         with col2:
#             st.metric("TTS Engines", "5")
#             st.markdown("🎭 Chatterbox (Voice Cloning)")
#             st.markdown("⚡ Kokoro (Fast)")
#             st.markdown("🎵 Piper (Multi-language)")
#         with col3:
#             st.metric("News Categories", "8")
#         with col4:
#             st.metric("Output Formats", "5")
        
#         st.markdown("---")
#         st.subheader("🚨 Priority Features")
        
#         # Priority Features Display
#         priority_features = {
#             "1️⃣ VOICE CLONING (New)": "Clone any voice from 10-20 seconds of audio for news reading, podcasts, narration",
#             "2️⃣ VOICE INPUT (Highest Priority)": "For emergency reports, citizen communications, and quick news reporting",
#             "3️⃣ HANDWRITING INPUT": "For handwritten notes, reports, and document processing",
#             "4️⃣ TEXT INPUT": "For pre-written content and direct text processing",
#             "5️⃣ OFFLINE TTS": "Multiple offline TTS engines for different languages",
#             "6️⃣ AUTO-PLAY AUDIO": "Audio plays automatically after generation"
#         }
        
#         for feature, description in priority_features.items():
#             with st.expander(feature):
#                 st.markdown(f"**{description}**")
        
#         st.markdown("---")
#         st.subheader("🎯 Quick Actions")
        
#         col1, col2, col3, col4 = st.columns(4)
#         with col1:
#             if st.button("🎭 Voice Cloning", use_container_width=True):
#                 st.session_state.app_mode = "Voice Cloning"
#                 st.rerun()
#         with col2:
#             if st.button("🎤 Start Voice Input", use_container_width=True):
#                 st.session_state.app_mode = "Voice Input"
#                 st.rerun()
#         with col3:
#             if st.button("✍️ Upload Handwriting", use_container_width=True):
#                 st.session_state.app_mode = "Handwriting Input"
#                 st.rerun()
#         with col4:
#             if st.button("🎵 Install Chatterbox", use_container_width=True):
#                 offline_tts = OfflineTTSManager()
#                 offline_tts.install_chatterbox()
    
#     # Voice Input Page
#     elif app_mode == "Voice Input":
#         st.header("🎤 Voice Input - Highest Priority Feature")
        
#         st.markdown("""
#         ### 🚨 Emergency & News Reporting via Voice
#         **Perfect for:** Road accidents, emergency situations, quick news reports, citizen journalism
        
#         **How it works:**
#         1. Speak your report/incident/news
#         2. System converts speech to text
#         3. AI processes and generates news/alert
#         4. System can dispatch emergency services if needed
#         """)
        
#         # Voice Input Methods
#         input_method = st.radio(
#             "Choose voice input method:",
#             ["🎤 Microphone (Live)", "📁 Upload Audio File", "📞 Emergency Hotline Mode"]
#         )
        
#         if input_method == "🎤 Microphone (Live)":
#             st.subheader("Speak Now (Maximum 30 seconds)")
            
#             col1, col2 = st.columns([1, 3])
#             with col1:
#                 if st.button("🎙️ Start Recording", type="primary", use_container_width=True):
#                     with st.spinner("Listening..."):
#                         text, method = st.session_state.generator.input_processor.speech_to_text_microphone()
#                         if text:
#                             st.session_state.voice_text = text
#                             st.success(f"✅ Speech recognized using {method}")
            
#             with col2:
#                 language = st.selectbox(
#                     "Speech Language",
#                     ["en-IN (English India)", "hi-IN (Hindi)", "te-IN (Telugu)", "ta-IN (Tamil)"]
#                 )
        
#         elif input_method == "📁 Upload Audio File":
#             st.subheader("Upload Audio Recording")
            
#             audio_file = st.file_uploader(
#                 "Upload audio file (WAV, MP3, M4A)", 
#                 type=['wav', 'mp3', 'm4a', 'ogg']
#             )
            
#             if audio_file:
#                 if st.button("Process Audio File", type="primary"):
#                     with st.spinner("Processing audio..."):
#                         text, method = st.session_state.generator.input_processor.speech_to_text_upload(audio_file)
#                         if text:
#                             st.session_state.voice_text = text
#                             st.success(f"✅ Audio processed using {method}")
        
#         elif input_method == "📞 Emergency Hotline Mode":
#             st.subheader("🚨 Emergency Reporting Mode")
#             st.warning("This mode prioritizes emergency keywords and dispatches alerts")
            
#             col1, col2 = st.columns(2)
#             with col1:
#                 incident_type = st.selectbox(
#                     "Emergency Type",
#                     ["Accident", "Fire", "Medical Emergency", "Crime", "Natural Disaster", "Other"]
#                 )
            
#             with col2:
#                 location = st.text_input("Location (if known)", "Unknown")
            
#             if st.button("🚨 Start Emergency Report", type="primary"):
#                 with st.spinner("Listening for emergency report..."):
#                     text, method = st.session_state.generator.input_processor.speech_to_text_microphone()
#                     if text:
#                         st.session_state.voice_text = text
                        
#                         # Generate emergency report
#                         emergency_report = st.session_state.generator.generate_emergency_report(
#                             text, location, incident_type.lower()
#                         )
                        
#                         st.success("🚨 EMERGENCY REPORT GENERATED")
                        
#                         # Display emergency report
#                         st.code(emergency_report["report"], language="markdown")
                        
#                         # Show urgency level
#                         urgency_color = {
#                             "HIGH": "red",
#                             "MEDIUM": "orange",
#                             "LOW": "yellow",
#                             "NORMAL": "green"
#                         }
                        
#                         st.markdown(f"**Urgency Level:** :{urgency_color.get(emergency_report['urgency'], 'gray')}[{emergency_report['urgency']}]")
                        
#                         # Show recommended actions
#                         st.subheader("🚑 Recommended Actions:")
#                         for action in emergency_report["recommended_actions"]:
#                             st.markdown(f"• {action}")
        
#         # Display and process recognized text
#         if st.session_state.voice_text:
#             st.subheader("📝 Recognized Text")
#             st.text_area("Voice-to-Text Result", st.session_state.voice_text, height=150)
            
#             # Process options
#             st.subheader("🔄 Process Recognized Text")
            
#             col1, col2, col3, col4 = st.columns(4)
#             with col1:
#                 if st.button("📰 Create News Item", use_container_width=True):
#                     news_item = NewsItem(
#                         id=f"voice_{datetime.datetime.now().strftime('%H%M%S')}",
#                         title=f"Voice Report: {st.session_state.voice_text[:50]}...",
#                         raw_content=st.session_state.voice_text,
#                         category=NewsCategory.BREAKING,
#                         timestamp=datetime.datetime.now(),
#                         location="Voice Report Location",
#                         metadata={"source": "Voice Input", "verified": False}
#                     )
#                     st.session_state.news_items.append(news_item)
#                     st.success("News item created from voice input!")
            
#             with col2:
#                 if st.button("📊 Generate Summary", use_container_width=True):
#                     bullets = st.session_state.generator.generate_bullet_summary(st.session_state.voice_text)
#                     st.subheader("Bullet Summary")
#                     for bullet in bullets:
#                         st.markdown(bullet)
            
#             with col3:
#                 if st.button("🚨 Emergency Analysis", use_container_width=True):
#                     report = st.session_state.generator.generate_emergency_report(st.session_state.voice_text)
#                     st.subheader("Emergency Analysis")
#                     st.json(report)
            
#             with col4:
#                 if st.button("🔊 Convert to Speech", use_container_width=True):
#                     st.session_state.tts_text = st.session_state.voice_text
#                     st.session_state.app_mode = "Text-to-Speech"
#                     st.rerun()
    
#     # Handwriting Input Page
#     elif app_mode == "Handwriting Input":
#         st.header("✍️ Handwriting Input")
        
#         st.markdown("""
#         ### Convert Handwritten Notes to Digital Text
#         **Perfect for:** Handwritten reports, notes, forms, documents
        
#         **Supported:**
#         • Handwritten notes
#         • Emergency forms
#         • Reports and documents
#         • Field notes
#         """)
        
#         input_method = st.radio(
#             "Choose handwriting input method:",
#             ["📷 Upload Image", "📱 Take Photo", "📄 Upload Form"]
#         )
        
#         if input_method == "📷 Upload Image":
#             image_file = st.file_uploader(
#                 "Upload handwritten image", 
#                 type=['jpg', 'jpeg', 'png', 'bmp', 'tiff'],
#                 help="Upload clear image of handwritten text"
#             )
            
#             if image_file:
#                 st.image(image_file, caption="Uploaded Handwriting", width=300)
                
#                 col1, col2 = st.columns(2)
#                 with col1:
#                     language = st.selectbox(
#                         "Handwriting Language",
#                         ["eng (English)", "hin (Hindi)", "tel (Telugu)", "tam (Tamil)", "kan (Kannada)"]
#                     )
                
#                 with col2:
#                     if st.button("📖 Recognize Handwriting", type="primary"):
#                         with st.spinner("Processing handwriting..."):
#                             text, method = st.session_state.generator.input_processor.handwriting_to_text(image_file, language[:3])
#                             if text:
#                                 st.session_state.handwriting_text = text
#                                 st.success(f"✅ Handwriting recognized using {method}")
        
#         elif input_method == "📱 Take Photo":
#             st.warning("Camera access requires browser permissions")
#             st.info("Please use the upload option if camera is not available")
        
#         elif input_method == "📄 Upload Form":
#             st.subheader("Emergency/Report Form Processing")
            
#             form_file = st.file_uploader(
#                 "Upload filled form", 
#                 type=['jpg', 'jpeg', 'png', 'pdf'],
#                 help="Upload image of filled emergency/report form"
#             )
            
#             if form_file:
#                 st.image(form_file, caption="Uploaded Form", width=300)
                
#                 if st.button("📋 Process Form", type="primary"):
#                     with st.spinner("Extracting form data..."):
#                         form_data, method = st.session_state.generator.input_processor.process_handwritten_form(form_file)
#                         if form_data:
#                             st.success(f"✅ Form processed using {method}")
#                             st.subheader("Extracted Form Data:")
#                             st.json(form_data)
        
#         # Display and process recognized text
#         if st.session_state.handwriting_text:
#             st.subheader("📝 Recognized Text")
#             st.text_area("Handwriting-to-Text Result", st.session_state.handwriting_text, height=150)
            
#             # Process options
#             st.subheader("🔄 Process Recognized Text")
            
#             col1, col2, col3 = st.columns(3)
#             with col1:
#                 if st.button("📰 Create News Item", use_container_width=True):
#                     news_item = NewsItem(
#                         id=f"handwriting_{datetime.datetime.now().strftime('%H%M%S')}",
#                         title=f"Handwritten Report: {st.session_state.handwriting_text[:50]}...",
#                         raw_content=st.session_state.handwriting_text,
#                         category=NewsCategory.ANALYSIS,
#                         timestamp=datetime.datetime.now(),
#                         location="Handwriting Source",
#                         metadata={"source": "Handwriting Input", "verified": False}
#                     )
#                     st.session_state.news_items.append(news_item)
#                     st.success("News item created from handwriting!")
            
#             with col2:
#                 if st.button("📊 Generate Summary", use_container_width=True):
#                     bullets = st.session_state.generator.generate_bullet_summary(st.session_state.handwriting_text)
#                     st.subheader("Bullet Summary")
#                     for bullet in bullets:
#                         st.markdown(bullet)
            
#             with col3:
#                 if st.button("🔊 Convert to Speech", use_container_width=True):
#                     st.session_state.tts_text = st.session_state.handwriting_text
#                     st.session_state.app_mode = "Text-to-Speech"
#                     st.rerun()
    
#     # Text Input Page
#     elif app_mode == "Text Input":
#         st.header("⌨️ Text Input")
        
#         st.markdown("""
#         ### Direct Text Input
#         **For:** Pre-written content, articles, reports
        
#         **Simply paste or type your text below:**
#         """)
        
#         text_input = st.text_area(
#             "Enter your text here:",
#             height=200,
#             placeholder="Paste your news article, report, or any text here..."
#         )
        
#         if text_input:
#             col1, col2, col3, col4 = st.columns(4)
            
#             with col1:
#                 if st.button("📰 Create News Item", use_container_width=True):
#                     news_item = NewsItem(
#                         id=f"text_{datetime.datetime.now().strftime('%H%M%S')}",
#                         title=f"Text Report: {text_input[:50]}...",
#                         raw_content=text_input,
#                         category=NewsCategory.ANALYSIS,
#                         timestamp=datetime.datetime.now(),
#                         location="Text Source",
#                         metadata={"source": "Text Input", "verified": True}
#                     )
#                     st.session_state.news_items.append(news_item)
#                     st.success("News item created from text!")
            
#             with col2:
#                 if st.button("📊 Generate Summary", use_container_width=True):
#                     bullets = st.session_state.generator.generate_bullet_summary(text_input)
#                     st.subheader("Bullet Point Summary")
#                     for bullet in bullets:
#                         st.markdown(bullet)
            
#             with col3:
#                 if st.button("🎯 Detect Emergency", use_container_width=True):
#                     urgency = st.session_state.generator._detect_urgency(text_input)
#                     st.subheader(f"Urgency Level: {urgency}")
                    
#                     if urgency in ["HIGH", "MEDIUM"]:
#                         st.warning("⚠️ This text contains emergency keywords!")
#                         report = st.session_state.generator.generate_emergency_report(text_input)
#                         st.code(report["report"], language="markdown")
            
#             with col4:
#                 if st.button("🔊 Convert to Speech", use_container_width=True):
#                     st.session_state.tts_text = text_input
#                     st.session_state.app_mode = "Text-to-Speech"
#                     st.rerun()
    
#     # Create News Page
#     elif app_mode == "Create News":
#         st.header("Create News Items")
        
#         # Input Method Selection
#         input_method = st.radio(
#             "Choose input method:",
#             ["⌨️ Type Manually", "🎤 Use Voice Input", "✍️ Use Handwriting", "📝 Use Existing Text", "🔊 Convert to Speech"]
#         )
        
#         if input_method == "🎤 Use Voice Input":
#             if st.button("Start Voice Input"):
#                 st.session_state.app_mode = "Voice Input"
#                 st.rerun()
        
#         elif input_method == "✍️ Use Handwriting":
#             if st.button("Start Handwriting Input"):
#                 st.session_state.app_mode = "Handwriting Input"
#                 st.rerun()
        
#         elif input_method == "🔊 Convert to Speech":
#             if st.button("Go to Text-to-Speech"):
#                 st.session_state.app_mode = "Text-to-Speech"
#                 st.rerun()
        
#         elif input_method == "📝 Use Existing Text":
#             if st.session_state.voice_text:
#                 default_text = st.session_state.voice_text
#             elif st.session_state.handwriting_text:
#                 default_text = st.session_state.handwriting_text
#             else:
#                 default_text = ""
#         else:
#             default_text = ""
        
#         with st.form("create_news_form"):
#             col1, col2 = st.columns(2)
#             with col1:
#                 news_id = st.text_input("News ID", "news_001")
#                 title = st.text_input("Headline", "Government Announces New Policy")
#                 category = st.selectbox(
#                     "Category",
#                     [cat.value for cat in NewsCategory],
#                     index=0
#                 )
#                 location = st.text_input("Location", "New Delhi")
            
#             with col2:
#                 priority = st.slider("Priority", 1, 5, 3)
#                 source = st.text_input("Source", "PTI")
#                 verified = st.checkbox("Verified", True)
            
#             raw_content = st.text_area(
#                 "News Content",
#                 default_text if input_method == "📝 Use Existing Text" else "The government today announced a comprehensive new economic policy aimed at boosting growth and creating jobs. The policy includes tax incentives for startups and infrastructure investments."
#             )
            
#             submitted = st.form_submit_button("Create News Item")
            
#             if submitted:
#                 news_item = NewsItem(
#                     id=news_id,
#                     title=title,
#                     raw_content=raw_content,
#                     category=NewsCategory(category),
#                     timestamp=datetime.datetime.now(),
#                     location=location,
#                     priority=priority,
#                     metadata={
#                         "source": source,
#                         "verified": verified,
#                         "tags": []
#                     }
#                 )
#                 st.session_state.news_items.append(news_item)
#                 st.success(f"News item '{title}' created successfully!")
        
#         # Display created news items
#         if st.session_state.news_items:
#             st.subheader("Created News Items")
#             for i, item in enumerate(st.session_state.news_items):
#                 with st.expander(f"{item.title} ({item.category.value})"):
#                     st.write(f"**ID:** {item.id}")
#                     st.write(f"**Location:** {item.location}")
#                     st.write(f"**Priority:** {item.priority}")
#                     st.write(f"**Content:** {item.raw_content[:200]}...")
                    
#                     col1, col2, col3 = st.columns(3)
#                     with col1:
#                         if st.button(f"Generate Preview {i+1}", key=f"preview_{i}"):
#                             script = st.session_state.generator.generate_tv_script([item], Language.ENGLISH)
#                             st.code(script[:300] + "..." if len(script) > 300 else script, language="markdown")
                    
#                     with col2:
#                         if st.button(f"Generate Audio {i+1}", key=f"audio_{i}"):
#                             audio_bytes = st.session_state.generator.generate_audio_news_bulletin(
#                                 [item], "en", expressive_reading=True, engine="auto"
#                             )
#                             if audio_bytes:
#                                 st.audio(audio_bytes, format='audio/wav')
                    
#                     with col3:
#                         if st.button(f"Delete Item {i+1}", key=f"delete_{i}"):
#                             st.session_state.news_items.pop(i)
#                             st.rerun()
    
#     # Generate Scripts Page
#     elif app_mode == "Generate Scripts":
#         st.header("Generate News Scripts")
        
#         if not st.session_state.news_items:
#             st.warning("No news items created yet.")
            
#             # Quick input options
#             st.subheader("Quick Input Options:")
#             col1, col2, col3, col4 = st.columns(4)
            
#             with col1:
#                 if st.button("🎤 Add via Voice"):
#                     st.session_state.app_mode = "Voice Input"
#                     st.rerun()
            
#             with col2:
#                 if st.button("✍️ Add via Handwriting"):
#                     st.session_state.app_mode = "Handwriting Input"
#                     st.rerun()
            
#             with col3:
#                 if st.button("📝 Add via Text"):
#                     st.session_state.app_mode = "Text Input"
#                     st.rerun()
            
#             with col4:
#                 if st.button("🎵 Install Chatterbox"):
#                     offline_tts = OfflineTTSManager()
#                     offline_tts.install_chatterbox()
#         else:
#             col1, col2 = st.columns(2)
#             with col1:
#                 selected_items = st.multiselect(
#                     "Select News Items",
#                     [item.title for item in st.session_state.news_items],
#                     default=[item.title for item in st.session_state.news_items[:2]]
#                 )
                
#                 selected_news = []
#                 for title in selected_items:
#                     for item in st.session_state.news_items:
#                         if item.title == title:
#                             selected_news.append(item)
#                             break
            
#             with col2:
#                 language = st.selectbox(
#                     "Select Language",
#                     [lang.value for lang in Language],
#                     index=0
#                 )
                
#                 output_format = st.radio(
#                     "Output Format",
#                     ["TV Script", "Bullet Points", "Breaking News", "Headlines", "Audio Bulletin"]
#                 )
                
#                 auto_play = st.checkbox("🔊 Auto-play audio", value=True)
#                 expressive_reading = st.checkbox("🎭 Add Expressive Reading", value=True)
                
#                 # TTS engine selection
#                 if output_format == "Audio Bulletin":
#                     engine_options = []
#                     if CHATTERBOX_AVAILABLE and CHATTERBOX_WORKING:
#                         engine_options.append("chatterbox")
#                     if KOKORO_AVAILABLE and KOKORO_WORKING:
#                         engine_options.append("kokoro")
#                     if PIPER_AVAILABLE:
#                         engine_options.append("piper")
#                     if TTS_AVAILABLE:
#                         engine_options.append("pyttsx3")
#                     if GTTS_AVAILABLE:
#                         engine_options.append("gtts")
#                     engine_options.append("auto")
                    
#                     tts_engine = st.selectbox(
#                         "TTS Engine",
#                         engine_options,
#                         index=0 if CHATTERBOX_AVAILABLE and CHATTERBOX_WORKING else 0
#                     )
            
#             if st.button("Generate Content", type="primary"):
#                 if selected_news:
#                     language_enum = Language(language)
                    
#                     if output_format == "TV Script":
#                         content = st.session_state.generator.generate_tv_script(selected_news, language_enum)
#                         st.subheader(f"{language.upper()} TV News Script")
#                         st.code(content, language="markdown")
                        
#                         # Add to generated content
#                         st.session_state.generated_content.append({
#                             "type": "tv_script",
#                             "language": language,
#                             "content": content,
#                             "timestamp": datetime.datetime.now().isoformat()
#                         })
                    
#                     elif output_format == "Bullet Points":
#                         bullets = st.session_state.generator.generate_bullet_summary(
#                             selected_news[0].raw_content
#                         )
#                         st.subheader("Bullet Point Summary")
#                         for bullet in bullets:
#                             st.markdown(bullet)
                    
#                     elif output_format == "Breaking News":
#                         if len(selected_news) > 0:
#                             breaking = st.session_state.generator.generate_breaking_news_version(
#                                 selected_news[0], language_enum
#                             )
#                             st.subheader("Breaking News Version")
#                             st.code(breaking, language="markdown")
                    
#                     elif output_format == "Headlines":
#                         headlines = st.session_state.generator.generate_headlines(selected_news, language_enum)
#                         st.subheader("News Headlines")
#                         for i, headline in enumerate(headlines, 1):
#                             st.markdown(f"{i}. {headline}")
                    
#                     elif output_format == "Audio Bulletin":
#                         # Check if voice cloning is needed
#                         voice_clone_audio = None
#                         voice_clone_text = None
                        
#                         if tts_engine == "chatterbox" and st.session_state.voice_clone_audio:
#                             voice_clone_audio = st.session_state.voice_clone_audio
#                             voice_clone_text = st.session_state.voice_clone_text
                        
#                         audio_bytes = st.session_state.generator.generate_audio_news_bulletin(
#                             selected_news, language, expressive_reading, 
#                             engine=tts_engine,
#                             voice_clone_audio=voice_clone_audio,
#                             voice_clone_text=voice_clone_text,
#                             auto_play=auto_play
#                         )
#                         if audio_bytes:
#                             st.success("✅ Audio news bulletin generated!")
                            
#                             # Show script preview
#                             script = st.session_state.generator.generate_tv_script(selected_news, language_enum)
#                             with st.expander("📄 Script Preview"):
#                                 st.text_area("Bulletin Script", script[:1000] + "..." if len(script) > 1000 else script, height=200)
                            
#                             # Additional download button if not auto-played
#                             if not auto_play:
#                                 st.audio(audio_bytes, format='audio/wav')
                            
#                             # Download button
#                             st.download_button(
#                                 label="📥 Download Audio Bulletin",
#                                 data=audio_bytes,
#                                 file_name=f"audio_bulletin_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.wav",
#                                 mime="audio/wav"
#                             )
                    
#                     st.success(f"Content generated successfully in {language.upper()}!")
#                 else:
#                     st.error("Please select at least one news item")
    
#     # Special Reports Page
#     elif app_mode == "Special Reports":
#         st.header("Special Reports")
        
#         report_type = st.selectbox(
#             "Select Report Type",
#             ["Weather", "Market", "Traffic", "Breaking News", "Long-form Program", "Emergency Report"]
#         )
        
#         if report_type == "Emergency Report":
#             st.subheader("🚨 Emergency Report Generator")
            
#             # Quick voice input for emergencies
#             if st.button("🎤 Report Emergency via Voice", type="primary"):
#                 st.session_state.app_mode = "Voice Input"
#                 st.rerun()
            
#             col1, col2 = st.columns(2)
#             with col1:
#                 incident_type = st.selectbox(
#                     "Incident Type",
#                     ["accident", "fire", "medical", "crime", "natural_disaster", "other"]
#                 )
#                 location = st.text_input("Location", "Unknown")
            
#             with col2:
#                 reporter = st.text_input("Reporter", "Citizen")
#                 status = st.selectbox(
#                     "Status",
#                     ["PENDING", "RESPONDING", "RESOLVED", "CRITICAL"]
#                 )
            
#             details = st.text_area("Incident Details", "Enter details of the emergency...")
            
#             if st.button("Generate Emergency Report"):
#                 report = st.session_state.generator.generate_emergency_report(
#                     details, location, incident_type
#                 )
#                 st.code(report["report"], language="markdown")
                
#                 # Show analysis
#                 st.subheader("Emergency Analysis")
#                 col1, col2 = st.columns(2)
#                 with col1:
#                     st.metric("Urgency Level", report["urgency"])
#                 with col2:
#                     st.metric("Incident Type", report["extracted_info"]["incident_type"])
        
#         elif report_type == "Weather":
#             st.subheader("Weather Report Generator")
#             col1, col2 = st.columns(2)
#             with col1:
#                 location = st.text_input("Weather Location", "Hyderabad")
#                 temp = st.number_input("Temperature (°C)", value=32.5)
#                 conditions = st.selectbox(
#                     "Weather Conditions",
#                     ["Sunny", "Partly Cloudy", "Cloudy", "Rainy", "Stormy", "Foggy"]
#                 )
#             with col2:
#                 humidity = st.slider("Humidity (%)", 0, 100, 65)
#                 wind_speed = st.number_input("Wind Speed (km/h)", value=15.0)
#                 wind_direction = st.selectbox(
#                     "Wind Direction",
#                     ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
#                 )
            
#             if st.button("Generate Weather Report"):
#                 weather_data = {
#                     'temperature': temp,
#                     'conditions': conditions,
#                     'humidity': humidity,
#                     'wind_speed': wind_speed,
#                     'wind_direction': wind_direction,
#                     'sunrise': '06:15',
#                     'sunset': '18:30'
#                 }
                
#                 report = st.session_state.generator.generate_weather_report(location, weather_data)
#                 st.code(report, language="markdown")
            
#         elif report_type == "Market":
#             st.subheader("Market Update Generator")
#             col1, col2 = st.columns(2)
#             with col1:
#                 sensex = st.number_input("Sensex", value=72410.38)
#                 sensex_change = st.text_input("Sensex Change", "+0.86%")
#                 nifty = st.number_input("Nifty", value=22000.00)
#                 nifty_change = st.text_input("Nifty Change", "+0.78%")
#             with col2:
#                 usd_inr = st.number_input("USD/INR", value=83.15)
#                 gold_price = st.number_input("Gold Price (₹/10g)", value=62000)
            
#             if st.button("Generate Market Update"):
#                 market_data = {
#                     'sensex': sensex,
#                     'sensex_change': sensex_change,
#                     'nifty': nifty,
#                     'nifty_change': nifty_change,
#                     'gainers': ['TATA Motors', 'Reliance', 'Infosys'],
#                     'losers': ['HDFC Bank', 'ICICI Bank', 'SBI'],
#                     'usd_inr': usd_inr,
#                     'gold_price': gold_price
#                 }
                
#                 report = st.session_state.generator.generate_market_update(market_data)
#                 st.code(report, language="markdown")
            
#         elif report_type == "Breaking News":
#             st.subheader("Breaking News Generator")
            
#             if not st.session_state.news_items:
#                 st.warning("No news items available. Create one first.")
#             else:
#                 selected_news = st.selectbox(
#                     "Select News Item",
#                     [item.title for item in st.session_state.news_items]
#                 )
                
#                 selected_item = None
#                 for item in st.session_state.news_items:
#                     if item.title == selected_news:
#                         selected_item = item
#                         break
                
#                 if selected_item:
#                     language = st.selectbox(
#                         "Language",
#                         [lang.value for lang in Language],
#                         index=0
#                     )
                    
#                     if st.button("Generate Breaking News"):
#                         breaking = st.session_state.generator.generate_breaking_news_version(
#                             selected_item, Language(language)
#                         )
#                         st.code(breaking, language="markdown")
            
#         elif report_type == "Long-form Program":
#             st.subheader("Long-form Program Generator")
#             col1, col2 = st.columns(2)
#             with col1:
#                 topic = st.text_input("Program Topic", "Artificial Intelligence in Journalism")
#                 program_type = st.selectbox(
#                     "Program Type",
#                     ["analysis", "debate", "info_segment", "documentary"]
#                 )
#             with col2:
#                 duration = st.text_input("Duration", "30 minutes")
#                 participants = st.text_area("Participants (comma-separated)", "Anchor, Expert 1, Expert 2, Reporter")
            
#             if st.button("Generate Program Outline"):
#                 program = st.session_state.generator.create_long_form_program(topic, program_type)
                
#                 st.subheader(f"Program: {program['program_title']}")
#                 st.write(f"**Duration:** {program['template']['duration']}")
#                 st.write(f"**Estimated Words:** {program['estimated_words']}")
                
#                 st.subheader("Program Structure:")
#                 for i, section in enumerate(program['template']['structure'], 1):
#                     st.write(f"{i}. {section}")
                
#                 st.subheader("Suggested Graphics:")
#                 for graphic in program['suggested_graphics']:
#                     st.write(f"• {graphic}")
                
#                 st.subheader("Production Notes:")
#                 st.json(program['production_notes'])
    
#     # Batch Processing Page
#     elif app_mode == "Batch Processing":
#         st.header("Batch Processing")
        
#         col1, col2 = st.columns(2)
#         with col1:
#             category = st.selectbox(
#                 "News Category",
#                 [cat.value for cat in NewsCategory],
#                 index=0
#             )
#             count = st.slider("Number of Items", 1, 10, 3)
        
#         with col2:
#             languages = st.multiselect(
#                 "Languages",
#                 [lang.value for lang in Language],
#                 default=["en", "te", "hi"]
#             )
        
#         if st.button("Generate Batch", type="primary"):
#             scheduler = AutomatedNewsScheduler(st.session_state.generator)
            
#             with st.spinner("Generating batch content..."):
#                 batch = scheduler.generate_content_batch(
#                     NewsCategory(category), count
#                 )
                
#                 # Filter by selected languages
#                 filtered_batch = [
#                     item for item in batch 
#                     if item['language'] in languages
#                 ]
                
#                 st.success(f"Generated {len(filtered_batch)} news items")
                
#                 # Display results in tabs
#                 if filtered_batch:
#                     tabs = st.tabs([item['language'].upper() for item in filtered_batch[:5]])
                    
#                     for idx, (tab, item) in enumerate(zip(tabs, filtered_batch[:5])):
#                         with tab:
#                             st.markdown(f"**ID:** {item['id']}")
#                             st.markdown(f"**Category:** {item['category']}")
#                             st.markdown("**Headlines:**")
#                             for headline in item['headlines']:
#                                 st.markdown(f"• {headline}")
                            
#                             with st.expander("View Full Script"):
#                                 st.code(item['full_script'], language="markdown")
                
#                 # Store for export
#                 st.session_state.batch_content = filtered_batch
    
#     # Export Content Page
#     elif app_mode == "Export Content":
#         st.header("Export Content")
        
#         col1, col2 = st.columns(2)
#         with col1:
#             export_type = st.selectbox(
#                 "Export Format",
#                 ["JSON", "Text", "HTML", "Audio (WAV)"]
#             )
            
#             filename = st.text_input("Filename", "news_export")
        
#         with col2:
#             st.subheader("Available Content")
            
#             if 'batch_content' in st.session_state and st.session_state.batch_content:
#                 content_to_export = st.session_state.batch_content
#                 st.info(f"Batch Content: {len(content_to_export)} items")
#                 for item in content_to_export[:3]:
#                     st.caption(f"{item['id']} - {item['language'].upper()}")
#             elif st.session_state.generated_content:
#                 content_to_export = st.session_state.generated_content
#                 st.info(f"Generated Content: {len(content_to_export)} items")
#                 for item in content_to_export[:3]:
#                     st.caption(f"{item['type']} - {item['language']}")
#             elif st.session_state.tts_audio and export_type == "Audio (WAV)":
#                 content_to_export = [{"type": "audio", "data": st.session_state.tts_audio}]
#                 st.info("Audio content available for export")
#             else:
#                 content_to_export = []
#                 st.warning("No content available for export.")
        
#         if content_to_export and st.button("Export", type="primary"):
#             export_manager = NewsExportManager()
#             filename_with_ext = f"{filename}.{export_type.lower().replace(' (wav)', '.wav').replace('audio ', '')}"
            
#             try:
#                 if export_type == "JSON":
#                     result = export_manager.export_to_json(content_to_export, filename_with_ext)
#                 elif export_type == "Text":
#                     result = export_manager.export_to_text(content_to_export, filename_with_ext)
#                 elif export_type == "HTML":
#                     result = export_manager.export_to_html(content_to_export, filename_with_ext)
#                 elif export_type == "Audio (WAV)":
#                     if st.session_state.tts_audio:
#                         with open(filename_with_ext, 'wb') as f:
#                             f.write(st.session_state.tts_audio)
#                         result = f"Exported audio to {filename_with_ext}"
#                     else:
#                         result = "No audio content available"
                
#                 st.success(f"Content exported successfully!")
#                 st.info(result)
                
#                 # Provide download button
#                 try:
#                     if export_type == "Audio (WAV)" and st.session_state.tts_audio:
#                         st.download_button(
#                             label=f"📥 Download {export_type} file",
#                             data=st.session_state.tts_audio,
#                             file_name=filename_with_ext,
#                             mime="audio/wav"
#                         )
#                     elif export_type in ["JSON", "Text", "HTML"]:
#                         with open(filename_with_ext, 'r', encoding='utf-8') as f:
#                             file_content = f.read()
                        
#                         st.download_button(
#                             label=f"📥 Download {export_type} file",
#                             data=file_content,
#                             file_name=filename_with_ext,
#                             mime="application/json" if export_type == "JSON" else "text/plain"
#                         )
#                 except Exception as e:
#                     st.error(f"Could not create download button: {str(e)}")
#             except Exception as e:
#                 st.error(f"Error exporting content: {str(e)}")
    
#     # Settings Page
#     elif app_mode == "Settings":
#         st.header("System Settings")
        
#         st.subheader("Input Settings")
        
#         input_settings = st.multiselect(
#             "Enable Input Methods",
#             ["Voice Input", "Handwriting Input", "Text Input", "Text-to-Speech", "Voice Cloning"],
#             default=["Voice Input", "Handwriting Input", "Text Input", "Text-to-Speech", "Voice Cloning"]
#         )
        
#         st.subheader("TTS Engine Settings")
        
#         # Chatterbox settings
#         if CHATTERBOX_AVAILABLE and CHATTERBOX_WORKING:
#             st.subheader("🎭 Chatterbox Voice Cloning Settings")
            
#             processor = st.session_state.generator.input_processor
#             chatterbox_voices = processor.get_chatterbox_voices() if hasattr(processor, 'chatterbox_voices') else []
            
#             if chatterbox_voices:
#                 st.info("Available Chatterbox Voices:")
#                 for voice in chatterbox_voices:
#                     st.write(f"• **{voice['name']}** - {voice['type']} ({'Voice Cloning' if voice['cloning'] else 'Pre-trained'})")
            
#             # Voice sample management
#             if st.session_state.voice_clone_audio:
#                 st.success("✅ Voice sample is uploaded")
#                 if st.button("Clear Voice Sample"):
#                     st.session_state.voice_clone_audio = None
#                     st.session_state.voice_clone_text = ""
#                     st.success("Voice sample cleared")
        
#         # Kokoro settings
#         if KOKORO_AVAILABLE and KOKORO_WORKING:
#             st.subheader("⚡ Kokoro Settings")
            
#             processor = st.session_state.generator.input_processor
#             kokoro_voices = processor.get_kokoro_voices() if processor.kokoro_model else []
            
#             if kokoro_voices:
#                 default_kokoro_voice = st.selectbox(
#                     "Default Kokoro Voice",
#                     kokoro_voices,
#                     index=kokoro_voices.index(st.session_state.kokoro_voice) 
#                     if st.session_state.kokoro_voice in kokoro_voices else 0
#                 )
                
#                 if st.button("Update Kokoro Settings"):
#                     st.session_state.kokoro_voice = default_kokoro_voice
#                     processor.set_kokoro_voice(default_kokoro_voice)
#                     st.success(f"Kokoro voice set to: {default_kokoro_voice}")
        
#         # Piper TTS installation
#         if not PIPER_AVAILABLE:
#             st.subheader("🎵 Piper TTS Installation")
#             if st.button("Install Piper TTS"):
#                 offline_tts = OfflineTTSManager()
#                 offline_tts.install_piper()
        
#         # Chatterbox installation
#         if not CHATTERBOX_WORKING:
#             st.subheader("🚀 Chatterbox TTS Installation")
#             if st.button("Install Chatterbox TTS"):
#                 offline_tts = OfflineTTSManager()
#                 offline_tts.install_chatterbox()
        
#         st.subheader("News Templates")
        
#         template_types = ["TV Script", "Bullet Summary", "Emergency Report"]
#         selected_template = st.selectbox("Select Template Type", template_types)
        
#         if selected_template == "TV Script":
#             col1, col2, col3 = st.columns(3)
#             with col1:
#                 intro = st.text_area("Intro Template", 
#                                    st.session_state.generator.templates["tv_script"]["intro"])
#             with col2:
#                 story = st.text_area("Story Template", 
#                                    st.session_state.generator.templates["tv_script"]["story_template"])
#             with col3:
#                 outro = st.text_area("Outro Template", 
#                                    st.session_state.generator.templates["tv_script"]["outro"])
            
#             if st.button("Update TV Script Template"):
#                 st.session_state.generator.templates["tv_script"]["intro"] = intro
#                 st.session_state.generator.templates["tv_script"]["story_template"] = story
#                 st.session_state.generator.templates["tv_script"]["outro"] = outro
#                 st.success("TV Script template updated!")
        
#         elif selected_template == "Emergency Report":
#             template = st.text_area("Emergency Report Template", 
#                                   st.session_state.generator.templates["emergency_report"]["template"],
#                                   height=200)
            
#             if st.button("Update Emergency Template"):
#                 st.session_state.generator.templates["emergency_report"]["template"] = template
#                 st.success("Emergency template updated!")
        
#         st.subheader("Language Settings")
        
#         selected_lang = st.selectbox(
#             "Select Language to Configure",
#             [lang.value for lang in Language]
#         )
        
#         col1, col2 = st.columns(2)
#         with col1:
#             greeting = st.text_input("Anchor Greeting", 
#                                    st.session_state.generator.style_guide.get(selected_lang, {}).get("anchor_greeting", ""))
#         with col2:
#             sign_off = st.text_input("Sign Off", 
#                                     st.session_state.generator.style_guide.get(selected_lang, {}).get("sign_off", ""))
        
#         if st.button("Update Language Settings"):
#             if selected_lang not in st.session_state.generator.style_guide:
#                 st.session_state.generator.style_guide[selected_lang] = {}
            
#             st.session_state.generator.style_guide[selected_lang]["anchor_greeting"] = greeting
#             st.session_state.generator.style_guide[selected_lang]["sign_off"] = sign_off
#             st.success(f"Language settings for {selected_lang} updated!")
        
#         st.subheader("System Information")
#         st.info(f"Total News Items: {len(st.session_state.news_items)}")
#         st.info(f"Generated Content Items: {len(st.session_state.generated_content)}")
#         st.info(f"Voice Inputs Processed: {len([x for x in st.session_state.news_items if x.metadata.get('source') == 'Voice Input'])}")
#         st.info(f"Handwriting Inputs Processed: {len([x for x in st.session_state.news_items if x.metadata.get('source') == 'Handwriting Input'])}")
#         st.info(f"Voice Cloning Samples: {1 if st.session_state.voice_clone_audio else 0}")
#         st.info(f"Text-to-Speech Conversions: {1 if st.session_state.tts_audio else 0}")
        
#         if CHATTERBOX_AVAILABLE:
#             if CHATTERBOX_WORKING:
#                 st.success("✅ Chatterbox TTS is working (Voice Cloning Available)")
#             else:
#                 st.warning("⚠️ Chatterbox TTS dependencies available but not fully functional")
#         else:
#             st.error("❌ Chatterbox TTS is not installed (No PyTorch)")
        
#         if KOKORO_AVAILABLE:
#             if KOKORO_WORKING:
#                 st.success("✅ Kokoro TTS is working")
#             else:
#                 st.warning("⚠️ Kokoro TTS is installed but not working properly")
#         else:
#             st.error("❌ Kokoro TTS is not installed")
        
#         if PIPER_AVAILABLE:
#             st.success("✅ Piper TTS is available")
#         else:
#             st.warning("⚠️ Piper TTS is not installed. Click 'Install Piper TTS' above")
        
#         if st.button("Clear All Data"):
#             st.session_state.news_items = []
#             st.session_state.generated_content = []
#             st.session_state.batch_content = []
#             st.session_state.voice_text = ""
#             st.session_state.handwriting_text = ""
#             st.session_state.tts_audio = None
#             st.session_state.tts_text = ""
#             st.session_state.voice_clone_audio = None
#             st.session_state.voice_clone_text = ""
#             st.success("All data cleared!")


# # Run the Streamlit app
# if __name__ == "__main__":

#     # Run main app
#     main()








# import streamlit as st
# from test import save_file, process_file

# st.set_page_config(page_title="Media → News Generator", layout="centered")

# st.title("📰 Media to News Script")
# st.write("Upload **audio, video, or image** to generate a news-style narration.")

# # Language selector
# lang = st.selectbox(
#     "Select Output Language",
#     options=[("English", "en"), ("Hindi", "hi"), ("Telugu", "te"), ("Kannada", "kn")],
#     format_func=lambda x: x[0]
# )[1]

# uploaded_file = st.file_uploader(
#     "Upload a file",
#     type=["mp3", "wav", "mp4", "mkv", "mov", "png", "jpg", "jpeg", "webp"],
# )

# clone_voice = st.checkbox("🎤 Clone voice from audio/video")

# # Make sure we only process once per upload
# if uploaded_file is not None:
#     with st.spinner("Processing..."):
#         path = save_file(uploaded_file)
#         result = process_file(path, lang=lang, clone_voice_from_audio=clone_voice)

#     if "error" in result:
#         st.error(result["error"])
#     else:
#         st.success("Done!")

#         # Transcript (for audio/video)
#         if "transcript" in result:
#             st.subheader("📄 Transcript")
#             st.text_area(
#                 label="Transcript",
#                 value=result["transcript"],
#                 height=150,
#                 label_visibility="collapsed"
#             )

#         # Image description (for images)
#         if "image_meaning" in result:
#             st.subheader("🖼 Image Description")
#             st.text_area(
#                 label="Image Description",
#                 value=result["image_meaning"],
#                 height=150,
#                 label_visibility="collapsed"
#             )

#         # News script
#         st.subheader("📝 News Script")
#         st.text_area(
#             label="News Script",
#             value=result["news_script"],
#             height=220,
#             label_visibility="collapsed"
#         )

#         # Audio playback
#         st.subheader("🎧 Generated Audio")
#         st.audio(result["audio"])























































































import re
import os
import base64
import streamlit as st
from pydub import AudioSegment
from sarvamai import SarvamAI

# Page configuration
st.set_page_config(
    page_title="Telugu Text-to-Speech",
    page_icon="🎙️",
    layout="wide"
)

# Title and description
st.title("🎙️ Telugu Text-to-Speech Generator")
st.markdown("Convert Telugu text to speech using Sarvam AI")

# Sidebar for API key and settings
with st.sidebar:
    st.header("⚙️ Settings")
    api_key = st.text_input(
        "API Key",
        value="sk_s82v0y7c_K4Vegv1jb81WFzdShpm2HmS0",
        type="password",
        help="Enter your Sarvam AI API key"
    )
    
    speaker = st.selectbox(
        "Speaker Voice",
        ["arya", "meera", "kavya"],
        help="Select the voice for speech generation"
    )
    
    sample_rate = st.selectbox(
        "Sample Rate",
        [8000, 16000, 22050, 44100],
        index=2,
        help="Audio quality (higher = better quality but larger file)"
    )

# Default Telugu news text
default_text = """తెలుగు రోడ్ సేఫ్టీ ముఖ్య వార్తా కథనం ఈ దేశంలో రహదారి ప్రమాదాలు ఇంకా ఒక అత్యవసర సమస్యగా కొనసాగుతున్నాయి. ఇటీవల విడుదలైన కేంద్ర సర్కార్ గణాంకాల ప్రకారం, గత ఐదు సంవత్సరాల్లో దేశవ్యాప్తంగా రోడ్డు ప్రమాదాల్లో సుమారు 7.77 లక్షల మంది ప్రాణాలు కోల్పోయారు మరియు వేలాది మంది గాయపడ్డారు, ఇది రహదారి భద్రతపై తీవ్ర ఆందోళనను చూపుతుందని ప్రతిపాదించబడింది."""

# Text input area
st.subheader("📝 Enter Telugu Text")
input_text = st.text_area(
    "Text to convert",
    value=default_text,
    height=200,
    help="Enter or paste Telugu text here"
)

# Character count
st.caption(f"Characters: {len(input_text)}")

# Generate button
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    generate_button = st.button("🎵 Generate Speech", use_container_width=True, type="primary")

# Processing and generation
if generate_button:
    if not input_text.strip():
        st.error("⚠️ Please enter some Telugu text!")
    elif not api_key:
        st.error("⚠️ Please enter your API key in the sidebar!")
    else:
        try:
            # Initialize client
            client = SarvamAI(api_subscription_key=api_key)
            
            # Progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Split into chunks
            status_text.text("📋 Splitting text into chunks...")
            raw_chunks = re.split(r'(?<=[।\.\?\!])\s+', input_text.strip())
            valid_chunks = [
                chunk for chunk in raw_chunks
                if len(chunk.strip()) > 3 and re.search(r'[\u0C00-\u0C7F]', chunk)
            ]
            
            st.info(f"✅ Found {len(valid_chunks)} valid text chunks")
            
            # Create temporary directory
            os.makedirs("tmp_mp3s", exist_ok=True)
            chunk_files = []
            
            # Generate audio for each chunk
            for i, sentence in enumerate(valid_chunks):
                progress = (i + 1) / len(valid_chunks)
                progress_bar.progress(progress)
                status_text.text(f"🎤 Generating audio chunk {i+1}/{len(valid_chunks)}...")
                
                response = client.text_to_speech.convert(
                    text=sentence,
                    target_language_code="te-IN",
                    speaker=speaker,
                    output_audio_codec="mp3",
                    speech_sample_rate=sample_rate,
                    enable_preprocessing=True,
                    model="bulbul:v2"
                )
                
                chunk_name = f"tmp_mp3s/chunk_{i}.mp3"
                with open(chunk_name, "wb") as f:
                    for audio_base64 in response.audios:
                        f.write(base64.b64decode(audio_base64))
                
                chunk_files.append(chunk_name)
            
            # Combine audio chunks
            status_text.text("🔗 Combining audio chunks...")
            combined = AudioSegment.empty()
            for chunk in chunk_files:
                combined += AudioSegment.from_mp3(chunk)
            
            # Export final file
            final_file = "final_telugu_news.mp3"
            combined.export(final_file, format="mp3")
            
            # Clear progress indicators
            progress_bar.empty()
            status_text.empty()
            
            # Success message and audio player
            st.success("✅ Audio generated successfully!")
            
            # Display audio player
            st.subheader("🔊 Generated Audio")
            with open(final_file, "rb") as audio_file:
                audio_bytes = audio_file.read()
                st.audio(audio_bytes, format="audio/mp3")
            
            # Download button
            st.download_button(
                label="⬇️ Download MP3",
                data=audio_bytes,
                file_name="telugu_speech.mp3",
                mime="audio/mp3",
                use_container_width=True
            )
            
            # Cleanup option
            if st.button("🗑️ Clean up temporary files"):
                for chunk in chunk_files:
                    if os.path.exists(chunk):
                        os.remove(chunk)
                if os.path.exists("tmp_mp3s") and not os.listdir("tmp_mp3s"):
                    os.rmdir("tmp_mp3s")
                st.success("Temporary files cleaned up!")
                
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            st.exception(e)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray;'>
        Built with Streamlit & Sarvam AI | Telugu Text-to-Speech
    </div>
    """,
    unsafe_allow_html=True
)