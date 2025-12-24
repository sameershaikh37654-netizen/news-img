# """
# Voice Cloning Web App with Streamlit and Coqui TTS
# Upload audio, enter text, and clone voices with an intuitive interface
# Now with Indian language support
# """

# import streamlit as st
# from TTS.api import TTS
# import torch
# import os
# import tempfile
# from TTS.tts.configs.xtts_config import XttsConfig
# from TTS.tts.models.xtts import XttsAudioConfig  # Add missing class

# # Page configuration
# st.set_page_config(
#     page_title="Voice Cloning Studio - Indian Languages",
#     page_icon="🎙️",
#     layout="wide"
# )

# # Custom CSS
# st.markdown("""
#     <style>
#     .main-header {
#         font-size: 3rem;
#         color: #FF4B4B;
#         text-align: center;
#         margin-bottom: 1rem;
#     }
#     .sub-header {
#         text-align: center;
#         color: #666;
#         margin-bottom: 2rem;
#     }
#     .stButton>button {
#         width: 100%;
#         height: 3rem;
#         font-size: 1.2rem;
#     }
#     .stProgress > div > div > div > div {
#         background-color: #FF4B4B;
#     }
#     .indian-flag {
#         color: #FF9933;
#         font-weight: bold;
#     }
#     </style>
# """, unsafe_allow_html=True)

# @st.cache_resource(show_spinner=False)
# def load_tts_model():
#     """Load TTS model safely with PyTorch 2.6+"""
#     device = "cuda" if torch.cuda.is_available() else "cpu"
    
#     # Add all required classes to safe globals
#     safe_classes = [
#         XttsConfig,
#         XttsAudioConfig,
#         'TTS.tts.models.xtts.XttsAudioConfig',
#         'TTS.tts.configs.xtts_config.XttsConfig',
#     ]
    
#     try:
#         # Try loading with safe globals context
#         with torch.serialization.safe_globals(safe_classes):
#             tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
#     except Exception as e:
#         # If that fails, try the older approach with weights_only=False
#         st.warning("⚠️ Using alternative loading method...")
#         import warnings
#         warnings.filterwarnings("ignore", category=UserWarning)
        
#         # Manually set weights_only to False as a fallback
#         original_load = torch.load
#         def custom_load(*args, **kwargs):
#             if 'weights_only' in kwargs:
#                 kwargs['weights_only'] = False
#             else:
#                 kwargs = {**kwargs, 'weights_only': False}
#             return original_load(*args, **kwargs)
        
#         torch.load = custom_load
#         tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
#         torch.load = original_load  # Restore original
    
#     return tts, device

# def clone_voice(tts, reference_audio_path, text_to_speak, output_path, language="en"):
#     """Clone voice and generate speech"""
#     try:
#         # Validate audio duration (optional but recommended)
#         import soundfile as sf
#         audio_info = sf.info(reference_audio_path)
#         duration = audio_info.duration
        
#         if duration < 2:
#             return False, "Audio is too short (minimum 2 seconds)"
#         if duration > 60:
#             return False, "Audio is too long (maximum 60 seconds)"
        
#         # Generate audio
#         tts.tts_to_file(
#             text=text_to_speak,
#             speaker_wav=reference_audio_path,
#             language=language,
#             file_path=output_path
#         )
#         return True, f"Audio generated successfully! (Reference: {duration:.1f}s)"
#     except Exception as e:
#         return False, f"Error: {str(e)}"

# def validate_text(text):
#     """Validate input text"""
#     if not text or text.strip() == "":
#         return False, "Text cannot be empty"
#     if len(text) > 2000:
#         return False, "Text is too long (maximum 2000 characters)"
#     return True, "Text is valid"

# def main():
#     # Header with Indian focus
#     st.markdown('<h1 class="main-header">🎙️ Voice Cloning Studio <span class="indian-flag">🇮🇳</span></h1>', unsafe_allow_html=True)
#     st.markdown('<p class="sub-header">Clone any voice with AI-powered text-to-speech | Now with Indian language support</p>', unsafe_allow_html=True)
    
#     # Sidebar for settings
#     with st.sidebar:
#         st.header("⚙️ Settings")
        
#         # Language selection with Indian languages prominently featured
#         language_options = {
#             # Indian Languages
#             "Hindi (हिन्दी)": "hi",
#             "Bengali (বাংলা)": "bn",
#             "Telugu (తెలుగు)": "te",
#             "Tamil (தமிழ்)": "ta",
#             "Marathi (मराठी)": "mr",
#             "Gujarati (ગુજરાતી)": "gu",
#             "Kannada (ಕನ್ನಡ)": "kn",
#             "Malayalam (മലയാളം)": "ml",
#             "Punjabi (ਪੰਜਾਬੀ)": "pa",
#             "Odia (ଓଡ଼ିଆ)": "or",
#             "Urdu (اردو)": "ur",
#             "Sanskrit (संस्कृतम्)": "sa",
#             "Assamese (অসমীয়া)": "as",
#             "Maithili (मैथिली)": "mai",
#             "Santali (ᱥᱟᱱᱛᱟᱲᱤ)": "sat",
            
#             # Other popular languages
#             "English": "en",
#             "Spanish": "es", 
#             "French": "fr",
#             "German": "de",
#             "Arabic": "ar",
#             "Chinese": "zh-cn",
#             "Japanese": "ja",
#             "Korean": "ko",
#             "Russian": "ru"
#         }
        
#         # Group Indian languages separately
#         indian_languages = {k: v for k, v in language_options.items() if k in [
#             "Hindi (हिन्दी)", "Bengali (বাংলা)", "Telugu (తెలుగు)", "Tamil (தமிழ்)", 
#             "Marathi (मराठी)", "Gujarati (ગુજરાતી)", "Kannada (ಕನ್ನಡ)", "Malayalam (മലയാളം)",
#             "Punjabi (ਪੰਜਾਬੀ)", "Odia (ଓଡ଼ିଆ)", "Urdu (اردو)", "Sanskrit (संस्कृतम्)",
#             "Assamese (অসমীয়া)", "Maithili (मैथिली)", "Santali (ᱥᱟᱱᱛᱟᱲᱤ)"
#         ]}
        
#         other_languages = {k: v for k, v in language_options.items() if k not in indian_languages}
        
#         # Create a custom selectbox with Indian languages first
#         st.subheader("🌍 Language Selection")
        
#         # Indian languages section
#         st.markdown("#### 🇮🇳 Indian Languages")
#         selected_indian_lang = st.selectbox(
#             "Select Indian Language",
#             list(indian_languages.keys()),
#             index=0,
#             help="Choose an Indian language for voice cloning",
#             key="indian_lang"
#         )
        
#         # Other languages section
#         st.markdown("#### 🌐 Other Languages")
#         selected_other_lang = st.selectbox(
#             "Select Other Language",
#             list(other_languages.keys()),
#             index=0,
#             help="Choose other supported languages",
#             key="other_lang"
#         )
        
#         # Radio button to choose between Indian and other languages
#         language_category = st.radio(
#             "Language Category",
#             ["Indian Languages", "Other Languages"],
#             horizontal=True
#         )
        
#         if language_category == "Indian Languages":
#             language = indian_languages[selected_indian_lang]
#             selected_lang_name = selected_indian_lang.split(" (")[0]  # Clean name for file
#         else:
#             language = other_languages[selected_other_lang]
#             selected_lang_name = selected_other_lang
        
#         # Add language-specific tips
#         st.markdown("---")
#         st.markdown("### 📝 Indian Language Tips")
#         st.markdown("""
#         - **Hindi/Urdu**: Works best with Devanagari script
#         - **South Indian languages**: Clear pronunciation yields best results
#         - **Bengali/Odia**: Use proper Unicode characters
#         - **Punjabi**: Both Gurmukhi and Shahmukhi scripts supported
#         """)
        
#         # Add advanced options
#         st.markdown("---")
#         st.subheader("Advanced Options")
        
#         # Clear cache button
#         if st.button("🔄 Clear Model Cache"):
#             st.cache_resource.clear()
#             st.success("Cache cleared! Model will reload on next generation.")
        
#         # Device info
#         st.markdown("---")
#         if torch.cuda.is_available():
#             st.success(f"✅ GPU Detected: {torch.cuda.get_device_name(0)}")
#             st.info(f"🎯 VRAM Available: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
#         else:
#             st.warning("⚠️ CPU Mode: Generation will be slower")
        
#         st.markdown("---")
#         st.markdown("### 📋 Tips for Best Results")
#         st.markdown("""
#         - Use clear audio (6-30 seconds)
#         - Minimal background noise
#         - WAV format recommended
#         - Natural speaking tone
#         - Single speaker only
#         - Avoid music or effects
#         """)
        
#         # Indian language keyboard help
#         with st.expander("🔤 Indian Language Typing Help"):
#             st.markdown("""
#             **Online Keyboard Tools:**
#             - [Google Input Tools](https://www.google.com/inputtools/try/)
#             - [Quillpad](https://www.quillpad.in/)
#             - [Branah](https://www.branah.com/indian-language-keyboards)
            
#             **Mobile Apps:**
#             - Google Indic Keyboard
#             - SwiftKey with Indian languages
#             - Gboard with Indian language packs
#             """)
    
#     # Main content
#     col1, col2 = st.columns([1, 1])
    
#     with col1:
#         st.header("📤 Upload Reference Audio")
        
#         # Language-specific guidance
#         if language in ["hi", "bn", "te", "ta", "mr", "gu", "kn", "ml", "pa", "or", "ur"]:
#             st.info(f"🎙️ **Indian Voice Tip**: For best {selected_lang_name} cloning, use audio with clear pronunciation and natural accent.")
        
#         uploaded_file = st.file_uploader(
#             "Choose an audio file to clone",
#             type=["wav", "mp3"],
#             help="Upload a clear voice sample (2-60 seconds recommended)"
#         )
        
#         if uploaded_file is not None:
#             # Display file info
#             file_extension = uploaded_file.name.split('.')[-1].lower()
#             st.success(f"✅ {uploaded_file.name} uploaded successfully!")
            
#             # Display audio player
#             st.audio(uploaded_file, format=f"audio/{file_extension}")
            
#             # File info
#             file_size_mb = uploaded_file.size / (1024 * 1024)
#             st.caption(f"📊 **File Info:** {uploaded_file.name} | {file_size_mb:.2f} MB | {file_extension.upper()} format")
            
#             # Try to get duration if it's a WAV file
#             if file_extension == "wav":
#                 try:
#                     import soundfile as sf
#                     with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
#                         tmp.write(uploaded_file.getvalue())
#                         tmp_path = tmp.name
#                         audio_info = sf.info(tmp_path)
#                         st.caption(f"⏱️ **Duration:** {audio_info.duration:.1f} seconds")
#                         os.unlink(tmp_path)
#                 except:
#                     pass
    
#     with col2:
#         st.header("✍️ Enter Text to Speak")
        
#         # Language-specific placeholder text
#         placeholders = {
#             "hi": "नमस्ते! मैं आपकी क्लोन की हुई आवाज़ हूँ। आज मैं आपकी कैसे मदद कर सकता हूँ?",
#             "bn": "নমস্কার! আমি আপনার ক্লোন করা ভয়েস। আজকে আমি আপনাকে কিভাবে সাহায্য করতে পারি?",
#             "te": "నమస్కారం! నేను మీ క్లోన్ చేసిన వాయిస్. ఈరోజు నేను మీకు ఎలా సహాయం చేయగలను?",
#             "ta": "வணக்கம்! நான் உங்கள் குளோன் செய்யப்பட்ட குரல். இன்று நான் உங்களுக்கு எவ்வாறு உதவ முடியும்?",
#             "mr": "नमस्कार! मी तुमचा क्लोन केलेला आवाज आहे. आज मी तुमची कशी मदत करू शकतो?",
#             "en": "Hello! I'm your cloned voice. How can I help you today?"
#         }
        
#         placeholder_text = placeholders.get(language, "Hello! I'm your cloned voice. How can I help you today?")
        
#         # Text input with character counter
#         text_input = st.text_area(
#             f"What should the cloned voice say in {selected_lang_name}?",
#             height=200,
#             placeholder=placeholder_text,
#             help=f"Enter the text in {selected_lang_name} you want to be spoken in the cloned voice"
#         )
        
#         if text_input:
#             char_count = len(text_input)
#             word_count = len(text_input.split())
            
#             col_a, col_b = st.columns(2)
#             with col_a:
#                 st.caption(f"📝 **Characters:** {char_count}")
#             with col_b:
#                 st.caption(f"🔤 **Words:** {word_count}")
            
#             # Validation
#             is_valid, validation_msg = validate_text(text_input)
#             if not is_valid:
#                 st.warning(f"⚠️ {validation_msg}")
    
#     # Generate button section
#     st.markdown("---")
#     st.markdown("## 🎬 Generate Cloned Voice")
    
#     # Language info banner
#     if language in indian_languages.values():
#         st.info(f"🇮🇳 **Generating {selected_lang_name} Voice** - Using XTTS-v2 multilingual model for Indian language support")
    
#     # Check if requirements are met
#     can_generate = uploaded_file is not None and text_input.strip() != ""
    
#     if not can_generate:
#         if uploaded_file is None:
#             st.info("📤 Please upload a reference audio file to begin")
#         if text_input.strip() == "":
#             st.info("✍️ Please enter text for the voice to speak")
    
#     # Generate button
#     col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
#     with col_btn2:
#         generate_btn = st.button(
#             f"🚀 Generate {selected_lang_name} Voice",
#             use_container_width=True,
#             type="primary" if can_generate else "secondary",
#             disabled=not can_generate
#         )
    
#     # Generation process
#     if generate_btn and can_generate:
#         # Create progress container
#         progress_container = st.container()
        
#         with progress_container:
#             # Step 1: Loading model
#             st.markdown("### 🔄 Step 1: Loading AI Model")
#             model_progress = st.progress(0)
            
#             try:
#                 tts, device = load_tts_model()
#                 model_progress.progress(100)
#                 st.success(f"✅ Model loaded on {device.upper()}")
#             except Exception as e:
#                 st.error(f"❌ Failed to load model: {str(e)}")
#                 st.stop()
            
#             # Step 2: Processing audio
#             st.markdown("### 🎵 Step 2: Processing Reference Audio")
#             audio_progress = st.progress(0)
            
#             # Save uploaded file temporarily
#             with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_input:
#                 tmp_input.write(uploaded_file.getvalue())
#                 tmp_input_path = tmp_input.name
            
#             audio_progress.progress(50)
            
#             # Create output file
#             output_path = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
            
#             audio_progress.progress(100)
            
#             # Step 3: Generating speech
#             st.markdown("### 🎙️ Step 3: Generating Cloned Voice")
#             gen_progress = st.progress(0)
            
#             try:
#                 success, message = clone_voice(
#                     tts,
#                     tmp_input_path,
#                     text_input,
#                     output_path,
#                     language=language
#                 )
#                 gen_progress.progress(100)
                
#                 if success:
#                     st.success(f"✅ {message}")
                    
#                     # Display results
#                     st.markdown("---")
#                     st.markdown(f"## 🔊 Generated {selected_lang_name} Audio")
                    
#                     # Play audio
#                     with open(output_path, "rb") as audio_file:
#                         audio_bytes = audio_file.read()
#                         st.audio(audio_bytes, format="audio/wav")
                    
#                     # Audio info
#                     import soundfile as sf
#                     output_info = sf.info(output_path)
#                     st.caption(f"⏱️ **Generated Duration:** {output_info.duration:.1f} seconds")
                    
#                     # Download button
#                     col_dl1, col_dl2, col_dl3 = st.columns([1, 2, 1])
#                     with col_dl2:
#                         st.download_button(
#                             label=f"⬇️ Download {selected_lang_name} Audio",
#                             data=audio_bytes,
#                             file_name=f"cloned_voice_{selected_lang_name.replace(' ', '_')}.wav",
#                             mime="audio/wav",
#                             use_container_width=True,
#                             icon="💾"
#                         )
                    
#                     # Success tips
#                     with st.expander("🎯 Tips for better results"):
#                         if language in indian_languages.values():
#                             st.markdown(f"""
#                             ### {selected_lang_name} Specific Tips:
#                             - **Pronunciation**: Ensure proper diction in the reference audio
#                             - **Accent**: Model works with various Indian accents
#                             - **Script**: Use correct Unicode characters for best results
#                             - **Sentence Length**: Keep sentences natural and conversational
#                             """)
#                         else:
#                             st.markdown("""
#                             - **Audio Quality:** Use studio-quality recordings for best results
#                             - **Text Length:** Keep sentences under 30 seconds for optimal clarity
#                             - **Emotion:** The model captures emotional tone from the reference
#                             - **Multiple Speakers:** For multi-speaker cloning, use separate samples
#                             """)
                    
#                 else:
#                     st.error(f"❌ {message}")
#                     st.info("💡 Try a different audio sample or shorter text")
            
#             except Exception as e:
#                 st.error(f"❌ Generation failed: {str(e)}")
            
#             finally:
#                 # Cleanup temporary files
#                 try:
#                     os.unlink(tmp_input_path)
#                     if os.path.exists(output_path) and not success:
#                         os.unlink(output_path)
#                 except:
#                     pass
    
#     # Indian languages showcase
#     st.markdown("---")
#     st.markdown("## 🇮🇳 Supported Indian Languages")
    
#     indian_lang_data = [
#         {"Language": "Hindi", "Native": "हिन्दी", "Code": "hi", "Speakers": "~615M", "Region": "North India"},
#         {"Language": "Bengali", "Native": "বাংলা", "Code": "bn", "Speakers": "~272M", "Region": "West Bengal, Bangladesh"},
#         {"Language": "Telugu", "Native": "తెలుగు", "Code": "te", "Speakers": "~96M", "Region": "Andhra Pradesh, Telangana"},
#         {"Language": "Tamil", "Native": "தமிழ்", "Code": "ta", "Speakers": "~85M", "Region": "Tamil Nadu, Sri Lanka"},
#         {"Language": "Marathi", "Native": "मराठी", "Code": "mr", "Speakers": "~99M", "Region": "Maharashtra"},
#         {"Language": "Gujarati", "Native": "ગુજરાતી", "Code": "gu", "Speakers": "~62M", "Region": "Gujarat"},
#     ]
    
#     # Display Indian languages in a nice table
#     col_i1, col_i2, col_i3 = st.columns(3)
#     for i, lang in enumerate(indian_lang_data):
#         col = [col_i1, col_i2, col_i3][i % 3]
#         with col:
#             st.markdown(f"""
#             **{lang['Language']}** ({lang['Native']})
#             - Code: `{lang['Code']}`
#             - Speakers: {lang['Speakers']}
#             - Region: {lang['Region']}
#             """)
    
#     # Footer with model info
#     st.markdown("---")
#     col_f1, col_f2, col_f3 = st.columns(3)
    
#     with col_f1:
#         st.markdown("**🤖 Model**")
#         st.caption("XTTS-v2 by Coqui")
#         st.caption("15+ Indian languages")
    
#     with col_f2:
#         st.markdown("**⚡ Backend**")
#         st.caption(f"PyTorch {torch.__version__}")
#         st.caption("Multilingual support")
    
#     with col_f3:
#         st.markdown("**📚 Language Support**")
#         st.caption("15 Indian languages")
#         st.caption("10+ global languages")
    
#     st.markdown("---")
#     st.markdown("""
#     <div style='text-align: center; color: #666; padding: 20px;'>
#         <p>Powered by <strong>Coqui TTS</strong> • Built with <strong>Streamlit</strong> • Voice Cloning Studio v3.0 🇮🇳</p>
#         <p><small>💡 <em>Now with comprehensive Indian language support. Use responsibly. Respect copyright and privacy.</em></small></p>
#     </div>
#     """, unsafe_allow_html=True)

# if __name__ == "__main__":
#     # Add required imports at runtime
#     try:
#         import soundfile
#     except ImportError:
#         st.warning("Installing required audio libraries...")
#         import subprocess
#         import sys
#         subprocess.check_call([sys.executable, "-m", "pip", "install", "soundfile", "librosa"])
    
#     # Run the app
#     main()















































# # """
# # Voice Cloning Web App with Streamlit and Coqui TTS
# # Upload English audio, enter text in Indian languages, and clone voices across languages
# # Advanced cross-lingual voice cloning with Indian language support (Hindi + Telugu + Marathi)
# # Fixed version with complete language support
# # """

# # import streamlit as st
# # from TTS.api import TTS
# # import torch
# # import os
# # import tempfile
# # import soundfile as sf
# # import warnings
# # from pathlib import Path

# # # Page configuration
# # st.set_page_config(
# #     page_title="Cross-Lingual Voice Cloning Studio",
# #     page_icon="🌐",
# #     layout="wide"
# # )

# # # Custom CSS
# # st.markdown("""
# #     <style>
# #     .main-header {
# #         font-size: 3rem;
# #         color: #4B8BBE;
# #         text-align: center;
# #         margin-bottom: 1rem;
# #     }
# #     .sub-header {
# #         text-align: center;
# #         color: #306998;
# #         margin-bottom: 2rem;
# #     }
# #     .stButton>button {
# #         width: 100%;
# #         height: 3rem;
# #         font-size: 1.2rem;
# #         background: linear-gradient(45deg, #4B8BBE, #306998);
# #     }
# #     .success-box {
# #         background-color: #d4edda;
# #         border: 1px solid #c3e6cb;
# #         border-radius: 5px;
# #         padding: 15px;
# #         margin: 10px 0;
# #     }
# #     .warning-box {
# #         background-color: #fff3cd;
# #         border: 1px solid #ffeaa7;
# #         border-radius: 5px;
# #         padding: 15px;
# #         margin: 10px 0;
# #     }
# #     .info-box {
# #         background-color: #d1ecf1;
# #         border: 1px solid #bee5eb;
# #         border-radius: 5px;
# #         padding: 15px;
# #         margin: 10px 0;
# #     }
# #     .language-badge {
# #         display: inline-block;
# #         padding: 2px 8px;
# #         border-radius: 12px;
# #         background-color: #306998;
# #         color: white;
# #         font-size: 0.8rem;
# #         margin-right: 5px;
# #     }
# #     .indian-language-badge {
# #         display: inline-block;
# #         padding: 2px 8px;
# #         border-radius: 12px;
# #         background-color: #FF9933;
# #         color: white;
# #         font-size: 0.8rem;
# #         margin-right: 5px;
# #     }
# #     </style>
# # """, unsafe_allow_html=True)

# # @st.cache_resource(show_spinner=False)
# # def load_tts_model():
# #     """Load TTS model safely with PyTorch 2.6+"""
# #     device = "cuda" if torch.cuda.is_available() else "cpu"
    
# #     try:
# #         # Load XTTS-v2 model for cross-lingual cloning
# #         st.info("Loading XTTS-v2 model for cross-lingual voice cloning...")
# #         tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
# #         return tts, device
# #     except Exception as e:
# #         st.error(f"Failed to load model: {str(e)}")
        
# #         # Fallback loading method
# #         try:
# #             warnings.filterwarnings("ignore", category=UserWarning)
# #             original_load = torch.load
            
# #             def custom_load(*args, **kwargs):
# #                 kwargs['weights_only'] = False
# #                 return original_load(*args, **kwargs)
            
# #             torch.load = custom_load
# #             tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
# #             torch.load = original_load
# #             return tts, device
# #         except Exception as e2:
# #             st.error(f"Complete failure: {str(e2)}")
# #             return None, device

# # def clone_voice_crosslingual(tts, reference_audio_path, text_to_speak, output_path, 
# #                             source_language="en", target_language="hi"):
# #     """
# #     Clone voice from source language to target language
# #     Supports cross-lingual voice cloning (e.g., English → Hindi/Telugu/Marathi)
# #     """
# #     try:
# #         # Validate audio duration
# #         audio_info = sf.info(reference_audio_path)
# #         duration = audio_info.duration
        
# #         if duration < 3:
# #             return False, "Audio is too short (minimum 3 seconds for cross-lingual)"
# #         if duration > 60:
# #             return False, "Audio is too long (maximum 60 seconds)"
        
# #         # Generate audio with cross-lingual cloning
# #         tts.tts_to_file(
# #             text=text_to_speak,
# #             speaker_wav=reference_audio_path,
# #             language=target_language,  # Target language for output
# #             file_path=output_path
# #         )
        
# #         return True, f"Cross-lingual audio generated! (English → {target_language.upper()}, Reference: {duration:.1f}s)"
# #     except Exception as e:
# #         return False, f"Error in cross-lingual cloning: {str(e)}"

# # def get_supported_languages():
# #     """Return languages supported by XTTS-v2 for cross-lingual cloning"""
    
# #     # SOURCE languages (where your voice sample comes from)
# #     source_languages = {
# #         "English": "en",
# #         "Spanish": "es",
# #         "French": "fr",
# #         "German": "de",
# #         "Italian": "it",
# #         "Portuguese": "pt"
# #     }
    
# #     # TARGET languages (what you want to generate) - UPDATED WITH MARATHI
# #     target_languages = {
# #         # Indian Languages
# #         "Hindi (हिन्दी)": "hi",
# #         "Telugu (తెలుగు)": "te",
# #         "Marathi (मराठी)": "mr",  # ADDED MARATHI SUPPORT
# #         # Other supported languages (can be mixed with Indian languages)
# #         "English": "en",
# #         "Spanish (Español)": "es",
# #         "French (Français)": "fr",
# #         "German (Deutsch)": "de",
# #         "Chinese (中文)": "zh-cn",
# #         "Japanese (日本語)": "ja",
# #         "Korean (한국어)": "ko",
# #         "Arabic (العربية)": "ar",
# #         "Russian (Русский)": "ru"
# #     }
    
# #     return source_languages, target_languages

# # def get_language_examples():
# #     """Return example texts for different Indian languages - UPDATED WITH MARATHI"""
# #     examples = {
# #         "hi": {
# #             "text": "नमस्ते! यह आपकी क्लोन की गई आवाज है। मैं अंग्रेजी आवाज से हिंदी में बोल रहा हूँ।",
# #             "meaning": "Hello! This is your cloned voice. I'm speaking Hindi from an English voice sample.",
# #             "english_sample": "Hello, this is a test recording for voice cloning purposes."
# #         },
# #         "te": {
# #             "text": "నమస్కారం! ఇది మీ క్లోన్ చేయబడిన వాయిస్. నేను ఇంగ్లీష్ వాయిస్ నుండి తెలుగులో మాట్లాడుతున్నాను.",
# #             "meaning": "Hello! This is your cloned voice. I'm speaking Telugu from an English voice sample.",
# #             "english_sample": "Hello, this is a test recording for voice cloning purposes."
# #         },
# #         "mr": {  # ADDED MARATHI EXAMPLE
# #             "text": "नमस्कार! ही तुमची क्लोन केलेली आवाज आहे. मी इंग्रजी आवाजावरून मराठीत बोलत आहे.",
# #             "meaning": "Hello! This is your cloned voice. I'm speaking Marathi from an English voice sample.",
# #             "english_sample": "Hello, this is a test recording for voice cloning purposes."
# #         },
# #         "en": {
# #             "text": "Hello! This is your cloned voice. I can speak multiple languages from your English voice sample.",
# #             "meaning": "",
# #             "english_sample": "Hello, this is a test recording for voice cloning purposes."
# #         },
# #         "es": {
# #             "text": "¡Hola! Esta es tu voz clonada. Puedo hablar varios idiomas desde tu muestra de voz en inglés.",
# #             "meaning": "Hello! This is your cloned voice. I can speak multiple languages from your English voice sample.",
# #             "english_sample": "Hello, this is a test recording for voice cloning purposes."
# #         },
# #         "fr": {
# #             "text": "Bonjour ! C'est votre voix clonée. Je peux parler plusieurs langues à partir de votre échantillon vocal anglais.",
# #             "meaning": "Hello! This is your cloned voice. I can speak multiple languages from your English voice sample.",
# #             "english_sample": "Hello, this is a test recording for voice cloning purposes."
# #         },
# #         "zh-cn": {
# #             "text": "你好！这是你克隆的声音。我可以从你的英语声音样本中说多种语言。",
# #             "meaning": "Hello! This is your cloned voice. I can speak multiple languages from your English voice sample.",
# #             "english_sample": "Hello, this is a test recording for voice cloning purposes."
# #         }
# #     }
# #     return examples

# # def validate_audio_file(audio_path):
# #     """Validate audio file for cross-lingual cloning"""
# #     try:
# #         info = sf.info(audio_path)
        
# #         # Check duration
# #         if info.duration < 3:
# #             return False, "Audio too short (need at least 3 seconds for good cloning)"
# #         if info.duration > 60:
# #             return False, "Audio too long (max 60 seconds)"
        
# #         # Check sample rate (XTTS works best with 16-24kHz)
# #         if info.samplerate < 16000 or info.samplerate > 48000:
# #             return True, f"Sample rate {info.samplerate}Hz may need resampling (optimal: 16-24kHz)"
        
# #         return True, f"Audio valid: {info.duration:.1f}s, {info.samplerate}Hz"
# #     except Exception as e:
# #         return False, f"Cannot read audio file: {str(e)}"

# # def main():
# #     # Header
# #     st.markdown('<h1 class="main-header">🌐 Cross-Lingual Voice Cloning Studio</h1>', unsafe_allow_html=True)
# #     st.markdown('<p class="sub-header">Clone English voices to speak Indian languages | Advanced cross-lingual AI technology</p>', unsafe_allow_html=True)
    
# #     # Language badges display
# #     st.markdown("""
# #     <div style="text-align: center; margin-bottom: 20px;">
# #         <span class="indian-language-badge">Hindi (हिन्दी) ✔</span>
# #         <span class="indian-language-badge">Telugu (తెలుగు) ✔</span>
# #         <span class="indian-language-badge">Marathi (मराठी) ✔</span>
# #         <span class="language-badge">English</span>
# #         <span class="language-badge">Spanish</span>
# #         <span class="language-badge">French</span>
# #         <span class="language-badge">17+ Languages</span>
# #     </div>
# #     """, unsafe_allow_html=True)
    
# #     # Initialize session state
# #     if 'generated_audio' not in st.session_state:
# #         st.session_state.generated_audio = None
# #     if 'generated_text' not in st.session_state:
# #         st.session_state.generated_text = ""
# #     if 'target_language' not in st.session_state:
# #         st.session_state.target_language = "hi"
    
# #     # Get language configurations
# #     source_languages, target_languages = get_supported_languages()
# #     examples = get_language_examples()
    
# #     # Main layout
# #     col1, col2 = st.columns([1, 1])
    
# #     with col1:
# #         st.header("🎤 Step 1: Upload English Voice Sample")
        
# #         st.markdown("""
# #         <div class="info-box">
# #         <strong>📢 How cross-lingual cloning works:</strong><br>
# #         1. Upload an <strong>English</strong> voice sample (3-30 seconds)<br>
# #         2. The AI learns the speaker's voice characteristics<br>
# #         3. Generate speech in <strong>Hindi, Telugu, Marathi or other languages</strong><br>
# #         4. Same voice, different language!
# #         </div>
# #         """, unsafe_allow_html=True)
        
# #         # Upload audio
# #         uploaded_file = st.file_uploader(
# #             "Upload English voice recording (WAV/MP3 format)",
# #             type=["wav", "mp3", "ogg"],
# #             help="Upload clear English speech for best cross-lingual cloning"
# #         )
        
# #         if uploaded_file is not None:
# #             # Save to temp file
# #             file_extension = uploaded_file.name.split('.')[-1].lower()
# #             temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_extension}")
# #             temp_file.write(uploaded_file.getvalue())
# #             temp_file.close()
            
# #             # Validate audio
# #             st.audio(uploaded_file.getvalue(), format=f"audio/{file_extension}")
            
# #             # Show audio info
# #             try:
# #                 audio_info = sf.info(temp_file.name)
# #                 duration = audio_info.duration
# #                 samplerate = audio_info.samplerate
                
# #                 st.success(f"✅ Audio uploaded successfully!")
# #                 st.caption(f"**Duration:** {duration:.1f} seconds | **Sample Rate:** {samplerate} Hz")
                
# #                 # Quality indicators
# #                 if duration < 6:
# #                     st.warning("⚠️ Short sample - for best results use 6-20 seconds")
# #                 elif duration > 30:
# #                     st.info("ℹ️ Long sample - using first 30 seconds for optimal quality")
                
# #                 # Store in session state
# #                 st.session_state.audio_path = temp_file.name
# #                 st.session_state.audio_duration = duration
                
# #             except Exception as e:
# #                 st.error(f"❌ Cannot read audio file: {str(e)}")
    
# #     with col2:
# #         st.header("🌍 Step 2: Select Target Language")
        
# #         # Target language selection
# #         st.subheader("Choose output language:")
# #         target_lang = st.selectbox(
# #             "Select language for generated speech",
# #             options=list(target_languages.keys()),
# #             index=0,
# #             help="Choose which language you want the cloned voice to speak in"
# #         )
        
# #         # Get language code
# #         target_lang_code = target_languages[target_lang]
# #         st.session_state.target_language = target_lang_code
        
# #         # Language info
# #         st.markdown(f"<div class='info-box'><strong>Selected:</strong> {target_lang}</div>", 
# #                    unsafe_allow_html=True)
        
# #         # Show example for selected language
# #         if target_lang_code in examples:
# #             example = examples[target_lang_code]
# #             st.subheader("✍️ Example text for this language:")
            
# #             # Text input with example
# #             default_text = example["text"]
            
# #             user_text = st.text_area(
# #                 f"Enter text in {target_lang.split(' (')[0]}",
# #                 value=default_text,
# #                 height=150,
# #                 help=f"Write what you want the cloned voice to say in {target_lang}"
# #             )
            
# #             st.session_state.generated_text = user_text
            
# #             if example.get("meaning"):
# #                 st.caption(f"**English meaning:** {example['meaning']}")
            
# #             # Character counter
# #             char_count = len(user_text)
# #             st.caption(f"📝 **Characters:** {char_count}/1000")
            
# #             if char_count > 500:
# #                 st.info("ℹ️ For longer texts, consider generating in segments")
    
# #     # Advanced settings
# #     with st.expander("⚙️ Advanced Settings & Model Info"):
# #         col_a1, col_a2 = st.columns(2)
        
# #         with col_a1:
# #             st.subheader("Model Information")
# #             st.markdown("""
# #             **XTTS-v2 Cross-Lingual Features:**
# #             - Voice cloning from English to multiple languages
# #             - Zero-shot voice adaptation
# #             - 17+ supported languages including Hindi, Telugu & Marathi
# #             - Optimal: 6+ seconds of clear English speech
# #             - Output: 24kHz high-quality audio
# #             """)
            
# #             # Device info
# #             device = "cuda" if torch.cuda.is_available() else "cpu"
# #             if device == "cuda":
# #                 st.success(f"✅ GPU: {torch.cuda.get_device_name(0)}")
# #             else:
# #                 st.warning("⚠️ CPU mode - slower generation")
        
# #         with col_a2:
# #             st.subheader("Quality Tips")
# #             st.markdown("""
# #             **For best cross-lingual results:**
# #             - Use clear English speech (no background noise)
# #             - Natural speaking tone (not too fast/slow)
# #             - Single speaker only
# #             - Avoid music or sound effects
# #             - WAV format recommended (16-24kHz)
# #             - 6-20 seconds optimal length
# #             """)
            
# #             # Clear cache button
# #             if st.button("🔄 Clear Model Cache"):
# #                 st.cache_resource.clear()
# #                 st.success("Cache cleared - model will reload")
    
# #     # Generate section
# #     st.markdown("---")
# #     st.header("🚀 Step 3: Generate Cross-Lingual Voice")
    
# #     # Check if ready to generate
# #     can_generate = ('audio_path' in st.session_state and 
# #                     st.session_state.generated_text.strip() != "")
    
# #     if not can_generate:
# #         if 'audio_path' not in st.session_state:
# #             st.info("📤 Please upload an English voice sample first")
# #         elif not st.session_state.generated_text.strip():
# #             st.info("✍️ Please enter text for the voice to speak")
    
# #     # Generate button
# #     col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
# #     with col_btn2:
# #         generate_btn = st.button(
# #             f"✨ Generate {target_lang.split(' (')[0]} Voice from English Sample",
# #             type="primary" if can_generate else "secondary",
# #             disabled=not can_generate,
# #             use_container_width=True
# #         )
    
# #     # Generation process
# #     if generate_btn and can_generate:
# #         with st.spinner("Loading AI model for cross-lingual cloning..."):
# #             tts, device = load_tts_model()
            
# #             if tts is None:
# #                 st.error("❌ Failed to load TTS model")
# #                 st.stop()
            
# #             st.success(f"✅ Model loaded on {device.upper()}")
        
# #         # Create progress bar
# #         progress_bar = st.progress(0)
# #         status_text = st.empty()
        
# #         # Step 1: Processing audio
# #         status_text.text("🔍 Processing English voice sample...")
# #         progress_bar.progress(25)
        
# #         # Validate audio again
# #         is_valid, valid_msg = validate_audio_file(st.session_state.audio_path)
# #         if not is_valid:
# #             st.error(f"❌ Audio validation failed: {valid_msg}")
# #             st.stop()
        
# #         # Step 2: Preparing for generation
# #         status_text.text("⚙️ Setting up cross-lingual synthesis...")
# #         progress_bar.progress(50)
        
# #         # Create output file
# #         output_path = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
        
# #         # Step 3: Generate speech
# #         status_text.text(f"🎙️ Generating {target_lang} speech from English voice...")
# #         progress_bar.progress(75)
        
# #         try:
# #             success, message = clone_voice_crosslingual(
# #                 tts,
# #                 st.session_state.audio_path,
# #                 st.session_state.generated_text,
# #                 output_path,
# #                 source_language="en",
# #                 target_language=st.session_state.target_language
# #             )
            
# #             progress_bar.progress(100)
            
# #             if success:
# #                 status_text.text("✅ Generation complete!")
                
# #                 # Store in session state
# #                 st.session_state.generated_audio = output_path
                
# #                 # Display results
# #                 st.markdown("---")
# #                 st.markdown(f"## 🔊 Generated {target_lang} Audio")
                
# #                 # Play audio
# #                 with open(output_path, "rb") as f:
# #                     audio_bytes = f.read()
# #                     st.audio(audio_bytes, format="audio/wav")
                
# #                 # Audio info
# #                 try:
# #                     output_info = sf.info(output_path)
# #                     st.caption(f"**Duration:** {output_info.duration:.1f}s | **Sample Rate:** {output_info.samplerate}Hz")
# #                 except:
# #                     pass
                
# #                 # Download button
# #                 st.download_button(
# #                     label=f"💾 Download {target_lang} Audio",
# #                     data=audio_bytes,
# #                     file_name=f"crosslingual_clone_{target_lang_code}.wav",
# #                     mime="audio/wav",
# #                     icon="⬇️"
# #                 )
                
# #                 # Success message
# #                 st.markdown(f"""
# #                 <div class="success-box">
# #                 <strong>✅ Cross-lingual voice cloning successful!</strong><br>
# #                 • English voice → {target_lang} speech<br>
# #                 • Same voice characteristics, different language<br>
# #                 • AI has learned the speaker's voice from English sample
# #                 </div>
# #                 """, unsafe_allow_html=True)
                
# #                 # Try another language button
# #                 st.markdown("---")
# #                 st.subheader("🔄 Try Another Language")
                
# #                 col_l1, col_l2, col_l3, col_l4 = st.columns(4)
# #                 with col_l1:
# #                     if st.button("Try Hindi", use_container_width=True):
# #                         st.session_state.target_language = "hi"
# #                         st.rerun()
# #                 with col_l2:
# #                     if st.button("Try Telugu", use_container_width=True):
# #                         st.session_state.target_language = "te"
# #                         st.rerun()
# #                 with col_l3:
# #                     if st.button("Try Marathi", use_container_width=True):
# #                         st.session_state.target_language = "mr"
# #                         st.rerun()
# #                 with col_l4:
# #                     if st.button("Try Spanish", use_container_width=True):
# #                         st.session_state.target_language = "es"
# #                         st.rerun()
                
# #             else:
# #                 st.error(f"❌ Generation failed: {message}")
                
# #         except Exception as e:
# #             st.error(f"❌ Error during generation: {str(e)}")
# #             st.info("💡 Try a different voice sample or shorter text")
        
# #         finally:
# #             # Cleanup temporary input file
# #             try:
# #                 if 'audio_path' in st.session_state and os.path.exists(st.session_state.audio_path):
# #                     os.unlink(st.session_state.audio_path)
# #             except:
# #                 pass
    
# #     # Demo section if no audio uploaded
# #     if 'audio_path' not in st.session_state:
# #         st.markdown("---")
# #         st.header("🎯 Quick Demo")
        
# #         col_d1, col_d2, col_d3, col_d4 = st.columns(4)
        
# #         with col_d1:
# #             st.subheader("🇺🇸 English Sample")
# #             st.markdown("""
# #             **Sample text:**
# #             > "Hello, this is a demonstration of voice cloning technology."
            
# #             **Use this to clone to:**
# #             - Hindi, Telugu, Marathi, etc.
# #             """)
        
# #         with col_d2:
# #             st.subheader("🇮🇳 Hindi Output")
# #             st.markdown("""
# #             **Generated text:**
# #             > "नमस्ते, यह आवाज क्लोनिंग तकनीक का प्रदर्शन है।"
            
# #             **Same voice speaking Hindi**
# #             """)
        
# #         with col_d3:
# #             st.subheader("🇮🇳 Telugu Output")
# #             st.markdown("""
# #             **Generated text:**
# #             > "నమస్కారం, ఇది వాయిస్ క్లోనింగ్ టెక్నాలజీ ప్రదర్శన."
            
# #             **Same voice speaking Telugu**
# #             """)
            
# #         with col_d4:
# #             st.subheader("🇮🇳 Marathi Output")
# #             st.markdown("""
# #             **Generated text:**
# #             > "नमस्कार, ही आवाज क्लोनिंग तंत्रज्ञानाची प्रात्यक्षिक आहे."
            
# #             **Same voice speaking Marathi**
# #             """)
    
# #     # How it works section
# #     with st.expander("🤖 How Cross-Lingual Voice Cloning Works"):
# #         st.markdown("""
# #         ### 🧠 AI Technology Behind Cross-Lingual Cloning
        
# #         **1. Voice Identity Extraction**
# #         - The AI analyzes your English voice sample
# #         - Extracts unique voice characteristics: pitch, tone, rhythm, timbre
# #         - Creates a "voice fingerprint" that's language-independent
        
# #         **2. Language-Specific Synthesis**
# #         - Uses the target language's phonetics and pronunciation rules
# #         - Applies your voice characteristics to the new language
# #         - Maintains natural accent and intonation
        
# #         **3. XTTS-v2 Architecture**
# #         - Multilingual training on 17+ languages
# #         - Zero-shot voice adaptation (no fine-tuning needed)
# #         - Cross-lingual transfer learning
# #         - High-quality 24kHz output
# #         - Now supports Hindi, Telugu and Marathi!
        
# #         **4. Supported Use Cases**
# #         - English speakers creating content in Indian languages
# #         - Voiceovers for multilingual videos
# #         - Accessibility tools for different language speakers
# #         - Language learning applications
# #         """)
    
# #     # Footer
# #     st.markdown("---")
# #     col_f1, col_f2, col_f3 = st.columns(3)
    
# #     with col_f1:
# #         st.markdown("**🤖 AI Model**")
# #         st.caption("XTTS-v2 Multilingual")
# #         st.caption("Coqui TTS Framework")
# #         st.caption("PyTorch Backend")
    
# #     with col_f2:
# #         st.markdown("**🌍 Languages**")
# #         st.caption("Cross-lingual cloning")
# #         st.caption("English → Hindi/Telugu/Marathi")
# #         st.caption("17+ languages supported")
    
# #     with col_f3:
# #         st.markdown("**⚡ Performance**")
# #         st.caption(f"Device: {'GPU' if torch.cuda.is_available() else 'CPU'}")
# #         st.caption("6s min sample length")
# #         st.caption("24kHz output quality")
    
# #     st.markdown("---")
# #     st.markdown("""
# #     <div style='text-align: center; color: #666; padding: 20px;'>
# #         <p><strong>Cross-Lingual Voice Cloning Studio</strong> • v2.1 • Powered by Coqui XTTS-v2</p>
# #         <p><small>💡 Upload English voice, generate Hindi/Telugu/Marathi/global language speech • Use responsibly</small></p>
# #     </div>
# #     """, unsafe_allow_html=True)

# # if __name__ == "__main__":
# #     # Ensure required packages
# #     try:
# #         import soundfile
# #     except ImportError:
# #         st.warning("Installing audio libraries...")
# #         import subprocess
# #         subprocess.run(["pip", "install", "soundfile", "librosa", "torchaudio"])
    
# #     # Run app
# #     main()













































# import streamlit as st
# import requests
# import os
# import tempfile

# # ---------------- CONFIG ----------------
# ELEVENLABS_API_URL = "https://api.elevenlabs.io/v1"

# st.set_page_config(
#     page_title="ElevenLabs Voice Cloning Studio",
#     page_icon="🎙️",
#     layout="wide"
# )

# # ---------------- UI ----------------
# st.title("🎙️ ElevenLabs Voice Cloning Studio")
# st.caption("Hindi • Telugu • Marathi • English • 30+ languages")

# # ---------------- SIDEBAR ----------------
# api_key = st.sidebar.text_input(
#     "🔑 ElevenLabs API Key",
#     type="password"
# )

# if not api_key:
#     st.warning("Please enter your ElevenLabs API key")
#     st.stop()

# headers = {
#     "xi-api-key": api_key
# }

# # ---------------- FUNCTIONS ----------------
# def create_cloned_voice(api_key, audio_file, voice_name):
#     url = f"{ELEVENLABS_API_URL}/voices/add"

#     files = {
#         "files": (audio_file.name, audio_file, audio_file.type)
#     }

#     data = {
#         "name": voice_name,
#         "description": "Cloned via Streamlit app"
#     }

#     response = requests.post(
#         url,
#         headers={"xi-api-key": api_key},
#         data=data,
#         files=files
#     )

#     if response.status_code == 200:
#         return True, response.json()["voice_id"]
#     else:
#         return False, response.text


# def generate_speech(api_key, voice_id, text):
#     url = f"{ELEVENLABS_API_URL}/text-to-speech/{voice_id}"

#     payload = {
#         "text": text,
#         "model_id": "eleven_multilingual_v2",
#         "voice_settings": {
#             "stability": 0.55,
#             "similarity_boost": 0.75,
#             "style": 0.3,
#             "use_speaker_boost": True
#         }
#     }

#     response = requests.post(
#         url,
#         headers={
#             "xi-api-key": api_key,
#             "Content-Type": "application/json"
#         },
#         json=payload
#     )

#     if response.status_code == 200:
#         return True, response.content
#     else:
#         return False, response.text


# # ---------------- MAIN ----------------
# mode = st.radio(
#     "Choose Mode",
#     ["Text-to-Speech", "Voice Cloning"]
# )

# text = st.text_area(
#     "Enter text (Hindi / Telugu / Marathi / English)",
#     height=150
# )

# # ---------------- TEXT TO SPEECH ----------------
# if mode == "Text-to-Speech":
#     voice_id = st.text_input(
#         "Enter Voice ID (from ElevenLabs dashboard)",
#         value="21m00Tcm4TlvDq8ikWAM"
#     )

#     if st.button("Generate Voice"):
#         success, result = generate_speech(api_key, voice_id, text)

#         if success:
#             st.audio(result, format="audio/mp3")
#             st.download_button(
#                 "Download Audio",
#                 result,
#                 "tts_output.mp3",
#                 "audio/mp3"
#             )
#         else:
#             st.error(result)

# # ---------------- VOICE CLONING ----------------
# else:
#     uploaded_audio = st.file_uploader(
#         "Upload voice sample (10–60 sec clear audio)",
#         type=["wav", "mp3", "m4a"]
#     )

#     voice_name = st.text_input("Voice Name", "My Cloned Voice")

#     if st.button("Clone Voice & Generate Speech"):
#         if not uploaded_audio:
#             st.warning("Upload audio sample first")
#             st.stop()

#         with st.spinner("Cloning voice..."):
#             success, voice_id_or_error = create_cloned_voice(
#                 api_key,
#                 uploaded_audio,
#                 voice_name
#             )

#         if not success:
#             st.error(voice_id_or_error)
#             st.stop()

#         st.success(f"Voice cloned successfully ✅ (Voice ID: {voice_id_or_error})")

#         with st.spinner("Generating speech..."):
#             success, audio = generate_speech(
#                 api_key,
#                 voice_id_or_error,
#                 text
#             )

#         if success:
#             st.audio(audio, format="audio/mp3")
#             st.download_button(
#                 "Download Audio",
#                 audio,
#                 "cloned_voice.mp3",
#                 "audio/mp3"
#             )
#         else:
#             st.error(audio)


























import os
import tempfile
import subprocess
import time
import base64
from dotenv import load_dotenv

from openai import OpenAI
from elevenlabs.client import ElevenLabs

# ------------------------------
# ENV SETUP
# ------------------------------

load_dotenv()

ELEVENLABS_KEY = os.getenv("ELEVENLABS_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

assert ELEVENLABS_KEY, "Set ELEVENLABS_API_KEY in .env"
assert OPENAI_API_KEY, "Set OPENAI_API_KEY in .env"

openai_client = OpenAI(api_key=OPENAI_API_KEY)
elevenlabs_client = ElevenLabs(api_key=ELEVENLABS_KEY)

# ✅ FIXED NEWS ANCHOR VOICE
DEFAULT_NEWS_VOICE_ID = "21m00Tcm4TlvDq8ikWAM"

# ------------------------------
# FILE HELPERS
# ------------------------------

def save_file(uploaded_file) -> str:
    suffix = os.path.splitext(uploaded_file.name)[1].lower()
    path = tempfile.NamedTemporaryFile(delete=False, suffix=suffix).name
    with open(path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return path

def is_video(path: str) -> bool:
    return path.endswith((".mp4", ".mkv", ".mov", ".avi", ".webm"))

def is_audio(path: str) -> bool:
    return path.endswith((".wav", ".mp3", ".m4a", ".ogg", ".flac"))

def is_image(path: str) -> bool:
    return path.endswith((".png", ".jpg", ".jpeg", ".webp"))

LANGUAGE_MAP = {
    "en": "English",
    "hi": "Hindi",
    "te": "Telugu",
    "kn": "Kannada",
}

# ------------------------------
# VIDEO UTILITIES
# ------------------------------

def extract_audio(video_path: str) -> str:
    audio_path = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
    subprocess.run(
        ["ffmpeg", "-y", "-i", video_path, "-ac", "1", "-ar", "16000", audio_path],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=True,
    )
    return audio_path

def extract_key_frames(video_path: str, every_n_seconds=2):
    frame_dir = tempfile.mkdtemp()
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-i", video_path,
            "-vf", f"fps=1/{every_n_seconds}",
            f"{frame_dir}/frame_%03d.jpg"
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=True,
    )
    return sorted(
        os.path.join(frame_dir, f)
        for f in os.listdir(frame_dir)
        if f.endswith(".jpg")
    )

# ------------------------------
# OPENAI – STT & VISION
# ------------------------------

def transcribe_audio(audio_path: str) -> str:
    with open(audio_path, "rb") as f:
        result = openai_client.audio.transcriptions.create(
            file=f,
            model="whisper-1"
        )
    return result.text.strip()

def analyze_image(image_path: str) -> str:
    with open(image_path, "rb") as img:
        image_base64 = base64.b64encode(img.read()).decode("utf-8")

    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a professional news analyst. "
                    "Describe clearly, factually, and neutrally what is visible."
                ),
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this image."},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_base64}"
                        },
                    },
                ],
            },
        ],
        max_tokens=300,
        temperature=0.2,
    )
    return response.choices[0].message.content.strip()

# ------------------------------
# NEWS SCRIPT GENERATION
# ------------------------------

def generate_news_script(text: str, lang: str = "en") -> str:
    language = LANGUAGE_MAP.get(lang, "English")

    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    f"You are a professional TV news anchor. "
                    f"Write the final news script strictly in {language}. "
                    "Use HEADLINE, LEAD, and BODY. "
                    "Formal, broadcast-ready language only."
                ),
            },
            {
                "role": "user",
                "content": text,
            },
        ],
        temperature=0.25,
        max_tokens=450,
    )
    return response.choices[0].message.content.strip()

# ------------------------------
# ELEVENLABS – VOICE
# ------------------------------

def clone_voice(audio_file_path: str, voice_name: str) -> str:
    with open(audio_file_path, "rb") as f:
        voice = elevenlabs_client.voices.ivc.create(
            name=voice_name,
            files=[f]
        )
    return voice.voice_id

def text_to_speech(text: str, voice_id: str | None = None) -> str:
    """
    Always uses fixed news anchor voice unless cloning is enabled
    """
    output_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3").name

    final_voice_id = voice_id or DEFAULT_NEWS_VOICE_ID

    audio = elevenlabs_client.text_to_speech.convert(
        text=text,
        voice_id=final_voice_id,
        model_id="eleven_multilingual_v2"
    )

    with open(output_path, "wb") as f:
        for chunk in audio:
            f.write(chunk)

    return output_path

# ------------------------------
# MAIN PIPELINE
# ------------------------------

def process_file(
    path: str,
    lang: str = "en",
    clone_voice_from_audio: bool = False,
    voice_id: str | None = None
) -> dict:
    """
    Process uploaded media and generate news audio using fixed anchor voice
    """

    try:
        # ---------- VIDEO ----------
        if is_video(path):
            audio = extract_audio(path)
            transcript = transcribe_audio(audio)
            frames = extract_key_frames(path)

            visuals = [analyze_image(f) for f in frames[:5]]
            visuals_text = "\n".join(visuals)

            combined = (
                "Visuals:\n" + visuals_text +
                "\n\nTranscript:\n" + transcript
            )

            script = generate_news_script(combined, lang)

            final_voice_id = (
                clone_voice(audio, f"video_voice_{int(time.time())}")
                if clone_voice_from_audio
                else None
            )

            audio_out = text_to_speech(script, final_voice_id)

            return {
                "type": "video",
                "language": lang,
                "transcript": transcript,
                "news_script": script,
                "audio": audio_out,
            }

        # ---------- AUDIO ----------
        elif is_audio(path):
            transcript = transcribe_audio(path)
            script = generate_news_script(transcript, lang)

            final_voice_id = (
                clone_voice(path, f"audio_voice_{int(time.time())}")
                if clone_voice_from_audio
                else None
            )

            audio_out = text_to_speech(script, final_voice_id)

            return {
                "type": "audio",
                "language": lang,
                "transcript": transcript,
                "news_script": script,
                "audio": audio_out,
            }

        # ---------- IMAGE ----------
        elif is_image(path):
            meaning = analyze_image(path)
            script = generate_news_script(meaning, lang)
            audio_out = text_to_speech(script)

            return {
                "type": "image",
                "language": lang,
                "image_meaning": meaning,
                "news_script": script,
                "audio": audio_out,
            }

        return {"error": "Unsupported file type"}

    except subprocess.CalledProcessError:
        return {"error": "FFmpeg error. Ensure FFmpeg is installed."}
    except Exception as e:
        return {"error": f"Processing error: {str(e)}"}
