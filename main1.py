# import os
# import json
# import tempfile
# import subprocess
# import time
# import base64
# from dotenv import load_dotenv
# from openai import OpenAI, RateLimitError
# from elevenlabs.client import ElevenLabs

# load_dotenv()

# # ------------------------------
# # Setup
# # ------------------------------

# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")

# assert OPENAI_API_KEY
# assert ELEVENLABS_API_KEY

# openai_client = OpenAI(api_key=OPENAI_API_KEY)
# elevenlabs_client = ElevenLabs(api_key=ELEVENLABS_API_KEY)

# VOICE_STORE = "saved_voices.json"

# LANGUAGE_MAP = {
#     "en": "English",
#     "hi": "Hindi",
#     "te": "Telugu",
#     "kn": "Kannada",
# }

# # ------------------------------
# # Rate-limit safe retry
# # ------------------------------

# def openai_retry(call, retries=5, delay=0.5):
#     for i in range(retries):
#         try:
#             return call()
#         except RateLimitError:
#             if i == retries - 1:
#                 raise
#             time.sleep(delay * (2 ** i))

# # ------------------------------
# # Voice persistence
# # ------------------------------

# def load_voice_store():
#     if not os.path.exists(VOICE_STORE):
#         return {}
#     with open(VOICE_STORE, "r") as f:
#         return json.load(f)

# def save_voice_store(store):
#     with open(VOICE_STORE, "w") as f:
#         json.dump(store, f, indent=2)

# def get_saved_voice(voice_key):
#     store = load_voice_store()
#     return store.get(voice_key)


# def clone_and_save_voice(audio_path, voice_key):
#     store = load_voice_store()
#     with open(audio_path, "rb") as f:
#         voice = elevenlabs_client.voices.ivc.create(
#             name=voice_key,
#             files=[f]
#         )
#     store[voice_key] = voice.voice_id
#     save_voice_store(store)
#     return voice.voice_id

# # ------------------------------
# # File helpers
# # ------------------------------

# def is_video(p): return p.lower().endswith((".mp4", ".mkv", ".mov", ".avi", ".webm"))
# def is_audio(p): return p.lower().endswith((".wav", ".mp3", ".m4a", ".ogg", ".flac"))
# def is_image(p): return p.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))

# # ------------------------------
# # FFmpeg helpers
# # ------------------------------

# def video_has_audio(video_path):
#     result = subprocess.run(
#         ["ffmpeg", "-i", video_path],
#         stderr=subprocess.PIPE,
#         stdout=subprocess.DEVNULL,
#         text=True,
#     )
#     return "Audio:" in result.stderr

# def extract_audio(video_path):
#     if not video_has_audio(video_path):
#         return None

#     audio_path = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
#     subprocess.run(
#         [
#             "ffmpeg", "-y", "-loglevel", "error",
#             "-i", video_path,
#             "-vn", "-ac", "1", "-ar", "16000",
#             audio_path
#         ],
#         check=True
#     )
#     return audio_path

# def extract_key_frames(video_path, every_n_seconds=2):
#     frame_dir = tempfile.mkdtemp()
#     subprocess.run(
#         [
#             "ffmpeg", "-y",
#             "-i", video_path,
#             "-vf", f"fps=1/{every_n_seconds},scale=640:-1",
#             f"{frame_dir}/frame_%03d.jpg"
#         ],
#         stdout=subprocess.DEVNULL,
#         stderr=subprocess.DEVNULL,
#         check=True,
#     )
#     return [
#         os.path.join(frame_dir, f)
#         for f in sorted(os.listdir(frame_dir))
#         if f.endswith(".jpg")
#     ]

# # ------------------------------
# # STT
# # ------------------------------

# def transcribe_audio(audio_path):
#     if not audio_path:
#         return "[No audible sound detected.]"

#     def call():
#         with open(audio_path, "rb") as f:
#             return openai_client.audio.transcriptions.create(
#                 file=f,
#                 model="whisper-1"
#             )

#     return openai_retry(call).text.strip()

# # ------------------------------
# # FIX-2 Batched Vision
# # ------------------------------

# def analyze_frames_batch(frames):
#     content = [{"type": "text", "text": "Describe what is visible in each image."}]

#     for frame in frames[:5]:
#         with open(frame, "rb") as f:
#             content.append({
#                 "type": "image_url",
#                 "image_url": {
#                     "url": "data:image/jpeg;base64," +
#                            base64.b64encode(f.read()).decode()
#                 }
#             })

#     def call():
#         return openai_client.chat.completions.create(
#             model="gpt-4o-mini",
#             messages=[
#                 {"role": "system", "content": "Neutral news image analysis."},
#                 {"role": "user", "content": content},
#             ],
#             max_tokens=400,
#             temperature=0.2,
#         )

#     return openai_retry(call).choices[0].message.content.strip()

# # ------------------------------
# # News script
# # ------------------------------

# def generate_news_script(text, lang):
#     language = LANGUAGE_MAP.get(lang, "English")

#     def call():
#         return openai_client.chat.completions.create(
#             model="gpt-4o-mini",
#             messages=[
#                 {
#                     "role": "system",
#                     "content": (
#                         f"Write a TV news report in {language}. "
#                         "Include HEADLINE, LEAD, BODY."
#                     ),
#                 },
#                 {"role": "user", "content": text},
#             ],
#             max_tokens=450,
#             temperature=0.25,
#         )

#     return openai_retry(call).choices[0].message.content.strip()

# # ------------------------------
# # TTS
# # ------------------------------

# def text_to_speech(text, lang, voice_id=None):
#     output = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3").name

#     default_voice = {
#         "en": "21m00Tcm4TlvDq8ikWAM",
#         "hi": "TxGEqnHWrfWFTfGW9XjX",
#     }.get(lang, "21m00Tcm4TlvDq8ikWAM")

#     voice = voice_id or default_voice

#     audio = elevenlabs_client.text_to_speech.convert(
#         text=text,
#         voice_id=voice,
#         model_id="eleven_multilingual_v2"
#     )

#     with open(output, "wb") as f:
#         for chunk in audio:
#             f.write(chunk)

#     return output

# def list_all_voices():
#     voices = elevenlabs_client.voices.get()
#     print("=== All Available Voices ===")
#     for v in voices:
#         name = getattr(v, "name", "<no name>")
#         vid  = getattr(v, "voice_id", "<no voice_id>")
#         print(f"Name: {name}   |   Voice ID: {vid}")
#     print("============================")

# # ------------------------------
# # MAIN PIPELINE (YOUR RULES)
# # ------------------------------

# def process_file(
#     path,
#     lang="en",
#     reuse_saved_voice=False,
#     clone_voice_from_audio=False
# ):
#     DEFAULT_VOICE_KEY = f"default_news_voice_{lang}"
#     audio = None
#     lists = list_all_voices()
#     print(896,lists)

#     if is_video(path):
#         audio = extract_audio(path)
#         transcript = transcribe_audio(audio)
#         visuals = analyze_frames_batch(extract_key_frames(path))
#         content = f"VISUAL:\n{visuals}\n\nAUDIO:\n{transcript}"

#     elif is_audio(path):
#         audio = path
#         content = transcribe_audio(path)

#     elif is_image(path):
#         content = analyze_frames_batch([path])

#     else:
#         return {"error": "Unsupported file"}

#     script = generate_news_script(content, lang)

#     voice_id = None

#     # 1️⃣ Explicit reuse
#     # if reuse_saved_voice:
#     #     voice_id = get_saved_voice(DEFAULT_VOICE_KEY)
#     # if reuse_saved_voice:
#     #     voice_id = get_saved_voice(DEFAULT_VOICE_KEY)

#     #     if not voice_id:
#     #         return {
#     #             "error": (
#     #                 "You selected 'Reuse saved voice' but no voice has been saved yet. "
#     #                 "Please clone a voice at least once."
#     #             )
#     #         }
#     if reuse_saved_voice:
#         voice_id = get_saved_voice(DEFAULT_VOICE_KEY)

#         if not voice_id:
#             voice_id = None  # fallback to default


#     # 2️⃣ Explicit clone
#     elif clone_voice_from_audio and audio:
#         voice_id = clone_and_save_voice(audio, DEFAULT_VOICE_KEY)

#     audio_out = text_to_speech(script, lang, voice_id)

#     return {
#         "news_script": script,
#         "audio": audio_out,
#     }


















import os
import json
import tempfile
import subprocess
import time
import base64
from dotenv import load_dotenv
from openai import OpenAI, RateLimitError
from elevenlabs.client import ElevenLabs

load_dotenv()

# ------------------------------
# Setup
# ------------------------------

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")

assert OPENAI_API_KEY, "OPENAI_API_KEY not found in .env"
assert ELEVENLABS_API_KEY, "ELEVENLABS_API_KEY not found in .env"

openai_client = OpenAI(api_key=OPENAI_API_KEY)
elevenlabs_client = ElevenLabs(api_key=ELEVENLABS_API_KEY)

VOICE_STORE = "saved_voices.json"

LANGUAGE_MAP = {
    "en": "English",
    "hi": "Hindi",
    "te": "Telugu",
    "kn": "Kannada",
}

# ------------------------------
# Rate-limit safe retry
# ------------------------------

def openai_retry(call, retries=5, delay=0.5):
    for i in range(retries):
        try:
            return call()
        except RateLimitError:
            if i == retries - 1:
                raise
            time.sleep(delay * (2 ** i))

# ------------------------------
# Voice persistence
# ------------------------------

def load_voice_store():
    if not os.path.exists(VOICE_STORE):
        return {}
    with open(VOICE_STORE, "r", encoding="utf-8") as f:
        return json.load(f)

def save_voice_store(store):
    with open(VOICE_STORE, "w", encoding="utf-8") as f:
        json.dump(store, f, indent=2)

def get_saved_voice(voice_key):
    store = load_voice_store()
    return store.get(voice_key)

def clone_and_save_voice(audio_path, voice_key):
    store = load_voice_store()
    with open(audio_path, "rb") as f:
        voice = elevenlabs_client.voices.ivc.create(
            name=voice_key,
            files=[f]
        )
    store[voice_key] = voice.voice_id
    save_voice_store(store)
    return voice.voice_id

# ------------------------------
# File helpers
# ------------------------------

def is_video(p): 
    return p.lower().endswith((".mp4", ".mkv", ".mov", ".avi", ".webm", ".mpeg", ".mpg"))

def is_audio(p): 
    return p.lower().endswith((".wav", ".mp3", ".m4a", ".ogg", ".flac", ".aac"))

def is_image(p): 
    return p.lower().endswith((".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"))

# ------------------------------
# FFmpeg helpers
# ------------------------------

def video_has_audio(video_path):
    try:
        result = subprocess.run(
            ["ffprobe", "-i", video_path, "-show_streams", "-select_streams", "a", "-loglevel", "error"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        return bool(result.stdout.strip())
    except Exception:
        return False

def extract_audio(video_path):
    if not video_has_audio(video_path):
        return None

    audio_path = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
    try:
        subprocess.run(
            [
                "ffmpeg", "-y", "-loglevel", "error",
                "-i", video_path,
                "-vn", "-ac", "1", "-ar", "16000",
                audio_path
            ],
            check=True
        )
        return audio_path
    except Exception as e:
        print(f"Error extracting audio: {e}")
        return None

def extract_key_frames(video_path, every_n_seconds=2):
    frame_dir = tempfile.mkdtemp()
    try:
        subprocess.run(
            [
                "ffmpeg", "-y", "-loglevel", "error",
                "-i", video_path,
                "-vf", f"fps=1/{every_n_seconds},scale=640:-1",
                f"{frame_dir}/frame_%03d.jpg"
            ],
            check=True,
        )
    except Exception as e:
        print(f"Error extracting frames: {e}")
        return []
    
    frames = []
    for f in sorted(os.listdir(frame_dir)):
        if f.endswith(".jpg"):
            frames.append(os.path.join(frame_dir, f))
    
    # Limit to 5 frames to avoid API limits
    return frames[:5]

# ------------------------------
# STT (Speech to Text)
# ------------------------------

def transcribe_audio(audio_path):
    if not audio_path or not os.path.exists(audio_path):
        return "[No audible sound detected.]"

    def call():
        with open(audio_path, "rb") as f:
            return openai_client.audio.transcriptions.create(
                file=f,
                model="whisper-1",
                response_format="text"
            )

    try:
        return openai_retry(call).strip()
    except Exception as e:
        print(f"Transcription error: {e}")
        return "[Error transcribing audio.]"

# ------------------------------
# Vision Analysis
# ------------------------------

def analyze_frames_batch(frames):
    if not frames:
        return "[No images to analyze.]"
    
    content = [{"type": "text", "text": "Describe what is visible in each image in detail."}]

    for frame in frames:
        try:
            with open(frame, "rb") as f:
                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": "data:image/jpeg;base64," +
                               base64.b64encode(f.read()).decode()
                    }
                })
        except Exception as e:
            print(f"Error reading frame {frame}: {e}")
            continue

    def call():
        return openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a news analyst. Describe what you see in these images objectively."},
                {"role": "user", "content": content},
            ],
            max_tokens=500,
            temperature=0.2,
        )

    try:
        return openai_retry(call).choices[0].message.content.strip()
    except Exception as e:
        print(f"Vision analysis error: {e}")
        return "[Error analyzing images.]"

# ------------------------------
# News script generation
# ------------------------------

def generate_news_script(text, lang="en"):
    language = LANGUAGE_MAP.get(lang, "English")

    def call():
        return openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        f"You are a professional news anchor. Write a concise TV news report in {language}. "
                        "Structure it with: HEADLINE (one line), LEAD (brief summary), BODY (detailed report), "
                        "and ENDING (concluding remark). Keep it engaging and factual."
                    ),
                },
                {"role": "user", "content": text},
            ],
            max_tokens=600,
            temperature=0.3,
        )

    try:
        return openai_retry(call).choices[0].message.content.strip()
    except Exception as e:
        print(f"Script generation error: {e}")
        return f"Error generating news script in {language}."

# ------------------------------
# TTS (Text to Speech)
# ------------------------------

def text_to_speech(text, lang="en", voice_id=None):
    output = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3").name

    # Default voices for different languages
    default_voices = {
        "en": "21m00Tcm4TlvDq8ikWAM",  # Rachel (English)
        "hi": "TxGEqnHWrfWFTfGW9XjX",   # Hindi voice
        "te": "TxGEqnHWrfWFTfGW9XjX",   # Telugu (using same as Hindi for now)
        "kn": "TxGEqnHWrfWFTfGW9XjX",   # Kannada (using same as Hindi for now)
    }

    voice = voice_id or default_voices.get(lang, "21m00Tcm4TlvDq8ikWAM")

    try:
        audio = elevenlabs_client.text_to_speech.convert(
            text=text,
            voice_id=voice,
            model_id="eleven_multilingual_v2",
            output_format="mp3_44100_128"
        )

        with open(output, "wb") as f:
            for chunk in audio:
                if chunk:
                    f.write(chunk)

        return output
    except Exception as e:
        print(f"TTS error: {e}")
        # Fallback: create empty audio file
        with open(output, "wb") as f:
            pass
        return output

# ------------------------------
# Main Processing Pipeline
# ------------------------------

def process_file(
    file_path,
    lang="en",
    reuse_saved_voice=False,
    clone_voice_from_audio=False
):
    """
    Process media file and generate news audio.
    
    Args:
        file_path: Path to media file
        lang: Language code ('en', 'hi', 'te', 'kn')
        reuse_saved_voice: Whether to use previously saved voice
        clone_voice_from_audio: Whether to clone voice from input audio
    
    Returns:
        Dictionary with news_script and audio_path
    """
    
    if not os.path.exists(file_path):
        return {"error": f"File not found: {file_path}"}
    
    DEFAULT_VOICE_KEY = f"default_news_voice_{lang}"
    audio = None
    content = ""
    
    print(f"Processing file: {file_path}")
    print(f"File type - Video: {is_video(file_path)}, Audio: {is_audio(file_path)}, Image: {is_image(file_path)}")
    
    # Process based on file type
    if is_video(file_path):
        print("Processing video file...")
        # Extract audio if exists
        audio = extract_audio(file_path)
        if audio:
            print(f"Extracted audio to: {audio}")
            transcript = transcribe_audio(audio)
        else:
            transcript = "[No audio track in video.]"
        
        # Extract and analyze frames
        frames = extract_key_frames(file_path)
        print(f"Extracted {len(frames)} frames")
        visuals = analyze_frames_batch(frames)
        
        content = f"VIDEO ANALYSIS:\nVisual Content: {visuals}\n\nAudio Transcript: {transcript}"
        
    elif is_audio(file_path):
        print("Processing audio file...")
        audio = file_path
        transcript = transcribe_audio(file_path)
        content = f"AUDIO TRANSCRIPT:\n{transcript}"
        
    elif is_image(file_path):
        print("Processing image file...")
        visuals = analyze_frames_batch([file_path])
        content = f"IMAGE ANALYSIS:\n{visuals}"
        
    else:
        return {"error": "Unsupported file format. Please provide video, audio, or image."}
    
    print("Generating news script...")
    script = generate_news_script(content, lang)
    
    voice_id = None
    
    # Handle voice selection
    if reuse_saved_voice:
        voice_id = get_saved_voice(DEFAULT_VOICE_KEY)
        if voice_id:
            print(f"Using saved voice: {voice_id}")
        else:
            print("No saved voice found, using default")
    
    elif clone_voice_from_audio and audio:
        print("Cloning voice from audio...")
        voice_id = clone_and_save_voice(audio, DEFAULT_VOICE_KEY)
        print(f"Voice cloned and saved with ID: {voice_id}")
    
    print("Generating speech...")
    audio_out = text_to_speech(script, lang, voice_id)
    
    # Cleanup temporary files
    if audio and audio != file_path and os.path.exists(audio):
        try:
            os.remove(audio)
        except:
            pass
    
    return {
        "success": True,
        "news_script": script,
        "audio_path": audio_out,
        "content_summary": content[:200] + "..." if len(content) > 200 else content
    }