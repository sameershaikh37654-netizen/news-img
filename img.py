import os
import json
import base64
import mimetypes
import subprocess
from datetime import datetime
from typing import Literal, Any, Dict, Optional, List
import tempfile

import streamlit as st
from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError
from openai import OpenAI


# -----------------------------
# 1) Schema: Universal News Script JSON
# -----------------------------
class NewsScript(BaseModel):
    headline: str = Field(..., min_length=5)
    location_date: str = Field(..., description="Format: '<Location> | <DD Mon YYYY>'")
    what_happened: str
    why_it_matters: str
    who_is_affected: str
    next_steps_official_response: str

    language: str = "en"
    confidence: float = Field(0.65, ge=0.0, le=1.0)
    safety_flags: list[str] = Field(default_factory=list)
    notes_for_editor: str = ""
    source_type: Literal["text", "image", "audio", "video"] = "text"


NEWS_JSON_SCHEMA: Dict[str, Any] = {
    "name": "news_script",
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "headline": {"type": "string"},
            "location_date": {"type": "string"},
            "what_happened": {"type": "string"},
            "why_it_matters": {"type": "string"},
            "who_is_affected": {"type": "string"},
            "next_steps_official_response": {"type": "string"},
            "language": {"type": "string"},
            "confidence": {"type": "number"},
            "safety_flags": {"type": "array", "items": {"type": "string"}},
            "notes_for_editor": {"type": "string"},
            "source_type": {"type": "string", "enum": ["text", "image", "audio", "video"]},
        },
        "required": [
            "headline",
            "location_date",
            "what_happened",
            "why_it_matters",
            "who_is_affected",
            "next_steps_official_response",
            "language",
            "confidence",
            "safety_flags",
            "notes_for_editor",
            "source_type",
        ],
    },
    "strict": True,
}


# -----------------------------
# 2) TV NEWS COMMAND TYPES System
# -----------------------------
TV_COMMAND_TYPES = {
    "BREAKING NEWS": {
        "icon": "🚨",
        "color": "#ff4444",
        "description": "Urgent breaking news report",
        "tv_style": "BREAKING NEWS BANNER",
        "voice_tone": "Urgent, immediate, dramatic",
        "instructions": [
            "START with 'Breaking News' banner",
            "Use dramatic, urgent language",
            "Keep points short and impactful",
            "Focus on WHAT just happened",
            "Emphasize time sensitivity (JUST IN, MOMENTS AGO)",
            "End with 'Stay tuned for updates'"
        ]
    },
    "LIVE REPORT": {
        "icon": "📡",
        "color": "#ffaa00",
        "description": "Live from location reporting",
        "tv_style": "LIVE TAG + LOCATION",
        "voice_tone": "Present tense, descriptive, energetic",
        "instructions": [
            "START with 'Live from [Location]'",
            "Use present tense throughout",
            "Describe what you're SEEING right now",
            "Include ambient sounds/atmosphere",
            "Mention crowd reactions if visible",
            "End with 'Reporting live from [Location]'"
        ]
    },
    "PRIME TIME": {
        "icon": "🌟",
        "color": "#ffcc00",
        "description": "Prime time detailed analysis",
        "tv_style": "PRIME TIME SPECIAL REPORT",
        "voice_tone": "Authoritative, analytical, in-depth",
        "instructions": [
            "Start with 'This Prime Time Special Report'",
            "Provide background and context",
            "Include expert analysis",
            "Use graphics-ready bullet points",
            "Add 'Why this matters' section",
            "End with 'What to expect next'"
        ]
    },
    "HEADLINE NEWS": {
        "icon": "📰",
        "color": "#44aaff",
        "description": "Top stories bulletin",
        "tv_style": "HEADLINE NEWS BULLETIN",
        "voice_tone": "Clear, concise, professional",
        "instructions": [
            "Start with 'Top Stories This Hour'",
            "3-5 key headlines only",
            "Each headline: 10-15 words max",
            "Include time/location for each",
            "Use 'Also in the news' for additional",
            "End with 'More details after these messages'"
        ]
    },
    "SPORTS NEWS": {
        "icon": "⚽",
        "color": "#00aa44",
        "description": "Sports highlights and scores",
        "tv_style": "SPORTS CENTER",
        "voice_tone": "Energetic, exciting, competitive",
        "instructions": [
            "Start with 'Sports Update' or 'Game Day'",
            "Lead with main result/score",
            "Include key moments/highlights",
            "Use sports terminology appropriately",
            "Mention records/achievements",
            "End with upcoming matches/events"
        ]
    },
    "WEATHER REPORT": {
        "icon": "🌧️",
        "color": "#8844ff",
        "description": "Weather forecast and alerts",
        "tv_style": "WEATHER ALERT",
        "voice_tone": "Clear, informative, sometimes urgent",
        "instructions": [
            "Start with 'Weather Update' or 'Weather Alert'",
            "Current conditions first",
            "Today's forecast next",
            "Warning/alerts if any",
            "Weekend outlook",
            "End with 'Stay prepared/Stay safe'"
        ]
    },
    "BUSINESS NEWS": {
        "icon": "📈",
        "color": "#00cccc",
        "description": "Financial markets and economy",
        "tv_style": "MARKET UPDATE",
        "voice_tone": "Professional, factual, numbers-focused",
        "instructions": [
            "Start with 'Market Update' or 'Business Brief'",
            "Lead with major indices/numbers",
            "Key company news",
            "Economic indicators",
            "Expert analysis/commentary",
            "End with 'Tomorrow's outlook'"
        ]
    },
    "ENTERTAINMENT NEWS": {
        "icon": "🎬",
        "color": "#ff66aa",
        "description": "Celebrity, movies, TV shows",
        "tv_style": "ENTERTAINMENT TONIGHT",
        "voice_tone": "Light, engaging, gossipy (but professional)",
        "instructions": [
            "Start with 'Entertainment News'",
            "Lead with biggest celebrity story",
            "Movie/TV show updates",
            "Award show news",
            "Social media trends",
            "End with 'Coming soon/premieres'"
        ]
    },
    "HEALTH NEWS": {
        "icon": "🏥",
        "color": "#ff6666",
        "description": "Medical updates and health alerts",
        "tv_style": "HEALTH WATCH",
        "voice_tone": "Caring, informative, reassuring",
        "instructions": [
            "Start with 'Health Watch' or 'Medical Update'",
            "Lead with important health news",
            "Doctor/expert advice",
            "Prevention tips",
            "Hospital/medical facility updates",
            "End with 'Stay healthy' reminder"
        ]
    },
    "CRIME REPORT": {
        "icon": "🚔",
        "color": "#333333",
        "description": "Police, crime, investigations",
        "tv_style": "CRIME WATCH",
        "voice_tone": "Serious, factual, avoid sensationalism",
        "instructions": [
            "Start with 'Crime Report' or 'Police Update'",
            "Stick to confirmed facts only",
            "Avoid graphic details",
            "Police statements/quotes",
            "Community safety advice",
            "End with 'Anyone with information...'"
        ]
    },
    "POLITICAL NEWS": {
        "icon": "🏛️",
        "color": "#6666ff",
        "description": "Government, elections, policy",
        "tv_style": "POLITICAL BRIEFING",
        "voice_tone": "Neutral, balanced, authoritative",
        "instructions": [
            "Start with 'Political Update'",
            "Present multiple perspectives",
            "Include official statements",
            "Policy implications",
            "Election updates if applicable",
            "End with 'Developments to watch'"
        ]
    },
    "TECH NEWS": {
        "icon": "💻",
        "color": "#00aaff",
        "description": "Technology, gadgets, internet",
        "tv_style": "TECH UPDATE",
        "voice_tone": "Futuristic, innovative, explanatory",
        "instructions": [
            "Start with 'Tech News' or 'Digital Update'",
            "Lead with major tech announcement",
            "Explain in simple terms",
            "Impact on users",
            "Future implications",
            "End with 'What's next in tech'"
        ]
    }
}


def get_tv_command_instructions(selected_commands: List[str]) -> str:
    """Generate TV-style instructions based on selected command types"""
    if not selected_commands:
        return ""
    
    instructions = ["\n**📺 TV NEWS COMMAND INSTRUCTIONS:**"]
    
    for cmd in selected_commands:
        if cmd in TV_COMMAND_TYPES:
            info = TV_COMMAND_TYPES[cmd]
            instructions.append(f"\n{info['icon']} **{cmd}** ({info['tv_style']}):")
            instructions.append(f"  **Voice Tone:** {info['voice_tone']}")
            for inst in info["instructions"]:
                instructions.append(f"  • {inst}")
    
    return "\n".join(instructions)


# -----------------------------
# 3) Auto Model Selection
# -----------------------------
def get_optimal_model(
    source_type: str,
    selected_commands: List[str],
    file_size_mb: float = 0
) -> str:
    """
    Automatically select the best model based on input type and requirements.
    """
    
    # Complex scenarios that need gpt-4o
    complex_scenarios = [
        source_type in ["image", "video"],
        "CRIME REPORT" in selected_commands,
        "BREAKING NEWS" in selected_commands,
        file_size_mb > 50,
        len(selected_commands) >= 2,
    ]
    
    if any(complex_scenarios):
        return "gpt-4o"
    else:
        return "gpt-4o-mini"


# -----------------------------
# 4) Helpers
# -----------------------------
def ensure_dirs():
    os.makedirs("output", exist_ok=True)
    os.makedirs("temp", exist_ok=True)


def today_str():
    return datetime.now().strftime("%d %b %Y")


def file_to_data_url(file_path: str) -> str:
    mime, _ = mimetypes.guess_type(file_path)
    if not mime:
        mime = "application/octet-stream"
    with open(file_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime};base64,{b64}"


def save_upload_to_temp(uploaded_file) -> str:
    ensure_dirs()
    ext = os.path.splitext(uploaded_file.name)[1] or ""
    path = os.path.join("temp", f"upload_{int(datetime.now().timestamp())}{ext}")
    with open(path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return path


def extract_audio_from_video(video_path: str, out_wav_path: str) -> None:
    """Requires ffmpeg installed on your system."""
    try:
        cmd = [
            "ffmpeg",
            "-y",
            "-i", video_path,
            "-vn",
            "-acodec", "pcm_s16le",
            "-ar", "16000",
            "-ac", "1",
            "-loglevel", "error",
            out_wav_path
        ]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=60)
        if result.returncode != 0:
            raise RuntimeError(f"FFmpeg failed: {result.stderr}")
    except subprocess.TimeoutExpired:
        raise RuntimeError("FFmpeg timeout - video might be too long or corrupted")
    except FileNotFoundError:
        raise RuntimeError("FFmpeg not found. Please install ffmpeg on your system.")


def make_client(api_key: str) -> OpenAI:
    return OpenAI(api_key=api_key)


def transcribe_audio(client: OpenAI, audio_path: str, language_hint: Optional[str] = None) -> str:
    try:
        with open(audio_path, "rb") as f:
            resp = client.audio.transcriptions.create(
                model="whisper-1",
                file=f,
                language=language_hint,
                response_format="text"
            )
        return resp
    except Exception as e:
        raise RuntimeError(f"Transcription failed: {str(e)}")


def analyze_image_with_gpt4_vision(client: OpenAI, image_data_url: str, prompt: str) -> str:
    """Analyze image using GPT-4 Vision API"""
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": image_data_url,
                                "detail": "high"
                            }
                        }
                    ]
                }
            ],
            max_tokens=1000,
            temperature=0.2
        )
        return response.choices[0].message.content
    except Exception as e:
        raise RuntimeError(f"Image analysis failed: {str(e)}")


def extract_frames_from_video(video_path: str, num_frames: int = 3) -> List[str]:
    """Extract key frames from video for analysis"""
    ensure_dirs()
    
    # Get video duration
    cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        video_path
    ]
    
    try:
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        duration = float(result.stdout.strip())
        
        # Extract frames at intervals
        frame_paths = []
        for i in range(num_frames):
            timestamp = (duration / (num_frames + 1)) * (i + 1)
            frame_path = os.path.join("temp", f"frame_{int(datetime.now().timestamp())}_{i}.jpg")
            
            cmd = [
                "ffmpeg",
                "-y",
                "-ss", str(timestamp),
                "-i", video_path,
                "-vframes", "1",
                "-q:v", "2",
                "-loglevel", "error",
                frame_path
            ]
            
            subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if os.path.exists(frame_path):
                frame_paths.append(frame_path)
        
        return frame_paths
    except Exception as e:
        raise RuntimeError(f"Failed to extract video frames: {str(e)}")


def analyze_video_with_gpt4_vision(client: OpenAI, video_path: str) -> str:
    """Analyze video by extracting and analyzing key frames"""
    try:
        # Extract frames
        frame_paths = extract_frames_from_video(video_path, num_frames=2)
        
        if not frame_paths:
            return "Unable to extract frames from video for analysis."
        
        # Convert frames to data URLs
        frame_data_urls = [file_to_data_url(frame_path) for frame_path in frame_paths]
        
        # Prepare content with multiple images
        content = [
            {"type": "text", "text": "Analyze these video frames for TV news reporting. Look for:"},
            {"type": "text", "text": "1. Key events or actions visible"},
            {"type": "text", "text": "2. People, vehicles, or objects involved"},
            {"type": "text", "text": "3. Location clues and environment"},
            {"type": "text", "text": "4. Any text, signs, or important visual elements"},
            {"type": "text", "text": "5. Emotional tone and atmosphere"},
        ]
        
        for i, data_url in enumerate(frame_data_urls):
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": data_url,
                    "detail": "high"
                }
            })
            if i < len(frame_paths) - 1:
                content.append({"type": "text", "text": f"Frame {i+2}:"})
        
        # Call GPT-4 Vision
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": content
                }
            ],
            max_tokens=2000,
            temperature=0.3
        )
        
        # Cleanup temporary frame files
        for frame_path in frame_paths:
            try:
                os.remove(frame_path)
            except:
                pass
        
        return response.choices[0].message.content
        
    except Exception as e:
        raise RuntimeError(f"Video analysis failed: {str(e)}")


def generate_tv_news_script_multilingual(
    client: OpenAI,
    model: str,
    source_type: Literal["text", "image", "audio", "video"],
    command_types: List[str],
    languages: List[str],
    *,
    raw_text: Optional[str] = None,
    image_data_url: Optional[str] = None,
    video_path: Optional[str] = None,
    transcript: Optional[str] = None,
    town: Optional[str] = None,
    incident_time: Optional[str] = None,
) -> Dict[str, str]:
    """
    Generates TV news scripts in multiple languages simultaneously.
    Returns dictionary with language codes as keys and scripts as values.
    """
    
    # Language names mapping
    language_names = {
        "en": "English",
        "hi": "Hindi",
        "te": "Telugu"
    }
    
    # Prepare the input content (same for all languages)
    user_content = []
    
    # Add location and time hints if provided
    location_info = ""
    if town:
        location_info += f"Location: {town}\n"
    if incident_time:
        location_info += f"Time: {incident_time}\n"
    
    if location_info:
        user_content.append({"type": "text", "text": location_info})
    
    user_content.append({"type": "text", "text": f"Broadcast Date: {today_str()}\n\n"})

    # Add the main content based on source type
    if source_type == "text":
        user_content.append({"type": "text", "text": f"RAW TEXT:\n{raw_text or ''}"})
    
    elif source_type == "image":
        # First analyze the image
        analysis_prompt = """Analyze this image for TV news reporting. Provide details for:
1. What is happening in the scene?
2. Who is involved (approximate number, demographics)?
3. Key objects, vehicles, infrastructure visible?
4. Any text, signs, logos visible?
5. Location setting (indoor/outdoor, urban/rural)?
6. Time of day/weather conditions?
7. Visible emotions or actions?
8. Potential news story/angle?

Provide detailed analysis for TV news script."""
        
        image_analysis = analyze_image_with_gpt4_vision(client, image_data_url, analysis_prompt)
        user_content.append({"type": "text", "text": f"IMAGE ANALYSIS FOR TV NEWS:\n{image_analysis}\n\n"})
        user_content.append({"type": "image_url", "image_url": {"url": image_data_url, "detail": "high"}})
    
    elif source_type == "video":
        # Analyze video frames
        video_analysis = analyze_video_with_gpt4_vision(client, video_path)
        user_content.append({"type": "text", "text": f"VIDEO ANALYSIS FOR TV NEWS:\n{video_analysis}\n\n"})
        
        if transcript:
            user_content.append({"type": "text", "text": f"AUDIO TRANSCRIPT:\n{transcript}\n\n"})
    
    elif source_type == "audio":
        user_content.append({"type": "text", "text": f"TRANSCRIPT:\n{transcript or ''}"})
    
    else:
        raise ValueError("Invalid source_type")

    # Get command-specific instructions
    command_instructions = get_tv_command_instructions(command_types)
    
    # Determine the main command type
    main_command = command_types[0] if command_types else "HEADLINE NEWS"
    
    # Build TV news script prompt based on command type
    if main_command == "BREAKING NEWS":
        tv_format = """
📺 **TV NEWS FORMAT - BREAKING NEWS:**
---
[BREAKING NEWS BANNER APPEARS]
[URGENT NEWS THEME MUSIC]

ANCHOR: (Urgent tone)
"Breaking News! We're interrupting our regular programming with this important update..."

SCRIPT FORMAT:
1. BREAKING NEWS ALERT: [One line dramatic headline]
2. WHAT WE KNOW: [3-4 bullet points of confirmed facts]
3. LOCATION & TIME: [Where and when this happened]
4. OFFICIAL RESPONSE: [Police/Govt statements if available]
5. WHAT'S NEXT: [Ongoing developments]
6. STAY TUNED: [Update promise]

[VISUALS: Show location maps, relevant images, lower third with "BREAKING"]
"""
    
    elif main_command == "LIVE REPORT":
        tv_format = """
📺 **TV NEWS FORMAT - LIVE REPORT:**
---
[LIVE TAG ON SCREEN: "LIVE FROM [LOCATION]"]
[REPORTER STANDUP WITH BACKGROUND ACTIVITY]

REPORTER: (Present tense, energetic)
"Good [morning/afternoon/evening], I'm [Reporter Name] reporting live from [Location] where..."

SCRIPT FORMAT:
1. LIVE OPENING: [Set the scene - what you're seeing]
2. CURRENT SITUATION: [3-4 bullet points of what's happening NOW]
3. EYEWITNESS ACCOUNTS: [If available]
4. BACKGROUND: [How this started]
5. OFFICIAL PRESENCE: [Police/emergency services on scene]
6. LIVE SIGN-OFF: [What to expect next, back to studio]

[VISUALS: Show live footage, reporter standup, location graphics]
"""
    
    elif main_command == "PRIME TIME":
        tv_format = """
📺 **TV NEWS FORMAT - PRIME TIME SPECIAL:**
---
[PRIME TIME SPECIAL GRAPHIC]
[THEME MUSIC]

ANCHOR: (Authoritative tone)
"This Prime Time Special Report brings you an in-depth look at..."

SCRIPT FORMAT:
1. SPECIAL REPORT INTRO: [Why this story matters tonight]
2. BACKGROUND & CONTEXT: [History leading up to this]
3. KEY DEVELOPMENTS: [3-4 bullet points of main events]
4. EXPERT ANALYSIS: [Insights from specialists]
5. IMPACT: [Who is affected and how]
6. WHAT'S NEXT: [Future implications]
7. PRIME TIME CLOSE: [Summary and call to action]

[VISUALS: Graphics package, expert interviews, historical footage]
"""
    
    elif main_command == "HEADLINE NEWS":
        tv_format = """
📺 **TV NEWS FORMAT - HEADLINE NEWS BULLETIN:**
---
[HEADLINE NEWS GRAPHIC]
[UPBEAT NEWS THEME]

ANCHOR: (Clear, professional)
"Top stories this hour..."

SCRIPT FORMAT:
1. HEADLINE 1: [Main story - 15-20 words]
2. HEADLINE 2: [Second story - 15-20 words]
3. HEADLINE 3: [Third story - 15-20 words]
4. ALSO IN THE NEWS: [Brief mentions of 2-3 more stories]
5. WEATHER CHECK: [Brief weather mention if relevant]
6. COMING UP: [Tease next segment]

[VISUALS: Headline graphics, story images, lower third updates]
"""
    
    elif main_command == "SPORTS NEWS":
        tv_format = """
📺 **TV NEWS FORMAT - SPORTS UPDATE:**
---
[SPORTS GRAPHIC]
[ENERGETIC SPORTS THEME]

SPORTS ANCHOR: (Energetic)
"In sports today..."

SCRIPT FORMAT:
1. TOP SPORTS STORY: [Main game/result]
2. SCOREBOARD: [Key scores/results]
3. HIGHLIGHTS: [3-4 bullet points of key moments]
4. PLAYER PERFORMANCE: [Standout players]
5. UPCOMING EVENTS: [Next games/matches]
6. SPORTS SIGN-OFF: [Looking ahead]

[VISUALS: Game footage, score graphics, player close-ups]
"""
    
    elif main_command == "WEATHER REPORT":
        tv_format = """
📺 **TV NEWS FORMAT - WEATHER UPDATE:**
---
[WEATHER GRAPHIC]
[CALM WEATHER THEME]

WEATHER ANCHOR: (Clear, informative)
"Here's your latest weather update..."

SCRIPT FORMAT:
1. CURRENT CONDITIONS: [Temperature, conditions now]
2. TODAY'S FORECAST: [High/low, precipitation chance]
3. ALERTS/WARNINGS: [If any severe weather]
4. EXTENDED FORECAST: [Next 3 days]
5. WEEKEND OUTLOOK: [Weekend weather]
6. WEATHER TIP: [Preparation/safety advice]

[VISUALS: Radar maps, temperature graphics, forecast icons]
"""
    
    elif main_command == "BUSINESS NEWS":
        tv_format = """
📺 **TV NEWS FORMAT - MARKET UPDATE:**
---
[BUSINESS GRAPHIC]
[PROFESSIONAL THEME]

BUSINESS ANCHOR: (Professional, factual)
"In business news today..."

SCRIPT FORMAT:
1. MARKET NUMBERS: [Key indices up/down]
2. TOP BUSINESS STORY: [Major company news]
3. ECONOMIC INDICATORS: [Key economic data]
4. EXPERT ANALYSIS: [Market analyst insights]
5. CONSUMER IMPACT: [How this affects viewers]
6. TOMORROW'S WATCH: [What to watch for]

[VISUALS: Stock ticker, charts, CEO interviews]
"""
    
    else:  # Default format for other commands
        tv_format = """
📺 **TV NEWS FORMAT - STANDARD REPORT:**
---
[NEWS GRAPHIC]
[NEWS THEME MUSIC]

ANCHOR: (Professional tone)
"In our top story today..."

SCRIPT FORMAT:
1. STORY INTRO: [What happened]
2. KEY DETAILS: [3-4 bullet points of important facts]
3. WHO IS INVOLVED: [People/parties involved]
4. OFFICIAL STATEMENTS: [Quotes from authorities]
5. COMMUNITY IMPACT: [How this affects people]
6. WHAT'S NEXT: [Next steps/developments]

[VISUALS: Story images, location maps, relevant graphics]
"""

    # Create system prompt for multilingual generation
    system_prompt = f"""
You are a professional TV news anchor and scriptwriter for a major television news channel.

YOUR TASK: Create broadcast-ready TV news scripts in MULTIPLE LANGUAGES simultaneously.

{command_instructions}

FORMAT REQUIREMENTS:
{tv_format}

LANGUAGE INSTRUCTIONS:
For each language, adapt the script to:
1. Use appropriate cultural references and idioms
2. Follow the language's natural speech patterns
3. Use proper terminology for TV broadcasting in that language
4. Maintain the same professional tone across all languages

FORMAT FOR OUTPUT:
Generate the script in the following structure for EACH language:

[LANGUAGE: English]
[SCRIPT START]
[Complete TV news script in English with proper formatting]
[SCRIPT END]

[LANGUAGE: Hindi]
[SCRIPT START]
[Complete TV news script in Hindi with proper formatting]
[SCRIPT END]

[LANGUAGE: Telugu]
[SCRIPT START]
[Complete TV news script in Telugu with proper formatting]
[SCRIPT END]

ADDITIONAL RULES FOR ALL SCRIPTS:
- Use natural, conversational language for TV
- Include visual cues in [BRACKETS] for producers
- Keep bullet points concise (15-25 words each)
- Use present tense for LIVE reports
- Include time references (TODAY, THIS MORNING, etc.)
- Add location details prominently
- End with proper sign-off
- Include lower third suggestions [LOWER THIRD: text]
- Suggest appropriate music/banners
- DO NOT include markdown formatting in final output
- Script should be ready for teleprompter

GENERATE COMPLETE TV NEWS SCRIPTS READY FOR BROADCAST IN ALL REQUESTED LANGUAGES.
"""

    try:
        # Call OpenAI
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ],
            temperature=0.3,
            max_tokens=4000  # Increased for multiple languages
        )

        output = response.choices[0].message.content
        
        # Parse the output to extract scripts for each language
        scripts = {}
        
        # Split by language markers
        sections = output.split('[LANGUAGE:')
        
        for section in sections[1:]:  # Skip first empty section
            if not section.strip():
                continue
            
            # Extract language name and script
            lines = section.split('\n')
            language_line = lines[0].strip()
            
            # Find the language code
            lang_code = None
            for code, name in language_names.items():
                if name.lower() in language_line.lower():
                    lang_code = code
                    break
            
            if not lang_code:
                continue
            
            # Find script content between markers
            script_content = []
            in_script = False
            
            for line in lines[1:]:
                if '[SCRIPT START]' in line:
                    in_script = True
                    continue
                elif '[SCRIPT END]' in line:
                    break
                elif in_script:
                    script_content.append(line)
            
            if script_content:
                # Format the script
                formatted_script = format_tv_script('\n'.join(script_content), lang_code, main_command)
                scripts[lang_code] = formatted_script
        
        # Ensure we have all requested languages
        for lang in languages:
            if lang not in scripts:
                scripts[lang] = f"Script for {language_names.get(lang, lang)} could not be generated."
        
        return scripts
        
    except Exception as e:
        raise RuntimeError(f"Failed to generate multilingual TV news scripts: {str(e)}")


def format_tv_script(raw_script: str, language: str, command_type: str) -> str:
    """Format TV script with proper headers and cleanup"""
    
    # Language headers
    language_headers = {
        "en": "📺 ENGLISH TV NEWS SCRIPT",
        "hi": "📺 हिंदी टीवी न्यूज़ स्क्रिप्ट",
        "te": "📺 తెలుగు టీవీ న్యూస్ స్క్రిప్ట్"
    }
    
    # Add timestamp and command type header
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    header = f"""
{'='*70}
{language_headers.get(language, '📺 TV NEWS SCRIPT')} | {timestamp}
🎯 COMMAND: {command_type} | LANGUAGE: {language}
{'='*70}

"""
    
    # Basic formatting cleanup
    lines = raw_script.split('\n')
    formatted_lines = []
    
    in_script = False
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # Remove markdown if present
        if line.startswith('#') or line.startswith('**') or line.startswith('*') and not line.startswith('***'):
            line = line.replace('#', '').replace('**', '').replace('*', '')
            line = line.strip()
        
        # Add proper formatting
        if line.upper().startswith('ANCHOR:') or line.upper().startswith('REPORTER:') or \
           line.upper().startswith('SPORTS ANCHOR:') or line.upper().startswith('WEATHER ANCHOR:') or \
           line.upper().startswith('BUSINESS ANCHOR:'):
            formatted_lines.append("")
            formatted_lines.append(line.upper())
        elif any(x in line for x in ['[LOWER THIRD:', '[VISUALS:', '[GRAPHIC:', '[MUSIC:', '[BANNER:']):
            formatted_lines.append(f"  {line}")
        elif line.isdigit() and len(line) <= 2 and in_script:  # Bullet point numbers
            continue  # Skip numeric bullets
        elif line.startswith('•') or line.startswith('-'):
            formatted_lines.append(f"  • {line[1:].strip()}")
            in_script = True
        elif in_script and line and not line.startswith('['):
            # Append to last bullet point
            if formatted_lines and formatted_lines[-1].startswith('  •'):
                formatted_lines[-1] = formatted_lines[-1] + ' ' + line
            else:
                formatted_lines.append(line)
        else:
            formatted_lines.append(line)
    
    # Join with proper spacing
    formatted_output = '\n'.join(formatted_lines)
    
    return header + formatted_output


def save_tv_script_output_multilingual(scripts: Dict[str, str], source_type: str, command_types: List[str]) -> Dict[str, str]:
    """Save TV script outputs to text files for each language."""
    ensure_dirs()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    saved_paths = {}
    
    for lang_code, script in scripts.items():
        # Create filename
        cmd_str = command_types[0] if command_types else "standard"
        out_path = os.path.join("output", f"tv_news_{cmd_str}_{lang_code}_{ts}.txt")
        
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(script)
        
        saved_paths[lang_code] = out_path
    
    return saved_paths


def display_tv_script_multilingual(scripts: Dict[str, str], command_types: List[str], model_used: str):
    """Display TV scripts in multiple languages with tabs"""
    
    # Show model info
    st.markdown(f"### 🤖 Model Used: `{model_used}`")
    
    # Show command type badges
    if command_types:
        st.markdown("### 🎯 TV News Command Applied")
        main_cmd = command_types[0]
        if main_cmd in TV_COMMAND_TYPES:
            info = TV_COMMAND_TYPES[main_cmd]
            st.markdown(f"""
            <div style="
                background-color: {info['color']}20;
                border-left: 4px solid {info['color']};
                padding: 15px;
                border-radius: 5px;
                margin: 10px 0;
            ">
            <h3>{info['icon']} {main_cmd}</h3>
            <p><strong>TV Style:</strong> {info['tv_style']}</p>
            <p><strong>Voice Tone:</strong> {info['voice_tone']}</p>
            <p><strong>Description:</strong> {info['description']}</p>
            </div>
            """, unsafe_allow_html=True)
    
    st.divider()
    
    # Create tabs for each language
    if scripts:
        tab_names = []
        for lang in scripts.keys():
            if lang == "en":
                tab_names.append("🇬🇧 English")
            elif lang == "hi":
                tab_names.append("🇮🇳 Hindi")
            elif lang == "te":
                tab_names.append("🇮🇳 Telugu")
            else:
                tab_names.append(f"🌐 {lang.upper()}")
        
        tabs = st.tabs(tab_names)
        
        for idx, (lang_code, script) in enumerate(scripts.items()):
            with tabs[idx]:
                # Language-specific styling
                if lang_code == "hi":
                    st.markdown("""
                    <style>
                    .hindi-script {
                        font-family: 'Nirmala UI', 'Mangal', 'Arial Unicode MS', sans-serif;
                        font-size: 16px;
                        line-height: 1.6;
                    }
                    </style>
                    """, unsafe_allow_html=True)
                    script_class = "hindi-script"
                elif lang_code == "te":
                    st.markdown("""
                    <style>
                    .telugu-script {
                        font-family: 'Nirmala UI', 'Gautami', 'Arial Unicode MS', sans-serif;
                        font-size: 16px;
                        line-height: 1.6;
                    }
                    </style>
                    """, unsafe_allow_html=True)
                    script_class = "telugu-script"
                else:
                    script_class = "english-script"
                
                # Display script with formatting
                lines = script.split('\n')
                
                for line in lines:
                    line = line.strip()
                    if not line:
                        st.markdown("<br>", unsafe_allow_html=True)
                        continue
                        
                    # Header section
                    if '='*70 in line:
                        st.markdown(f"`{line}`")
                    elif 'TV NEWS SCRIPT' in line or 'न्यूज़ स्क्रिप्ट' in line or 'న్యూస్ స్క్రిప్ట్' in line:
                        st.markdown(f"## {line}")
                    elif 'COMMAND:' in line:
                        st.info(line)
                        
                    # Anchor/Reporter lines
                    elif any(prefix in line.upper() for prefix in ['ANCHOR:', 'REPORTER:', 'SPORTS ANCHOR:', 'WEATHER ANCHOR:', 'BUSINESS ANCHOR:']):
                        st.markdown(f"""
                        <div class="{script_class}" style="
                            background-color: #f0f2f6;
                            padding: 10px;
                            border-radius: 5px;
                            margin: 10px 0;
                            border-left: 4px solid #0066cc;
                            font-family: {'monospace' if lang_code == 'en' else 'inherit'};
                        ">
                        <strong>{line}</strong>
                        </div>
                        """, unsafe_allow_html=True)
                        
                    # Visual/Technical cues
                    elif line.startswith('[') and line.endswith(']'):
                        st.markdown(f"""
                        <div class="{script_class}" style="
                            background-color: #fff3cd;
                            color: #856404;
                            padding: 8px;
                            border-radius: 3px;
                            margin: 5px 0;
                            font-family: monospace;
                            font-size: 0.9em;
                        ">
                        🎬 {line}
                        </div>
                        """, unsafe_allow_html=True)
                        
                    # Bullet points
                    elif line.startswith('•'):
                        st.markdown(f"""
                        <div class="{script_class}" style="
                            margin-left: 20px;
                            margin-bottom: 5px;
                            padding-left: 10px;
                            border-left: 2px solid #ddd;
                        ">
                        {line}
                        </div>
                        """, unsafe_allow_html=True)
                        
                    # Regular text
                    else:
                        st.markdown(f"""
                        <div class="{script_class}">
                        {line}
                        </div>
                        """, unsafe_allow_html=True)


# -----------------------------
# 5) Streamlit UI
# -----------------------------
load_dotenv()
ensure_dirs()

st.set_page_config(
    page_title="TV News Script Generator - Multilingual",
    layout="wide",
    page_icon="📺"
)

st.title("📺 TV News Script Generator - Multilingual")
st.caption("Create Professional TV News Scripts in English, Hindi & Telugu from Any Source")

with st.sidebar:
    st.header("⚙️ TV News Settings")
    
    st.subheader("🔐 OpenAI Configuration")
    env_key = os.getenv("OPENAI_API_KEY", "")
    api_key = st.text_input("API Key", value=env_key, type="password", help="Your OpenAI API key")
    
    # Auto model info
    st.markdown("**🤖 Auto Model Selection**")
    st.info("""
    Model is automatically selected:
    - Videos/Images → gpt-4o (vision)
    - Breaking News → gpt-4o
    - Crime Reports → gpt-4o
    - Large files → gpt-4o
    - Simple text → gpt-4o-mini
    """)
    
    st.divider()
    
    st.subheader("🌐 Language Selection")
    st.markdown("Select languages for TV news scripts:")
    
    languages = st.multiselect(
        "Choose languages:",
        options=["en", "hi", "te"],
        default=["en", "hi", "te"],
        format_func=lambda x: {
            "en": "🇬🇧 English",
            "hi": "🇮🇳 Hindi",
            "te": "🇮🇳 Telugu"
        }[x],
        help="Scripts will be generated in all selected languages"
    )
    
    if not languages:
        st.warning("Please select at least one language")
    
    st.divider()
    
    st.subheader("🎬 TV News Commands")
    st.markdown("Select the type of TV news report:")
    
    selected_commands = st.multiselect(
        "Choose news type (first is main command):",
        options=list(TV_COMMAND_TYPES.keys()),
        format_func=lambda x: f"{TV_COMMAND_TYPES[x]['icon']} {x}",
        help="First selection is the main news type. Others add additional elements.",
        max_selections=3
    )
    
    # Show TV style preview
    if selected_commands:
        main_cmd = selected_commands[0]
        if main_cmd in TV_COMMAND_TYPES:
            info = TV_COMMAND_TYPES[main_cmd]
            st.markdown(f"""
            <div style="
                background-color: {info['color']}20;
                padding: 10px;
                border-radius: 5px;
                margin: 10px 0;
            ">
            <h4>Selected: {info['icon']} {main_cmd}</h4>
            <p><strong>TV Style:</strong> {info['tv_style']}</p>
            <p><strong>Voice Tone:</strong> {info['voice_tone']}</p>
            </div>
            """, unsafe_allow_html=True)
    
    st.divider()
    
    st.subheader("🗺️ News Context")
    town = st.text_input("Location", placeholder="e.g., Downtown City Center", help="Location for news report")
    incident_time = st.text_input("Time Reference", placeholder="e.g., this morning / around 3 PM", help="When it happened")
    
    lang_hint = st.selectbox(
        "Audio Language",
        ["auto", "en", "te", "hi", "kn", "ta", "mr"],
        index=0,
        help="Language hint for audio transcription"
    )
    
    st.divider()
    
    # Quick Command Guide
    st.markdown("### 📋 Quick Command Guide")
    
    cmd_categories = {
        "Urgent": ["BREAKING NEWS", "LIVE REPORT"],
        "Scheduled": ["PRIME TIME", "HEADLINE NEWS"],
        "Specialized": ["SPORTS NEWS", "WEATHER REPORT", "BUSINESS NEWS"],
        "Other": ["ENTERTAINMENT NEWS", "HEALTH NEWS", "CRIME REPORT", "POLITICAL NEWS", "TECH NEWS"]
    }
    
    for category, cmds in cmd_categories.items():
        st.markdown(f"**{category}**")
        for cmd in cmds:
            if cmd in TV_COMMAND_TYPES:
                info = TV_COMMAND_TYPES[cmd]
                st.markdown(f"{info['icon']} {cmd}")

# Main interface
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📥 1) News Source Input")
    
    input_type = st.radio(
        "Select source type:",
        ["text", "image", "audio", "video"],
        horizontal=True,
        captions=["Text/Notes", "Photos", "Audio clips", "Video footage"],
        help="Choose the type of source material"
    )
    
    if input_type == "text":
        st.markdown("**Enter news content:**")
        raw_text = st.text_area(
            "News content",
            height=250,
            placeholder="Paste reporter notes, eyewitness accounts, press releases, or any text content here...",
            label_visibility="collapsed"
        )
        uploaded = None
        
    else:
        file_types = {
            "image": ["jpg", "jpeg", "png", "webp"],
            "audio": ["mp3", "wav", "m4a", "ogg"],
            "video": ["mp4", "mov", "avi", "mkv"]
        }
        
        uploaded = st.file_uploader(
            f"Upload {input_type.upper()} file",
            type=file_types[input_type],
            help=f"Upload {input_type} for TV news analysis"
        )
        
        if uploaded:
            file_size = uploaded.size / (1024 * 1024)  # MB
            if file_size > 100:
                st.warning(f"File size ({file_size:.1f} MB) is large. Processing may take time.")
            
            # Show preview
            if input_type == "image":
                st.image(uploaded, caption="News Image", use_column_width=True)
            elif input_type == "audio":
                st.audio(uploaded, format=f"audio/{uploaded.name.split('.')[-1]}")
            elif input_type == "video":
                st.video(uploaded)
        
        raw_text = ""

with col2:
    st.subheader("🚀 2) Generate TV News Scripts")
    
    # Show language selection preview
    if languages:
        lang_display = []
        for lang in languages:
            if lang == "en":
                lang_display.append("🇬🇧 English")
            elif lang == "hi":
                lang_display.append("🇮🇳 Hindi")
            elif lang == "te":
                lang_display.append("🇮🇳 Telugu")
        
        st.markdown(f"""
        <div style="
            background-color: #e8f4fd;
            padding: 15px;
            border-radius: 10px;
            margin: 10px 0;
            text-align: center;
            border: 2px solid #0066cc;
        ">
        <h4>🌐 Scripts will be generated in:</h4>
        <p style='font-size: 18px;'>{' | '.join(lang_display)}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Show selected command preview
    if selected_commands:
        main_cmd = selected_commands[0]
        if main_cmd in TV_COMMAND_TYPES:
            info = TV_COMMAND_TYPES[main_cmd]
            st.markdown(f"""
            <div style="
                background-color: {info['color']}15;
                padding: 15px;
                border-radius: 10px;
                margin: 10px 0;
                text-align: center;
                border: 2px solid {info['color']};
            ">
            <h3>{info['icon']} {main_cmd}</h3>
            <p style='color: #666;'>{info['tv_style']}</p>
            </div>
            """, unsafe_allow_html=True)
    
    if st.button(
        "📺 GENERATE MULTILINGUAL TV NEWS SCRIPTS",
        use_container_width=True,
        type="primary",
        disabled=not api_key.strip() or not languages
    ):
        if not api_key.strip():
            st.error("Please enter your OpenAI API key in the sidebar.")
            st.stop()
        
        if not languages:
            st.error("Please select at least one language in the sidebar.")
            st.stop()
        
        # Validation
        if input_type == "text" and not raw_text.strip():
            st.error("Please enter some text content.")
            st.stop()
        elif input_type != "text" and uploaded is None:
            st.error(f"Please upload an {input_type} file.")
            st.stop()
        
        client = make_client(api_key.strip())
        
        # Determine file size
        file_size_mb = 0
        if uploaded:
            file_size_mb = uploaded.size / (1024 * 1024)
        
        # Auto-select model
        selected_model = get_optimal_model(input_type, selected_commands, file_size_mb)
        
        # Show model selection info
        st.info(f"🤖 **Auto-selected model:** `{selected_model}`")
        
        # Create progress containers
        progress_bar = st.progress(0)
        status_text = st.empty()
        step_container = st.container()
        
        try:
            # Step 1: Processing input
            status_text.text("🔄 Processing news source...")
            progress_bar.progress(10)
            
            transcript = None
            video_path = None
            image_data_url = None
            
            with step_container:
                with st.expander("🔍 Processing Details", expanded=True):
                    if input_type == "text":
                        st.success("✓ Text content loaded")
                        st.info(f"Character count: {len(raw_text)}")
                        
                    elif input_type == "image":
                        st.success("✓ News image uploaded")
                        img_path = save_upload_to_temp(uploaded)
                        image_data_url = file_to_data_url(img_path)
                        st.info("Image ready for TV news analysis")
                        
                    elif input_type == "audio":
                        st.success("✓ Audio clip uploaded")
                        audio_path = save_upload_to_temp(uploaded)
                        status_text.text("🔄 Transcribing audio...")
                        progress_bar.progress(30)
                        
                        with st.spinner("Transcribing audio for news script..."):
                            lang = None if lang_hint == "auto" else lang_hint
                            transcript = transcribe_audio(client, audio_path, lang)
                        
                        st.success(f"✓ Transcription complete ({len(transcript)} characters)")
                        st.markdown("**Transcript preview:**")
                        st.code(transcript[:500] + ("..." if len(transcript) > 500 else ""), language="text")
                        
                    elif input_type == "video":
                        st.success("✓ Video footage uploaded")
                        video_path = save_upload_to_temp(uploaded)
                        
                        # Extract audio and transcribe
                        status_text.text("🔄 Extracting audio from video...")
                        progress_bar.progress(20)
                        
                        wav_path = os.path.join("temp", f"extracted_{int(datetime.now().timestamp())}.wav")
                        with st.spinner("Extracting audio for transcription..."):
                            extract_audio_from_video(video_path, wav_path)
                        
                        st.success("✓ Audio extracted")
                        
                        status_text.text("🔄 Transcribing video audio...")
                        progress_bar.progress(40)
                        
                        with st.spinner("Transcribing video content..."):
                            lang = None if lang_hint == "auto" else lang_hint
                            transcript = transcribe_audio(client, wav_path, lang)
                        
                        st.success(f"✓ Transcription complete ({len(transcript)} characters)")
                        st.markdown("**Transcript preview:**")
                        st.code(transcript[:500] + ("..." if len(transcript) > 500 else ""), language="text")
            
            # Step 2: Generating TV news scripts in multiple languages
            status_text.text(f"📺 Creating TV news scripts in {len(languages)} languages...")
            progress_bar.progress(70)
            
            with step_container:
                with st.expander("🤖 AI News Analysis", expanded=True):
                    with st.spinner(f"Analyzing content and creating broadcast-ready TV scripts in {len(languages)} languages..."):
                        tv_scripts = generate_tv_news_script_multilingual(
                            client=client,
                            model=selected_model,
                            source_type=input_type,
                            command_types=selected_commands,
                            languages=languages,
                            raw_text=raw_text if input_type == "text" else None,
                            image_data_url=image_data_url if input_type == "image" else None,
                            video_path=video_path if input_type == "video" else None,
                            transcript=transcript if input_type in ["audio", "video"] else None,
                            town=town or None,
                            incident_time=incident_time or None,
                        )
            
            # Step 3: Finalizing
            status_text.text("💾 Saving TV scripts...")
            progress_bar.progress(90)
            
            script_paths = save_tv_script_output_multilingual(tv_scripts, input_type, selected_commands)
            
            status_text.text("✅ TV News Scripts Ready!")
            progress_bar.progress(100)
            
            # Display results
            st.success(f"### 📋 TV News Scripts Generated Successfully in {len(tv_scripts)} Languages!")
            
            # Show formatted TV scripts in tabs
            display_tv_script_multilingual(tv_scripts, selected_commands, selected_model)
            
            st.divider()
            
            # Download section
            st.markdown("### 📥 Download Scripts")
            
            # Create columns for download buttons
            cols = st.columns(len(tv_scripts))
            
            for idx, (lang_code, script) in enumerate(tv_scripts.items()):
                with cols[idx]:
                    # Language labels
                    lang_labels = {
                        "en": "English",
                        "hi": "Hindi",
                        "te": "Telugu"
                    }
                    
                    lang_name = lang_labels.get(lang_code, lang_code.upper())
                    
                    # Download button
                    if lang_code in script_paths:
                        with open(script_paths[lang_code], "rb") as f:
                            st.download_button(
                                label=f"📥 {lang_name} Script",
                                data=f,
                                file_name=os.path.basename(script_paths[lang_code]),
                                mime="text/plain",
                                use_container_width=True
                            )
                    
                    # Copy to clipboard button
                    if st.button(f"📋 Copy {lang_name}", key=f"copy_{lang_code}", use_container_width=True):
                        st.session_state[f"copied_script_{lang_code}"] = script
                        st.success(f"{lang_name} script copied to clipboard!")
            
            # Combined download option
            st.divider()
            st.markdown("### 🌐 Combined Download")
            
            # Create a ZIP file with all scripts
            import zipfile
            from io import BytesIO
            
            zip_buffer = BytesIO()
            with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                for lang_code, script in tv_scripts.items():
                    lang_name = lang_labels.get(lang_code, lang_code.upper())
                    filename = f"tv_news_{lang_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
                    zip_file.writestr(filename, script)
            
            zip_buffer.seek(0)
            
            st.download_button(
                label="📦 Download All Scripts (ZIP)",
                data=zip_buffer,
                file_name=f"tv_news_scripts_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
                mime="application/zip",
                use_container_width=True
            )
            
            # Broadcast tips
            st.info("""
            **📺 Broadcast Ready Features:**
            - Anchor/reporter dialogue formatted in multiple languages
            - Visual cues in [brackets] for producers
            - Lower third suggestions included
            - Music/banner recommendations
            - Teleprompter-friendly formatting
            - Cultural adaptations for each language
            """)
            
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            st.info("""
            **Troubleshooting tips:**
            1. Check your API key is valid
            2. Ensure ffmpeg is installed for video/audio processing
            3. Try smaller files if processing fails
            4. Check your internet connection
            5. For large videos, consider extracting audio separately
            """)

# TV News Format Guide
st.markdown("---")
st.markdown("## 📖 TV News Script Format Guide")

col_guide1, col_guide2, col_guide3 = st.columns(3)

with col_guide1:
    st.markdown("""
    ### 🎯 Key TV News Commands
    
    **🚨 BREAKING NEWS**
    - Urgent updates
    - Dramatic tone
    - Breaking banner
    
    **📡 LIVE REPORT**
    - Present tense
    - Location tags
    - Live visuals
    
    **🌟 PRIME TIME**
    - In-depth analysis
    - Expert interviews
    - Special graphics
    """)

with col_guide2:
    st.markdown("""
    ### 🌐 Multilingual Support
    
    **🇬🇧 ENGLISH**
    - Professional broadcast English
    - Standard TV news terminology
    
    **🇮🇳 HINDI**
    - हिंदी समाचार भाषा
    - Cultural context adaptation
    - Regional terminology
    
    **🇮🇳 TELUGU**
    - తెలుగు వార్తా భాష
    - సాంస్కృతిక సందర్భం
    - ప్రాంతీయ పదజాలం
    """)

with col_guide3:
    st.markdown("""
    ### 🎬 Specialized Formats
    
    **⚽ SPORTS NEWS**
    - Scores & highlights
    - Player performances
    - Upcoming events
    
    **🌧️ WEATHER REPORT**
    - Current conditions
    - Forecast
    - Alerts/warnings
    
    **📈 BUSINESS NEWS**
    - Market numbers
    - Company updates
    - Economic impact
    """)

# Footer
st.markdown("---")
st.caption("""
🔹 **TV News Script Generator - Multilingual** v4.0 | 
📺 Professional Broadcast Scripts in English, Hindi & Telugu | 
🎯 12 TV News Commands: BREAKING NEWS, LIVE REPORT, PRIME TIME, HEADLINE NEWS, SPORTS, WEATHER, BUSINESS, ENTERTAINMENT, HEALTH, CRIME, POLITICAL, TECH | 
📁 Supports: Text, Images, Audio, Video | 
🤖 Auto Model Selection: gpt-4o / gpt-4o-mini | 
🌐 Languages: English 🇬🇧, Hindi 🇮🇳, Telugu 🇮🇳
""")

































































# import os
# import json
# import base64
# import mimetypes
# import subprocess
# from datetime import datetime
# from typing import Literal, Any, Dict, Optional, List
# import tempfile

# import streamlit as st
# from dotenv import load_dotenv
# from pydantic import BaseModel, Field, ValidationError
# from openai import OpenAI


# # -----------------------------
# # 1) Schema: Universal News Script JSON
# # -----------------------------
# class NewsScript(BaseModel):
#     headline: str = Field(..., min_length=5)
#     location_date: str = Field(..., description="Format: '<Location> | <DD Mon YYYY>'")
#     what_happened: str
#     why_it_matters: str
#     who_is_affected: str
#     next_steps_official_response: str

#     language: str = "en"
#     confidence: float = Field(0.65, ge=0.0, le=1.0)
#     safety_flags: list[str] = Field(default_factory=list)
#     notes_for_editor: str = ""
#     source_type: Literal["text", "image", "audio", "video"] = "text"


# NEWS_JSON_SCHEMA: Dict[str, Any] = {
#     "name": "news_script",
#     "schema": {
#         "type": "object",
#         "additionalProperties": False,
#         "properties": {
#             "headline": {"type": "string"},
#             "location_date": {"type": "string"},
#             "what_happened": {"type": "string"},
#             "why_it_matters": {"type": "string"},
#             "who_is_affected": {"type": "string"},
#             "next_steps_official_response": {"type": "string"},
#             "language": {"type": "string"},
#             "confidence": {"type": "number"},
#             "safety_flags": {"type": "array", "items": {"type": "string"}},
#             "notes_for_editor": {"type": "string"},
#             "source_type": {"type": "string", "enum": ["text", "image", "audio", "video"]},
#         },
#         "required": [
#             "headline",
#             "location_date",
#             "what_happened",
#             "why_it_matters",
#             "who_is_affected",
#             "next_steps_official_response",
#             "language",
#             "confidence",
#             "safety_flags",
#             "notes_for_editor",
#             "source_type",
#         ],
#     },
#     "strict": True,
# }


# # -----------------------------
# # 2) TV NEWS COMMAND TYPES System
# # -----------------------------
# TV_COMMAND_TYPES = {
#     "BREAKING NEWS": {
#         "icon": "🚨",
#         "color": "#ff4444",
#         "description": "Urgent breaking news report",
#         "tv_style": "BREAKING NEWS BANNER",
#         "voice_tone": "Urgent, immediate, dramatic",
#         "instructions": [
#             "START with 'Breaking News' banner",
#             "Use dramatic, urgent language",
#             "Keep points short and impactful",
#             "Focus on WHAT just happened",
#             "Emphasize time sensitivity (JUST IN, MOMENTS AGO)",
#             "End with 'Stay tuned for updates'"
#         ]
#     },
#     "LIVE REPORT": {
#         "icon": "📡",
#         "color": "#ffaa00",
#         "description": "Live from location reporting",
#         "tv_style": "LIVE TAG + LOCATION",
#         "voice_tone": "Present tense, descriptive, energetic",
#         "instructions": [
#             "START with 'Live from [Location]'",
#             "Use present tense throughout",
#             "Describe what you're SEEING right now",
#             "Include ambient sounds/atmosphere",
#             "Mention crowd reactions if visible",
#             "End with 'Reporting live from [Location]'"
#         ]
#     },
#     "PRIME TIME": {
#         "icon": "🌟",
#         "color": "#ffcc00",
#         "description": "Prime time detailed analysis",
#         "tv_style": "PRIME TIME SPECIAL REPORT",
#         "voice_tone": "Authoritative, analytical, in-depth",
#         "instructions": [
#             "Start with 'This Prime Time Special Report'",
#             "Provide background and context",
#             "Include expert analysis",
#             "Use graphics-ready bullet points",
#             "Add 'Why this matters' section",
#             "End with 'What to expect next'"
#         ]
#     },
#     "HEADLINE NEWS": {
#         "icon": "📰",
#         "color": "#44aaff",
#         "description": "Top stories bulletin",
#         "tv_style": "HEADLINE NEWS BULLETIN",
#         "voice_tone": "Clear, concise, professional",
#         "instructions": [
#             "Start with 'Top Stories This Hour'",
#             "3-5 key headlines only",
#             "Each headline: 10-15 words max",
#             "Include time/location for each",
#             "Use 'Also in the news' for additional",
#             "End with 'More details after these messages'"
#         ]
#     },
#     "SPORTS NEWS": {
#         "icon": "⚽",
#         "color": "#00aa44",
#         "description": "Sports highlights and scores",
#         "tv_style": "SPORTS CENTER",
#         "voice_tone": "Energetic, exciting, competitive",
#         "instructions": [
#             "Start with 'Sports Update' or 'Game Day'",
#             "Lead with main result/score",
#             "Include key moments/highlights",
#             "Use sports terminology appropriately",
#             "Mention records/achievements",
#             "End with upcoming matches/events"
#         ]
#     },
#     "WEATHER REPORT": {
#         "icon": "🌧️",
#         "color": "#8844ff",
#         "description": "Weather forecast and alerts",
#         "tv_style": "WEATHER ALERT",
#         "voice_tone": "Clear, informative, sometimes urgent",
#         "instructions": [
#             "Start with 'Weather Update' or 'Weather Alert'",
#             "Current conditions first",
#             "Today's forecast next",
#             "Warning/alerts if any",
#             "Weekend outlook",
#             "End with 'Stay prepared/Stay safe'"
#         ]
#     },
#     "BUSINESS NEWS": {
#         "icon": "📈",
#         "color": "#00cccc",
#         "description": "Financial markets and economy",
#         "tv_style": "MARKET UPDATE",
#         "voice_tone": "Professional, factual, numbers-focused",
#         "instructions": [
#             "Start with 'Market Update' or 'Business Brief'",
#             "Lead with major indices/numbers",
#             "Key company news",
#             "Economic indicators",
#             "Expert analysis/commentary",
#             "End with 'Tomorrow's outlook'"
#         ]
#     },
#     "ENTERTAINMENT NEWS": {
#         "icon": "🎬",
#         "color": "#ff66aa",
#         "description": "Celebrity, movies, TV shows",
#         "tv_style": "ENTERTAINMENT TONIGHT",
#         "voice_tone": "Light, engaging, gossipy (but professional)",
#         "instructions": [
#             "Start with 'Entertainment News'",
#             "Lead with biggest celebrity story",
#             "Movie/TV show updates",
#             "Award show news",
#             "Social media trends",
#             "End with 'Coming soon/premieres'"
#         ]
#     },
#     "HEALTH NEWS": {
#         "icon": "🏥",
#         "color": "#ff6666",
#         "description": "Medical updates and health alerts",
#         "tv_style": "HEALTH WATCH",
#         "voice_tone": "Caring, informative, reassuring",
#         "instructions": [
#             "Start with 'Health Watch' or 'Medical Update'",
#             "Lead with important health news",
#             "Doctor/expert advice",
#             "Prevention tips",
#             "Hospital/medical facility updates",
#             "End with 'Stay healthy' reminder"
#         ]
#     },
#     "CRIME REPORT": {
#         "icon": "🚔",
#         "color": "#333333",
#         "description": "Police, crime, investigations",
#         "tv_style": "CRIME WATCH",
#         "voice_tone": "Serious, factual, avoid sensationalism",
#         "instructions": [
#             "Start with 'Crime Report' or 'Police Update'",
#             "Stick to confirmed facts only",
#             "Avoid graphic details",
#             "Police statements/quotes",
#             "Community safety advice",
#             "End with 'Anyone with information...'"
#         ]
#     },
#     "POLITICAL NEWS": {
#         "icon": "🏛️",
#         "color": "#6666ff",
#         "description": "Government, elections, policy",
#         "tv_style": "POLITICAL BRIEFING",
#         "voice_tone": "Neutral, balanced, authoritative",
#         "instructions": [
#             "Start with 'Political Update'",
#             "Present multiple perspectives",
#             "Include official statements",
#             "Policy implications",
#             "Election updates if applicable",
#             "End with 'Developments to watch'"
#         ]
#     },
#     "TECH NEWS": {
#         "icon": "💻",
#         "color": "#00aaff",
#         "description": "Technology, gadgets, internet",
#         "tv_style": "TECH UPDATE",
#         "voice_tone": "Futuristic, innovative, explanatory",
#         "instructions": [
#             "Start with 'Tech News' or 'Digital Update'",
#             "Lead with major tech announcement",
#             "Explain in simple terms",
#             "Impact on users",
#             "Future implications",
#             "End with 'What's next in tech'"
#         ]
#     }
# }


# def get_tv_command_instructions(selected_commands: List[str]) -> str:
#     """Generate TV-style instructions based on selected command types"""
#     if not selected_commands:
#         return ""
    
#     instructions = ["\n**📺 TV NEWS COMMAND INSTRUCTIONS:**"]
    
#     for cmd in selected_commands:
#         if cmd in TV_COMMAND_TYPES:
#             info = TV_COMMAND_TYPES[cmd]
#             instructions.append(f"\n{info['icon']} **{cmd}** ({info['tv_style']}):")
#             instructions.append(f"  **Voice Tone:** {info['voice_tone']}")
#             for inst in info["instructions"]:
#                 instructions.append(f"  • {inst}")
    
#     return "\n".join(instructions)


# # -----------------------------
# # 3) Language Support
# # -----------------------------
# LANGUAGE_SUPPORT = {
#     "en": {
#         "name": "English",
#         "voice_style": "Clear, professional, broadcast-ready",
#         "example_headline": "Breaking News: Major Development in City Center",
#         "anchor_intro": "ANCHOR: Good evening, here are the top stories...",
#     },
#     "hi": {
#         "name": "Hindi",
#         "voice_style": "स्पष्ट, पेशेवर, प्रसारण के लिए तैयार",
#         "example_headline": "ब्रेकिंग न्यूज़: शहर के केंद्र में बड़ा विकास",
#         "anchor_intro": "एंकर: नमस्कार, यहां हैं आज की प्रमुख खबरें...",
#     },
#     "te": {
#         "name": "Telugu",
#         "voice_style": "స్పష్టమైన, ప్రొఫెషనల్, ప్రసారానికి సిద్ధంగా ఉండే",
#         "example_headline": "బ్రేకింగ్ న్యూస్: సిటీ సెంటర్లో పెద్ద అభివృద్ధి",
#         "anchor_intro": "యాంకర్: శుభ సాయంత్రం, ఇక్కడ ఉన్నాయి నేటి ముఖ్య వార్తలు...",
#     }
# }


# # -----------------------------
# # 4) Auto Model Selection
# # -----------------------------
# def get_optimal_model(
#     source_type: str,
#     selected_commands: List[str],
#     file_size_mb: float = 0
# ) -> str:
#     """
#     Automatically select the best model based on input type and requirements.
#     """
#     # Always use gpt-4o for highest quality news scripts
#     return "gpt-4o"


# # -----------------------------
# # 5) Helpers
# # -----------------------------
# def ensure_dirs():
#     os.makedirs("output", exist_ok=True)
#     os.makedirs("temp", exist_ok=True)


# def today_str():
#     return datetime.now().strftime("%d %b %Y")


# def file_to_data_url(file_path: str) -> str:
#     mime, _ = mimetypes.guess_type(file_path)
#     if not mime:
#         mime = "application/octet-stream"
#     with open(file_path, "rb") as f:
#         b64 = base64.b64encode(f.read()).decode("utf-8")
#     return f"data:{mime};base64,{b64}"


# def save_upload_to_temp(uploaded_file) -> str:
#     ensure_dirs()
#     ext = os.path.splitext(uploaded_file.name)[1] or ""
#     path = os.path.join("temp", f"upload_{int(datetime.now().timestamp())}{ext}")
#     with open(path, "wb") as f:
#         f.write(uploaded_file.getbuffer())
#     return path


# def extract_audio_from_video(video_path: str, out_wav_path: str) -> None:
#     """Requires ffmpeg installed on your system."""
#     try:
#         cmd = [
#             "ffmpeg",
#             "-y",
#             "-i", video_path,
#             "-vn",
#             "-acodec", "pcm_s16le",
#             "-ar", "16000",
#             "-ac", "1",
#             "-loglevel", "error",
#             out_wav_path
#         ]
#         result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=60)
#         if result.returncode != 0:
#             raise RuntimeError(f"FFmpeg failed: {result.stderr}")
#     except subprocess.TimeoutExpired:
#         raise RuntimeError("FFmpeg timeout - video might be too long or corrupted")
#     except FileNotFoundError:
#         raise RuntimeError("FFmpeg not found. Please install ffmpeg on your system.")


# def make_client(api_key: str) -> OpenAI:
#     return OpenAI(api_key=api_key)


# def transcribe_audio(client: OpenAI, audio_path: str, language_hint: Optional[str] = None) -> str:
#     try:
#         with open(audio_path, "rb") as f:
#             resp = client.audio.transcriptions.create(
#                 model="whisper-1",
#                 file=f,
#                 language=language_hint,
#                 response_format="text"
#             )
#         return resp
#     except Exception as e:
#         raise RuntimeError(f"Transcription failed: {str(e)}")


# def analyze_image_with_gpt4_vision(client: OpenAI, image_data_url: str, prompt: str) -> str:
#     """Analyze image using GPT-4 Vision API"""
#     try:
#         response = client.chat.completions.create(
#             model="gpt-4o",
#             messages=[
#                 {
#                     "role": "user",
#                     "content": [
#                         {"type": "text", "text": prompt},
#                         {
#                             "type": "image_url",
#                             "image_url": {
#                                 "url": image_data_url,
#                                 "detail": "high"
#                             }
#                         }
#                     ]
#                 }
#             ],
#             max_tokens=1000,
#             temperature=0.2
#         )
#         return response.choices[0].message.content
#     except Exception as e:
#         raise RuntimeError(f"Image analysis failed: {str(e)}")


# def analyze_video_with_gpt4_vision(client: OpenAI, video_path: str) -> str:
#     """Analyze video by extracting and analyzing key frames"""
#     try:
#         # Extract frames
#         frame_paths = extract_frames_from_video(video_path, num_frames=2)
        
#         if not frame_paths:
#             return "Unable to extract frames from video for analysis."
        
#         # Convert frames to data URLs
#         frame_data_urls = [file_to_data_url(frame_path) for frame_path in frame_paths]
        
#         # Prepare content with multiple images
#         content = [
#             {"type": "text", "text": "Analyze these video frames for TV news reporting. Look for:"},
#             {"type": "text", "text": "1. Key events or actions visible"},
#             {"type": "text", "text": "2. People, vehicles, or objects involved"},
#             {"type": "text", "text": "3. Location clues and environment"},
#             {"type": "text", "text": "4. Any text, signs, or important visual elements"},
#             {"type": "text", "text": "5. Emotional tone and atmosphere"},
#         ]
        
#         for i, data_url in enumerate(frame_data_urls):
#             content.append({
#                 "type": "image_url",
#                 "image_url": {
#                     "url": data_url,
#                     "detail": "high"
#                 }
#             })
#             if i < len(frame_paths) - 1:
#                 content.append({"type": "text", "text": f"Frame {i+2}:"})
        
#         # Call GPT-4 Vision
#         response = client.chat.completions.create(
#             model="gpt-4o",
#             messages=[
#                 {
#                     "role": "user",
#                     "content": content
#                 }
#             ],
#             max_tokens=2000,
#             temperature=0.3
#         )
        
#         # Cleanup temporary frame files
#         for frame_path in frame_paths:
#             try:
#                 os.remove(frame_path)
#             except:
#                 pass
        
#         return response.choices[0].message.content
        
#     except Exception as e:
#         raise RuntimeError(f"Video analysis failed: {str(e)}")


# def extract_frames_from_video(video_path: str, num_frames: int = 3) -> List[str]:
#     """Extract key frames from video for analysis"""
#     ensure_dirs()
    
#     # Get video duration
#     cmd = [
#         "ffprobe",
#         "-v", "error",
#         "-show_entries", "format=duration",
#         "-of", "default=noprint_wrappers=1:nokey=1",
#         video_path
#     ]
    
#     try:
#         result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
#         duration = float(result.stdout.strip())
        
#         # Extract frames at intervals
#         frame_paths = []
#         for i in range(num_frames):
#             timestamp = (duration / (num_frames + 1)) * (i + 1)
#             frame_path = os.path.join("temp", f"frame_{int(datetime.now().timestamp())}_{i}.jpg")
            
#             cmd = [
#                 "ffmpeg",
#                 "-y",
#                 "-ss", str(timestamp),
#                 "-i", video_path,
#                 "-vframes", "1",
#                 "-q:v", "2",
#                 "-loglevel", "error",
#                 frame_path
#             ]
            
#             subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
#             if os.path.exists(frame_path):
#                 frame_paths.append(frame_path)
        
#         return frame_paths
#     except Exception as e:
#         raise RuntimeError(f"Failed to extract video frames: {str(e)}")


# def generate_high_quality_news_script(
#     client: OpenAI,
#     model: str,
#     source_type: Literal["text", "image", "audio", "video"],
#     command_types: List[str],
#     language: str,
#     *,
#     raw_text: Optional[str] = None,
#     image_data_url: Optional[str] = None,
#     video_path: Optional[str] = None,
#     transcript: Optional[str] = None,
#     town: Optional[str] = None,
#     incident_time: Optional[str] = None,
# ) -> str:
#     """
#     Generates HIGH-QUALITY TV news scripts directly from input.
#     Focuses on creating professional, detailed news scripts based on input.
#     """
    
#     # Prepare the input content
#     user_content = []
    
#     # Add location and time hints if provided
#     location_info = ""
#     if town:
#         location_info += f"Location: {town}\n"
#     if incident_time:
#         location_info += f"Time: {incident_time}\n"
    
#     if location_info:
#         user_content.append({"type": "text", "text": location_info})
    
#     user_content.append({"type": "text", "text": f"Broadcast Date: {today_str()}\n\n"})

#     # Add the main content based on source type
#     if source_type == "text":
#         user_content.append({"type": "text", "text": f"NEWS CONTENT INPUT:\n{raw_text or ''}"})
    
#     elif source_type == "image":
#         # Analyze the image for news reporting
#         analysis_prompt = """Analyze this image for HIGH-QUALITY TV news reporting. Provide detailed analysis for:
# 1. EXACTLY what is happening in the scene
# 2. SPECIFIC details about people involved (number, approximate ages, activities)
# 3. IDENTIFIABLE objects, vehicles, infrastructure
# 4. ALL visible text, signs, logos, labels
# 5. PRECISE location setting and environment details
# 6. Time of day/weather conditions visible
# 7. Visible emotions, actions, interactions
# 8. NEWSWORTHY angles and story potential

# Provide ULTRA-DETAILED analysis suitable for professional TV news script."""
        
#         image_analysis = analyze_image_with_gpt4_vision(client, image_data_url, analysis_prompt)
#         user_content.append({"type": "text", "text": f"DETAILED IMAGE ANALYSIS FOR TV NEWS:\n{image_analysis}\n\n"})
#         user_content.append({"type": "image_url", "image_url": {"url": image_data_url, "detail": "high"}})
    
#     elif source_type == "video":
#         # Analyze video frames
#         video_analysis = analyze_video_with_gpt4_vision(client, video_path)
#         user_content.append({"type": "text", "text": f"DETAILED VIDEO ANALYSIS FOR TV NEWS:\n{video_analysis}\n\n"})
        
#         if transcript:
#             user_content.append({"type": "text", "text": f"AUDIO TRANSCRIPT:\n{transcript}\n\n"})
    
#     elif source_type == "audio":
#         user_content.append({"type": "text", "text": f"AUDIO TRANSCRIPT:\n{transcript or ''}"})
    
#     else:
#         raise ValueError("Invalid source_type")

#     # Get command-specific instructions
#     command_instructions = get_tv_command_instructions(command_types)
    
#     # Determine the main command type
#     main_command = command_types[0] if command_types else "HEADLINE NEWS"
    
#     # Language-specific instructions
#     lang_info = LANGUAGE_SUPPORT.get(language, LANGUAGE_SUPPORT["en"])
    
#     # Build TV news script prompt for HIGH QUALITY output
#     system_prompt = f"""
# You are a SENIOR TV news anchor and scriptwriter for a major national television news channel.
# Your expertise: Creating ULTRA-QUALITY, PROFESSIONAL, BROADCAST-READY TV news scripts.

# **YOUR MISSION:** Generate the HIGHEST QUALITY TV news script based SOLELY on the provided input.
# **CRITICAL RULE:** DO NOT invent or add fictional details. Use ONLY the information provided in the input.
# **LANGUAGE:** Generate script in {lang_info['name']} ({language})

# **SCRIPT QUALITY REQUIREMENTS (NON-NEGOTIABLE):**
# 1. **FACT-BASED:** Use ONLY facts from input. NO speculation.
# 2. **DETAIL-ORIENTED:** Include specific details from input (names, numbers, times, locations).
# 3. **PROFESSIONAL STRUCTURE:** Proper TV news format with anchor/reporter dialogue.
# 4. **VISUAL DESCRIPTIONS:** Clear [VISUAL CUES] for producers.
# 5. **BROADCAST READY:** Ready for immediate teleprompter use.

# **TV NEWS FORMAT FOR {main_command}:**
# [PROFESSIONAL NEWS GRAPHIC]
# [APPROPRIATE THEME MUSIC]

# {lang_info['anchor_intro']}

# **SCRIPT STRUCTURE:**
# 1. MAIN HEADLINE: Based on key input fact
# 2. DETAILED BACKGROUND: Specific context from input
# 3. KEY DEVELOPMENTS: Important details from input
# 4. WHO IS INVOLVED: Specific people/groups from input
# 5. LOCATION DETAILS: Exact location info from input
# 6. OFFICIAL/EXPERT INPUT: If provided in input
# 7. WHAT'S NEXT: Based on input information
# 8. SIGN-OFF: Professional closing

# {command_instructions}

# **ADDITIONAL RULES:**
# - Use ONLY information provided in the input
# - Make script NATURAL and CONVERSATIONAL for TV
# - Include [BRACKETED VISUAL CUES] for producers
# - Suggest [LOWER THIRD: text] graphics
# - Keep sentences SHORT and CLEAR for teleprompter
# - Add [MUSIC: suggestion] and [GRAPHIC: suggestion]
# - END with proper anchor sign-off
# - NO markdown in final output
# - Maximum detail from input, minimum invention

# **REMEMBER:** Quality comes from USING INPUT DETAILS, not inventing new ones.
# Generate a COMPLETE, PROFESSIONAL, BROADCAST-READY TV NEWS SCRIPT.
# """

#     try:
#         # Call OpenAI with emphasis on using input details
#         response = client.chat.completions.create(
#             model=model,
#             messages=[
#                 {"role": "system", "content": system_prompt},
#                 {"role": "user", "content": user_content}
#             ],
#             temperature=0.2,  # Lower temperature for more factual output
#             max_tokens=3000  # More tokens for detailed scripts
#         )

#         output = response.choices[0].message.content

#         # Format cleanup for TV news
#         lines = output.split('\n')
#         formatted_lines = []
        
#         in_script = False
#         for line in lines:
#             line = line.strip()
#             if not line:
#                 continue
            
#             # Remove markdown if present
#             if line.startswith('#') or line.startswith('**') or line.startswith('*') and not line.startswith('***'):
#                 line = line.replace('#', '').replace('**', '').replace('*', '')
#                 line = line.strip()
            
#             # Add proper formatting
#             if any(x in line for x in ['ANCHOR:', 'REPORTER:', 'एंकर:', 'యాంకర్:', 'रिपोर्टर:', 'రిపోర్టర్:']):
#                 formatted_lines.append("")
#                 formatted_lines.append(line.upper())
#             elif any(x in line for x in ['[LOWER THIRD:', '[VISUALS:', '[GRAPHIC:', '[MUSIC:', '[BANNER:']):
#                 formatted_lines.append(f"  {line}")
#             elif line.isdigit() and len(line) <= 2 and in_script:
#                 continue
#             elif line.startswith('•') or line.startswith('-'):
#                 formatted_lines.append(f"  • {line[1:].strip()}")
#                 in_script = True
#             elif in_script and line and not line.startswith('['):
#                 if formatted_lines and formatted_lines[-1].startswith('  •'):
#                     formatted_lines[-1] = formatted_lines[-1] + ' ' + line
#                 else:
#                     formatted_lines.append(line)
#             else:
#                 formatted_lines.append(line)
        
#         # Join with proper spacing
#         output = '\n'.join(formatted_lines)
        
#         # Add timestamp and command type header
#         timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
#         header = f"""
# {'='*70}
# 📺 TV NEWS SCRIPT | {timestamp}
# 🎯 COMMAND: {main_command}
# 🌐 LANGUAGE: {lang_info['name']} ({language})
# {'='*70}

# """
        
#         return header + output
        
#     except Exception as e:
#         raise RuntimeError(f"Failed to generate TV news script: {str(e)}")


# def generate_multilingual_news_scripts(
#     client: OpenAI,
#     model: str,
#     source_type: str,
#     command_types: List[str],
#     *,
#     raw_text: Optional[str] = None,
#     image_data_url: Optional[str] = None,
#     video_path: Optional[str] = None,
#     transcript: Optional[str] = None,
#     town: Optional[str] = None,
#     incident_time: Optional[str] = None,
# ) -> Dict[str, str]:
#     """Generate news scripts in multiple languages"""
#     scripts = {}
    
#     languages = ["en", "hi", "te"]  # English, Hindi, Telugu
    
#     for lang in languages:
#         try:
#             script = generate_high_quality_news_script(
#                 client=client,
#                 model=model,
#                 source_type=source_type,
#                 command_types=command_types,
#                 language=lang,
#                 raw_text=raw_text,
#                 image_data_url=image_data_url,
#                 video_path=video_path,
#                 transcript=transcript,
#                 town=town,
#                 incident_time=incident_time,
#             )
#             scripts[lang] = script
#         except Exception as e:
#             scripts[lang] = f"Error generating {LANGUAGE_SUPPORT[lang]['name']} script: {str(e)}"
    
#     return scripts


# def save_tv_script_output(tv_script: str, language: str, source_type: str, command_types: List[str]) -> str:
#     """Save TV script output to a text file."""
#     ensure_dirs()
#     ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    
#     # Create filename
#     cmd_str = command_types[0] if command_types else "standard"
#     lang_code = language
#     out_path = os.path.join("output", f"tv_news_{cmd_str}_{lang_code}_{ts}.txt")
    
#     with open(out_path, "w", encoding="utf-8") as f:
#         f.write(tv_script)
    
#     return out_path


# def display_tv_script(tv_script: str, language: str, command_types: List[str], model_used: str):
#     """Display TV script with proper formatting"""
#     lang_info = LANGUAGE_SUPPORT.get(language, LANGUAGE_SUPPORT["en"])
    
#     # Show header
#     st.markdown(f"### 📺 {lang_info['name']} TV News Script")
#     st.markdown(f"**🤖 Model Used:** `{model_used}` | **🎯 Style:** `{lang_info['voice_style']}`")
    
#     # Show command type
#     if command_types:
#         main_cmd = command_types[0]
#         if main_cmd in TV_COMMAND_TYPES:
#             info = TV_COMMAND_TYPES[main_cmd]
#             st.markdown(f"""
#             <div style="
#                 background-color: {info['color']}20;
#                 border-left: 4px solid {info['color']};
#                 padding: 10px;
#                 border-radius: 5px;
#                 margin: 10px 0;
#             ">
#             <strong>{info['icon']} {main_cmd}</strong> - {info['description']}
#             </div>
#             """, unsafe_allow_html=True)
    
#     st.divider()
    
#     # Display the script
#     st.markdown("##### 📋 Script Content:")
    
#     lines = tv_script.split('\n')
    
#     for line in lines:
#         line = line.strip()
#         if not line:
#             st.markdown("<br>", unsafe_allow_html=True)
#             continue
            
#         # Header section
#         if '='*70 in line:
#             st.code(line)
#         elif 'TV NEWS SCRIPT' in line or 'COMMAND:' in line or 'LANGUAGE:' in line:
#             st.info(line)
            
#         # Anchor/Reporter lines
#         elif any(prefix in line for prefix in ['ANCHOR:', 'REPORTER:', 'एंकर:', 'యాంకర్:', 'रिपोर्टर:', 'రిపోర్టర్:']):
#             st.markdown(f"""
#             <div style="
#                 background-color: #e8f4f8;
#                 padding: 12px;
#                 border-radius: 5px;
#                 margin: 8px 0;
#                 border-left: 4px solid #0066cc;
#                 font-family: 'Arial', sans-serif;
#                 font-weight: bold;
#             ">
#             {line}
#             </div>
#             """, unsafe_allow_html=True)
            
#         # Visual/Technical cues
#         elif line.startswith('[') and line.endswith(']'):
#             st.markdown(f"""
#             <div style="
#                 background-color: #fff3cd;
#                 color: #856404;
#                 padding: 8px;
#                 border-radius: 3px;
#                 margin: 5px 0;
#                 font-family: 'Courier New', monospace;
#                 font-size: 0.9em;
#             ">
#             🎬 {line}
#             </div>
#             """, unsafe_allow_html=True)
            
#         # Bullet points
#         elif line.startswith('•'):
#             st.markdown(f"""
#             <div style="
#                 margin-left: 20px;
#                 margin-bottom: 5px;
#                 padding-left: 10px;
#                 border-left: 2px solid #4CAF50;
#                 background-color: #f9f9f9;
#                 padding: 8px;
#                 border-radius: 3px;
#             ">
#             {line}
#             </div>
#             """, unsafe_allow_html=True)
            
#         # Regular text (long paragraphs)
#         elif len(line) > 80 and not any(x in line for x in [':', '[', ']', '•']):
#             st.markdown(f"""
#             <div style="
#                 padding: 10px;
#                 background-color: #f8f9fa;
#                 border-radius: 5px;
#                 margin: 5px 0;
#                 line-height: 1.6;
#             ">
#             {line}
#             </div>
#             """, unsafe_allow_html=True)
            
#         # Other text
#         elif line:
#             st.markdown(line)


# # -----------------------------
# # 6) Streamlit UI - SIMPLE VERSION
# # -----------------------------
# load_dotenv()
# ensure_dirs()

# st.set_page_config(
#     page_title="TV News Generator",
#     layout="wide",
#     page_icon="📺"
# )

# # Simple UI
# st.title("📺 TV News Script Generator")
# st.markdown("Create professional TV news scripts in English, Hindi & Telugu")

# # Sidebar - Simple Settings
# with st.sidebar:
#     st.header("Settings")
    
#     # API Key
#     env_key = os.getenv("OPENAI_API_KEY", "")
#     api_key = st.text_input("OpenAI API Key", value=env_key, type="password")
    
#     st.divider()
    
#     # TV News Type
#     st.subheader("News Type")
#     selected_commands = st.selectbox(
#         "Select news format:",
#         options=list(TV_COMMAND_TYPES.keys()),
#         format_func=lambda x: f"{TV_COMMAND_TYPES[x]['icon']} {x}",
#         help="Choose the type of TV news report"
#     )
    
#     if selected_commands:
#         info = TV_COMMAND_TYPES[selected_commands]
#         st.caption(f"**Style:** {info['tv_style']}")
#         st.caption(f"**Tone:** {info['voice_tone']}")
    
#     st.divider()
    
#     # Location/Time
#     st.subheader("News Details")
#     town = st.text_input("Location", placeholder="e.g., Mumbai, Delhi, Bangalore")
#     incident_time = st.text_input("Time", placeholder="e.g., today morning, yesterday evening")
    
#     st.divider()
    
#     # Languages to generate
#     st.subheader("Languages")
#     generate_english = st.checkbox("English", value=True)
#     generate_hindi = st.checkbox("Hindi", value=True)
#     generate_telugu = st.checkbox("Telugu", value=True)

# # Main Content Area - Simple
# st.header("📥 Input News Content")

# # Input method
# input_type = st.radio(
#     "Choose input type:",
#     ["Text", "Image", "Audio", "Video"],
#     horizontal=True
# )

# input_content = None
# uploaded_file = None

# if input_type == "Text":
#     input_content = st.text_area(
#         "Enter news content:",
#         height=200,
#         placeholder="Paste news content, reporter notes, eyewitness accounts, or any relevant information here...",
#         help="Enter detailed information for high-quality news script"
#     )
# else:
#     file_extensions = {
#         "Image": ["jpg", "jpeg", "png", "webp"],
#         "Audio": ["mp3", "wav", "m4a", "ogg"],
#         "Video": ["mp4", "mov", "avi", "mkv"]
#     }
    
#     uploaded_file = st.file_uploader(
#         f"Upload {input_type} file",
#         type=file_extensions[input_type]
#     )
    
#     if uploaded_file:
#         if input_type == "Image":
#             st.image(uploaded_file, caption="News Image", use_column_width=True)
#         elif input_type == "Audio":
#             st.audio(uploaded_file)
#         elif input_type == "Video":
#             st.video(uploaded_file)

# # Generate Button
# st.divider()

# if st.button(
#     "🚀 GENERATE TV NEWS SCRIPTS",
#     type="primary",
#     use_container_width=True,
#     disabled=not api_key.strip()
# ):
#     if not api_key.strip():
#         st.error("Please enter your OpenAI API key.")
#         st.stop()
    
#     if input_type == "Text" and not input_content:
#         st.error("Please enter news content.")
#         st.stop()
#     elif input_type != "Text" and not uploaded_file:
#         st.error(f"Please upload an {input_type} file.")
#         st.stop()
    
#     # Prepare for processing
#     client = make_client(api_key.strip())
#     selected_model = get_optimal_model(input_type.lower(), [selected_commands])
    
#     # Show processing status
#     with st.spinner("🔄 Processing input and generating high-quality news scripts..."):
#         try:
#             # Process input based on type
#             transcript = None
#             video_path = None
#             image_data_url = None
#             raw_text = input_content if input_type == "Text" else ""
            
#             if input_type == "Image" and uploaded_file:
#                 img_path = save_upload_to_temp(uploaded_file)
#                 image_data_url = file_to_data_url(img_path)
#             elif input_type == "Audio" and uploaded_file:
#                 audio_path = save_upload_to_temp(uploaded_file)
#                 transcript = transcribe_audio(client, audio_path)
#             elif input_type == "Video" and uploaded_file:
#                 video_path = save_upload_to_temp(uploaded_file)
#                 wav_path = os.path.join("temp", f"extracted_{int(datetime.now().timestamp())}.wav")
#                 extract_audio_from_video(video_path, wav_path)
#                 transcript = transcribe_audio(client, wav_path)
            
#             # Generate scripts for selected languages
#             languages_to_generate = []
#             if generate_english:
#                 languages_to_generate.append("en")
#             if generate_hindi:
#                 languages_to_generate.append("hi")
#             if generate_telugu:
#                 languages_to_generate.append("te")
            
#             # Create tabs for each language
#             tabs = st.tabs([f"{LANGUAGE_SUPPORT[lang]['name']} Script" for lang in languages_to_generate])
            
#             for idx, lang in enumerate(languages_to_generate):
#                 with tabs[idx]:
#                     try:
#                         # Generate individual script
#                         script = generate_high_quality_news_script(
#                             client=client,
#                             model=selected_model,
#                             source_type=input_type.lower(),
#                             command_types=[selected_commands],
#                             language=lang,
#                             raw_text=raw_text,
#                             image_data_url=image_data_url,
#                             video_path=video_path,
#                             transcript=transcript,
#                             town=town or None,
#                             incident_time=incident_time or None,
#                         )
                        
#                         # Display script
#                         display_tv_script(script, lang, [selected_commands], selected_model)
                        
#                         # Save and download option
#                         script_path = save_tv_script_output(script, lang, input_type.lower(), [selected_commands])
                        
#                         col1, col2 = st.columns(2)
#                         with col1:
#                             with open(script_path, "rb") as f:
#                                 st.download_button(
#                                     label=f"📥 Download {LANGUAGE_SUPPORT[lang]['name']} Script",
#                                     data=f,
#                                     file_name=os.path.basename(script_path),
#                                     mime="text/plain",
#                                     use_container_width=True
#                                 )
#                         with col2:
#                             if st.button(f"📋 Copy {LANGUAGE_SUPPORT[lang]['name']} Script", use_container_width=True, key=f"copy_{lang}"):
#                                 st.session_state[f"script_{lang}"] = script
#                                 st.success(f"{LANGUAGE_SUPPORT[lang]['name']} script copied to clipboard!")
                        
#                     except Exception as e:
#                         st.error(f"Failed to generate {LANGUAGE_SUPPORT[lang]['name']} script: {str(e)}")
            
#             st.success("✅ All news scripts generated successfully!")
            
#         except Exception as e:
#             st.error(f"❌ Error: {str(e)}")

# # Simple Footer
# st.divider()
# st.caption("📺 TV News Script Generator | Supports English, Hindi & Telugu | High-Quality Broadcast Scripts")