import streamlit as st
import openai
import subprocess
import os
import json
from pathlib import Path
import tempfile
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Page config
st.set_page_config(
    page_title="Auto News Summarizer",
    page_icon="🎬",
    layout="wide"
)

st.title("🎬 Automatic News Video Summarizer")
st.markdown("Automatically extracts 30-second news highlights when person speaks")

# Initialize session state
if 'processing' not in st.session_state:
    st.session_state.processing = False
if 'last_summary' not in st.session_state:
    st.session_state.last_summary = None
if 'ffmpeg_available' not in st.session_state:
    st.session_state.ffmpeg_available = None

def check_ffmpeg():
    """Check if FFmpeg is installed"""
    try:
        subprocess.run(['ffmpeg', '-version'], 
                      stdout=subprocess.PIPE, 
                      stderr=subprocess.PIPE, 
                      check=True)
        return True
    except:
        return False

def check_ffprobe():
    """Check if FFprobe is installed (part of FFmpeg)"""
    try:
        subprocess.run(['ffprobe', '-version'], 
                      stdout=subprocess.PIPE, 
                      stderr=subprocess.PIPE, 
                      check=True)
        return True
    except:
        return False

# Check FFmpeg once and store in session state
if st.session_state.ffmpeg_available is None:
    st.session_state.ffmpeg_available = check_ffmpeg()
    st.session_state.ffprobe_available = check_ffprobe()

# Sidebar for API key and settings
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Get API key from .env or allow manual input
    default_api_key = os.getenv("OPENAI_API_KEY", "")
    if default_api_key:
        api_key = st.text_input("OpenAI API Key", 
                               value=default_api_key, 
                               type="password",
                               help="API key loaded from .env file")
        st.success("✅ API key loaded from environment")
    else:
        api_key = st.text_input("OpenAI API Key", 
                               type="password",
                               help="Enter your OpenAI API key or set OPENAI_API_KEY in .env file")
        st.warning("⚠️ No API key found in .env file")
    
    st.markdown("---")
    st.header("📋 Auto Settings")
    target_duration = st.slider("Target Duration (seconds)", 20, 60, 30)
    
    st.info("🤖 AI will automatically select the most important news segments")
    
    st.markdown("---")
    st.header("🔧 Advanced Settings")
    
    # Advanced options
    with st.expander("Advanced Options"):
        min_segment_duration = st.slider("Minimum segment duration (seconds)", 
                                         2, 10, 3)
        max_segments = st.slider("Maximum segments to extract", 3, 10, 5)
        add_transitions = st.checkbox("Add smooth transitions", value=True)
    
    st.markdown("---")
    st.markdown("### 📖 How it works")
    st.markdown("""
    1. Upload your video
    2. AI detects speech segments
    3. Extracts important news
    4. Creates 30-sec summary
    5. Download result
    """)
    
    # System check
    st.markdown("---")
    st.markdown("### 🔍 System Check")
    
    ffmpeg_status = "✅ Found" if st.session_state.ffmpeg_available else "❌ Missing"
    ffprobe_status = "✅ Found" if st.session_state.ffprobe_available else "❌ Missing"
    
    st.write(f"FFmpeg: {ffmpeg_status}")
    st.write(f"FFprobe: {ffprobe_status}")
    st.write(f"OpenAI API: {'✅ Configured' if api_key else '❌ Required'}")
    
    # Show FFmpeg installation help if needed
    if not st.session_state.ffmpeg_available:
        with st.expander("FFmpeg Installation Guide"):
            st.markdown("""
            **Install FFmpeg:**
            
            **Windows:**
            ```bash
            # Using Chocolatey
            choco install ffmpeg
            
            # Or download from ffmpeg.org
            ```
            
            **macOS:**
            ```bash
            brew install ffmpeg
            ```
            
            **Ubuntu/Debian:**
            ```bash
            sudo apt update
            sudo apt install ffmpeg
            ```
            """)

def extract_audio(video_path, audio_path):
    """Extract audio from video using FFmpeg"""
    cmd = [
        'ffmpeg', '-i', video_path,
        '-vn', '-acodec', 'libmp3lame',
        '-ar', '16000', '-ac', '1',
        '-y', audio_path
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        error_msg = result.stderr.decode()[:500]
        st.error(f"FFmpeg error: {error_msg}")
        raise Exception("Audio extraction failed")
    return audio_path

def get_video_duration(video_path):
    """Get video duration in seconds"""
    if not st.session_state.ffprobe_available:
        file_size_mb = os.path.getsize(video_path) / (1024 * 1024)
        return min(file_size_mb * 5, 600)
    
    cmd = [
        'ffprobe', '-v', 'error',
        '-show_entries', 'format=duration',
        '-of', 'default=noprint_wrappers=1:nokey=1',
        video_path
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode == 0:
        try:
            return float(result.stdout)
        except:
            return 300
    return 300

def transcribe_audio(audio_path, api_key):
    """Transcribe audio using OpenAI Whisper - supports all languages"""
    client = openai.OpenAI(api_key=api_key)
    
    with open(audio_path, 'rb') as audio_file:
        try:
            transcript = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                response_format="verbose_json",
                timestamp_granularities=["segment"]
            )
            return transcript
        except openai.AuthenticationError:
            st.error("❌ Invalid API key. Please check your OpenAI API key.")
            raise
        except openai.RateLimitError:
            st.error("❌ API rate limit exceeded. Please try again later.")
            raise
        except Exception as e:
            st.error(f"❌ Transcription failed: {str(e)}")
            raise

def analyze_content_auto(transcript, api_key, target_duration, video_duration, max_segments=5):
    """Automatically analyze and extract important news segments - works with ANY language"""
    client = openai.OpenAI(api_key=api_key)
    
    # Prepare transcript text with timestamps - FIX: Access attributes instead of dict keys
    segments_text = "\n".join([
        f"[{seg.start:.2f}s - {seg.end:.2f}s]: {seg.text}"
        for seg in transcript.segments
    ])
    
    # Detect language from transcript
    detected_language = transcript.language if hasattr(transcript, 'language') else "unknown"
    
    prompt = f"""You are an expert video editor. Analyze this {video_duration:.1f}-second transcript and automatically identify the MOST IMPORTANT news segments for a {target_duration}-second summary.

TRANSCRIPT (Language: {detected_language}):
{segments_text}

CRITICAL INSTRUCTIONS:
1. This transcript may be in ANY language (English, Tamil, Hindi, Spanish, etc.) - process it regardless
2. Identify segments containing KEY NEWS, IMPORTANT ANNOUNCEMENTS, or CRITICAL INFORMATION
3. Focus on segments where a person is actively speaking important content
4. Select {max_segments} most newsworthy segments
5. Total duration should be close to {target_duration} seconds
6. Each segment must be at least 3 seconds long
7. Maintain chronological order
8. NO overlapping segments

Return ONLY a valid JSON array with this EXACT format (no markdown, no code blocks, no extra text):

[
  {{"start_time": 0.5, "end_time": 8.3, "reason": "Opening key announcement", "priority": 5, "text": "transcript text here"}},
  {{"start_time": 45.2, "end_time": 52.7, "reason": "Critical data/news point", "priority": 4, "text": "transcript text here"}}
]

IMPORTANT: Return ONLY the JSON array. Do NOT include any other text, explanations, or markdown formatting."""

    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {
                    "role": "system", 
                    "content": "You are an expert video editor specializing in news content extraction. You MUST return ONLY valid JSON arrays, nothing else. You can process transcripts in ANY language."
                },
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=2000
        )
        
        result = response.choices[0].message.content.strip()
        
        # Clean up response - remove any markdown or extra text
        # Find JSON array in the response
        start_idx = result.find('[')
        end_idx = result.rfind(']')
        
        if start_idx == -1 or end_idx == -1:
            st.error("AI did not return valid JSON. Attempting fallback...")
            return create_fallback_segments(transcript, target_duration, max_segments)
        
        json_str = result[start_idx:end_idx+1]
        
        try:
            segments = json.loads(json_str)
            
            # Validate segments
            if not isinstance(segments, list) or len(segments) == 0:
                st.warning("Invalid segment format. Using fallback method...")
                return create_fallback_segments(transcript, target_duration, max_segments)
            
            # Ensure all required fields exist
            for seg in segments:
                if not all(key in seg for key in ['start_time', 'end_time', 'priority']):
                    st.warning("Missing required fields. Using fallback method...")
                    return create_fallback_segments(transcript, target_duration, max_segments)
            
            return segments
            
        except json.JSONDecodeError as e:
            st.error(f"JSON parsing failed: {e}")
            st.info("Using fallback segment selection...")
            return create_fallback_segments(transcript, target_duration, max_segments)
        
    except Exception as e:
        st.error(f"AI analysis error: {str(e)}")
        st.info("Using fallback segment selection...")
        return create_fallback_segments(transcript, target_duration, max_segments)

def create_fallback_segments(transcript, target_duration, max_segments):
    """Fallback method: automatically select segments from transcript"""
    segments = []
    
    if not hasattr(transcript, 'segments') or len(transcript.segments) == 0:
        raise ValueError("No transcript segments available")
    
    # Calculate how many segments we need
    total_duration = 0
    segment_duration = target_duration / max_segments
    
    # Select evenly distributed segments from the transcript
    total_transcript_segments = len(transcript.segments)
    step = max(1, total_transcript_segments // max_segments)
    
    selected_indices = []
    for i in range(0, total_transcript_segments, step):
        if len(selected_indices) >= max_segments:
            break
        selected_indices.append(i)
    
    # Create segment objects - FIX: Access attributes instead of dict keys
    for idx in selected_indices:
        seg = transcript.segments[idx]
        segment_obj = {
            'start_time': seg.start,
            'end_time': seg.end,
            'priority': 3,
            'reason': 'Auto-selected segment',
            'text': seg.text
        }
        segments.append(segment_obj)
        total_duration += (seg.end - seg.start)
        
        if total_duration >= target_duration:
            break
    
    st.info(f"✅ Fallback: Selected {len(segments)} segments automatically")
    return segments

def create_summary_video(input_video, segments, output_path, min_duration=3, add_transitions=False):
    """Create summarized video by concatenating selected segments"""
    
    # Filter out very short segments
    valid_segments = []
    for seg in segments:
        duration = seg['end_time'] - seg['start_time']
        if duration >= min_duration:
            valid_segments.append(seg)
    
    if not valid_segments:
        raise ValueError("No valid segments found for summary")
    
    # Create a temporary directory for segment files
    with tempfile.TemporaryDirectory() as temp_dir:
        segment_files = []
        
        # Extract each segment
        for i, seg in enumerate(valid_segments):
            segment_path = os.path.join(temp_dir, f"segment_{i:03d}.mp4")
            
            start_time = seg['start_time']
            end_time = seg['end_time']
            duration = end_time - start_time
            
            # Extract segment
            cmd = [
                'ffmpeg', '-i', input_video,
                '-ss', str(start_time),
                '-t', str(duration),
                '-c:v', 'libx264', '-preset', 'fast',
                '-c:a', 'aac',
                '-avoid_negative_ts', 'make_zero',
                '-y', segment_path
            ]
            
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30)
            if result.returncode != 0:
                # Try simpler method
                cmd_simple = [
                    'ffmpeg', '-i', input_video,
                    '-ss', str(start_time),
                    '-t', str(duration),
                    '-c', 'copy',
                    '-y', segment_path
                ]
                subprocess.run(cmd_simple, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30)
            
            segment_files.append(segment_path)
        
        # Create concat file
        concat_file = os.path.join(temp_dir, 'concat.txt')
        with open(concat_file, 'w', encoding='utf-8') as f:
            for seg_file in segment_files:
                f.write(f"file '{seg_file}'\n")
        
        # Concatenate segments
        cmd = [
            'ffmpeg', '-f', 'concat',
            '-safe', '0',
            '-i', concat_file,
            '-c:v', 'libx264', '-preset', 'fast',
            '-c:a', 'aac',
            '-movflags', '+faststart',
            '-y', output_path
        ]
        
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=60)
        if result.returncode != 0:
            st.error(f"Concatenation failed: {result.stderr.decode()[:500]}")
            raise Exception("Video creation failed")
    
    return output_path, valid_segments

# Main interface
col1, col2 = st.columns([1, 1])

with col1:
    st.header("📤 Upload Video")
    uploaded_file = st.file_uploader(
        "Choose a video file",
        type=['mp4', 'mov', 'avi', 'mkv', 'webm'],
        help="Upload any video - AI will auto-detect speech and extract news"
    )
    
    if uploaded_file:
        file_size_mb = uploaded_file.size / (1024 * 1024)
        
        col_info1, col_info2 = st.columns(2)
        with col_info1:
            st.metric("File Size", f"{file_size_mb:.1f} MB")
        
        st.video(uploaded_file)
        
        if file_size_mb > 200:
            st.warning("⚠️ Large file detected. Processing may take longer.")

with col2:
    st.header("🎯 Auto News Summary")
    
    if uploaded_file:
        if not api_key:
            st.error("🔑 Please enter your OpenAI API key in the sidebar")
        elif not st.session_state.ffmpeg_available:
            st.error("❌ FFmpeg not found. Please install FFmpeg to use this tool.")
        else:
            if st.button("🚀 Auto-Generate News Summary", 
                        type="primary", 
                        disabled=st.session_state.processing,
                        use_container_width=True):
                
                st.session_state.processing = True
                
                try:
                    with tempfile.TemporaryDirectory() as temp_dir:
                        
                        # Save uploaded video
                        video_path = os.path.join(temp_dir, "input_video.mp4")
                        with open(video_path, 'wb') as f:
                            f.write(uploaded_file.read())
                        
                        video_duration = get_video_duration(video_path)
                        
                        # Progress tracking
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        # Step 1: Extract audio
                        status_text.text("🎵 Step 1/4: Extracting audio...")
                        progress_bar.progress(25)
                        audio_path = os.path.join(temp_dir, "audio.mp3")
                        extract_audio(video_path, audio_path)
                        
                        # Step 2: Transcribe (supports all languages)
                        status_text.text("📝 Step 2/4: Transcribing speech (any language)...")
                        progress_bar.progress(50)
                        transcript = transcribe_audio(audio_path, api_key)
                        
                        # Show detected language
                        detected_lang = transcript.language if hasattr(transcript, 'language') else "unknown"
                        st.info(f"🌍 Detected language: {detected_lang.upper()}")
                        
                        # Step 3: Auto-analyze content
                        status_text.text("🤖 Step 3/4: AI analyzing & selecting important news...")
                        progress_bar.progress(75)
                        segments = analyze_content_auto(
                            transcript, api_key, 
                            target_duration, video_duration, max_segments
                        )
                        
                        # Step 4: Create summary video
                        status_text.text("✂️ Step 4/4: Creating news summary video...")
                        progress_bar.progress(90)
                        output_path = os.path.join(temp_dir, "summary.mp4")
                        output_path, selected_segments = create_summary_video(
                            video_path, segments, output_path, 
                            min_segment_duration, add_transitions
                        )
                        
                        progress_bar.progress(100)
                        status_text.text("✅ News summary created!")
                        
                        # Calculate stats
                        total_summary_duration = sum(
                            seg['end_time'] - seg['start_time'] 
                            for seg in selected_segments
                        )
                        
                        # Display results
                        st.success(f"""
                        🎉 Auto News Summary Complete!
                        - **Duration:** {total_summary_duration:.1f} seconds
                        - **Segments:** {len(selected_segments)}
                        - **Compression:** {(video_duration/total_summary_duration):.1f}x
                        """)
                        
                        # Show summary video
                        with open(output_path, 'rb') as f:
                            video_bytes = f.read()
                            
                            st.subheader("📺 News Summary Preview")
                            st.video(video_bytes)
                            
                            # Download button
                            col_dl1, col_dl2 = st.columns([1, 2])
                            with col_dl1:
                                st.download_button(
                                    label="⬇️ Download Summary",
                                    data=video_bytes,
                                    file_name=f"news_summary_{target_duration}s.mp4",
                                    mime="video/mp4",
                                    use_container_width=True
                                )
                            
                            with col_dl2:
                                file_size_mb = len(video_bytes) / (1024 * 1024)
                                st.caption(f"File size: {file_size_mb:.1f} MB")
                        
                        # Show segment details
                        with st.expander("📋 View Selected Segments"):
                            for i, seg in enumerate(selected_segments, 1):
                                duration = seg['end_time'] - seg['start_time']
                                
                                col_seg1, col_seg2 = st.columns([1, 4])
                                
                                with col_seg1:
                                    st.metric(f"Segment {i}", f"{duration:.1f}s")
                                
                                with col_seg2:
                                    priority_stars = "⭐" * seg.get('priority', 3)
                                    st.markdown(f"""
                                    **Time:** {seg['start_time']:.1f}s - {seg['end_time']:.1f}s  
                                    **Priority:** {priority_stars}  
                                    **Reason:** {seg.get('reason', 'Auto-selected')}
                                    """)
                                    if 'text' in seg and seg['text']:
                                        with st.expander("View transcript"):
                                            st.caption(seg['text'])
                                
                                st.divider()
                        
                        # Store in session state
                        st.session_state.last_summary = {
                            'video_bytes': video_bytes,
                            'segments': selected_segments,
                            'original_duration': video_duration,
                            'summary_duration': total_summary_duration
                        }
                
                except subprocess.TimeoutExpired:
                    st.error("⏱️ Processing timed out. Try with a shorter video.")
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
                    st.info("💡 Try with clearer audio or adjust settings in sidebar.")
                
                finally:
                    st.session_state.processing = False
    
    else:
        st.info("👈 Upload a video to automatically extract important news")

# Custom CSS
st.markdown("""
<style>
    .stButton > button {
        width: 100%;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: bold;
    }
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
    .stVideo {
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
    <h4>🎯 Automatic News Detection</h4>
    <p>✅ Supports ALL languages (English, Tamil, Hindi, Spanish, etc.)</p>
    <p>✅ AI automatically selects important news segments</p>
    <p>✅ Creates 30-second summaries instantly</p>
    <br>
    <p><strong>🔒 Privacy:</strong> Your videos are processed securely and not stored</p>
    <p style='font-size: 0.8em; margin-top: 20px; color: #888;'>Powered by OpenAI Whisper & GPT-4</p>
</div>
""", unsafe_allow_html=True)