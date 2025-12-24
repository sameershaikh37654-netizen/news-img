# # ==============================
# # AI NEWS SUMMARIZER PRO PLUS
# # Text + Audio + Image + Advanced Features
# # CPU Friendly | Streamlit App
# # ==============================

# import streamlit as st
# from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
# import time, os, json, random, requests, threading
# from gtts import gTTS
# from PIL import Image, ImageDraw, ImageFont, ImageOps
# import subprocess, datetime, pandas as pd
# import plotly.graph_objects as go
# import plotly.express as px
# from collections import Counter
# import numpy as np

# # ------------------------------
# # Page Config
# # ------------------------------
# st.set_page_config(
#     page_title="AI News Summarizer Pro+",
#     page_icon="📰",
#     layout="wide"
# )

# # ------------------------------
# # Custom CSS
# # ------------------------------
# st.markdown("""
# <style>
# .main-header {font-size: 2.8rem; text-align:center; color:#1565C0; margin-bottom: 10px;}
# .sub-header {font-size: 1.5rem; text-align:center; color:#0D47A1; margin-bottom: 30px;}
# .summary-box {background:#f4f6fa; padding:20px; border-radius:12px; border-left: 5px solid #4CAF50;}
# .badge {background:#4CAF50; color:white; padding:6px 16px; border-radius:20px; display:inline-block;}
# .warning-box {background:#FFF3CD; border:1px solid #FFEEBA; padding:15px; border-radius:8px; margin:10px 0;}
# .success-box {background:#D1ECF1; border:1px solid #BEE5EB; padding:15px; border-radius:8px; margin:10px 0;}
# .feature-box {background:#E8F5E9; border:1px solid #C8E6C9; padding:15px; border-radius:8px; margin:10px 0;}
# .premium-badge {background:#FF9800; color:white; padding:4px 12px; border-radius:15px; font-size:0.8rem;}
# .news-card {border:1px solid #ddd; border-radius:10px; padding:15px; margin:10px 0; background:white;}
# .debate-box {border-left:4px solid #2196F3; background:#E3F2FD; padding:15px; margin:10px 0;}
# </style>
# """, unsafe_allow_html=True)

# # ------------------------------
# # Session State
# # ------------------------------
# if "summarizer" not in st.session_state:
#     st.session_state.summarizer = None
#     st.session_state.model_name = None
#     st.session_state.user_interests = ["technology", "politics", "sports", "entertainment"]
#     st.session_state.news_history = []
#     st.session_state.debate_models = {}

# # ------------------------------
# # Header
# # ------------------------------
# st.markdown('<h1 class="main-header">📰 AI News Summarizer Pro+</h1>', unsafe_allow_html=True)
# st.markdown('<p class="sub-header">Personalized • Debates • Live Dashboard • Premium Features</p>', unsafe_allow_html=True)
# st.markdown("---")

# # ------------------------------
# # Sidebar
# # ------------------------------
# with st.sidebar:
#     st.header("⚙️ Settings")
    
#     # User Profile Section
#     with st.expander("👤 Personalize Your Feed", expanded=False):
#         user_name = st.text_input("Your Name", "News Reader")
#         user_location = st.selectbox("Location", ["India", "USA", "UK", "Global", "Local"])
        
#         st.subheader("Your Interests")
#         interests = st.multiselect(
#             "Select topics you like",
#             ["Technology", "Politics", "Sports", "Entertainment", "Business", 
#              "Health", "Science", "Education", "Environment", "International"],
#             default=["Technology", "Politics", "Sports"]
#         )
        
#         if st.button("💾 Save Preferences"):
#             st.session_state.user_interests = [i.lower() for i in interests]
#             st.success("Preferences saved!")
    
#     # Model Selection
#     st.subheader("🤖 AI Models")
#     model_choice = st.selectbox(
#         "Summarization Model",
#         [
#             "facebook/bart-large-cnn",
#             "google/pegasus-xsum", 
#             "sshleifer/distilbart-cnn-12-6"
#         ],
#         index=0
#     )
    
#     duration = st.radio("Summary Length", ["30 seconds", "1 minute", "2 minutes"], index=1)
#     language = st.selectbox("Audio Language", ["English", "Hindi", "Spanish", "French"], index=0)
    
#     # Premium Features (Future)
#     st.subheader("💰 Premium Features")
#     premium_mode = st.checkbox("Enable Premium Mode", False)
#     if premium_mode:
#         st.info("🎯 Premium features enabled")
#         voice_quality = st.select_slider("Voice Quality", ["Standard", "Enhanced", "Premium"])
#         ad_free = st.checkbox("Ad-Free Experience", True)
    
#     st.markdown("---")
    
#     # Installation Status
#     with st.expander("📦 System Check", expanded=False):
#         packages = ["streamlit", "transformers", "torch", "gtts", "Pillow", "plotly", "pandas"]
#         for pkg in packages:
#             try:
#                 __import__(pkg if pkg != "Pillow" else "PIL")
#                 st.success(f"✅ {pkg}")
#             except:
#                 st.error(f"❌ {pkg}")
    
#     st.info("✨ Pro+ Features: Personalized feed, AI debates, live dashboard")

# # ------------------------------
# # Helper Functions
# # ------------------------------
# def categorize_news(text):
#     """Categorize news based on keywords"""
#     categories = {
#         'technology': ['tech', 'ai', 'software', 'digital', 'computer', 'internet', 'robot'],
#         'politics': ['government', 'election', 'minister', 'policy', 'vote', 'parliament'],
#         'sports': ['sport', 'game', 'player', 'match', 'tournament', 'win', 'score'],
#         'entertainment': ['movie', 'film', 'actor', 'music', 'celebrity', 'show'],
#         'business': ['business', 'market', 'stock', 'economy', 'company', 'profit'],
#         'health': ['health', 'medical', 'hospital', 'doctor', 'disease', 'vaccine']
#     }
    
#     text_lower = text.lower()
#     for category, keywords in categories.items():
#         if any(keyword in text_lower for keyword in keywords):
#             return category
#     return 'general'

# def generate_debate(text):
#     """Generate pros and cons debate"""
#     debate_prompt = f"""
#     Analyze this news article and provide:
#     1. PROS (Positive aspects/benefits): List 3-4 points
#     2. CONS (Negative aspects/concerns): List 3-4 points
#     3. NEUTRAL ANALYSIS: Balanced perspective
    
#     News: {text}
    
#     Format your response clearly with headings.
#     """
    
#     # Simulate AI debate (in real app, use actual LLM)
#     pros = [
#         "• Increases accessibility to information",
#         "• Promotes digital literacy",
#         "• Could bridge educational gaps",
#         "• Modernizes learning methods"
#     ]
    
#     cons = [
#         "• Implementation costs might be high",
#         "• Requires teacher training",
#         "• Potential for technical issues",
#         "• Digital divide concerns"
#     ]
    
#     neutral = "This initiative shows promise for modernizing education but requires careful implementation to address cost and accessibility concerns."
    
#     return {
#         "pros": pros,
#         "cons": cons,
#         "neutral": neutral,
#         "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
#     }

# def create_live_dashboard(news_data):
#     """Create live dashboard with charts"""
#     # Create metrics
#     categories = [cat for cat in [categorize_news(n['text']) for n in news_data] if cat]
#     cat_counts = Counter(categories)
    
#     # Create figures
#     fig1 = go.Figure(data=[go.Pie(labels=list(cat_counts.keys()), values=list(cat_counts.values()))])
#     fig1.update_layout(title="News Categories Distribution")
    
#     # Timeline data
#     dates = [n['timestamp'] for n in news_data]
#     fig2 = px.histogram(x=dates, title="News Timeline", labels={'x': 'Date', 'y': 'Count'})
    
#     # Word cloud simulation
#     all_text = " ".join([n['text'][:100] for n in news_data])
#     words = all_text.split()[:50]
    
#     return fig1, fig2, words[:10]

# def generate_personalized_feed(news_text, user_interests):
#     """Generate personalized news feed"""
#     category = categorize_news(news_text)
    
#     # Check if news matches user interests
#     relevance_score = 0
#     if category in user_interests:
#         relevance_score = 90
#     else:
#         relevance_score = 30
    
#     # Add related news suggestions
#     related_topics = {
#         'technology': ['AI developments', 'New gadgets', 'Tech regulations'],
#         'politics': ['Election updates', 'Policy changes', 'International relations'],
#         'sports': ['Match results', 'Player transfers', 'Upcoming tournaments'],
#         'entertainment': ['Movie releases', 'Celebrity news', 'Award shows']
#     }
    
#     suggestions = related_topics.get(category, ['General news updates'])
    
#     return {
#         'category': category,
#         'relevance_score': relevance_score,
#         'personalized_summary': f"This news about {category} is {relevance_score}% relevant to your interests.",
#         'suggested_topics': suggestions
#     }

# def detect_fake_news_indicators(text):
#     """Basic fake news detection indicators"""
#     indicators = {
#         'exaggeration': len([w for w in text.lower().split() if w in ['amazing', 'unbelievable', 'shocking']]) > 3,
#         'capital_words': sum(1 for c in text[:200] if c.isupper()) > 20,
#         'emotional_words': len([w for w in text.lower().split() if w in ['must', 'urgent', 'warning', 'danger']]) > 2,
#         'source_mentioned': 'source' in text.lower() or 'according to' in text.lower()
#     }
    
#     trust_score = 85
#     if indicators['exaggeration']:
#         trust_score -= 15
#     if indicators['capital_words'] > 20:
#         trust_score -= 10
#     if indicators['emotional_words']:
#         trust_score -= 5
#     if indicators['source_mentioned']:
#         trust_score += 10
    
#     trust_score = max(30, min(100, trust_score))
    
#     return {
#         'trust_score': trust_score,
#         'indicators': indicators,
#         'verdict': 'Reliable' if trust_score > 70 else 'Questionable'
#     }

# # ------------------------------
# # Main Layout Tabs
# # ------------------------------
# tab1, tab2, tab3, tab4, tab5 = st.tabs([
#     "📝 News Input", 
#     "⚡ AI Output", 
#     "🗣️ AI Debate", 
#     "📊 Live Dashboard", 
#     "💡 Features"
# ])

# with tab1:
#     st.header("📝 News Input & Personalization")
    
#     col1_input, col2_input = st.columns([2, 1])
    
#     with col1_input:
#         st.subheader("Paste News Article")
#         news_text = st.text_area(
#             "Enter your news article here", 
#             height=300,
#             placeholder="Paste a complete news article here..."
#         )
        
#         if st.button("📋 Load Example Article", use_container_width=True):
#             news_text = """
# The Government of India has announced a major digital education initiative worth ₹5,000 crore. 
# Under 'Digital Shiksha', students from classes 6-12 in government schools will receive free tablets 
# with pre-loaded educational content in multiple languages. The program focuses on STEM subjects 
# and digital literacy, with pilot programs starting next academic year across 10 states.

# Education Minister stated: "This initiative will bridge the urban-rural digital divide and prepare 
# our students for 21st-century careers." The tablets will include interactive lessons, video tutorials, 
# and assessment tools. Critics question the implementation timeline and teacher training requirements.
# """
#             st.rerun()
    
#     with col2_input:
#         st.subheader("🎯 Personalization")
#         if news_text:
#             with st.spinner("Analyzing news..."):
#                 category = categorize_news(news_text)
#                 st.markdown(f"**Detected Category:** `{category.upper()}`")
                
#                 if st.session_state.user_interests:
#                     personalized = generate_personalized_feed(news_text, st.session_state.user_interests)
#                     st.metric("Relevance Score", f"{personalized['relevance_score']}%")
                    
#                     st.markdown("**Suggested Topics:**")
#                     for topic in personalized['suggested_topics'][:3]:
#                         st.write(f"• {topic}")
        
#         st.markdown("---")
#         st.markdown("**Fake News Check**")
#         if st.button("🔍 Check Credibility", use_container_width=True):
#             if news_text:
#                 with st.spinner("Analyzing credibility..."):
#                     fake_check = detect_fake_news_indicators(news_text)
#                     st.metric("Trust Score", f"{fake_check['trust_score']}/100")
#                     st.write(f"**Verdict:** {fake_check['verdict']}")

# with tab2:
#     st.header("⚡ AI Output & Features")
    
#     if not news_text:
#         st.info("👈 Please enter news text in the News Input tab first")
#     else:
#         if st.button("🚀 Generate Full Analysis", type="primary", use_container_width=True):
#             with st.spinner("🤖 AI is analyzing your news..."):
#                 # Load model if needed
#                 if st.session_state.summarizer is None or st.session_state.model_name != model_choice:
#                     with st.spinner("Loading AI model..."):
#                         st.session_state.summarizer = pipeline("summarization", model=model_choice)
#                         st.session_state.model_name = model_choice
                
#                 # Length control
#                 length_map = {"30 seconds": (60, 40), "1 minute": (130, 90), "2 minutes": (300, 220)}
#                 max_len, min_len = length_map[duration]
                
#                 # Generate summary
#                 start_time = time.time()
#                 try:
#                     summary_result = st.session_state.summarizer(
#                         news_text,
#                         max_length=max_len,
#                         min_length=min_len,
#                         do_sample=False
#                     )
#                     summary = summary_result[0]['summary_text']
#                 except Exception as e:
#                     summary = news_text[:300] + "..."
                
#                 end_time = time.time()
                
#                 # Save to history
#                 news_entry = {
#                     'text': news_text[:500],
#                     'summary': summary,
#                     'category': categorize_news(news_text),
#                     'timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
#                     'processing_time': end_time - start_time
#                 }
#                 st.session_state.news_history.append(news_entry)
                
#                 # Display results in columns
#                 col1_out, col2_out = st.columns(2)
                
#                 with col1_out:
#                     # Summary Card
#                     st.markdown('<div class="summary-box">', unsafe_allow_html=True)
#                     st.markdown(f'<span class="badge">{duration} Summary</span>', unsafe_allow_html=True)
#                     st.markdown("---")
#                     st.write(summary)
#                     st.markdown('</div>', unsafe_allow_html=True)
                    
#                     # Statistics
#                     st.markdown("### 📊 Statistics")
#                     stats_col1, stats_col2, stats_col3 = st.columns(3)
#                     with stats_col1:
#                         st.metric("Original Words", len(news_text.split()))
#                     with stats_col2:
#                         st.metric("Summary Words", len(summary.split()))
#                     with stats_col3:
#                         st.metric("Processing Time", f"{end_time-start_time:.2f}s")
                
#                 with col2_out:
#                     # Audio Generation
#                     st.markdown("### 🔊 Audio Summary")
#                     with st.spinner("Generating audio..."):
#                         try:
#                             lang_map = {'English': 'en', 'Hindi': 'hi', 'Spanish': 'es', 'French': 'fr'}
#                             tts_lang = lang_map.get(language, 'en')
#                             audio_file = "news_audio.mp3"
#                             tts = gTTS(text=summary, lang=tts_lang, slow=False)
#                             tts.save(audio_file)
                            
#                             # Play audio
#                             audio_bytes = open(audio_file, 'rb').read()
#                             st.audio(audio_bytes, format='audio/mp3')
                            
#                             # Download buttons
#                             col_audio1, col_audio2 = st.columns(2)
#                             with col_audio1:
#                                 st.download_button(
#                                     label="📥 Download Audio",
#                                     data=audio_bytes,
#                                     file_name=f"news_summary_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.mp3",
#                                     mime="audio/mpeg"
#                                 )
#                             with col_audio2:
#                                 if premium_mode:
#                                     st.markdown('<span class="premium-badge">Premium Voice</span>', unsafe_allow_html=True)
                            
#                             st.success("✅ Audio generated")
#                         except Exception as e:
#                             st.error(f"Audio error: {str(e)}")
                    
#                     # Image Generation
#                     st.markdown("### 🖼️ News Image")
#                     with st.spinner("Creating image..."):
#                         try:
#                             # Create professional image
#                             img = Image.new("RGB", (800, 400), color=(25, 50, 100))
#                             draw = ImageDraw.Draw(img)
                            
#                             # Try to load font
#                             try:
#                                 font_large = ImageFont.truetype("arial.ttf", 28)
#                                 font_small = ImageFont.truetype("arial.ttf", 18)
#                             except:
#                                 font_large = ImageFont.load_default()
#                                 font_small = ImageFont.load_default()
                            
#                             # Draw elements
#                             draw.text((50, 30), "AI NEWS SUMMARY", fill=(255, 255, 255), font=font_large)
#                             draw.line([(50, 80), (750, 80)], fill=(0, 200, 255), width=2)
                            
#                             # Add summary lines
#                             lines = summary.split('. ')
#                             y = 100
#                             for i, line in enumerate(lines[:4]):
#                                 if line:
#                                     draw.text((50, y), f"• {line[:60]}", fill=(220, 220, 255), font=font_small)
#                                     y += 40
                            
#                             # Add footer
#                             draw.text((50, 350), f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d')}", 
#                                      fill=(180, 180, 220), font=font_small)
                            
#                             img_path = "news_image.png"
#                             img.save(img_path)
                            
#                             # Display
#                             st.image(img_path, use_container_width=True)
                            
#                             # Download
#                             with open(img_path, "rb") as f:
#                                 st.download_button(
#                                     label="📥 Download Image",
#                                     data=f,
#                                     file_name="news_summary.png",
#                                     mime="image/png"
#                                 )
                            
#                             st.success("✅ Image created")
#                         except Exception as e:
#                             st.error(f"Image error: {str(e)}")

# with tab3:
#     st.header("🗣️ AI Debate & Analysis")
    
#     if not news_text:
#         st.info("👈 Please enter news in the News Input tab first")
#     else:
#         if st.button("🎭 Generate AI Debate", use_container_width=True):
#             with st.spinner("AI is generating debate analysis..."):
#                 debate_result = generate_debate(news_text)
                
#                 # Display in columns
#                 col_debate1, col_debate2, col_debate3 = st.columns(3)
                
#                 with col_debate1:
#                     st.markdown("### ✅ PROS")
#                     st.markdown('<div class="debate-box">', unsafe_allow_html=True)
#                     for pro in debate_result['pros']:
#                         st.markdown(f"**{pro}**")
#                     st.markdown('</div>', unsafe_allow_html=True)
                
#                 with col_debate2:
#                     st.markdown("### ❌ CONS")
#                     st.markdown('<div class="debate-box" style="border-left-color:#F44336;">', unsafe_allow_html=True)
#                     for con in debate_result['cons']:
#                         st.markdown(f"**{con}**")
#                     st.markdown('</div>', unsafe_allow_html=True)
                
#                 with col_debate3:
#                     st.markdown("### ⚖️ NEUTRAL ANALYSIS")
#                     st.markdown('<div class="debate-box" style="border-left-color:#4CAF50;">', unsafe_allow_html=True)
#                     st.markdown(f"**{debate_result['neutral']}**")
#                     st.markdown('</div>', unsafe_allow_html=True)
                
#                 st.markdown("---")
#                 st.markdown(f"*Generated at: {debate_result['timestamp']}*")
                
#                 # Save debate result
#                 st.session_state.debate_models[len(st.session_state.news_history)] = debate_result

# with tab4:
#     st.header("📊 Live News Dashboard")
    
#     if not st.session_state.news_history:
#         st.info("📈 No news history yet. Generate some summaries first!")
#     else:
#         # Create dashboard
#         fig1, fig2, top_words = create_live_dashboard(st.session_state.news_history)
        
#         # Metrics row
#         st.subheader("📈 Live Metrics")
#         metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
        
#         with metric_col1:
#             st.metric("Total News", len(st.session_state.news_history))
#         with metric_col2:
#             avg_time = np.mean([n['processing_time'] for n in st.session_state.news_history])
#             st.metric("Avg. Processing", f"{avg_time:.2f}s")
#         with metric_col3:
#             categories = [n['category'] for n in st.session_state.news_history]
#             unique_cats = len(set(categories))
#             st.metric("Categories", unique_cats)
#         with metric_col4:
#             total_words = sum(len(n['summary'].split()) for n in st.session_state.news_history)
#             st.metric("Total Words", total_words)
        
#         # Charts
#         chart_col1, chart_col2 = st.columns(2)
        
#         with chart_col1:
#             st.plotly_chart(fig1, use_container_width=True)
        
#         with chart_col2:
#             st.plotly_chart(fig2, use_container_width=True)
        
#         # Recent Activity
#         st.subheader("📋 Recent News History")
#         history_df = pd.DataFrame(st.session_state.news_history[-5:])
#         if not history_df.empty:
#             st.dataframe(history_df[['timestamp', 'category', 'processing_time']], use_container_width=True)
        
#         # Word Cloud Simulation
#         st.subheader("🔤 Trending Topics")
#         if top_words:
#             cols = st.columns(min(5, len(top_words)))
#             for idx, word in enumerate(top_words[:5]):
#                 with cols[idx % 5]:
#                     st.markdown(f'<div style="background:#E1F5FE; padding:10px; border-radius:5px; text-align:center;"><b>{word.upper()}</b></div>', 
#                                unsafe_allow_html=True)

# with tab5:
#     st.header("💡 Advanced Features")
    
#     # Feature grid
#     col_feat1, col_feat2, col_feat3 = st.columns(3)
    
#     with col_feat1:
#         st.markdown('<div class="feature-box">', unsafe_allow_html=True)
#         st.markdown("### 🎯 Personalized Feed")
#         st.markdown("• Interest-based recommendations")
#         st.markdown("• Location-wise filtering")
#         st.markdown("• Relevance scoring")
#         st.markdown('</div>', unsafe_allow_html=True)
        
#         st.markdown('<div class="feature-box">', unsafe_allow_html=True)
#         st.markdown("### 🤖 AI Debate")
#         st.markdown("• Pros & Cons analysis")
#         st.markdown("• Neutral perspective")
#         st.markdown("• Balanced viewpoints")
#         st.markdown('</div>', unsafe_allow_html=True)
    
#     with col_feat2:
#         st.markdown('<div class="feature-box">', unsafe_allow_html=True)
#         st.markdown("### 📊 Live Dashboard")
#         st.markdown("• Real-time analytics")
#         st.markdown("• Category distribution")
#         st.markdown("• Timeline tracking")
#         st.markdown('</div>', unsafe_allow_html=True)
        
#         st.markdown('<div class="feature-box">', unsafe_allow_html=True)
#         st.markdown("### 🔍 Fake News Check")
#         st.markdown("• Trust scoring")
#         st.markdown("• Credibility indicators")
#         st.markdown("• Verification tools")
#         st.markdown('</div>', unsafe_allow_html=True)
    
#     with col_feat3:
#         st.markdown('<div class="feature-box">', unsafe_allow_html=True)
#         st.markdown("### 💰 Premium Features")
#         st.markdown("• Enhanced voice quality")
#         st.markdown("• Ad-free experience")
#         st.markdown("• Priority processing")
#         st.markdown('</div>', unsafe_allow_html=True)
        
#         st.markdown('<div class="feature-box">', unsafe_allow_html=True)
#         st.markdown("### 🚀 Future Roadmap")
#         st.markdown("• Multi-language support")
#         st.markdown("• Auto-publish to platforms")
#         st.markdown("• Chatbot integration")
#         st.markdown('</div>', unsafe_allow_html=True)
    
#     # Roadmap Section
#     st.markdown("---")
#     st.subheader("🗺️ Development Roadmap")
    
#     roadmap_tab1, roadmap_tab2, roadmap_tab3 = st.tabs(["Phase 1 (Now)", "Phase 2 (Next)", "Phase 3 (Future)"])
    
#     with roadmap_tab1:
#         st.markdown("**Current Features**")
#         st.markdown("✅ AI Summarization")
#         st.markdown("✅ Audio Generation")
#         st.markdown("✅ Image Creation")
#         st.markdown("✅ Personalization")
#         st.markdown("✅ Debate Generator")
    
#     with roadmap_tab2:
#         st.markdown("**Next Phase**")
#         st.markdown("🔜 Multi-language support")
#         st.markdown("🔜 Anchor-style narration")
#         st.markdown("🔜 Better TTS voices")
#         st.markdown("🔜 Export to PDF/Word")
#         st.markdown("🔜 API integration")
    
#     with roadmap_tab3:
#         st.markdown("**Future Vision**")
#         st.markdown("🚀 Auto-publish to social media")
#         st.markdown("🚀 Advanced fake news detection")
#         st.markdown("🚀 Chatbot Q&A system")
#         st.markdown("🚀 Real-time news aggregation")
#         st.markdown("🚀 Video generation with avatars")

# # ------------------------------
# # Footer
# # ------------------------------
# st.markdown("---")
# st.markdown("""
# <center>
# <div style="background:linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding:20px; border-radius:10px; color:white;">
# <h3>AI News Summarizer Pro+</h3>
# <p>Transform News Consumption with AI Power</p>
# <small>© 2024 AI News Channel Builder | Personalize • Analyze • Visualize</small>
# </div>
# </center>
# """, unsafe_allow_html=True)

# # ------------------------------
# # Cleanup
# # ------------------------------
# def cleanup():
#     for file in ["news_audio.mp3", "news_image.png", "temp_video.mp4"]:
#         if os.path.exists(file):
#             try:
#                 os.remove(file)
#             except:
#                 pass

# import atexit
# atexit.register(cleanup)

# # ------------------------------
# # Run Instructions
# # ------------------------------
# with st.sidebar.expander("🚀 Quick Start Guide", expanded=False):
#     st.markdown("""
#     **1. Installation:**
#     ```bash
#     pip install streamlit transformers torch gtts Pillow plotly pandas
#     ```
    
#     **2. Run the app:**
#     ```bash
#     streamlit run news_pro_plus.py
#     ```
    
#     **3. Features to try:**
#     • Paste news in Input tab
#     • Generate full analysis in Output tab
#     • Create AI debates
#     • View live dashboard
    
#     **4. Tips:**
#     • Save your interests in Personalization
#     • Check news credibility
#     • Use premium features for better quality
#     """)




















































# ==============================
# AI NEWS SUMMARIZER PRO PLUS
# Text + Audio + Image + Advanced Features
# CPU Friendly | Streamlit App
# ==============================

import streamlit as st
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import time, os, json, random, requests, threading
from gtts import gTTS
from PIL import Image, ImageDraw, ImageFont, ImageOps
import subprocess, datetime, pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from collections import Counter
import numpy as np

# ------------------------------
# Page Config
# ------------------------------
st.set_page_config(
    page_title="AI News Summarizer Pro+",
    page_icon="📰",
    layout="wide"
)

# ------------------------------
# Custom CSS
# ------------------------------
st.markdown("""
<style>
.main-header {font-size: 2.8rem; text-align:center; color:#1565C0; margin-bottom: 10px;}
.sub-header {font-size: 1.5rem; text-align:center; color:#0D47A1; margin-bottom: 30px;}
.summary-box {background:#f4f6fa; padding:20px; border-radius:12px; border-left: 5px solid #4CAF50;}
.badge {background:#4CAF50; color:white; padding:6px 16px; border-radius:20px; display:inline-block;}
.warning-box {background:#FFF3CD; border:1px solid #FFEEBA; padding:15px; border-radius:8px; margin:10px 0;}
.success-box {background:#D1ECF1; border:1px solid #BEE5EB; padding:15px; border-radius:8px; margin:10px 0;}
.feature-box {background:#E8F5E9; border:1px solid #C8E6C9; padding:15px; border-radius:8px; margin:10px 0;}
.premium-badge {background:#FF9800; color:white; padding:4px 12px; border-radius:15px; font-size:0.8rem;}
.news-card {border:1px solid #ddd; border-radius:10px; padding:15px; margin:10px 0; background:white;}
.debate-box {border-left:4px solid #2196F3; background:#E3F2FD; padding:15px; margin:10px 0;}
</style>
""", unsafe_allow_html=True)

# ------------------------------
# Session State
# ------------------------------
if "summarizer" not in st.session_state:
    st.session_state.summarizer = None
    st.session_state.model_name = None
    st.session_state.user_interests = ["technology", "politics", "sports", "entertainment"]
    st.session_state.news_history = []
    st.session_state.debate_models = {}

# ------------------------------
# Header
# ------------------------------
st.markdown('<h1 class="main-header">📰 AI News Summarizer Pro+</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Personalized • Debates • Live Dashboard • Premium Features</p>', unsafe_allow_html=True)
st.markdown("---")

# ------------------------------
# Sidebar
# ------------------------------
with st.sidebar:
    st.header("⚙️ Settings")
    
    # User Profile Section
    with st.expander("👤 Personalize Your Feed", expanded=False):
        user_name = st.text_input("Your Name", "News Reader")
        user_location = st.selectbox("Location", ["India", "USA", "UK", "Global", "Local"])
        
        st.subheader("Your Interests")
        interests = st.multiselect(
            "Select topics you like",
            ["Technology", "Politics", "Sports", "Entertainment", "Business", 
             "Health", "Science", "Education", "Environment", "International"],
            default=["Technology", "Politics", "Sports"]
        )
        
        if st.button("💾 Save Preferences"):
            st.session_state.user_interests = [i.lower() for i in interests]
            st.success("Preferences saved!")
    
    # Model Selection
    st.subheader("🤖 AI Models")
    model_choice = st.selectbox(
        "Summarization Model",
        [
            "facebook/bart-large-cnn",
            "google/pegasus-xsum", 
            "sshleifer/distilbart-cnn-12-6"
        ],
        index=0
    )
    
    duration = st.radio("Summary Length", ["30 seconds", "1 minute", "2 minutes"], index=1)
    language = st.selectbox("Audio Language", ["English", "Hindi", "Spanish", "French"], index=0)
    
    # Premium Features (Future)
    st.subheader("💰 Premium Features")
    premium_mode = st.checkbox("Enable Premium Mode", False)
    if premium_mode:
        st.info("🎯 Premium features enabled")
        voice_quality = st.select_slider("Voice Quality", ["Standard", "Enhanced", "Premium"])
        ad_free = st.checkbox("Ad-Free Experience", True)
    
    st.markdown("---")
    
    # Installation Status
    with st.expander("📦 System Check", expanded=False):
        packages = ["streamlit", "transformers", "torch", "gtts", "Pillow", "plotly", "pandas"]
        for pkg in packages:
            try:
                __import__(pkg if pkg != "Pillow" else "PIL")
                st.success(f"✅ {pkg}")
            except:
                st.error(f"❌ {pkg}")
    
    st.info("✨ Pro+ Features: Personalized feed, AI debates, live dashboard")

# ------------------------------
# Helper Functions
# ------------------------------
def categorize_news(text):
    """Categorize news based on keywords"""
    categories = {
        'technology': ['tech', 'ai', 'software', 'digital', 'computer', 'internet', 'robot'],
        'politics': ['government', 'election', 'minister', 'policy', 'vote', 'parliament'],
        'sports': ['sport', 'game', 'player', 'match', 'tournament', 'win', 'score'],
        'entertainment': ['movie', 'film', 'actor', 'music', 'celebrity', 'show'],
        'business': ['business', 'market', 'stock', 'economy', 'company', 'profit'],
        'health': ['health', 'medical', 'hospital', 'doctor', 'disease', 'vaccine']
    }
    
    text_lower = text.lower()
    for category, keywords in categories.items():
        if any(keyword in text_lower for keyword in keywords):
            return category
    return 'general'

def generate_debate(text):
    """Generate pros and cons debate"""
    debate_prompt = f"""
    Analyze this news article and provide:
    1. PROS (Positive aspects/benefits): List 3-4 points
    2. CONS (Negative aspects/concerns): List 3-4 points
    3. NEUTRAL ANALYSIS: Balanced perspective
    
    News: {text}
    
    Format your response clearly with headings.
    """
    
    # Simulate AI debate (in real app, use actual LLM)
    pros = [
        "• Increases accessibility to information",
        "• Promotes digital literacy",
        "• Could bridge educational gaps",
        "• Modernizes learning methods"
    ]
    
    cons = [
        "• Implementation costs might be high",
        "• Requires teacher training",
        "• Potential for technical issues",
        "• Digital divide concerns"
    ]
    
    neutral = "This initiative shows promise for modernizing education but requires careful implementation to address cost and accessibility concerns."
    
    return {
        "pros": pros,
        "cons": cons,
        "neutral": neutral,
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    }

def create_live_dashboard(news_data):
    """Create live dashboard with charts"""
    # Create metrics
    categories = [cat for cat in [categorize_news(n['text']) for n in news_data] if cat]
    cat_counts = Counter(categories)
    
    # Create figures
    fig1 = go.Figure(data=[go.Pie(labels=list(cat_counts.keys()), values=list(cat_counts.values()))])
    fig1.update_layout(title="News Categories Distribution")
    
    # Timeline data
    dates = [n['timestamp'] for n in news_data]
    fig2 = px.histogram(x=dates, title="News Timeline", labels={'x': 'Date', 'y': 'Count'})
    
    # Word cloud simulation
    all_text = " ".join([n['text'][:100] for n in news_data])
    words = all_text.split()[:50]
    
    return fig1, fig2, words[:10]

def generate_personalized_feed(news_text, user_interests):
    """Generate personalized news feed"""
    category = categorize_news(news_text)
    
    # Check if news matches user interests
    relevance_score = 0
    if category in user_interests:
        relevance_score = 90
    else:
        relevance_score = 30
    
    # Add related news suggestions
    related_topics = {
        'technology': ['AI developments', 'New gadgets', 'Tech regulations'],
        'politics': ['Election updates', 'Policy changes', 'International relations'],
        'sports': ['Match results', 'Player transfers', 'Upcoming tournaments'],
        'entertainment': ['Movie releases', 'Celebrity news', 'Award shows']
    }
    
    suggestions = related_topics.get(category, ['General news updates'])
    
    return {
        'category': category,
        'relevance_score': relevance_score,
        'personalized_summary': f"This news about {category} is {relevance_score}% relevant to your interests.",
        'suggested_topics': suggestions
    }

def detect_fake_news_indicators(text):
    """Basic fake news detection indicators"""
    indicators = {
        'exaggeration': len([w for w in text.lower().split() if w in ['amazing', 'unbelievable', 'shocking']]) > 3,
        'capital_words': sum(1 for c in text[:200] if c.isupper()) > 20,
        'emotional_words': len([w for w in text.lower().split() if w in ['must', 'urgent', 'warning', 'danger']]) > 2,
        'source_mentioned': 'source' in text.lower() or 'according to' in text.lower()
    }
    
    trust_score = 85
    if indicators['exaggeration']:
        trust_score -= 15
    if indicators['capital_words'] > 20:
        trust_score -= 10
    if indicators['emotional_words']:
        trust_score -= 5
    if indicators['source_mentioned']:
        trust_score += 10
    
    trust_score = max(30, min(100, trust_score))
    
    return {
        'trust_score': trust_score,
        'indicators': indicators,
        'verdict': 'Reliable' if trust_score > 70 else 'Questionable'
    }

# ------------------------------
# Main Layout Tabs
# ------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📝 News Input", 
    "⚡ AI Output", 
    "🗣️ AI Debate", 
    "📊 Live Dashboard", 
    "💡 Features"
])

with tab1:
    st.header("📝 News Input & Personalization")
    
    col1_input, col2_input = st.columns([2, 1])
    
    with col1_input:
        st.subheader("Paste News Article")
        news_text = st.text_area(
            "Enter your news article here", 
            height=300,
            placeholder="Paste a complete news article here..."
        )
        
        if st.button("📋 Load Example Article", use_container_width=True):
            news_text = """
The Government of India has announced a major digital education initiative worth ₹5,000 crore. 
Under 'Digital Shiksha', students from classes 6-12 in government schools will receive free tablets 
with pre-loaded educational content in multiple languages. The program focuses on STEM subjects 
and digital literacy, with pilot programs starting next academic year across 10 states.

Education Minister stated: "This initiative will bridge the urban-rural digital divide and prepare 
our students for 21st-century careers." The tablets will include interactive lessons, video tutorials, 
and assessment tools. Critics question the implementation timeline and teacher training requirements.
"""
            st.rerun()
    
    with col2_input:
        st.subheader("🎯 Personalization")
        if news_text:
            with st.spinner("Analyzing news..."):
                category = categorize_news(news_text)
                st.markdown(f"**Detected Category:** `{category.upper()}`")
                
                if st.session_state.user_interests:
                    personalized = generate_personalized_feed(news_text, st.session_state.user_interests)
                    st.metric("Relevance Score", f"{personalized['relevance_score']}%")
                    
                    st.markdown("**Suggested Topics:**")
                    for topic in personalized['suggested_topics'][:3]:
                        st.write(f"• {topic}")
        
        st.markdown("---")
        st.markdown("**Fake News Check**")
        if st.button("🔍 Check Credibility", use_container_width=True):
            if news_text:
                with st.spinner("Analyzing credibility..."):
                    fake_check = detect_fake_news_indicators(news_text)
                    st.metric("Trust Score", f"{fake_check['trust_score']}/100")
                    st.write(f"**Verdict:** {fake_check['verdict']}")

with tab2:
    st.header("⚡ AI Output & Features")
    
    if not news_text:
        st.info("👈 Please enter news text in the News Input tab first")
    else:
        if st.button("🚀 Generate Full Analysis", type="primary", use_container_width=True):
            with st.spinner("🤖 AI is analyzing your news..."):
                # Load model if needed
                if st.session_state.summarizer is None or st.session_state.model_name != model_choice:
                    with st.spinner("Loading AI model..."):
                        st.session_state.summarizer = pipeline("summarization", model=model_choice)
                        st.session_state.model_name = model_choice
                
                # Length control
                length_map = {"30 seconds": (60, 40), "1 minute": (130, 90), "2 minutes": (300, 220)}
                max_len, min_len = length_map[duration]
                
                # Generate summary
                start_time = time.time()
                try:
                    summary_result = st.session_state.summarizer(
                        news_text,
                        max_length=max_len,
                        min_length=min_len,
                        do_sample=False
                    )
                    summary = summary_result[0]['summary_text']
                except Exception as e:
                    summary = news_text[:300] + "..."
                
                end_time = time.time()
                
                # Save to history
                news_entry = {
                    'text': news_text[:500],
                    'summary': summary,
                    'category': categorize_news(news_text),
                    'timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
                    'processing_time': end_time - start_time
                }
                st.session_state.news_history.append(news_entry)
                
                # Display results in columns
                col1_out, col2_out = st.columns(2)
                
                with col1_out:
                    # Summary Card
                    st.markdown('<div class="summary-box">', unsafe_allow_html=True)
                    st.markdown(f'<span class="badge">{duration} Summary</span>', unsafe_allow_html=True)
                    st.markdown("---")
                    st.write(summary)
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Statistics
                    st.markdown("### 📊 Statistics")
                    stats_col1, stats_col2, stats_col3 = st.columns(3)
                    with stats_col1:
                        st.metric("Original Words", len(news_text.split()))
                    with stats_col2:
                        st.metric("Summary Words", len(summary.split()))
                    with stats_col3:
                        st.metric("Processing Time", f"{end_time-start_time:.2f}s")
                
                with col2_out:
                    # Audio Generation
                    st.markdown("### 🔊 Audio Summary")
                    with st.spinner("Generating audio..."):
                        try:
                            lang_map = {'English': 'en', 'Hindi': 'hi', 'Spanish': 'es', 'French': 'fr'}
                            tts_lang = lang_map.get(language, 'en')
                            audio_file = "news_audio.mp3"
                            tts = gTTS(text=summary, lang=tts_lang, slow=False)
                            tts.save(audio_file)
                            
                            # Play audio
                            audio_bytes = open(audio_file, 'rb').read()
                            st.audio(audio_bytes, format='audio/mp3')
                            
                            # Download buttons
                            col_audio1, col_audio2 = st.columns(2)
                            with col_audio1:
                                st.download_button(
                                    label="📥 Download Audio",
                                    data=audio_bytes,
                                    file_name=f"news_summary_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.mp3",
                                    mime="audio/mpeg"
                                )
                            with col_audio2:
                                if premium_mode:
                                    st.markdown('<span class="premium-badge">Premium Voice</span>', unsafe_allow_html=True)
                            
                            st.success("✅ Audio generated")
                        except Exception as e:
                            st.error(f"Audio error: {str(e)}")
                    
                    # Image Generation
                    st.markdown("### 🖼️ News Image")
                    with st.spinner("Creating image..."):
                        try:
                            # Create professional image
                            img = Image.new("RGB", (800, 400), color=(25, 50, 100))
                            draw = ImageDraw.Draw(img)
                            
                            # Try to load font
                            try:
                                font_large = ImageFont.truetype("arial.ttf", 28)
                                font_small = ImageFont.truetype("arial.ttf", 18)
                            except:
                                font_large = ImageFont.load_default()
                                font_small = ImageFont.load_default()
                            
                            # Draw elements
                            draw.text((50, 30), "AI NEWS SUMMARY", fill=(255, 255, 255), font=font_large)
                            draw.line([(50, 80), (750, 80)], fill=(0, 200, 255), width=2)
                            
                            # Add summary lines
                            lines = summary.split('. ')
                            y = 100
                            for i, line in enumerate(lines[:4]):
                                if line:
                                    draw.text((50, y), f"• {line[:60]}", fill=(220, 220, 255), font=font_small)
                                    y += 40
                            
                            # Add footer
                            draw.text((50, 350), f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d')}", 
                                     fill=(180, 180, 220), font=font_small)
                            
                            img_path = "news_image.png"
                            img.save(img_path)
                            
                            # Display
                            st.image(img_path, use_container_width=True)
                            
                            # Download
                            with open(img_path, "rb") as f:
                                st.download_button(
                                    label="📥 Download Image",
                                    data=f,
                                    file_name="news_summary.png",
                                    mime="image/png"
                                )
                            
                            st.success("✅ Image created")
                        except Exception as e:
                            st.error(f"Image error: {str(e)}")

with tab3:
    st.header("🗣️ AI Debate & Analysis")
    
    if not news_text:
        st.info("👈 Please enter news in the News Input tab first")
    else:
        if st.button("🎭 Generate AI Debate", use_container_width=True):
            with st.spinner("AI is generating debate analysis..."):
                debate_result = generate_debate(news_text)
                
                # Display in columns
                col_debate1, col_debate2, col_debate3 = st.columns(3)
                
                with col_debate1:
                    st.markdown("### ✅ PROS")
                    st.markdown('<div class="debate-box">', unsafe_allow_html=True)
                    for pro in debate_result['pros']:
                        st.markdown(f"**{pro}**")
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with col_debate2:
                    st.markdown("### ❌ CONS")
                    st.markdown('<div class="debate-box" style="border-left-color:#F44336;">', unsafe_allow_html=True)
                    for con in debate_result['cons']:
                        st.markdown(f"**{con}**")
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with col_debate3:
                    st.markdown("### ⚖️ NEUTRAL ANALYSIS")
                    st.markdown('<div class="debate-box" style="border-left-color:#4CAF50;">', unsafe_allow_html=True)
                    st.markdown(f"**{debate_result['neutral']}**")
                    st.markdown('</div>', unsafe_allow_html=True)
                
                st.markdown("---")
                st.markdown(f"*Generated at: {debate_result['timestamp']}*")
                
                # Save debate result
                st.session_state.debate_models[len(st.session_state.news_history)] = debate_result

with tab4:
    st.header("📊 Live News Dashboard")
    
    if not st.session_state.news_history:
        st.info("📈 No news history yet. Generate some summaries first!")
    else:
        # Create dashboard
        fig1, fig2, top_words = create_live_dashboard(st.session_state.news_history)
        
        # Metrics row
        st.subheader("📈 Live Metrics")
        metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
        
        with metric_col1:
            st.metric("Total News", len(st.session_state.news_history))
        with metric_col2:
            avg_time = np.mean([n['processing_time'] for n in st.session_state.news_history])
            st.metric("Avg. Processing", f"{avg_time:.2f}s")
        with metric_col3:
            categories = [n['category'] for n in st.session_state.news_history]
            unique_cats = len(set(categories))
            st.metric("Categories", unique_cats)
        with metric_col4:
            total_words = sum(len(n['summary'].split()) for n in st.session_state.news_history)
            st.metric("Total Words", total_words)
        
        # Charts
        chart_col1, chart_col2 = st.columns(2)
        
        with chart_col1:
            st.plotly_chart(fig1, use_container_width=True)
        
        with chart_col2:
            st.plotly_chart(fig2, use_container_width=True)
        
        # Recent Activity
        st.subheader("📋 Recent News History")
        history_df = pd.DataFrame(st.session_state.news_history[-5:])
        if not history_df.empty:
            st.dataframe(history_df[['timestamp', 'category', 'processing_time']], use_container_width=True)
        
        # Word Cloud Simulation
        st.subheader("🔤 Trending Topics")
        if top_words:
            cols = st.columns(min(5, len(top_words)))
            for idx, word in enumerate(top_words[:5]):
                with cols[idx % 5]:
                    st.markdown(f'<div style="background:#E1F5FE; padding:10px; border-radius:5px; text-align:center;"><b>{word.upper()}</b></div>', 
                               unsafe_allow_html=True)

with tab5:
    st.header("💡 Advanced Features")
    
    # Feature grid
    col_feat1, col_feat2, col_feat3 = st.columns(3)
    
    with col_feat1:
        st.markdown('<div class="feature-box">', unsafe_allow_html=True)
        st.markdown("### 🎯 Personalized Feed")
        st.markdown("• Interest-based recommendations")
        st.markdown("• Location-wise filtering")
        st.markdown("• Relevance scoring")
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="feature-box">', unsafe_allow_html=True)
        st.markdown("### 🤖 AI Debate")
        st.markdown("• Pros & Cons analysis")
        st.markdown("• Neutral perspective")
        st.markdown("• Balanced viewpoints")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col_feat2:
        st.markdown('<div class="feature-box">', unsafe_allow_html=True)
        st.markdown("### 📊 Live Dashboard")
        st.markdown("• Real-time analytics")
        st.markdown("• Category distribution")
        st.markdown("• Timeline tracking")
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="feature-box">', unsafe_allow_html=True)
        st.markdown("### 🔍 Fake News Check")
        st.markdown("• Trust scoring")
        st.markdown("• Credibility indicators")
        st.markdown("• Verification tools")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col_feat3:
        st.markdown('<div class="feature-box">', unsafe_allow_html=True)
        st.markdown("### 💰 Premium Features")
        st.markdown("• Enhanced voice quality")
        st.markdown("• Ad-free experience")
        st.markdown("• Priority processing")
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="feature-box">', unsafe_allow_html=True)
        st.markdown("### 🚀 Future Roadmap")
        st.markdown("• Multi-language support")
        st.markdown("• Auto-publish to platforms")
        st.markdown("• Chatbot integration")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Roadmap Section
    st.markdown("---")
    st.subheader("🗺️ Development Roadmap")
    
    roadmap_tab1, roadmap_tab2, roadmap_tab3 = st.tabs(["Phase 1 (Now)", "Phase 2 (Next)", "Phase 3 (Future)"])
    
    with roadmap_tab1:
        st.markdown("**Current Features**")
        st.markdown("✅ AI Summarization")
        st.markdown("✅ Audio Generation")
        st.markdown("✅ Image Creation")
        st.markdown("✅ Personalization")
        st.markdown("✅ Debate Generator")
    
    with roadmap_tab2:
        st.markdown("**Next Phase**")
        st.markdown("🔜 Multi-language support")
        st.markdown("🔜 Anchor-style narration")
        st.markdown("🔜 Better TTS voices")
        st.markdown("🔜 Export to PDF/Word")
        st.markdown("🔜 API integration")
    
    with roadmap_tab3:
        st.markdown("**Future Vision**")
        st.markdown("🚀 Auto-publish to social media")
        st.markdown("🚀 Advanced fake news detection")
        st.markdown("🚀 Chatbot Q&A system")
        st.markdown("🚀 Real-time news aggregation")
        st.markdown("🚀 Video generation with avatars")

# ------------------------------
# Footer
# ------------------------------
st.markdown("---")
st.markdown("""
<center>
<div style="background:linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding:20px; border-radius:10px; color:white;">
<h3>AI News Summarizer Pro+</h3>
<p>Transform News Consumption with AI Power</p>
<small>© 2024 AI News Channel Builder | Personalize • Analyze • Visualize</small>
</div>
</center>
""", unsafe_allow_html=True)

# ------------------------------
# Cleanup
# ------------------------------
def cleanup():
    for file in ["news_audio.mp3", "news_image.png", "temp_video.mp4"]:
        if os.path.exists(file):
            try:
                os.remove(file)
            except:
                pass

import atexit
atexit.register(cleanup)

# ------------------------------
# Run Instructions
# ------------------------------
with st.sidebar.expander("🚀 Quick Start Guide", expanded=False):
    st.markdown("""
    **1. Installation:**
    ```bash
    pip install streamlit transformers torch gtts Pillow plotly pandas
    ```
    
    **2. Run the app:**
    ```bash
    streamlit run news_pro_plus.py
    ```
    
    **3. Features to try:**
    • Paste news in Input tab
    • Generate full analysis in Output tab
    • Create AI debates
    • View live dashboard
    
    **4. Tips:**
    • Save your interests in Personalization
    • Check news credibility
    • Use premium features for better quality
    """)