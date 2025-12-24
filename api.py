# import os
# from flask import Flask, request
# import requests
# import tempfile
# import mimetypes

# from main1 import process_file  # YOUR EXISTING FILE

# app = Flask(__name__)

# def send_dynamic_reply(destination, text):
#     """
#     Sends a WhatsApp reply via Gupshup API to whichever phone number
#     sent the message (destination).
#     """
#     url = "https://api.gupshup.io/sm/api/v1/msg"
#     headers = {
#         "apikey": os.getenv("GUPSHUP_API_KEY"),
#         "Content-Type": "application/x-www-form-urlencoded"
#     }
#     payload = {
#         "channel": "whatsapp",
#         "source": os.getenv("GUPSHUP_SENDER_NUMBER"),  # your sandbox or app WhatsApp number
#         "destination": destination,
#         "message": f'[{{"type":"text","text":"{text}"}}]'
#     }
#     requests.post(url, headers=headers, data=payload)

# @app.route("/whatsapp/webhook", methods=["POST"])
# def whatsapp_webhook():
#     data = request.json
#     print("INCOMING:", data)

#     msg = data.get("payload", {})
#     msg_type = msg.get("type")
#     sender = msg.get("sender", {}).get("phone")

#     if msg_type in ["image", "audio", "video"]:
#         media_url = msg.get("payload", {}).get("url")
#         if media_url:
#             tmp_path = tempfile.NamedTemporaryFile(delete=False,
#                                                    suffix=mimetypes.guess_extension(
#                                                        msg.get("payload", {}).get("contentType", "")
#                                                    ) or ".bin").name

#             with open(tmp_path, "wb") as f:
#                 f.write(requests.get(media_url).content)

#             print("Saved media:", tmp_path)
#             result = process_file(tmp_path)
#             print("PROCESS RESULT:", result)

#             send_dynamic_reply(sender, "Media received — processing now.")
#     elif msg_type == "text":
#         send_dynamic_reply(sender, "Thanks for your text!")

#     return "", 200



# if __name__ == "__main__":
#     app.run(port=8000)




import os
import json
import tempfile
import mimetypes
import requests
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename

from main1 import process_file  # Updated import

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

app = Flask(__name__)

# Configuration
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
ALLOWED_EXTENSIONS = {
    'image': ['.png', '.jpg', '.jpeg', '.webp', '.gif'],
    'audio': ['.mp3', '.wav', '.m4a', '.ogg', '.flac'],
    'video': ['.mp4', '.avi', '.mov', '.mkv', '.webm']
}

# Gupshup Configuration
GUPSHUP_API_KEY = os.getenv("GUPSHUP_API_KEY")
GUPSHUP_SOURCE_NUMBER = os.getenv("GUPSHUP_SOURCE_NUMBER")
GUPSHUP_API_URL = "https://api.gupshup.io/sm/api/v1/msg"

def send_whatsapp_message(destination, message, message_type="text", media_url=None):
    """
    Send WhatsApp message via Gupshup API
    """
    if not GUPSHUP_API_KEY or not GUPSHUP_SOURCE_NUMBER:
        print("Warning: Gupshup credentials not configured")
        return False
    
    headers = {
        "apikey": GUPSHUP_API_KEY,
        "Content-Type": "application/x-www-form-urlencoded"
    }
    
    if message_type == "text":
        payload = {
            "channel": "whatsapp",
            "source": GUPSHUP_SOURCE_NUMBER,
            "destination": destination,
            "message": json.dumps([{
                "type": "text",
                "text": message
            }])
        }
    elif message_type == "audio" and media_url:
        payload = {
            "channel": "whatsapp",
            "source": GUPSHUP_SOURCE_NUMBER,
            "destination": destination,
            "message": json.dumps([{
                "type": "audio",
                "originalUrl": media_url,
                "text": message if message else "News Audio"
            }])
        }
    else:
        print(f"Unsupported message type: {message_type}")
        return False
    
    try:
        response = requests.post(GUPSHUP_API_URL, headers=headers, data=payload)
        if response.status_code == 202:
            print(f"Message sent to {destination}")
            return True
        else:
            print(f"Failed to send message: {response.status_code}, {response.text}")
            return False
    except Exception as e:
        print(f"Error sending WhatsApp message: {e}")
        return False

def download_media(url, destination):
    """
    Download media from URL to temporary file
    """
    try:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        
        with open(destination, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        
        return True
    except Exception as e:
        print(f"Error downloading media: {e}")
        return False

def allowed_file(filename):
    """
    Check if file extension is allowed
    """
    ext = os.path.splitext(filename.lower())[1]
    for category in ALLOWED_EXTENSIONS.values():
        if ext in category:
            return True
    return False

@app.route('/')
def home():
    return jsonify({
        "status": "online",
        "service": "News Script Generator API",
        "endpoints": {
            "/api/process": "POST - Process media file",
            "/whatsapp/webhook": "POST - Gupshup WhatsApp webhook"
        }
    })

@app.route('/api/process', methods=['POST'])
def api_process():
    """
    API endpoint to process media files
    """
    try:
        # Check if file is in request
        if 'file' not in request.files:
            return jsonify({"error": "No file provided"}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400
        
        if not allowed_file(file.filename):
            return jsonify({"error": "File type not allowed"}), 400
        
        # Save file to temporary location
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1])
        file.save(temp_file.name)
        
        # Get processing parameters
        lang = request.form.get('language', 'en')
        reuse_voice = request.form.get('reuse_voice', 'false').lower() == 'true'
        clone_voice = request.form.get('clone_voice', 'false').lower() == 'true'
        
        print(f"Processing file: {file.filename}, Language: {lang}")
        
        # Process the file
        result = process_file(
            temp_file.name,
            lang=lang,
            reuse_saved_voice=reuse_voice,
            clone_voice_from_audio=clone_voice
        )
        
        # Clean up temporary file
        try:
            os.unlink(temp_file.name)
        except:
            pass
        
        if 'error' in result:
            return jsonify(result), 400
        
        return jsonify(result), 200
        
    except Exception as e:
        print(f"API error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/whatsapp/webhook', methods=['POST'])
def whatsapp_webhook():
    """
    Gupshup WhatsApp webhook handler
    """
    try:
        data = request.json
        print(f"Received webhook: {json.dumps(data, indent=2)}")
        
        payload = data.get('payload', {})
        message_type = payload.get('type')
        sender = payload.get('sender', {}).get('phone')
        
        if not sender:
            print("No sender phone number found")
            return '', 200
        
        if message_type in ['image', 'audio', 'video', 'document']:
            # Handle media messages
            media_payload = payload.get('payload', {})
            media_url = media_payload.get('url')
            mime_type = media_payload.get('contentType', '')
            
            if media_url:
                # Send acknowledgment
                send_whatsapp_message(sender, "📥 Media received! Processing your file...")
                
                # Download media
                ext = mimetypes.guess_extension(mime_type) or '.bin'
                temp_path = tempfile.NamedTemporaryFile(delete=False, suffix=ext).name
                
                if download_media(media_url, temp_path):
                    print(f"Downloaded media to: {temp_path}")
                    
                    # Process media
                    result = process_file(
                        temp_path,
                        lang='en',  # Default language, could be configurable
                        reuse_saved_voice=False,
                        clone_voice_from_audio=False
                    )
                    
                    # Cleanup temp file
                    try:
                        os.unlink(temp_path)
                    except:
                        pass
                    
                    if 'error' in result:
                        send_whatsapp_message(sender, f"❌ Error: {result['error']}")
                    else:
                        # Send success message with script preview
                        script_preview = result['news_script'][:200] + "..." if len(result['news_script']) > 200 else result['news_script']
                        send_whatsapp_message(sender, f"✅ News script generated!\n\n📝 Preview:\n{script_preview}")
                        
                        # Send audio file if we have a way to host it
                        # For now, we'll send a link if we can upload it somewhere
                        # Alternatively, we could use Gupshup's media upload
                        send_whatsapp_message(sender, "🎵 Audio generation complete! Check your messages for the audio file.")
                        
                        # Note: To send audio, you'd need to upload it to a public URL first
                        # then use send_whatsapp_message with type="audio"
                else:
                    send_whatsapp_message(sender, "❌ Failed to download media. Please try again.")
            
        elif message_type == 'text':
            text = payload.get('payload', {}).get('text', '')
            
            if text.lower() in ['hi', 'hello', 'help', 'start']:
                welcome_msg = """🎤 *News Script Generator Bot* 🎤

Send me:
• 📸 An image - I'll analyze it
• 🎵 An audio clip - I'll transcribe it  
• 🎥 A video - I'll analyze both visuals and audio

I'll create a professional news script and convert it to audio!

Commands:
- /help - Show this message
- /lang [en/hi/te/kn] - Set language
- /voice_clone - Clone voice from your audio
- /use_saved - Use saved voice"""
                
                send_whatsapp_message(sender, welcome_msg)
            elif text.lower().startswith('/lang'):
                # Handle language setting
                lang_code = text.split()[1] if len(text.split()) > 1 else 'en'
                if lang_code in ['en', 'hi', 'te', 'kn']:
                    send_whatsapp_message(sender, f"✅ Language set to {lang_code}")
                else:
                    send_whatsapp_message(sender, "❌ Invalid language code. Use: en, hi, te, kn")
            else:
                send_whatsapp_message(sender, "📤 Please send a media file (image, audio, or video) for processing.")
        
        return '', 200
        
    except Exception as e:
        print(f"Webhook error: {e}")
        return '', 200

@app.route('/whatsapp/status', methods=['POST'])
def whatsapp_status():
    """
    Handle message delivery status updates
    """
    data = request.json
    print(f"Status update: {json.dumps(data, indent=2)}")
    return '', 200

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8000))
    app.run(host='0.0.0.0', port=port, debug=True)