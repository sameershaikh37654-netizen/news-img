import streamlit as st
from test import save_file, process_file

st.set_page_config(page_title="Media → News Generator", layout="centered")

st.title("📰 Media to News Script")
st.write("Upload **audio, video, or image** to generate a news-style narration.")

uploaded_file = st.file_uploader(
    "Upload a file",
    type=["mp3", "wav", "mp4", "mkv", "mov", "png", "jpg", "jpeg", "webp"]
)

if uploaded_file:
    with st.spinner("Processing..."):
        path = save_file(uploaded_file)
        result = process_file(path)

    if "error" in result:
        st.error(result["error"])
    else:
        st.success("Done!")

        if "transcript" in result:
            st.subheader("📄 Transcript")
            st.text_area(
                label="Transcript",
                value=result["transcript"],
                height=150,
                label_visibility="collapsed"
            )
        if "image_meaning" in result:
            st.subheader("🖼 Image Description")
            st.text_area(
                label="Image Description",
                value=result["image_meaning"],
                height=150,
                label_visibility="collapsed"
            )

        st.subheader("📝 News Script")
        st.text_area(
            label="News Script",
            value=result["news_script"],
            height=200,
            label_visibility="collapsed"
        )

        st.subheader("🎧 Generated Audio")
        st.audio(result["audio"])