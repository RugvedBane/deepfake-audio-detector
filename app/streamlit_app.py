import streamlit as st
import requests

st.markdown("""
<style>
/* Dark gradient background */
.stApp {
    background: linear-gradient(135deg, #0f0f1a 0%, #1a1a2e 50%, #16213e 100%);
}

/* Hide default Streamlit elements */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* Custom title styling */
h1 {
    background: linear-gradient(90deg, #667eea, #764ba2);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 3rem !important;
    font-weight: 800 !important;
}

/* Button styling */
.stButton > button {
    background: linear-gradient(90deg, #667eea, #764ba2);
    color: white;
    border: none;
    border-radius: 25px;
    padding: 0.5rem 2rem;
    font-weight: bold;
    width: 100%;
}

/* Input method tabs instead of dropdown */
.stSelectbox {
    display: none;
}
</style>
""", unsafe_allow_html=True)

st.set_page_config(page_title="Deepfake Audio Detector", page_icon="🎙️")
st.title("🎙️ Deepfake Audio Detector")
st.caption("Detect AI-generated fake audio using deep learning")

# Input method selector
input_method = st.radio(
    "Choose input method",
    ["Record Live", "Upload Audio File", "YouTube URL"],
    horizontal=True
)

if input_method == "Upload Audio File":
    file = st.file_uploader("Upload audio file", type=["wav", "mp3", "flac"])
    
    if file and st.button("Analyze"):
        with st.spinner("Analyzing audio..."):
            response = requests.post(
                "http://localhost:8000/predict",
                files={"file": file},
                params={"input_method": "upload"}
            )
            result = response.json()
        
        if result["prediction"] == "fake":
            st.error(f"⚠️ FAKE — {result['confidence']}% confidence")
        else:
            st.success(f"✅ REAL — {result['confidence']}% confidence")
        

        if "explanation" in result:
            st.subheader(f"🔍 Why we think it's {result['prediction']}:")
            st.write(result["explanation"])
        

elif input_method == "Record Live":
    audio = st.audio_input("Record your audio")
    
    if audio and st.button("Analyze"):
        with st.spinner("Analyzing..."):
            response = requests.post(
                "http://localhost:8000/predict",
                files={"file": audio},
                params={"input_method": "record"}
            )
            result = response.json()
            
        
        if result["prediction"] == "fake":
            st.error(f"⚠️ FAKE — {result['confidence']}% confidence")
        else:
            st.success(f"✅ REAL — {result['confidence']}% confidence")

        if "explanation" in result:
            st.subheader(f"🔍 Why we think it's {result['prediction']}:")
            st.write(result["explanation"])

elif input_method == "YouTube URL":
    url = st.text_input("Paste YouTube URL")
    
    if url and st.button("Analyze"):
        with st.spinner("Extracting and analyzing audio..."):
            response = requests.post(
                "http://localhost:8000/predict-youtube",
                params={"url": url}
            )
            result = response.json()
        
        if result["prediction"] == "fake":
            st.error(f"⚠️ FAKE — {result['confidence']}% confidence")
        else:
            st.success(f"✅ REAL — {result['confidence']}% confidence")

        if "explanation" in result:
            st.subheader(f"🔍 Why we think it's {result['prediction']}:")
            st.write(result["explanation"])
        