import streamlit as st
from deepface import DeepFace
import numpy as np
import cv2
import pandas as pd
import plotly.express as px
from textblob import TextBlob
st.sidebar.header("🧠 Choose Emotion Model")
emotion_model = st.sidebar.selectbox(
    "Select a model for emotion detection",
    options=["RealFace", "EmotionNet", "ExpressNet X1"],
    index=0
)
st.sidebar.markdown(f"**{emotion_model}** model selected for emotion detection.")
st.title("😊 Simple Sentiment & Emotion Detector")
tab = st.radio("Select Input Type:", ["Image Upload", "Webcam (Live Emotion)", "Text Sentiment"])

if tab == "Image Upload":
    st.subheader("Upload a face image")
    uploaded_file = st.file_uploader("Choose an image file (jpg, png)", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        st.image(img_rgb, caption="Uploaded Image", use_container_width=True)
        with st.spinner("Analyzing emotions..."):
            try:
                result = DeepFace.analyze(img_path=img_rgb, actions=["emotion"], enforce_detection=False)[0]
                emotions = result["emotion"]
                df = pd.DataFrame(emotions.items(), columns=["Emotion", "Score"])
                fig = px.bar(df, x="Emotion", y="Score", title="Emotion Scores")
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Error analyzing image: {e}")

elif tab == "Webcam (Live Emotion)":
    st.subheader("Webcam Live Emotion Detection")

    st.info("Detecting emotions live from your webcam. No frames saved or captured.")

    FRAME_WINDOW = st.image([])
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        st.error("⚠️ Could not open webcam.")
    else:
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    st.warning("Failed to grab frame.")
                    break
                # Convert frame to RGB
                img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Analyze emotion on current frame (non-blocking)
                try:
                    result = DeepFace.analyze(img_path=img_rgb, actions=["emotion"], enforce_detection=False)[0]
                    dominant_emotion = result["dominant_emotion"]
                    emotion_scores = result["emotion"]
                except Exception:
                    dominant_emotion = "No face detected"
                    emotion_scores = {}

                # Show webcam frame with dominant emotion text overlay
                cv2.putText(
                    frame,
                    f"Emotion: {dominant_emotion}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )
                FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        except Exception as e:
            st.error(f"Error during webcam processing: {e}")
        finally:
            cap.release()

elif tab == "Text Sentiment":
    st.subheader("Text Sentiment Analysis")
    user_text = st.text_area("Enter text to analyze sentiment", "")
    if st.button("Analyze Text"):
        if user_text.strip() == "":
            st.warning("Please enter some text to analyze.")
        else:
            analysis = TextBlob(user_text)
            polarity = analysis.sentiment.polarity
            subjectivity = analysis.sentiment.subjectivity
            sentiment = "Neutral"
            if polarity > 0:
                sentiment = "Positive"
            elif polarity < 0:
                sentiment = "Negative"

            st.write(f"**Sentiment:** {sentiment}")
            st.write(f"**Polarity Score:** {polarity:.3f}")
            st.write(f"**Subjectivity Score:** {subjectivity:.3f}")

            # Simple sentiment bar chart
            df = pd.DataFrame({
                "Metric": ["Polarity", "Subjectivity"],
                "Score": [polarity, subjectivity]
            })
            fig = px.bar(df, x="Metric", y="Score", range_y=[-1,1], title="Sentiment Scores")
            st.plotly_chart(fig, use_container_width=True)
st.markdown("---")
st.markdown(
    """
    Built by [Janarthanan S](https://janarthanan-portfolioresume.vercel.app/#hero) | 
    [GitHub](https://github.com/Janarthanan1324) | 
    [LinkedIn](https://www.linkedin.com/in/janarthanan-s-130058304)
    """
)
