import streamlit as st
import pandas as pd
from textblob import TextBlob
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from datetime import datetime
from PIL import Image

# Download NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# -------------------- PAGE SETUP --------------------
st.set_page_config(
    page_title="AI Tools Demo App 🤖",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# -------------------- HEADER --------------------
st.markdown("""
    <style>
    .title {
        font-size: 45px;
        text-align: center;
        font-weight: 700;
        color: #FF4081;
        text-shadow: 2px 2px 5px #00000033;
    }
    .subtitle {
        text-align: center;
        color: gray;
        font-style: italic;
        font-size: 18px;
    }
    .footer {
        text-align: center;
        color: #888;
        margin-top: 50px;
        font-size: 14px;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="title">🧠 AI Tools Text Analyzer</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Powered by TextBlob • NLTK • WordCloud • Streamlit</p>', unsafe_allow_html=True)
st.divider()

# -------------------- TEXT INPUT --------------------
st.subheader("✍️ Enter your text below")
user_text = st.text_area("Write or paste your text here:", height=160, placeholder="Try something like: I absolutely love this product! 😍")

# -------------------- ANALYZE BUTTON --------------------
if st.button("🚀 Analyze Now"):
    if user_text.strip():
        with st.spinner("Analyzing text... 🔍"):
            # Sentiment analysis
            blob = TextBlob(user_text)
            sentiment = blob.sentiment.polarity

            # Sentiment interpretation
            st.subheader("📊 Sentiment Result")
            if sentiment > 0:
                st.success(f"Positive Sentiment ({sentiment:.2f}) 😊")
            elif sentiment < 0:
                st.error(f"Negative Sentiment ({sentiment:.2f}) 😡")
            else:
                st.info(f"Neutral Sentiment ({sentiment:.2f}) 😐")

            # -------------------- WORD CLOUD --------------------
            stop_words = set(stopwords.words('english'))
            clean_text = ' '.join([word for word in user_text.split() if word.lower() not in stop_words])

            wordcloud = WordCloud(
                width=800,
                height=400,
                background_color='white',
                colormap='plasma'
            ).generate(clean_text)

            st.subheader("☁️ Word Cloud Visualization")
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis('off')
            st.pyplot(fig)

            st.balloons()

    else:
        st.warning("⚠️ Please enter some text to analyze.")

# -------------------- FOOTER --------------------
st.markdown(f"""
<div class="footer">
    <hr>
    <p>✨ Built with ❤️ by Lemick Jackson, Joseph Muthui, Racheal Wanjiru — AI Tools Assignment ({datetime.now().year}) ✨</p>
</div>
""", unsafe_allow_html=True)
