import streamlit as st
import re
import nltk

from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Download stopwords
nltk.download('stopwords')

# FAQ Questions
questions = [
    "What is RAM?",
    "What is ROM?",
    "What is Android?",
    "What is Artificial Intelligence?",
    "What is Machine Learning?",
    "What is Computer Vision?",
    "What is YOLOv8?",
    "What is Precision Farming?",
    "How does AI help in agriculture?",
    "What is Deep Learning?"
]

# FAQ Answers
answers = [
    "RAM is temporary memory used by a computer.",
    "ROM is permanent memory used to store instructions.",
    "Android is a mobile operating system developed by Google.",
    "Artificial Intelligence is the simulation of human intelligence in machines.",
    "Machine Learning is a subset of AI where systems learn from data.",
    "Computer Vision enables computers to understand images and videos.",
    "YOLOv8 is an object detection model used for object recognition.",
    "Precision Farming uses technology to improve agricultural productivity.",
    "AI helps agriculture through crop monitoring, disease detection and automated harvesting.",
    "Deep Learning is a subset of Machine Learning based on neural networks."
]

# NLP Preprocessing Function
def preprocess_text(text):

    # Convert to lowercase
    text = text.lower()

    # Remove punctuation and special characters
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)

    # Tokenization
    words = text.split()

    # Stop-word removal
    stop_words = set(stopwords.words('english'))

    filtered_words = []

    for word in words:
        if word not in stop_words:
            filtered_words.append(word)

    # Convert list back to sentence
    cleaned_text = " ".join(filtered_words)

    return cleaned_text


# Preprocess FAQ Questions
processed_questions = []

for question in questions:
    processed_questions.append(
        preprocess_text(question)
    )

# TF-IDF Vectorizer
vectorizer = TfidfVectorizer()

question_vectors = vectorizer.fit_transform(
    processed_questions
)

# Streamlit UI
st.set_page_config(
    page_title="FAQ Chatbot",
    page_icon="🤖"
)

st.title("🤖 FAQ Chatbot")

st.write(
    "Ask a question."
)

user_question = st.text_input(
    "Enter your Question"
)

if st.button("Get Answer"):

    if user_question.strip() == "":
        st.warning(
            "Please enter a question."
        )

    else:

        # Preprocess User Question
        cleaned_question = preprocess_text(
            user_question
        )

        # Convert user question into vector
        user_vector = vectorizer.transform(
            [cleaned_question]
        )

        # Similarity Calculation
        similarity = cosine_similarity(
            user_vector,
            question_vectors
        )

        # Best Match
        best_match_index = similarity.argmax()

        confidence = similarity[0][best_match_index]

        if confidence > 0.30:

            st.success(
                answers[best_match_index]
            )

            st.write(
                f"Similarity Score: {confidence:.2f}"
            )

        else:

            st.error(
                "Sorry, I don't know the answer."
            )