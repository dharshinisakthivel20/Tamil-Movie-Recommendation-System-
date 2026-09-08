import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 1. Page Styling & Header Configuration
st.set_page_config(page_title="Tamil Movie Recommender", page_icon="🎬", layout="centered")
st.title("🎬 Tamil Movie Recommendation Engine")
st.markdown("Developed by **DHARSHINI S** for Placement Showcase")
st.write("Select a movie from the dropdown to get AI-powered similar recommendations.")

# 2. Load the dataset that we downloaded
df = pd.read_csv('final_movies.csv')

# 3. Core Machine Learning Logic Setup (TF-IDF & Cosine Similarity)
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['tags'].fillna(''))
similarity = cosine_similarity(tfidf_matrix)

# 4. Interactive Dropdown Box for user selection
movie_list = df['MovieName'].values
selected_movie = st.selectbox("Choose a Tamil Movie:", movie_list)

# 5. Recommendation Logic Processing Trigger Button
if st.button('Show Recommendations'):
    try:
        movie_idx = df[df['MovieName'] == selected_movie].index[0]
        similarity_scores = list(enumerate(similarity[movie_idx]))
        sorted_movies = sorted(similarity_scores, key=lambda x: x[1], reverse=True)[1:6]
        
        st.success(f"Top 5 Recommendations if you like '{selected_movie}':")
        for item in sorted_movies:
            st.write(f"🍿 {df.iloc[item[0]]['MovieName']}")
            
    except Exception as e:
        st.error("Something went wrong! Please try another movie title.")
