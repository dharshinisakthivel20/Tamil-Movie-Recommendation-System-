# 🎬 Tamil Movie Recommendation System

An AI-powered Content-Based Movie Recommendation Engine developed using Machine Learning and NLP principles. This project processes a dataset of 300+ Tamil movies to predict contextually similar suggestions based on genres, directors, and lead actors.

Developed by **DHARSHINI S** for Placement Showcase.

---

## 🛠️ Tech Stack & Tools Used
* **Language:** Python 🐍
* **Data Libraries:** Pandas, NumPy
* **Machine Learning & NLP:** Scikit-Learn (TF-IDF Vectorizer, Cosine Similarity)
* **Web UI Framework:** Streamlit 🎈
* **Server Deployment:** LocalTunnel

---

## 🧠 Machine Learning Approach
1. **Data Preprocessing:** Consolidated 'Genre', 'Director', and 'Actor' columns into unique text tags.
2. **TF-IDF Vectorization:** Converted text tokens into mathematical vectors, assigning higher weights to unique attributes like specific directors or actors.
3. **Similarity Indexing:** Computed the angular distance between movie vectors using **Cosine Similarity** matrix layouts `(329, 329)` to pick the top 5 closest recommendations.

---

## 🚀 How to Run the Live Dashboard
1. Open the core notebook `Untitled0.ipynb` in Google Colab.
2. Upload the `Tamil_movies_dataset.csv` into the session files storage.
3. Click **Runtime -> Run all** to automate model calculations and initialize the Streamlit server dashboard.
4. Access the generated LocalTunnel web link and bypass the security prompt using the cloud session public IP password!
   
