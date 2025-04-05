from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

# Sample training data (replace with actual dataset)
corpus = ["This is a spam message", "Hello, how are you?", "Win a free iPhone now"]

# Train the TF-IDF vectorizer
tfidf = TfidfVectorizer()
tfidf.fit(corpus)  # <--- This step is necessary

# Save the trained vectorizer
with open('vectorizer.pkl', 'wb') as f:
    pickle.dump(tfidf, f)


