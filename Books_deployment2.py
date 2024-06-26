import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import pandas as pd
import numpy as np
from fuzzywuzzy import process

# Load the dataset
df = pd.read_csv("updated_books.csv")

# Drop unnecessary columns
df.drop(columns=["original_title"], axis=1, inplace=True)

# Fill missing values in 'original_publication_year' and 'language_code' columns
df["original_publication_year"].fillna(method="ffill", inplace=True)
df['language_code'].fillna(df['language_code'].mode()[0], inplace=True)

# Handle outliers in numeric columns by replacing them with the mean
for col in df.columns:
    if df[col].dtype == "int64" or df[col].dtype == "float64":
        q1 = np.quantile(df[col], 0.25)
        q3 = np.quantile(df[col], 0.75)
        mean = np.mean(df[col])
        iqr = q3 - q1
        upper_bound = q3 + 1.5 * iqr
        lower_bound = q1 - 1.5 * iqr
        df[col] = np.where((df[col] > upper_bound) | (df[col] < lower_bound), mean, df[col])

# TF-IDF Vectorization
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['title'])

# Train KMeans clustering model
num_clusters = 27  # You can adjust this value
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
clusters = kmeans.fit_predict(X)

# Function to recommend books
def recommend_books(book_title=None, author_name=None):
    if book_title:
        # Use FuzzyWuzzy to find the closest match for the book title
        match = process.extractOne(book_title, df['title'])
        if match:
            matched_title = match[0]
            title_vector = vectorizer.transform([matched_title])
            cluster_id = kmeans.predict(title_vector)[0]
            cluster_books = df[clusters == cluster_id]
            if author_name:
                cluster_books = cluster_books[cluster_books['authors'].str.contains(author_name, case=False)]
            return cluster_books['title'].head(10).values
        else:
            return []
    elif author_name:
        author_books = df[df['authors'].str.contains(author_name, case=False)]
        if len(author_books) > 0:
            return author_books['title'].head(10).values
        else:
            return []
    else:
        return []

# Streamlit app
def main():
    st.title('Book Recommendation System')

    # Input fields for book title and author name
    book_title = st.text_input('Enter Book Title (optional):', '')
    author_name = st.text_input('Enter Author Name (optional):', '')

    # Get recommendations if the user clicks the button
    if st.button('Get Recommendations'):
        if book_title or author_name:
            similar_books = recommend_books(book_title, author_name)
            if len(similar_books) > 0:
                st.write(f"Books similar to '{book_title}' or by '{author_name}':")
                for book in similar_books:
                    st.write("-", book)
            else:
                st.write("No similar books found")
        else:
            st.write("Please enter a book title or an author name")

if __name__ == '__main__':
    main()
