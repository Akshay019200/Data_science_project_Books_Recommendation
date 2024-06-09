import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from fuzzywuzzy import process

# Load the dataset
df = pd.read_csv("C:\\Users\Akshay\\model_deployment_books\\updated_books.csv")

df.drop(columns=["original_title"], axis=1, inplace=True)

# Fill missing values in 'original_publication_year' and 'language_code' columns
df["original_publication_year"].fillna(method="ffill", inplace=True)
df['language_code'].fillna(df['language_code'].mode()[0], inplace=True)

# Standardize numeric columns
numeric_cols = ['books_count', 'original_publication_year', 'average_rating', 'ratings_count', 'ratings_1', 'ratings_2', 'ratings_3', 'ratings_4', 'ratings_5']
df[numeric_cols] = df[numeric_cols].apply(lambda x: (x - x.mean()) / x.std())

# Function to recommend similar books
def recommend_similar_books(book_title, author_name, df, cosine_sim):
    if book_title:
        # Find a match for the book title
        book_titles = df['title'].tolist()
        match = process.extractOne(book_title, book_titles)
        
        if match:
            match_title = match[0]
            match_index = df[df['title'] == match_title].index[0]
            similarity_scores = list(enumerate(cosine_sim[match_index]))
            similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
            similar_books_indices = [index for index, _ in similarity_scores[1:11]]
            
            # Filter similar books by author name
            similar_books = df.iloc[similar_books_indices]
            if author_name:
                similar_books = similar_books[similar_books['authors'].str.contains(author_name, case=False)]
                
            return similar_books['title'].tolist()
        else:
            return []
    elif author_name:
        # Filter books by author name
        similar_books = df[df['authors'].str.contains(author_name, case=False)]
        return similar_books['title'].tolist()
    else:
        return []

# Streamlit app
def main():
    st.title('Book Recommendation System')

    # Input fields for book title and author name
    book_title = st.text_input('Enter Book Title:', '')
    author_name = st.text_input('Enter Author Name:', '')

    # CountVectorizer to transform titles into matrix
    count_vectorizer = CountVectorizer()
    title_matrix = count_vectorizer.fit_transform(df['title'])
    cosine_sim = cosine_similarity(title_matrix)

    # Get recommendations if the user clicks the button
    if st.button('Get Recommendations'):
        similar_books = recommend_similar_books(book_title, author_name, df, cosine_sim)
        if similar_books:
            st.write(f"Books similar to '{book_title}' or by '{author_name}':")
            for book in similar_books:
                st.write("-", book)
        else:
            st.write("No similar books found")

if __name__ == '__main__':
    main()
