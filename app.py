import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

# Load the pre-trained model
model = LogisticRegression()

# Load the dataset
df = pd.read_csv('news_dataset.csv')

# Create the feature matrix and target vector
X = df['text']
y = df['label']

# Fit the model
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(X)
model.fit(X, y)

# Function to classify news
def classify_news(news):
    news_vectorized = vectorizer.transform([news])
    prediction = model.predict(news_vectorized)
    return prediction[0]

# Streamlit app
def main():
    st.title("Real or Fake News Detection")
    st.write("Enter the news text to classify whether it's real or fake.")

    # Input text
    news_text = st.text_area("News Text", "")

    if st.button("Classify"):
        if news_text:
            # Classify the news
            prediction = classify_news(news_text)

            # Display the result
            if prediction == 'real':
                st.success("The news is classified as REAL.")
            else:
                st.error("The news is classified as FAKE.")
        else:
            st.warning("Please enter some news text.")

if __name__ == '__main__':
    main()
