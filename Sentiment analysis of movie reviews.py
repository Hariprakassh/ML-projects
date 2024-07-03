import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

# Load the IMDb movie reviews dataset
@st.cache
def load_data():
    df = pd.read_csv('imdb_reviews.csv')
    return df

df = load_data()

# Preprocess the data
X = df['review']
y = df['sentiment']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a model pipeline
model = make_pipeline(
    TfidfVectorizer(stop_words='english', max_features=5000),
    LogisticRegression()
)

# Train the model
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
st.write(f'Model Accuracy: {accuracy:.2f}')
st.write('Classification Report:')
st.write(classification_report(y_test, y_pred))

# Define the prediction function
def predict_sentiment(text):
    prediction = model.predict([text])
    return prediction[0]

# Streamlit interface
def main():
    st.title('Movie Review Sentiment Analysis')
    st.write('Enter a movie review text to predict its sentiment (positive or negative).')

    user_input = st.text_area('Review Text:')

    if st.button('Predict'):
        if user_input:
            sentiment = predict_sentiment(user_input)
            if sentiment == 1:
                st.write('**Sentiment:** Positive')
            else:
                st.write('**Sentiment:** Negative')
        else:
            st.write('Please enter a review text.')

if __name__ == '__main__':
    main()
