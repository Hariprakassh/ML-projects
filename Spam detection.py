import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score

# Load the dataset
@st.cache
def load_data():
    df = pd.read_csv('spam.csv', encoding='latin-1')
    df = df[['v1', 'v2']]
    df.columns = ['label', 'text']
    df['label'] = df['label'].map({'ham': 0, 'spam': 1})
    return df

df = load_data()

# Preprocess the data
X = df['text']
y = df['label']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a model pipeline
model = make_pipeline(TfidfVectorizer(stop_words='english'), MultinomialNB())

# Train the model
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
st.write(f'Accuracy: {accuracy:.2f}')

# Define the prediction function
def predict_spam(text):
    prediction = model.predict([text])
    return 'Spam' if prediction[0] == 1 else 'Not Spam'

# Streamlit interface
def main():
    st.title('Spam Detection Application')
    st.write('Enter a message to classify it as spam or not spam.')

    user_input = st.text_area('Message:')

    if st.button('Predict'):
        if user_input:
            result = predict_spam(user_input)
            st.write(f'The message is: **{result}**')
        else:
            st.write('Please enter a message.')

if __name__ == '__main__':
    main()
