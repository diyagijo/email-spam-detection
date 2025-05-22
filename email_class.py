import streamlit as st
import pandas as pd
import string
import requests
from io import StringIO
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans
import joblib
import logging
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Download NLTK data (for preprocessing)
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Configure logging for alerting system
logging.basicConfig(filename='spam_alerts.log', level=logging.INFO, 
                    format='%(asctime)s - %(message)s')

# Simulated scraping function
def scrape_emails():
    # For demonstration, we'll simulate scraping by loading a public dataset
    # In a real scenario, this could fetch from an API or email server
    url = "https://raw.githubusercontent.com/justmarkham/DAT8/master/data/sms.tsv"
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = StringIO(response.text)
        df = pd.read_csv(data, sep='\t', header=None, names=['label', 'text'])
        return df
    except Exception as e:
        st.error(f"Error scraping data: {e}")
        # Fallback to a small sample if scraping fails
        return pd.DataFrame({
            'label': ['ham', 'spam', 'ham'],
            'text': ['Hello, how are you?', 'Win a free prize now!', 'Meeting at 5 PM']
        })

# Enhanced text preprocessing function
def clean_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    words = text.split()
    words = [word for word in words if word not in stop_words]
    # Apply stemming
    ps = PorterStemmer()
    words = [ps.stem(word) for word in words]
    return ' '.join(words)

# Load or scrape dataset
@st.cache_data
def load_data():
    df = scrape_emails()
    df['text'] = df['text'].apply(clean_text)
    return df

# Main function
def main():
    # Load and preprocess data
    df = load_data()
    
    # Display dataset info
    st.sidebar.markdown("### Dataset Info")
    st.sidebar.write(f"Total emails: {len(df)}")
    st.sidebar.write(f"Spam: {len(df[df['label'] == 'spam'])}")
    st.sidebar.write(f"Ham: {len(df[df['label'] == 'ham'])}")
    
    # Clustering
    vectorizer = CountVectorizer()
    X_clustering = vectorizer.fit_transform(df['text'])
    kmeans = KMeans(n_clusters=2, random_state=42)
    df['cluster'] = kmeans.fit_predict(X_clustering)
    st.sidebar.markdown("### Clustering Results")
    st.sidebar.write(f"Cluster 0: {len(df[df['cluster'] == 0])} emails")
    st.sidebar.write(f"Cluster 1: {len(df[df['cluster'] == 1])} emails")
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        df['text'], df['label'], test_size=0.2, random_state=42
    )
    
    # Create and train pipeline
    model = Pipeline([
        ('vectorizer', CountVectorizer()),
        ('classifier', MultinomialNB())
    ])
    model.fit(X_train, y_train)
    
    # Evaluate model
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    
    # Save model
    joblib.dump(model, 'spam_classifier_model.pkl')
    
    # Streamlit UI
    st.title(" Email Spam Detection using NLP & Naive Bayes")
    st.write("Enter an email message below to check if it's spam or not.")
    
    user_input = st.text_area("Email Text", "")
    
    if st.button("Predict"):
        if user_input.strip():
            cleaned_input = clean_text(user_input)
            prediction = model.predict([cleaned_input])[0]
            st.markdown(f"### 🔍 Prediction: {'🚫 Spam' if prediction == 'spam' else '✅ Ham (Not Spam)'}")
            # Log spam detection
            if prediction == 'spam':
                logging.info(f"Spam detected: {user_input[:50]}...")
        else:
            st.error("Please enter some text to predict.")
    
    st.sidebar.markdown("### Model Info")
    st.sidebar.write(f"**Accuracy on test set**: {accuracy:.2f}")

if __name__ == "__main__":
    main()