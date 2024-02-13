import os
import re
from bs4 import BeautifulSoup
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sumy.parsers.plaintext import PlaintextParser
from sumy.summarizers.lsa import LsaSummarizer
from sumy.nlp.tokenizers import Tokenizer
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Step 1: Data Preprocessing
def extract_text_from_eml(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    soup = BeautifulSoup(content, 'html.parser')
    return soup.get_text()

# Step 2: Named Entity Recognition (NER)
nlp = spacy.load("en_core_web_sm")

def extract_entities(text):
    doc = nlp(text)
    entities = [ent.text for ent in doc.ents]
    return entities

# Step 3: Topic Modeling (LDA)
def topic_modeling(text_corpus):
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(text_corpus)
    lda = LatentDirichletAllocation(n_components=5, random_state=42)
    lda.fit(X)
    return lda

# Step 4: Sentiment Analysis
def sentiment_analysis(text):
    sia = SentimentIntensityAnalyzer()
    sentiment_score = sia.polarity_scores(text)['compound']
    return sentiment_score

# Step 5: Machine Learning Models
# Assuming you have a labeled dataset for training
def train_ml_model(features, labels):
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    return clf, accuracy

# Step 6: Text Summarization
def text_summarization(text):
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer = LsaSummarizer()
    summary = summarizer(parser.document, 3)  # 3 is the number of sentences in the summary
    return ' '.join(str(sent) for sent in summary)

# Step 7: Keyword Extraction
def keyword_extraction(text):
    # Implement your keyword extraction algorithm here
    pass

# Step 8: Post-Processing
# Manually review and refine the summaries

# Step 9: Visualization
def visualize_wordcloud(text):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.show()

# Step 10: Integration
# Integrate the summarization process into your workflow

# Example usage:
file_path = "path/to/your/email.eml"
text_content = extract_text_from_eml(file_path)
entities = extract_entities(text_content)
lda_model = topic_modeling([text_content])
sentiment_score = sentiment_analysis(text_content)
summary = text_summarization(text_content)
visualize_wordcloud(text_content)
