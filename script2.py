''' 
Data Preprocessing:
Extract and clean the text content from the .eml files.

Named Entity Recognition (NER):
Identify entities related to products, prices, quantities, retailers, and suppliers in the emails.

Text Summarization:
Apply text summarization techniques to generate concise summaries for each email or conversation.

Visualization (Optional):
Visualize key information using word clouds or other techniques.

Integration:
Integrate the summarization process into your workflow for easy analysis and decision-making.

---------------------------------------------------------------------------------------
Before running the code, make sure to install the required packages using:

pip install beautifulsoup4 spacy sumy nltk
python -m spacy download en_core_web_sm
---------------------------------------------------------------------------------------

'''


import os
import re
import email
import nltk
from bs4 import BeautifulSoup
import spacy
from sumy.parsers.plaintext import PlaintextParser
from sumy.summarizers.lsa import LsaSummarizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Step 1: Data Preprocessing
def extract_text_from_eml(file_path):
    with open(file_path, 'rb') as file:
        # Use chardet to automatically detect the encoding
        raw_data = file.read()
        result = chardet.detect(raw_data)
        encoding = result['encoding']

        # Decode the content using the detected encoding
        text_content = raw_data.decode(encoding, 'ignore')

    return text_content

# Step 2: Named Entity Recognition (NER)
nlp = spacy.load("en_core_web_sm")

def extract_entities(text):
    doc = nlp(text)
    entities = [ent.text for ent in doc.ents]
    return entities

# Step 3: Text Summarization
def text_summarization(text):
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer = LsaSummarizer()
    summary = summarizer(parser.document, 3)  # 3 is the number of sentences in the summary
    return ' '.join(str(sent) for sent in summary)

# Step 4: Sentiment Analysis
def sentiment_analysis(text):
    sia = SentimentIntensityAnalyzer()
    sentiment_score = sia.polarity_scores(text)['compound']
    return sentiment_score

# Step 5: Example Usage
file_path = "path/to/your/email.eml"
text_content = extract_text_from_eml(file_path)
entities = extract_entities(text_content)
summary = text_summarization(text_content)
sentiment_score = sentiment_analysis(text_content)

# Print the results
print("Entities:", entities)
print("Summary:", summary)
print("Sentiment Score:", sentiment_score)

