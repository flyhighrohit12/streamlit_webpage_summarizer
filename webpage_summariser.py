import streamlit as st
from newspaper import Article
from bs4 import BeautifulSoup
import requests
from transformers import pipeline
import nltk

# Download the punkt tokenizer
nltk.download('punkt')

# Function to get the article text using Newspaper3k
def get_article_text_newspaper(url):
    try:
        article = Article(url)
        article.download()
        article.parse()
        return article.text
    except:
        return None

# Function to get the article text using BeautifulSoup
def get_article_text_bs4(url):
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')
        paragraphs = soup.find_all('p')
        article_text = " ".join([para.get_text() for para in paragraphs])
        return article_text
    except:
        return None

# Function to summarize text using a transformer model
def summarize_text(text, max_length=200, min_length=50):
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    summaries = summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)
    summary_text = " ".join([summary['summary_text'] for summary in summaries])
    return summary_text.split('. ')

# Streamlit app
st.title("Webpage Summarizer")

# Input URL
url = st.text_input("Enter the URL of the webpage you want to summarize")

if url:
    try:
        # Try to get the article text using Newspaper3k
        article_text = get_article_text_newspaper(url)
        
        # If Newspaper3k fails, use BeautifulSoup
        if not article_text:
            article_text = get_article_text_bs4(url)
        
        # If we have the article text, summarize it
        if article_text:
            summary_points = summarize_text(article_text)
            
            # Display the summary
            st.subheader("Summary")
            for i, point in enumerate(summary_points, 1):
                st.write(f"{i}. {point}")
        else:
            st.error("Failed to extract article text.")
    except Exception as e:
        st.error(f"An error occurred: {e}")
