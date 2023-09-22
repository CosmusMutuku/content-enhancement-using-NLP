# app.py
from flask import Flask, request, jsonify
import nltk
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import textstat
from gensim.summarization import summarize


app = Flask(__name__)

@app.route('/enhance', methods=['POST'])
def enhance_content():
    content = request.json.get('content')

    # Data Preprocessing
    def preprocess_text(text):
        # Tokenization
        tokens = nltk.word_tokenize(text)

        # Remove stopwords
        stop_words = set(stopwords.words("english"))
        filtered_tokens = [word.lower() for word in tokens if word.isalnum() and word.lower() not in stop_words]

        # Rejoin tokens into a clean text
        cleaned_text = " ".join(filtered_tokens)

        return cleaned_text

    cleaned_content = preprocess_text(content)

    # Sentiment Analysis
    def analyze_sentiment(text):
        analyzer = SentimentIntensityAnalyzer()
        sentiment_scores = analyzer.polarity_scores(text)

        # Determine sentiment label
        if sentiment_scores['compound'] >= 0.05:
            sentiment_label = 'Positive'
        elif sentiment_scores['compound'] <= -0.05:
            sentiment_label = 'Negative'
        else:
            sentiment_label = 'Neutral'

        return sentiment_scores, sentiment_label

    sentiment_scores, sentiment_label = analyze_sentiment(cleaned_content)

    # Readability Assessment
    def assess_readability(text):
        # Using Flesch-Kincaid Grade Level as an example
        grade_level = textstat.flesch_kincaid_grade(text)

        return grade_level

    readability_grade = assess_readability(cleaned_content)

    # Automatic Summarization
    def generate_summary(text):
        # Using Gensim's extractive summarization
        summary = summarize(text)
        return summary

    summary = generate_summary(content)

    result = {
        'cleaned_content': cleaned_content,
        'sentiment_scores': sentiment_scores,
        'sentiment_label': sentiment_label,
        'readability_grade': readability_grade,
        'summary': summary
    }

    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
