import nltk
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import sent_tokenize, word_tokenize
import textstat
from gensim.summarization import summarize
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# Sample educational content (replace with your data collection logic)
educational_content = """
    Educational content is essential for learning. It helps students acquire knowledge
    and skills needed for their future. The quality of educational content matters a lot.
    It should be engaging and informative.
    """

# Data Preprocessing
def preprocess_text(text):
    # Tokenization
    sentences = sent_tokenize(text)
    tokens = [word_tokenize(sentence) for sentence in sentences]

    # Remove stopwords and punctuation
    stop_words = set(stopwords.words("english"))
    cleaned_tokens = [[word.lower() for word in sentence if word.isalnum() and word.lower() not in stop_words] for sentence in tokens]

    # Rejoin tokens into cleaned sentences
    cleaned_sentences = [" ".join(sentence) for sentence in cleaned_tokens]

    return cleaned_sentences

cleaned_sentences = preprocess_text(educational_content)

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

sentiment_scores, sentiment_label = analyze_sentiment(educational_content)

# Readability Assessment
def assess_readability(text):
    # Using Flesch-Kincaid Grade Level as an example
    grade_level = textstat.flesch_kincaid_grade(text)

    return grade_level

readability_grade = assess_readability(educational_content)

# Automatic Summarization
def generate_summary(text):
    # Using Gensim's extractive summarization
    summary = summarize(text)
    return summary

summary = generate_summary(educational_content)

# Part-of-Speech Tagging
def pos_tagging(text):
    tagged_words = nltk.pos_tag(word_tokenize(text))
    return tagged_words

pos_tags = pos_tagging(educational_content)

# Named Entity Recognition (NER)
def named_entity_recognition(text):
    entities = nltk.ne_chunk(nltk.pos_tag(word_tokenize(text)))
    return entities

ner_entities = named_entity_recognition(educational_content)

# Word Frequency Visualization (Word Cloud)
def create_wordcloud(text):
    wordcloud = WordCloud(width=800, height=400, background_color="white").generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.show()

# Visualize word frequencies
word_freq_text = " ".join(word_tokenize(educational_content))
create_wordcloud(word_freq_text)

# Print results
print("Original Content:")
print(educational_content)
print("\nCleaned Sentences:")
for sentence in cleaned_sentences:
    print(sentence)
print("\nSentiment Analysis:")
print(f"Sentiment Scores: {sentiment_scores}")
print(f"Sentiment Label: {sentiment_label}")
print("\nReadability Grade Level:")
print(f"Flesch-Kincaid Grade Level: {readability_grade}")
print("\nGenerated Summary:")
print(summary)
print("\nPart-of-Speech Tagging:")
for word, pos_tag in pos_tags:
    print(f"{word}: {pos_tag}")
print("\nNamed Entity Recognition (NER):")
print(ner_entities)
