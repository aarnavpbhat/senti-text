from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
import string
import nltk
nltk.download('vader_lexicon')
nltk.download('stopwords')


def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    punctuations = string.punctuation
    clean_text = [word.lower() for word in text.split() if word.lower() not in stop_words and word not in punctuations]
    return ' '.join(clean_text)

def analyze_sentiment(text):
    analyzer = SentimentIntensityAnalyzer()
    preprocessed_text = preprocess_text(text)
    scores = analyzer.polarity_scores(preprocessed_text)
    if scores['compound'] >= 0.05:
        return 'Positive'
    elif scores['compound'] <= -0.05:
        return 'Negative'
    else:
        return 'Neutral'
    
text1 = "I had a great day at the beach today. The weather was perfect and I had a lot of fun."
text2 = "I'm so disappointed with the service at this restaurant. The food was terrible and the staff was rude."
text3 = "I went to the grocery store today and bought some milk and bread."

sentiment1 = analyze_sentiment(text1)
sentiment2 = analyze_sentiment(text2)
sentiment3 = analyze_sentiment(text3)

print("Text 1 sentiment:", sentiment1)
print("Text 2 sentiment:", sentiment2)
print("Text 3 sentiment:", sentiment3)

