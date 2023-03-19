from flask import Flask, render_template, request
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
import string
import nltk

nltk.download('vader_lexicon')
nltk.download('stopwords')

app = Flask(__name__)

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

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    text = request.form['text']
    sentiment = analyze_sentiment(text)
    return render_template('index.html', text=text, sentiment=sentiment)

if __name__ == '__main__':
    app.run(debug=True)
