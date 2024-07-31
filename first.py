from flask import Flask, request, render_template, jsonify
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import string

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

def preprocess_text(text):
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = word_tokenize(text)
    words = [word for word in words if word.lower() not in stopwords.words('english')]
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]
    return ' '.join(words)

# Load the trained model and vectorizer
with open('sentiment_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('vectorizer.pkl', 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.form['text']
        processed_data = preprocess_text(data)
        vectorized_data = vectorizer.transform([processed_data])
        prediction = model.predict(vectorized_data)
        sentiment = 'positive' if prediction[0] == 4 else 'negative'
        sentiment_class = 'positive' if prediction[0] == 4 else 'negative'
        return render_template('index.html', sentiment=sentiment, sentiment_class=sentiment_class)
    except Exception as e:
        return render_template('index.html', sentiment='Error: ' + str(e), sentiment_class='error')

if __name__ == '__main__':
    app.run(debug=True, port=5001)
