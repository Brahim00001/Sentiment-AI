import sys
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QTextEdit, QPushButton
from PyQt5.QtCore import Qt
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

class SentimentAnalysisApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Sentiment Analysis Tool')
        self.setGeometry(100, 100, 400, 300)

        layout = QVBoxLayout()

        self.label = QLabel('Enter text for sentiment analysis:')
        layout.addWidget(self.label)

        self.textEdit = QTextEdit(self)
        layout.addWidget(self.textEdit)

        self.analyzeButton = QPushButton('Analyze Sentiment', self)
        self.analyzeButton.clicked.connect(self.analyzeSentiment)
        layout.addWidget(self.analyzeButton)

        self.resultLabel = QLabel('')
        layout.addWidget(self.resultLabel)

        self.setLayout(layout)

    def analyzeSentiment(self):
        text = self.textEdit.toPlainText()
        processed_text = preprocess_text(text)
        vectorized_text = vectorizer.transform([processed_text])
        prediction = model.predict(vectorized_text)
        sentiment = 'positive' if prediction[0] == 4 else 'negative'
        self.resultLabel.setText(f'Sentiment: {sentiment}')
        self.resultLabel.setAlignment(Qt.AlignCenter)
        self.resultLabel.setStyleSheet('font-size: 18px; color: green;' if sentiment == 'positive' else 'font-size: 18px; color: red;')

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = SentimentAnalysisApp()
    ex.show()
    sys.exit(app.exec_())
