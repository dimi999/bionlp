import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.svm import SVC
import re

class SVMStanceDetector:    
    def __init__(self, data_path='merged_covid_vaccine_tweets.csv'):
        self.data_path = data_path
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        self.svm_model = SVC(kernel='rbf', random_state=42)
        self.label_mapping = {1: 'Against', 2: 'Neutral', 3: 'Favor'}
        self.load_and_prepare_data()   

    def load_and_prepare_data(self):
        self.df = pd.read_csv(self.data_path)
        print(f"Data loaded! Shape: {self.df.shape}")
        
        self.df['cleaned_text'] = self.df['tweet_text'].apply(self.clean_text)
        self.df = self.df[self.df['cleaned_text'].str.len() > 0]
        
        X = self.vectorizer.fit_transform(self.df['cleaned_text'])
        y = self.df['label'].values
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2)
        
        print(f"Training set: {self.X_train.shape[0]} samples")
        print(f"Test set: {self.X_test.shape[0]} samples")
    
    def clean_text(self, text):
        text = str(text).lower()

        # Remove URLs, mentions, and special characters
        text = re.sub(r'http\S+|www\S+|@\w+|[^a-zA-Z\s]', '', text)
        return ' '.join(text.split())
    
    def train(self):
        self.svm_model.fit(self.X_train, self.y_train)
        
        y_pred = self.svm_model.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        print(f"Accuracy: {accuracy:.4f}")
        
        return accuracy
    
    def evaluate(self):
        y_pred = self.svm_model.predict(self.X_test)
        
        target_names = ['Against', 'Neutral', 'Favor']
        print("Classification Report:")
        print(classification_report(self.y_test, y_pred, target_names=target_names))
        
        cm = confusion_matrix(self.y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=target_names, yticklabels=target_names)
        plt.title('SVM Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.savefig('svm_confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def predict(self, tweet_text):
        cleaned_text = self.clean_text(tweet_text)
        X_new = self.vectorizer.transform([cleaned_text])
        prediction = self.svm_model.predict(X_new)[0]
        
        return {
            'text': tweet_text,
            'predicted_stance': self.label_mapping[prediction],
            'predicted_label': prediction
        }

detector = SVMStanceDetector()
detector.train()
detector.evaluate()