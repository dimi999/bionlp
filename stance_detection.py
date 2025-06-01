import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from textblob import TextBlob
import warnings
warnings.filterwarnings('ignore')

# Download required NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('vader_lexicon', quiet=True)
except:
    print("NLTK downloads may have failed, but continuing...")

class VaccineStanceDetector:
    """
    A comprehensive stance detection model for COVID-19 vaccine tweets.
    
    This class implements multiple machine learning approaches to classify
    tweet stance as Against (1), Neutral (2), or Favor (3) regarding COVID-19 vaccines.
    """
    
    def __init__(self, data_path='merged_covid_vaccine_tweets.csv'):
        """Initialize the stance detector with data loading and preprocessing."""
        self.data_path = data_path
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.vectorizer = None
        self.models = {}
        self.best_model = None
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()
        
        # Load and preprocess data
        self.load_data()
        self.preprocess_data()
        
    def load_data(self):
        """Load the merged COVID-19 vaccine tweets dataset."""
        try:
            self.df = pd.read_csv(self.data_path)
            print(f"Data loaded successfully! Shape: {self.df.shape}")
            print(f"Columns: {list(self.df.columns)}")
            
            # Display basic statistics
            print("\nDataset Overview:")
            print(self.df.info())
            print("\nLabel distribution:")
            print(self.df['label'].value_counts().sort_index())
            
            # Map labels to meaningful names for better understanding
            self.label_mapping = {1: 'Against', 2: 'Neutral', 3: 'Favor'}
            print("\nLabel meanings:")
            for key, value in self.label_mapping.items():
                count = (self.df['label'] == key).sum()
                print(f"{key}: {value} ({count} tweets)")
                
        except Exception as e:
            print(f"Error loading data: {e}")
            return None
    
    def clean_text(self, text):
        """
        Clean and preprocess text data.
        
        Args:
            text (str): Raw tweet text
            
        Returns:
            str: Cleaned text
        """
        if pd.isna(text):
            return ""
        
        # Convert to lowercase
        text = str(text).lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove user mentions and hashtags (but keep the content)
        text = re.sub(r'@\w+', '', text)
        text = re.sub(r'#(\w+)', r'\1', text)  # Keep hashtag content
        
        # Remove special characters and digits
        # pastreaza emojiuri
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    def extract_features(self, text):
        """
        Extract additional features from text.
        
        Args:
            text (str): Tweet text
            
        Returns:
            dict: Feature dictionary
        """
        features = {}
        
        # Length features
        features['char_count'] = len(text)
        features['word_count'] = len(text.split())
        
        # Sentiment features using TextBlob
        # de intrebat
        blob = TextBlob(text)
        features['sentiment_polarity'] = blob.sentiment.polarity
        features['sentiment_subjectivity'] = blob.sentiment.subjectivity
        
        # Vaccine-related keywords
        vaccine_keywords = ['vaccine', 'vaccination', 'pfizer', 'moderna', 'astrazeneca', 
                          'johnson', 'sinovac', 'sinopharm', 'sputnik', 'covaxin', 'covishield']
        features['vaccine_mentions'] = sum(1 for word in vaccine_keywords if word in text.lower())
        
        # Emotional indicators
        positive_words = ['safe', 'effective', 'good', 'great', 'excellent', 'hope', 'relief']
        negative_words = ['dangerous', 'unsafe', 'bad', 'terrible', 'fear', 'worry', 'risk']
        
        features['positive_sentiment'] = sum(1 for word in positive_words if word in text.lower())
        features['negative_sentiment'] = sum(1 for word in negative_words if word in text.lower())
        
        return features
    
    def preprocess_data(self):
        """Preprocess the dataset for model training."""
        if self.df is None:
            print("No data loaded!")
            return
        
        # Remove rows with missing tweet text
        self.df = self.df.dropna(subset=['tweet_text'])
        
        # Clean text
        print("Cleaning text data...")
        self.df['cleaned_text'] = self.df['tweet_text'].apply(self.clean_text)
        
        # Remove empty texts after cleaning
        self.df = self.df[self.df['cleaned_text'].str.len() > 0]
        
        # Extract additional features
        print("Extracting features...")
        feature_dicts = self.df['tweet_text'].apply(self.extract_features)
        feature_df = pd.DataFrame(list(feature_dicts))
        
        # Combine with main dataframe
        self.df = pd.concat([self.df.reset_index(drop=True), feature_df], axis=1)
        
        print(f"Data preprocessing complete! Final shape: {self.df.shape}")
        
        # Display class distribution
        plt.figure(figsize=(10, 6))
        
        plt.subplot(1, 2, 1)
        self.df['label'].value_counts().sort_index().plot(kind='bar')
        plt.title('Stance Distribution')
        plt.xlabel('Stance')
        plt.ylabel('Count')
        plt.xticks([0, 1, 2], ['Against', 'Neutral', 'Favor'], rotation=0)
        
        plt.subplot(1, 2, 2)
        self.df['word_count'].hist(bins=30, alpha=0.7)
        plt.title('Tweet Length Distribution')
        plt.xlabel('Word Count')
        plt.ylabel('Frequency')
        
        plt.tight_layout()
        plt.savefig('data_exploration.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def prepare_features(self):
        """Prepare feature vectors for machine learning."""
        print("Preparing features for model training...")
        
        # Text vectorization using TF-IDF
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2),  # Include bigrams
            min_df=2,
            max_df=0.8
        )
        
        # Fit and transform text data
        text_features = self.vectorizer.fit_transform(self.df['cleaned_text'])
        
        # Additional numerical features
        numerical_features = self.df[['char_count', 'word_count', 'sentiment_polarity', 
                                    'sentiment_subjectivity', 'vaccine_mentions',
                                    'positive_sentiment', 'negative_sentiment']].values
        
        # Combine text and numerical features
        from scipy.sparse import hstack
        X = hstack([text_features, numerical_features])
        y = self.df['label'].values
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"Training set size: {self.X_train.shape}")
        print(f"Test set size: {self.X_test.shape}")
    
    def train_models(self):
        """Train multiple machine learning models."""
        print("Training multiple models...")
        
        # Prepare features
        self.prepare_features()
        
        # Define models to train
        models_to_train = {
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'SVM': SVC(kernel='rbf', random_state=42),
            #'Naive Bayes': MultinomialNB(),
            'Gradient Boosting': GradientBoostingClassifier(random_state=42)
        }
        
        results = {}
        
        for name, model in models_to_train.items():
            print(f"\nTraining {name}...")
            
            # Train the model
            model.fit(self.X_train, self.y_train)
            
            # Make predictions
            y_pred = model.predict(self.X_test)
            
            # Calculate accuracy
            accuracy = accuracy_score(self.y_test, y_pred)
            
            # Store model and results
            self.models[name] = model
            results[name] = {
                'accuracy': accuracy,
                'predictions': y_pred,
                'model': model
            }
            
            print(f"{name} Accuracy: {accuracy:.4f}")
        
        # Find best model
        best_model_name = max(results.keys(), key=lambda x: results[x]['accuracy'])
        self.best_model = results[best_model_name]['model']
        
        print(f"\nBest Model: {best_model_name} with accuracy: {results[best_model_name]['accuracy']:.4f}")
        
        return results
    
    def evaluate_model(self, model_name=None):
        """
        Evaluate a specific model or the best model.
        
        Args:
            model_name (str): Name of the model to evaluate
        """
        if model_name is None:
            model = self.best_model
            name = "Best Model"
        else:
            model = self.models[model_name]
            name = model_name
        
        # Make predictions
        y_pred = model.predict(self.X_test)
        
        # Print detailed classification report
        print(f"\n{name} - Detailed Evaluation:")
        print("="*50)
        
        target_names = ['Against', 'Neutral', 'Favor']
        print(classification_report(self.y_test, y_pred, target_names=target_names))
        
        # Confusion Matrix
        cm = confusion_matrix(self.y_test, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=target_names, yticklabels=target_names)
        plt.title(f'{name} - Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.savefig(f'{name.lower().replace(" ", "_")}_confusion_matrix.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
        return y_pred
    
    def predict_stance(self, tweet_text):
        """
        Predict stance for a new tweet.
        
        Args:
            tweet_text (str): The tweet text to analyze
            
        Returns:
            dict: Prediction results
        """
        if self.best_model is None:
            print("No model trained yet! Please train models first.")
            return None
        
        # Clean the input text
        cleaned_text = self.clean_text(tweet_text)
        
        # Extract features
        features = self.extract_features(tweet_text)
        
        # Vectorize text
        text_vector = self.vectorizer.transform([cleaned_text])
        
        # Prepare numerical features
        numerical_features = np.array([[
            features['char_count'], features['word_count'], 
            features['sentiment_polarity'], features['sentiment_subjectivity'],
            features['vaccine_mentions'], features['positive_sentiment'], 
            features['negative_sentiment']
        ]])
        
        # Combine features
        from scipy.sparse import hstack
        X_new = hstack([text_vector, numerical_features])
        
        # Make prediction
        prediction = self.best_model.predict(X_new)[0]
        
        # Get prediction probabilities if available
        try:
            probabilities = self.best_model.predict_proba(X_new)[0]
            prob_dict = {
                'Against': probabilities[0] if len(probabilities) > 2 else 0,
                'Neutral': probabilities[1] if len(probabilities) > 1 else probabilities[0],
                'Favor': probabilities[2] if len(probabilities) > 2 else 0
            }
        except:
            prob_dict = None
        
        result = {
            'text': tweet_text,
            'cleaned_text': cleaned_text,
            'predicted_stance': self.label_mapping[prediction],
            'predicted_label': prediction,
            'features': features,
            'probabilities': prob_dict
        }
        
        return result
    
    def hyperparameter_tuning(self):
        """Perform hyperparameter tuning for the best models."""
        print("Performing hyperparameter tuning...")
        
        # Parameter grids for different models
        param_grids = {
            'Random Forest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5, 10]
            },
            'SVM': {
                'C': [0.1, 1, 10],
                'gamma': ['scale', 'auto'],
                'kernel': ['rbf', 'linear']
            },
            'Logistic Regression': {
                'C': [0.1, 1, 10],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear', 'saga']
            }
        }
        
        best_params = {}
        
        for model_name, param_grid in param_grids.items():
            print(f"\nTuning {model_name}...")
            
            # Get base model
            if model_name == 'Random Forest':
                base_model = RandomForestClassifier(random_state=42)
            elif model_name == 'SVM':
                base_model = SVC(random_state=42)
            elif model_name == 'Logistic Regression':
                base_model = LogisticRegression(random_state=42, max_iter=1000)
            
            # Perform grid search
            grid_search = GridSearchCV(
                base_model, param_grid, cv=5, 
                scoring='accuracy', n_jobs=-1, verbose=1
            )
            
            grid_search.fit(self.X_train, self.y_train)
            
            # Store best parameters
            best_params[model_name] = grid_search.best_params_
            
            # Update model with best parameters
            self.models[f"{model_name}_tuned"] = grid_search.best_estimator_
            
            print(f"Best parameters for {model_name}: {grid_search.best_params_}")
            print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
        
        return best_params
    
    def generate_model_report(self):
        """Generate a comprehensive model performance report."""
        print("Generating comprehensive model report...")
        
        report = {
            'dataset_info': {
                'total_samples': len(self.df),
                'training_samples': self.X_train.shape[0],
                'test_samples': self.X_test.shape[0],
                'features': self.X_train.shape[1],
                'class_distribution': self.df['label'].value_counts().to_dict()
            },
            'model_performance': {}
        }
        
        # Evaluate all models
        for name, model in self.models.items():
            y_pred = model.predict(self.X_test)
            accuracy = accuracy_score(self.y_test, y_pred)
            
            report['model_performance'][name] = {
                'accuracy': accuracy,
                'classification_report': classification_report(
                    self.y_test, y_pred, output_dict=True
                )
            }
        
        # Save report
        import json
        with open('model_report.json', 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print("Report saved to 'model_report.json'")
        
        return report

def main():
    """Main function to demonstrate the stance detection system."""
    print("COVID-19 Vaccine Stance Detection System")
    print("="*50)
    
    # Initialize the detector
    detector = VaccineStanceDetector()
    
    # Train models
    results = detector.train_models()
    
    # Evaluate the best model
    detector.evaluate_model()
    
    # Perform hyperparameter tuning
    best_params = detector.hyperparameter_tuning()
    
    # Generate comprehensive report
    report = detector.generate_model_report()
    
    # Test with sample tweets
    sample_tweets = [
        "I got my COVID vaccine today and I feel great! Thank you science!",
        "COVID vaccines are dangerous and causing serious side effects",
        "The vaccine rollout is happening slowly but steadily",
        "Pfizer and Moderna vaccines are highly effective against COVID-19",
        "I'm worried about the long-term effects of these vaccines"
    ]
    
    print("\n" + "="*50)
    print("Testing with Sample Tweets:")
    print("="*50)
    
    for i, tweet in enumerate(sample_tweets, 1):
        print(f"\nSample {i}: {tweet}")
        result = detector.predict_stance(tweet)
        if result:
            print(f"Predicted Stance: {result['predicted_stance']}")
            if result['probabilities']:
                print("Probabilities:")
                for stance, prob in result['probabilities'].items():
                    print(f"  {stance}: {prob:.3f}")
    
    print("\n" + "="*50)
    print("Training Complete! You can now use the detector.predict_stance() method")
    print("to classify new tweets.")
    
    return detector

if __name__ == "__main__":
    detector = main()