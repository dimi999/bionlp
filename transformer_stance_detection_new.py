"""
TensorFlow Transformer for COVID-19 Vaccine Stance Detection

A transformer implementation for classifying tweet stance as:
- Against (1), Neutral (2), or Favor (3) regarding COVID-19 vaccines
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder
import re
import warnings
import json
import time
warnings.filterwarnings('ignore')

# Try TensorFlow, fallback to scikit-learn if not available
try:
    import tensorflow as tf
    from tensorflow.keras.preprocessing.text import Tokenizer
    from tensorflow.keras.utils import pad_sequences
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import (
        Input, Embedding, Dense, Dropout, LayerNormalization,
        MultiHeadAttention, GlobalAveragePooling1D, Add
    )
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    TENSORFLOW_AVAILABLE = True
    tf.random.set_seed(42)
except ImportError:
    print("TensorFlow not available, using scikit-learn fallback")
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.neural_network import MLPClassifier
    TENSORFLOW_AVAILABLE = False

np.random.seed(42)

class TransformerStanceDetector:
    """
    Transformer-based stance detection model for COVID-19 vaccine tweets.
    Uses TensorFlow if available, otherwise falls back to scikit-learn.
    """
    
    def __init__(self, data_path='merged_covid_vaccine_tweets.csv'):
        """Initialize the detector."""
        self.data_path = data_path
        self.df = None
        self.model = None
        self.history = None
        self.using_tensorflow = TENSORFLOW_AVAILABLE        # Model parameters (improved for better performance)
        if self.using_tensorflow:
            self.max_len = 120  # Slightly increased for better context
            self.vocab_size = 8000  # Increased vocabulary size
            self.embed_dim = 128  # Increased embedding dimension
            self.num_heads = 8   # More attention heads
            self.ff_dim = 512    # Larger feed-forward dimension
            self.num_layers = 2  # Multiple transformer blocks
            self.epochs = 15     # More epochs with better regularization
            self.batch_size = 32
            self.tokenizer = None
        else:
            self.max_features = 5000
            self.epochs = 100
            self.vectorizer = None
        
        # Label mapping
        self.label_mapping = {1: 'Against', 2: 'Neutral', 3: 'Favor'}
        
        print(f"Using {'TensorFlow Transformer' if self.using_tensorflow else 'scikit-learn MLP'} backend")
        
        # Load and prepare data
        self.load_data()
    
    def load_data(self):
        """Load and prepare the dataset."""
        print("Loading data...")
        try:
            self.df = pd.read_csv(self.data_path)
            print(f"Data loaded: {self.df.shape}")
            
            # Clean text
            self.df['cleaned_text'] = self.df['tweet_text'].apply(self.clean_text)
            
            # Remove empty texts
            self.df = self.df[self.df['cleaned_text'].str.len() > 0]
            
            print(f"After cleaning: {self.df.shape}")
            print("\nLabel distribution:")
            print(self.df['label'].value_counts().sort_index())
            
        except Exception as e:
            print(f"Error loading data: {e}")
            return None
    def clean_text(self, text):
        """Enhanced text cleaning with better preprocessing."""
        if pd.isna(text):
            return ""
        
        text = str(text).lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        
        # Handle mentions and hashtags more carefully
        text = re.sub(r'@\w+', '[USER]', text)  # Replace mentions with token
        text = re.sub(r'#(\w+)', r'\1', text)  # Keep hashtag content
        
        # Handle common vaccine-related abbreviations
        text = re.sub(r'\bcovid19\b|\bcovid-19\b|\bcovid\b', 'covid', text)
        text = re.sub(r'\bvaccin\w*\b', 'vaccine', text)
        text = re.sub(r'\bpfizer\b|\bmoderna\b|\bj&j\b|\bjohnson\b', '[VACCINE_BRAND]', text)
        
        # Remove extra punctuation but keep some meaningful ones
        text = re.sub(r'[^\w\s!?.]', ' ', text)
        text = re.sub(r'[!]{2,}', '!', text)  # Normalize exclamations
        text = re.sub(r'[?]{2,}', '?', text)  # Normalize questions
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    def prepare_data_tensorflow(self):
        """Prepare data for TensorFlow training."""
        print("Preparing data for TensorFlow training...")
        
        # Get texts and labels
        texts = self.df['cleaned_text'].values
        labels = self.df['label'].values
        
        # Convert labels to 0-indexed
        labels = labels - 1  # Convert 1,2,3 to 0,1,2
        
        # Split data
        X_train, X_temp, y_train, y_temp = train_test_split(
            texts, labels, test_size=0.3, random_state=42, stratify=labels
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
        )
          # Tokenize with improved settings
        self.tokenizer = Tokenizer(
            num_words=self.vocab_size, 
            oov_token="<OOV>",
            lower=True,
            split=' ',
            char_level=False,
            filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'  # Keep more punctuation
        )
        self.tokenizer.fit_on_texts(X_train)
        
        # Convert to sequences and pad
        X_train_seq = pad_sequences(
            self.tokenizer.texts_to_sequences(X_train), 
            maxlen=self.max_len, padding='post'
        )
        X_val_seq = pad_sequences(
            self.tokenizer.texts_to_sequences(X_val), 
            maxlen=self.max_len, padding='post'
        )
        X_test_seq = pad_sequences(
            self.tokenizer.texts_to_sequences(X_test), 
            maxlen=self.max_len, padding='post'
        )
        
        # Apply data augmentation for minority class (Against = 0)
        minority_indices = np.where(y_train == 0)[0]
        if len(minority_indices) > 0:
            print(f"Augmenting minority class 'Against' with {len(minority_indices)} samples...")
            # Duplicate minority class samples with slight variations
            augment_factor = 3  # Increase "Against" samples
            
            for i in range(augment_factor):
                # Add noise to sequences (randomly mask some tokens)
                augmented_sequences = X_train_seq[minority_indices].copy()
                for seq in augmented_sequences:
                    # Randomly mask 10% of tokens
                    mask_indices = np.random.choice(
                        len(seq), size=int(0.1 * len(seq)), replace=False
                    )
                    seq[mask_indices] = 1  # Set to OOV token
                
                X_train_seq = np.vstack([X_train_seq, augmented_sequences])
                y_train = np.hstack([y_train, np.full(len(minority_indices), 0)])
        
        print(f"Training set: {X_train_seq.shape}")
        print(f"Validation set: {X_val_seq.shape}")
        print(f"Test set: {X_test_seq.shape}")
        print(f"Training label distribution after augmentation:")
        unique, counts = np.unique(y_train, return_counts=True)
        for label, count in zip(unique, counts):
            print(f"  {label}: {count}")
        
        return (X_train_seq, y_train), (X_val_seq, y_val), (X_test_seq, y_test)
    
    def prepare_data_sklearn(self):
        """Prepare data for scikit-learn training."""
        print("Preparing data for scikit-learn training...")
        
        # Get texts and labels
        texts = self.df['cleaned_text'].values
        labels = self.df['label'].values
        
        # Split data
        X_train, X_temp, y_train, y_temp = train_test_split(
            texts, labels, test_size=0.3, random_state=42, stratify=labels
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
        )
        
        # Vectorize
        self.vectorizer = TfidfVectorizer(max_features=self.max_features, stop_words='english')
        X_train_vec = self.vectorizer.fit_transform(X_train)
        X_val_vec = self.vectorizer.transform(X_val)
        X_test_vec = self.vectorizer.transform(X_test)
        
        print(f"Training set: {X_train_vec.shape}")
        print(f"Validation set: {X_val_vec.shape}")
        print(f"Test set: {X_test_vec.shape}")
        
        return (X_train_vec, y_train), (X_val_vec, y_val), (X_test_vec, y_test)
    
    def positional_encoding(self, length, depth):
        """Create positional encoding for transformer."""
        depth = depth / 2
        positions = np.arange(length)[:, np.newaxis]
        depths = np.arange(depth)[np.newaxis, :] / depth
        
        angle_rates = 1 / (10000**depths)
        angle_rads = positions * angle_rates
        
        pos_encoding = np.concatenate([np.sin(angle_rads), np.cos(angle_rads)], axis=-1)
        return tf.cast(pos_encoding, dtype=tf.float32)
    
    def focal_loss(self, alpha=1.0, gamma=2.0):
        """
        Focal loss implementation to handle class imbalance.
        """
        def focal_loss_fixed(y_true, y_pred):
            epsilon = tf.keras.backend.epsilon()
            y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
            
            # Convert to one-hot if needed
            y_true = tf.cast(y_true, tf.int32)
            y_true_one_hot = tf.one_hot(y_true, depth=3)
            
            # Calculate focal loss
            ce = -y_true_one_hot * tf.math.log(y_pred)
            weight = alpha * y_true_one_hot * tf.math.pow((1 - y_pred), gamma)
            fl = weight * ce
            
            return tf.reduce_mean(tf.reduce_sum(fl, axis=1))
        
        return focal_loss_fixed
    
    def transformer_block(self, inputs, num_heads, ff_dim, dropout_rate=0.1):
        """Create a transformer block with improved architecture."""
        # Multi-head self-attention
        attention_output = MultiHeadAttention(
            num_heads=num_heads,
            key_dim=self.embed_dim // num_heads,
            dropout=dropout_rate
        )(inputs, inputs)
        
        # Add & Norm
        attention_output = Dropout(dropout_rate)(attention_output)
        attention_output = Add()([inputs, attention_output])
        attention_output = LayerNormalization(epsilon=1e-6)(attention_output)
        
        # Feed-forward network
        ff_output = Dense(ff_dim, activation='gelu')(attention_output)  # GELU activation
        ff_output = Dropout(dropout_rate)(ff_output)
        ff_output = Dense(self.embed_dim)(ff_output)
        
        # Add & Norm
        ff_output = Dropout(dropout_rate)(ff_output)
        ff_output = Add()([attention_output, ff_output])
        ff_output = LayerNormalization(epsilon=1e-6)(ff_output)
        
        return ff_output
    
    def build_transformer_model(self):
        """Build an improved transformer model with multiple blocks."""
        print("Building Enhanced Transformer model...")
        
        # Input
        inputs = Input(shape=(self.max_len,))
        
        # Embedding
        embedding = Embedding(
            input_dim=self.vocab_size,
            output_dim=self.embed_dim,
            embeddings_regularizer=tf.keras.regularizers.l2(0.001)
        )(inputs)
        
        # Add positional encoding
        pos_encoding = self.positional_encoding(self.max_len, self.embed_dim)
        embedding = embedding + pos_encoding
        embedding = Dropout(0.1)(embedding)
        
        # Multiple transformer blocks
        x = embedding
        for i in range(self.num_layers):
            x = self.transformer_block(
                x, 
                num_heads=self.num_heads,
                ff_dim=self.ff_dim,
                dropout_rate=0.1 + i * 0.05  # Increasing dropout in deeper layers
            )
        
        # Global pooling with both average and max pooling
        avg_pool = GlobalAveragePooling1D()(x)
        max_pool = tf.keras.layers.GlobalMaxPooling1D()(x)
        
        # Concatenate different pooling strategies
        pooled = tf.keras.layers.Concatenate()([avg_pool, max_pool])
        pooled = Dropout(0.3)(pooled)
        
        # Classification head with residual connection
        dense1 = Dense(256, activation='gelu', 
                      kernel_regularizer=tf.keras.regularizers.l2(0.001))(pooled)
        dense1 = Dropout(0.4)(dense1)
        
        dense2 = Dense(128, activation='gelu',
                      kernel_regularizer=tf.keras.regularizers.l2(0.001))(dense1)
        dense2 = Dropout(0.3)(dense2)
        
        # Output layer
        outputs = Dense(3, activation='softmax',
                       kernel_regularizer=tf.keras.regularizers.l2(0.001))(dense2)
        
        model = Model(inputs=inputs, outputs=outputs)
          # Calculate class weights for balanced training
        if hasattr(self, 'df') and self.df is not None:
            class_counts = self.df['label'].value_counts().sort_index()
            total_samples = len(self.df)
            # More aggressive class weights to handle severe imbalance
            class_weights = {}
            for i, count in enumerate(class_counts.values):
                class_weights[i] = total_samples / (len(class_counts) * count)
            
            # Boost minority class (Against) even more
            class_weights[0] = class_weights[0] * 2.0  # Double the weight for "Against"
            
            self.class_weights = class_weights
            print(f"Class weights: {self.class_weights}")
        else:
            self.class_weights = {0: 1.0, 1: 1.0, 2: 1.0}
        
        # Improved optimizer with learning rate scheduling
        initial_learning_rate = 0.001
        lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=initial_learning_rate,
            decay_steps=1000,
            alpha=0.1
        )
          # Compile with focal loss to handle class imbalance
        model.compile(
            optimizer=Adam(learning_rate=lr_schedule, beta_1=0.9, beta_2=0.98, epsilon=1e-9),
            loss=self.focal_loss(alpha=1.0, gamma=2.0),  # Use focal loss instead
            metrics=['accuracy', 'sparse_categorical_crossentropy']
        )
        
        print("\nEnhanced Transformer Model Summary:")
        model.summary()
        
        return model
    
    def build_sklearn_model(self):
        """Build a scikit-learn MLP model."""
        print("Building MLP model...")
        
        model = MLPClassifier(
            hidden_layer_sizes=(512, 256, 128),
            activation='relu',
            solver='adam',
            alpha=0.001,
            learning_rate='adaptive',
            max_iter=self.epochs,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=10
        )
        
        return model
    
    def train_model(self):
        """Train the model."""
        print("Training model...")
        
        start_time = time.time()
        if self.using_tensorflow:
            # TensorFlow training
            (X_train, y_train), (X_val, y_val), (X_test, y_test) = self.prepare_data_tensorflow()
            
            # Build model
            self.model = self.build_transformer_model()
            
            # Enhanced callbacks for better training
            if TENSORFLOW_AVAILABLE:
                callbacks = [
                    EarlyStopping(
                        monitor='val_loss',
                        patience=5,  # Increased patience
                        restore_best_weights=True,
                        verbose=1,
                        min_delta=0.001
                    ),
                    ReduceLROnPlateau(
                        monitor='val_loss',
                        factor=0.3,  # More aggressive reduction
                        patience=3,
                        min_lr=1e-7,
                        verbose=1,
                        cooldown=1
                    ),
                    tf.keras.callbacks.ModelCheckpoint(
                        filepath='best_transformer_model.h5',
                        monitor='val_accuracy',
                        save_best_only=True,
                        verbose=1
                    )
                ]
            else:
                callbacks = []
            
            # Train with class weights and improved callbacks
            self.history = self.model.fit(
                X_train, y_train,
                batch_size=self.batch_size,
                epochs=self.epochs,
                validation_data=(X_val, y_val),
                class_weight=self.class_weights,  # Use class weights
                callbacks=callbacks,
                verbose=1
            )
            
            # Store test data
            self.X_test = X_test
            self.y_test = y_test
            
            # Plot training history
            self.plot_training()
        else:
            # Scikit-learn training
            (X_train, y_train), (X_val, y_val), (X_test, y_test) = self.prepare_data_sklearn()
            
            # Build and train model
            self.model = self.build_sklearn_model()
            self.model.fit(X_train, y_train)
            
            # Store test data
            self.X_test = X_test
            self.y_test = y_test
        
        training_time = time.time() - start_time
        
        print(f"\nTraining completed in {training_time:.2f} seconds!")
        
        return {
            'training_time': training_time,
            'model_type': 'Transformer' if self.using_tensorflow else 'MLP'
        }
    
    def plot_training(self):
        """Plot training history (TensorFlow only)."""
        if not self.using_tensorflow or self.history is None:
            return
        
        plt.figure(figsize=(12, 4))
        
        # Accuracy
        plt.subplot(1, 2, 1)
        plt.plot(self.history.history['accuracy'], label='Training')
        plt.plot(self.history.history['val_accuracy'], label='Validation')
        plt.title('Transformer Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)
        
        # Loss
        plt.subplot(1, 2, 2)
        plt.plot(self.history.history['loss'], label='Training')
        plt.plot(self.history.history['val_loss'], label='Validation')
        plt.title('Transformer Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('transformer_training.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def evaluate_model(self):
        """Evaluate the trained model."""
        if self.model is None:
            print("No model trained yet!")
            return None
        
        print("Evaluating model...")
        
        if self.using_tensorflow:
            # TensorFlow evaluation
            y_pred_prob = self.model.predict(self.X_test)
            y_pred = np.argmax(y_pred_prob, axis=1)
            
            # Convert back to original labels (1, 2, 3)
            y_true_original = self.y_test + 1
            y_pred_original = y_pred + 1
        else:
            # Scikit-learn evaluation
            y_pred_original = self.model.predict(self.X_test)
            y_true_original = self.y_test
        
        # Calculate metrics
        accuracy = accuracy_score(y_true_original, y_pred_original)
        f1 = f1_score(y_true_original, y_pred_original, average='weighted')
        
        print(f"\nModel Evaluation Results:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"F1-Score: {f1:.4f}")
        
        # Classification report
        target_names = ['Against', 'Neutral', 'Favor']
        print("\nClassification Report:")
        print(classification_report(y_true_original, y_pred_original, target_names=target_names))
        
        # Confusion matrix
        cm = confusion_matrix(y_true_original, y_pred_original)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=target_names, yticklabels=target_names)
        model_name = 'Transformer' if self.using_tensorflow else 'MLP'
        plt.title(f'{model_name} - Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.savefig(f'{model_name.lower()}_confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return {
            'accuracy': accuracy,
            'f1_score': f1,
            'predictions': y_pred_original.tolist(),
            'true_labels': y_true_original.tolist()
        }
    
    def predict_stance(self, tweet_text):
        """Predict stance for a new tweet."""
        if self.model is None:
            print("No model trained yet!")
            return None
        
        # Clean text
        cleaned_text = self.clean_text(tweet_text)
        
        if self.using_tensorflow:
            # TensorFlow prediction
            sequence = self.tokenizer.texts_to_sequences([cleaned_text])
            padded_sequence = pad_sequences(sequence, maxlen=self.max_len, padding='post')
            
            prediction_prob = self.model.predict(padded_sequence, verbose=0)[0]
            predicted_class = np.argmax(prediction_prob)
            predicted_label = predicted_class + 1  # Convert back to 1, 2, 3
            confidence = float(np.max(prediction_prob))
            
            probabilities = {
                'Against': float(prediction_prob[0]),
                'Neutral': float(prediction_prob[1]),
                'Favor': float(prediction_prob[2])
            }
        else:
            # Scikit-learn prediction
            vectorized = self.vectorizer.transform([cleaned_text])
            predicted_label = self.model.predict(vectorized)[0]
            
            # Get probabilities if available
            try:
                proba = self.model.predict_proba(vectorized)[0]
                confidence = float(np.max(proba))
                probabilities = {
                    'Against': float(proba[0]) if len(proba) > 0 else 0.0,
                    'Neutral': float(proba[1]) if len(proba) > 1 else 0.0,
                    'Favor': float(proba[2]) if len(proba) > 2 else 0.0
                }
            except:
                confidence = 1.0
                probabilities = {'Against': 0.0, 'Neutral': 0.0, 'Favor': 0.0}
                probabilities[self.label_mapping[predicted_label]] = 1.0
        
        return {
            'text': tweet_text,
            'cleaned_text': cleaned_text,
            'predicted_stance': self.label_mapping[predicted_label],
            'predicted_label': predicted_label,
            'confidence': confidence,
            'probabilities': probabilities
        }
    
    def generate_report(self):
        """Generate a comprehensive report."""
        if self.model is None:
            print("No model trained yet!")
            return None
        
        eval_results = self.evaluate_model()
        
        report = {
            'model_type': 'Transformer' if self.using_tensorflow else 'MLP',
            'backend': 'TensorFlow' if self.using_tensorflow else 'scikit-learn',
            'dataset_samples': len(self.df),
            'accuracy': eval_results['accuracy'],
            'f1_score': eval_results['f1_score']
        }
        
        if self.using_tensorflow:
            report.update({
                'vocab_size': self.vocab_size,
                'embedding_dim': self.embed_dim,
                'num_heads': self.num_heads,
                'ff_dim': self.ff_dim,
                'max_sequence_length': self.max_len,
                'epochs': self.epochs,
                'batch_size': self.batch_size,
                'total_parameters': self.model.count_params()
            })
        else:
            report.update({
                'max_features': self.max_features,
                'hidden_layers': self.model.hidden_layer_sizes,
                'max_iterations': self.epochs
            })
        
        # Save report
        filename = 'transformer_report.json' if self.using_tensorflow else 'mlp_report.json'
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"Report saved to '{filename}'")
        return report

def main():
    """Main function to run the transformer."""
    print("COVID-19 Vaccine Stance Detection - Transformer Model")
    print("="*60)
    
    # Initialize detector
    detector = TransformerStanceDetector()
    
    # Train model
    training_results = detector.train_model()
    
    # Evaluate model
    eval_results = detector.evaluate_model()
    
    # Generate report
    report = detector.generate_report()
    
    # Test with sample tweets
    sample_tweets = [
        "I got my COVID vaccine today and I feel great!",
        "COVID vaccines are dangerous and experimental",
        "The vaccine rollout is happening slowly",
        "Pfizer vaccine is highly effective",
        "I'm worried about vaccine side effects"
    ]
    
    print("\n" + "="*60)
    print("Testing with Sample Tweets:")
    print("="*60)
    
    for i, tweet in enumerate(sample_tweets, 1):
        result = detector.predict_stance(tweet)
        if result:
            print(f"\n{i}. Tweet: {tweet}")
            print(f"   Predicted: {result['predicted_stance']} (confidence: {result['confidence']:.3f})")
    
    print(f"\nTransformer training complete!")
    return detector

if __name__ == "__main__":
    detector = main()
