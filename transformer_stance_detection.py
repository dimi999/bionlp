import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Dense, Dropout, Input, Embedding, GlobalAveragePooling1D,
    MultiHeadAttention, LayerNormalization
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import re
import warnings
warnings.filterwarnings('ignore')

class TransformerBlock(tf.keras.layers.Layer):    
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.5):
        super().__init__()
        self.att = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential([
            Dense(ff_dim, activation="relu", kernel_regularizer=l2(0.01)),
            Dropout(rate),
            Dense(embed_dim, kernel_regularizer=l2(0.01))
        ])
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)

    def call(self, inputs, training=None):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        
        ffn_output = self.ffn(out1, training=training)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)


class TransformerStanceDetector:    
    def __init__(self, data_path='merged_covid_vaccine_tweets.csv'):
        self.data_path = data_path
        self.df = None
        self.tokenizer = None
        self.model = None
        
        # Anti-overfitting hyperparameters
        self.vocab_size = 3000  # Reduced from 5000
        self.maxlen = 80        # Reduced from 100
        self.embed_dim = 32     # Reduced from 64
        self.num_heads = 2      # Reduced from 4
        self.ff_dim = 64        # Reduced from 128
        self.dropout_rate = 0.5 # Increased from 0.2
    
    def load_data(self):
        self.df = pd.read_csv(self.data_path)
        
        self.df = self.df.dropna(subset=['tweet_text'])
        self.df['cleaned_text'] = self.df['tweet_text'].apply(self.clean_text)
        self.df = self.df[self.df['cleaned_text'].str.len() > 0]
        self.df['encoded_label'] = self.df['label'] - 1  # Convert 1,2,3 to 0,1,2
        
    def clean_text(self, text):
        text = str(text).lower()
        text = re.sub(r'http\S+|www\S+|@\w+|#\w+', '', text)
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        text = ' '.join(text.split())
        return text
    
    def prepare_data(self):
        self.tokenizer = Tokenizer(num_words=self.vocab_size, oov_token="<OOV>")
        self.tokenizer.fit_on_texts(self.df['cleaned_text'])
        
        sequences = self.tokenizer.texts_to_sequences(self.df['cleaned_text'])
        X = pad_sequences(sequences, maxlen=self.maxlen, padding='post')
        y = self.df['encoded_label'].values
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2)
        
        print(f"Training: {self.X_train.shape}, Test: {self.X_test.shape}")
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def build_model(self):
        inputs = Input(shape=(self.maxlen,))
        x = Embedding(self.vocab_size, self.embed_dim, embeddings_regularizer=l2(0.01))(inputs)
        
        positions = tf.range(start=0, limit=self.maxlen, delta=1)
        position_embeddings = Embedding(self.maxlen, self.embed_dim, embeddings_regularizer=l2(0.01))(positions)
        x = x + position_embeddings
        x = Dropout(self.dropout_rate)(x)
        
        x = TransformerBlock(self.embed_dim, self.num_heads, self.ff_dim, self.dropout_rate)(x)
        
        x = GlobalAveragePooling1D()(x)
        x = Dropout(self.dropout_rate)(x)
        x = Dense(16, activation="relu", kernel_regularizer=l2(0.01))(x) 
        x = Dropout(self.dropout_rate)(x)
        outputs = Dense(3, activation="softmax")(x)
        
        self.model = Model(inputs, outputs)
        self.model.compile(
            optimizer=Adam(learning_rate=0.0005),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"]
        )
        self.model.summary()        
        return self.model
    
    def train(self, epochs=100, batch_size=64):
        self.prepare_data()
        self.build_model()
        
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=8,          
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,         
                patience=4,
                min_lr=1e-6,
                verbose=1
            )
        ]
        
        self.history = self.model.fit(
            self.X_train, self.y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(self.X_test, self.y_test),
            callbacks=callbacks,
            verbose=1
        )
        
        return self.history
    
    def evaluate(self):
        y_pred_proba = self.model.predict(self.X_test)
        y_pred = np.argmax(y_pred_proba, axis=1)
        
        accuracy = accuracy_score(self.y_test, y_pred)
        print(f"Test Accuracy: {accuracy:.4f}")
        
        target_names = ['Against', 'Neutral', 'Favor']
        print(classification_report(self.y_test, y_pred, target_names=target_names))
        
        cm = confusion_matrix(self.y_test, y_pred)
        plt.figure(figsize=(8, 6))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion Matrix')
        plt.colorbar()
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names)
        plt.yticks(tick_marks, target_names)
        
        # Add text annotations
        thresh = cm.max() / 2.
        for i, j in np.ndindex(cm.shape):
            plt.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
        
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.tight_layout()
        plt.savefig('transformer_confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        _, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        ax1.plot(self.history.history['accuracy'], label='Training')
        ax1.plot(self.history.history['val_accuracy'], label='Validation')
        ax1.set_title('Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        
        ax2.plot(self.history.history['loss'], label='Training')
        ax2.plot(self.history.history['val_loss'], label='Validation')
        ax2.set_title('Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig('transformer_training_regularized.png', dpi=300, bbox_inches='tight')
        plt.show()        
        return y_pred, y_pred_proba


detector = TransformerStanceDetector()
detector.load_data()

detector.train(epochs=20, batch_size=64)
detector.evaluate()
