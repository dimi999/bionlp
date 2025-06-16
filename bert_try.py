"""
Simplified BERT Stance Detection for COVID-19 Vaccine Tweets
"""

import os
import warnings
import re
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
warnings.filterwarnings('ignore')

try:
    from transformers import TFAutoModelForSequenceClassification, AutoTokenizer
except ImportError:
    print("Install required packages: pip install transformers")
    exit(1)

def clean_text(text):
    text = re.sub(r'http\S+|@\w+|#(\w+)', r'\1', str(text))
    return re.sub(r'\s+', ' ', text).strip()

def load_data(file_path='merged_covid_vaccine_tweets.csv'):
    df = pd.read_csv(file_path)
    df['label'] = df['label'].map({1: 0, 2: 1, 3: 2})
    df['text'] = df['tweet_text'].apply(clean_text)
    df.dropna(subset=['text', 'label'], inplace=True)
    return df['text'].values, df['label'].values

def prepare_inputs(texts, labels, tokenizer, max_length=128):
    encodings = tokenizer(
        list(texts), truncation=True, padding='max_length',
        max_length=max_length, return_tensors='tf'
    )
    return encodings, tf.constant(labels, dtype=tf.int32)

def train_bert(texts, labels, epochs=3, batch_size=16, max_length=128):
    X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2)

    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    model = TFAutoModelForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)

    train_enc, train_lab = prepare_inputs(X_train, y_train, tokenizer, max_length)
    test_enc, test_lab = prepare_inputs(X_test, y_test, tokenizer, max_length)

    train_ds = tf.data.Dataset.from_tensor_slices((dict(train_enc), train_lab)).batch(batch_size)
    test_ds = tf.data.Dataset.from_tensor_slices((dict(test_enc), test_lab)).batch(batch_size)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=2e-5),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )

    model.fit(train_ds, validation_data=test_ds, epochs=epochs, verbose=1)

    preds = model.predict(test_ds).logits
    y_pred = np.argmax(preds, axis=1)

    print("\nTest Accuracy: {:.4f}".format(np.mean(y_pred == y_test)))
    print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=["Against", "Neutral", "Favor"]))
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

    return model, tokenizer

def predict(texts, model, tokenizer, max_length=128):
    clean_texts = [clean_text(t) for t in texts]
    encodings = tokenizer(clean_texts, truncation=True, padding='max_length', max_length=max_length, return_tensors='tf')
    logits = model(encodings).logits
    probs = tf.nn.softmax(logits, axis=-1)
    labels = tf.argmax(probs, axis=-1).numpy()
    label_names = ["Against", "Neutral", "Favor"]

    results = []
    for i, text in enumerate(texts):
        results.append({
            'text': text,
            'prediction': label_names[labels[i]],
            'confidence': float(tf.reduce_max(probs[i])),
            'probabilities': dict(zip(label_names, probs[i].numpy().tolist()))
        })
    return results

texts, labels = load_data()
model, tokenizer = train_bert(texts, labels)

model.save_pretrained('./simple_bert_model')
tokenizer.save_pretrained('./simple_bert_model')
