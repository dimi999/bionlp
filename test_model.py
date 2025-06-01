#!/usr/bin/env python3

from transformer_stance_detection_new import TransformerStanceDetector

# Test the current model
detector = TransformerStanceDetector()

# Check if model exists
if detector.model is not None:
    # Test with sample tweets
    test_tweets = [
        "I got my COVID vaccine today and I feel great!",
        "COVID vaccines are dangerous and experimental",
        "The vaccine rollout is happening slowly",
        "Pfizer vaccine is highly effective against COVID",
        "I'm worried about vaccine side effects"
    ]
    
    print("Testing current model predictions:")
    print("="*50)
    
    for i, tweet in enumerate(test_tweets, 1):
        result = detector.predict_stance(tweet)
        if result:
            print(f"{i}. Tweet: {tweet}")
            print(f"   Predicted: {result['predicted_stance']} (confidence: {result['confidence']:.3f})")
            print(f"   Probabilities: {result['probabilities']}")
            print()
else:
    print("No trained model found. Training a new model...")
    detector.train_model()
    detector.evaluate_model()
