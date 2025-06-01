"""
Model Comparison Script: SVM vs Transformer for Stance Detection

This script runs both SVM and Transformer models on the same dataset
and provides a comprehensive comparison of their performance.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import time
import json
import warnings
warnings.filterwarnings('ignore')

# Import our custom models
from svm_stance_detection import SVMStanceDetector
from transformer_stance_detection import TransformerStanceDetector

class ModelComparison:
    """Class to compare SVM and Transformer models for stance detection."""
    
    def __init__(self, data_path='merged_covid_vaccine_tweets.csv'):
        self.data_path = data_path
        self.svm_detector = None
        self.transformer_detector = None
        self.results = {}
    
    def run_svm_experiment(self):
        """Run the SVM experiment and collect results."""
        print("="*60)
        print("RUNNING SVM EXPERIMENT")
        print("="*60)
        
        start_time = time.time()
        
        # Initialize and train SVM
        self.svm_detector = SVMStanceDetector(self.data_path)
        
        # Train with default parameters first
        svm_accuracy = self.svm_detector.train_svm()
        
        # # Perform hyperparameter tuning
        # best_params = self.svm_detector.hyperparameter_tuning()
        
        # Evaluate the tuned model
        eval_results = self.svm_detector.evaluate_svm()
        
        end_time = time.time()
        training_time = end_time - start_time
        
        # Calculate detailed metrics
        y_pred = self.svm_detector.svm_model.predict(self.svm_detector.X_test)
        
        # Convert back to original labels
        y_true_original = [self.svm_detector.y_test[i] for i in range(len(self.svm_detector.y_test))]
        y_pred_original = [y_pred[i] for i in range(len(y_pred))]
        
        accuracy = accuracy_score(y_true_original, y_pred_original)
        f1 = f1_score(y_true_original, y_pred_original, average='weighted')
        precision = precision_score(y_true_original, y_pred_original, average='weighted')
        recall = recall_score(y_true_original, y_pred_original, average='weighted')
        
        self.results['svm'] = {
            'accuracy': accuracy,
            'f1_score': f1,
            'precision': precision,
            'recall': recall,
            'training_time': training_time,
            'model_type': 'Support Vector Machine',
            'predictions': y_pred_original,
            'true_labels': y_true_original
        }
        
        print(f"\nSVM Results:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"F1-Score: {f1:.4f}")
        print(f"Training Time: {training_time:.2f} seconds")
        
        return self.results['svm']
    
    def run_transformer_experiment(self):
        """Run the Transformer experiment and collect results."""
        print("\n" + "="*60)
        print("RUNNING TRANSFORMER EXPERIMENT")
        print("="*60)
        
        start_time = time.time()
        
        # Initialize and train Transformer
        self.transformer_detector = TransformerStanceDetector(self.data_path)
        
        # Train the model
        training_results = self.transformer_detector.train_transformer()
        
        # Evaluate the model
        eval_results = self.transformer_detector.evaluate_transformer()
        
        end_time = time.time()
        training_time = end_time - start_time
        
        accuracy = eval_results['accuracy']
        f1 = eval_results['f1_score']
        
        # Calculate precision and recall
        y_true = eval_results['true_labels']
        y_pred = eval_results['predictions']
        precision = precision_score(y_true, y_pred, average='weighted')
        recall = recall_score(y_true, y_pred, average='weighted')
        
        self.results['transformer'] = {
            'accuracy': accuracy,
            'f1_score': f1,
            'precision': precision,
            'recall': recall,
            'training_time': training_time,
            'model_type': 'Transformer (TensorFlow)',
            'predictions': y_pred,
            'true_labels': y_true,
            'training_results': training_results
        }
        
        print(f"\nTransformer Results:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"F1-Score: {f1:.4f}")
        print(f"Training Time: {training_time:.2f} seconds")
        
        return self.results['transformer']
    
    def compare_models(self):
        """Compare the performance of both models."""
        if 'svm' not in self.results or 'transformer' not in self.results:
            print("Both models need to be trained before comparison!")
            return None
        
        print("\n" + "="*60)
        print("MODEL COMPARISON RESULTS")
        print("="*60)
        
        # Create comparison DataFrame
        comparison_data = {
            'Metric': ['Accuracy', 'F1-Score', 'Precision', 'Recall', 'Training Time (s)'],
            'SVM': [
                self.results['svm']['accuracy'],
                self.results['svm']['f1_score'],
                self.results['svm']['precision'],
                self.results['svm']['recall'],
                self.results['svm']['training_time']
            ],
            'Transformer': [
                self.results['transformer']['accuracy'],
                self.results['transformer']['f1_score'],
                self.results['transformer']['precision'],
                self.results['transformer']['recall'],
                self.results['transformer']['training_time']
            ]
        }
        
        comparison_df = pd.DataFrame(comparison_data)
        print("\nPerformance Comparison:")
        print(comparison_df.to_string(index=False, float_format='%.4f'))
        
        # Calculate improvements
        print("\nImprovement Analysis:")
        for metric in ['accuracy', 'f1_score', 'precision', 'recall']:
            svm_val = self.results['svm'][metric]
            transformer_val = self.results['transformer'][metric]
            improvement = ((transformer_val - svm_val) / svm_val) * 100
            better_model = "Transformer" if transformer_val > svm_val else "SVM"
            print(f"{metric.replace('_', ' ').title()}: {better_model} is better by {abs(improvement):.2f}%")
        
        # Training time comparison
        svm_time = self.results['svm']['training_time']
        transformer_time = self.results['transformer']['training_time']
        time_ratio = transformer_time / svm_time
        print(f"Training Time: Transformer takes {time_ratio:.1f}x longer than SVM")
        
        return comparison_df
    
    def visualize_comparison(self):
        """Create visualizations comparing both models."""
        if 'svm' not in self.results or 'transformer' not in self.results:
            print("Both models need to be trained before visualization!")
            return None
        
        # Create comprehensive comparison plot
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. Performance Metrics Comparison
        metrics = ['Accuracy', 'F1-Score', 'Precision', 'Recall']
        svm_scores = [self.results['svm'][m.lower().replace('-', '_')] for m in metrics]
        transformer_scores = [self.results['transformer'][m.lower().replace('-', '_')] for m in metrics]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        axes[0, 0].bar(x - width/2, svm_scores, width, label='SVM', alpha=0.8, color='skyblue')
        axes[0, 0].bar(x + width/2, transformer_scores, width, label='Transformer', alpha=0.8, color='lightcoral')
        axes[0, 0].set_xlabel('Metrics')
        axes[0, 0].set_ylabel('Score')
        axes[0, 0].set_title('Performance Metrics Comparison')
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels(metrics, rotation=45)
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Add value labels on bars
        for i, (svm_score, transformer_score) in enumerate(zip(svm_scores, transformer_scores)):
            axes[0, 0].text(i - width/2, svm_score + 0.01, f'{svm_score:.3f}', 
                           ha='center', va='bottom', fontsize=9)
            axes[0, 0].text(i + width/2, transformer_score + 0.01, f'{transformer_score:.3f}', 
                           ha='center', va='bottom', fontsize=9)
        
        # 2. Training Time Comparison
        models = ['SVM', 'Transformer']
        times = [self.results['svm']['training_time'], self.results['transformer']['training_time']]
        
        bars = axes[0, 1].bar(models, times, color=['skyblue', 'lightcoral'], alpha=0.8)
        axes[0, 1].set_ylabel('Training Time (seconds)')
        axes[0, 1].set_title('Training Time Comparison')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Add value labels
        for bar, time_val in zip(bars, times):
            axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02*max(times), 
                           f'{time_val:.1f}s', ha='center', va='bottom')
        
        # 3. Confusion Matrix - SVM
        from sklearn.metrics import confusion_matrix
        
        svm_cm = confusion_matrix(self.results['svm']['true_labels'], 
                                 self.results['svm']['predictions'])
        sns.heatmap(svm_cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 2],
                   xticklabels=['Against', 'Neutral', 'Favor'],
                   yticklabels=['Against', 'Neutral', 'Favor'])
        axes[0, 2].set_title('SVM Confusion Matrix')
        axes[0, 2].set_xlabel('Predicted')
        axes[0, 2].set_ylabel('Actual')
        
        # 4. Confusion Matrix - Transformer
        transformer_cm = confusion_matrix(self.results['transformer']['true_labels'], 
                                        self.results['transformer']['predictions'])
        sns.heatmap(transformer_cm, annot=True, fmt='d', cmap='Reds', ax=axes[1, 0],
                   xticklabels=['Against', 'Neutral', 'Favor'],
                   yticklabels=['Against', 'Neutral', 'Favor'])
        axes[1, 0].set_title('Transformer Confusion Matrix')
        axes[1, 0].set_xlabel('Predicted')
        axes[1, 0].set_ylabel('Actual')
        
        # 5. Model Complexity Comparison
        complexity_metrics = ['Training Time', 'Model Complexity']
        
        # Normalize for comparison (0-1 scale)
        svm_complexity = [
            self.results['svm']['training_time'] / max(times),
            0.3  # SVM is relatively simple
        ]
        transformer_complexity = [
            self.results['transformer']['training_time'] / max(times),
            1.0  # Transformer is more complex
        ]
        
        x = np.arange(len(complexity_metrics))
        axes[1, 1].bar(x - width/2, svm_complexity, width, label='SVM', alpha=0.8, color='skyblue')
        axes[1, 1].bar(x + width/2, transformer_complexity, width, label='Transformer', alpha=0.8, color='lightcoral')
        axes[1, 1].set_xlabel('Aspects')
        axes[1, 1].set_ylabel('Relative Score (0-1)')
        axes[1, 1].set_title('Model Complexity Comparison')
        axes[1, 1].set_xticks(x)
        axes[1, 1].set_xticklabels(complexity_metrics)
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        # 6. Performance vs Time Trade-off
        accuracy_scores = [self.results['svm']['accuracy'], self.results['transformer']['accuracy']]
        training_times = [self.results['svm']['training_time'], self.results['transformer']['training_time']]
        
        axes[1, 2].scatter(training_times, accuracy_scores, 
                          c=['skyblue', 'lightcoral'], s=200, alpha=0.8, edgecolors='black')
        
        # Add model labels
        for i, model in enumerate(['SVM', 'Transformer']):
            axes[1, 2].annotate(model, (training_times[i], accuracy_scores[i]), 
                               xytext=(10, 10), textcoords='offset points', fontsize=12)
        
        axes[1, 2].set_xlabel('Training Time (seconds)')
        axes[1, 2].set_ylabel('Accuracy')
        axes[1, 2].set_title('Performance vs Training Time Trade-off')
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('model_comparison_comprehensive.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return fig
    
    def test_sample_predictions(self):
        """Test both models on sample tweets and compare predictions."""
        sample_tweets = [
            "I got my COVID vaccine today and I feel great! Thank you science!",
            "COVID vaccines are dangerous and causing serious side effects",
            "The vaccine rollout is happening slowly but steadily",
            "Pfizer and Moderna vaccines are highly effective against COVID-19",
            "I'm worried about the long-term effects of these vaccines",
            "Vaccination is the best way to protect ourselves and others",
            "The government is forcing us to take experimental vaccines",
            "Scientists have done extensive research on vaccine safety"
        ]
        
        print("\n" + "="*80)
        print("SAMPLE PREDICTIONS COMPARISON")
        print("="*80)
        
        comparison_results = []
        
        for i, tweet in enumerate(sample_tweets, 1):
            print(f"\n{i}. Tweet: {tweet}")
            print("-" * 60)
            
            # SVM prediction
            if self.svm_detector:
                svm_result = self.svm_detector.predict_stance(tweet)
                svm_stance = svm_result['predicted_stance'] if svm_result else "Error"
                svm_confidence = svm_result['confidence'] if svm_result else 0
            else:
                svm_stance, svm_confidence = "Not Available", 0
            
            # Transformer prediction
            if self.transformer_detector:
                transformer_result = self.transformer_detector.predict_stance(tweet)
                transformer_stance = transformer_result['predicted_stance'] if transformer_result else "Error"
                transformer_confidence = transformer_result['confidence'] if transformer_result else 0
            else:
                transformer_stance, transformer_confidence = "Not Available", 0
            
            print(f"SVM Prediction:         {svm_stance} (confidence: {svm_confidence:.3f})")
            print(f"Transformer Prediction: {transformer_stance} (confidence: {transformer_confidence:.3f})")
            
            # Check agreement
            agreement = "✓ AGREE" if svm_stance == transformer_stance else "✗ DISAGREE"
            print(f"Agreement: {agreement}")
            
            comparison_results.append({
                'tweet': tweet,
                'svm_prediction': svm_stance,
                'svm_confidence': svm_confidence,
                'transformer_prediction': transformer_stance,
                'transformer_confidence': transformer_confidence,
                'agreement': svm_stance == transformer_stance
            })
        
        # Summary statistics
        total_tweets = len(comparison_results)
        agreements = sum(1 for result in comparison_results if result['agreement'])
        agreement_rate = agreements / total_tweets * 100
        
        print(f"\n" + "="*60)
        print(f"PREDICTION AGREEMENT SUMMARY")
        print(f"="*60)
        print(f"Total sample tweets: {total_tweets}")
        print(f"Agreements: {agreements}")
        print(f"Disagreements: {total_tweets - agreements}")
        print(f"Agreement rate: {agreement_rate:.1f}%")
        
        return comparison_results
    
    def generate_comparison_report(self):
        """Generate a comprehensive comparison report."""
        if 'svm' not in self.results or 'transformer' not in self.results:
            print("Both models need to be trained before generating report!")
            return None
        
        print("Generating comprehensive comparison report...")
        
        # Create comprehensive report
        report = {
            'experiment_info': {
                'dataset': self.data_path,
                'comparison_timestamp': pd.Timestamp.now().isoformat(),
                'models_compared': ['SVM', 'Transformer (TensorFlow)']
            },
            'svm_results': self.results['svm'],
            'transformer_results': self.results['transformer'],
            'comparison_summary': {
                'better_accuracy': 'Transformer' if self.results['transformer']['accuracy'] > self.results['svm']['accuracy'] else 'SVM',
                'better_f1': 'Transformer' if self.results['transformer']['f1_score'] > self.results['svm']['f1_score'] else 'SVM',
                'faster_training': 'SVM' if self.results['svm']['training_time'] < self.results['transformer']['training_time'] else 'Transformer',
                'accuracy_difference': abs(self.results['transformer']['accuracy'] - self.results['svm']['accuracy']),
                'f1_difference': abs(self.results['transformer']['f1_score'] - self.results['svm']['f1_score']),
                'training_time_ratio': self.results['transformer']['training_time'] / self.results['svm']['training_time']
            }
        }
        
        # Save report
        with open('model_comparison_report.json', 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print("Comparison report saved to 'model_comparison_report.json'")
        
        # Print summary
        print("\n" + "="*60)
        print("FINAL COMPARISON SUMMARY")
        print("="*60)
        
        summary = report['comparison_summary']
        print(f"Best Accuracy: {summary['better_accuracy']}")
        print(f"Best F1-Score: {summary['better_f1']}")
        print(f"Faster Training: {summary['faster_training']}")
        print(f"Accuracy Difference: {summary['accuracy_difference']:.4f}")
        print(f"F1-Score Difference: {summary['f1_difference']:.4f}")
        print(f"Training Time Ratio (T/S): {summary['training_time_ratio']:.1f}x")
        
        return report
    
    def run_complete_comparison(self):
        """Run the complete model comparison pipeline."""
        print("Starting Complete Model Comparison Pipeline")
        print("="*80)
        
        # Run SVM experiment
        svm_results = self.run_svm_experiment()
        
        # Run Transformer experiment
        transformer_results = self.run_transformer_experiment()
        
        # Compare models
        comparison_df = self.compare_models()
        
        # Create visualizations
        self.visualize_comparison()
        
        # Test sample predictions
        prediction_comparison = self.test_sample_predictions()
        
        # Generate final report
        final_report = self.generate_comparison_report()
        
        print("\n" + "="*80)
        print("COMPLETE COMPARISON FINISHED!")
        print("="*80)
        print("Files generated:")
        print("- model_comparison_comprehensive.png")
        print("- model_comparison_report.json")
        print("- svm_model_report.json")
        print("- transformer_model_report.json")
        
        return {
            'svm_results': svm_results,
            'transformer_results': transformer_results,
            'comparison_df': comparison_df,
            'prediction_comparison': prediction_comparison,
            'final_report': final_report
        }

def main():
    """Main function to run the model comparison."""
    print("COVID-19 Vaccine Stance Detection - Model Comparison")
    print("SVM vs Transformer Architecture")
    print("="*80)
    
    # Initialize comparison
    comparator = ModelComparison()
    
    # Run complete comparison
    results = comparator.run_complete_comparison()
    
    return comparator, results

if __name__ == "__main__":
    comparator, results = main()