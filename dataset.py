import json
import csv
import pandas as pd

def parse_json_to_csv(json_file_path, output_csv_path):
    """
    Parse the sample_data.json file and convert it to CSV format
    compatible with the existing COVID-19 vaccine tweets dataset.
    """
    
    # Load the JSON data
    with open(json_file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    
    # Create a list to store the processed data
    processed_data = []
    
    for tweet in data:
        tweet_id = tweet.get('id', '')
        annotations = tweet.get('annotations', [])
        
        # If there are annotations, process them
        if annotations:
            for annotation in annotations:
                reason = annotation.get('Reason', '')
                stance = annotation.get('Stance', '')
                
                # Map stance to numeric labels to match the existing dataset
                # Based on the existing dataset: 1=Against, 2=Neutral, 3=Favor
                label_mapping = {
                    'Against': 1,
                    'Neutral': 2,
                    'Favor': 3
                }
                
                label = label_mapping.get(stance, 2)  # Default to neutral if stance not recognized
                
                # Create tweet text from the annotation information
                tweet_text = reason
                
                processed_data.append({
                    'tweet_id': tweet_id,
                    'label': label,
                    'tweet_text': tweet_text
                })
        else:
            # If no annotations, add with neutral label and empty text
            processed_data.append({
                'tweet_id': tweet_id,
                'label': 2,  # Neutral
                'tweet_text': ''
            })
    
    # Convert to DataFrame and save as CSV
    df = pd.DataFrame(processed_data)
    df.to_csv(output_csv_path, index=False, encoding='utf-8')
    
    print(f"Successfully converted {len(processed_data)} records to CSV format")
    print(f"Output saved to: {output_csv_path}")
    
    return df

def merge_with_existing_csv(new_csv_path, existing_csv_path, merged_output_path):
    """
    Merge the newly created CSV with the existing COVID-19 vaccine tweets CSV.
    """
    
    # Load both CSV files
    new_df = pd.read_csv(new_csv_path)
    existing_df = pd.read_csv(existing_csv_path, encoding='unicode_escape')
    
    print(f"New data: {len(new_df)} records")
    print(f"Existing data: {len(existing_df)} records")
    
    # Merge the datasets
    merged_df = pd.concat([existing_df, new_df], ignore_index=True)
    
    # Remove duplicates based on tweet_id if any
    merged_df = merged_df.drop_duplicates(subset=['tweet_id'], keep='first')
    
    # Save the merged dataset
    merged_df.to_csv(merged_output_path, index=False, encoding='utf-8')
    
    print(f"Merged dataset: {len(merged_df)} records")
    print(f"Merged output saved to: {merged_output_path}")
    
    return merged_df

# Main execution
if __name__ == "__main__":
    # File paths
    json_file = "sample_data.json"
    new_csv_file = "sample_data_converted.csv"
    existing_csv_file = "covid-19_vaccine_tweets_with_sentiment.csv"
    merged_csv_file = "merged_covid_vaccine_tweets.csv"
    
    # Convert JSON to CSV
    print("Converting JSON to CSV...")
    df = parse_json_to_csv(json_file, new_csv_file)
    
    # Display sample of converted data
    print("\nSample of converted data:")
    print(df.head())
    
    # Merge with existing CSV
    print("\nMerging with existing dataset...")
    merged_df = merge_with_existing_csv(new_csv_file, existing_csv_file, merged_csv_file)
    
    # Display statistics
    print("\nDataset Statistics:")
    print("Label distribution in merged dataset:")
    print(merged_df['label'].value_counts().sort_index())