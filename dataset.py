import json
import pandas as pd

def parse_json_to_csv(json_file_path, output_csv_path):
    with open(json_file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    
    processed_data = []
    
    for tweet in data:
        tweet_id = tweet.get('id', '')
        annotations = tweet.get('annotations', [])
        
        if annotations:
            for annotation in annotations:
                reason = annotation.get('Reason', '')
                stance = annotation.get('Stance', '')
                
                label_mapping = {
                    'Against': 1,
                    'Neutral': 2,
                    'Favor': 3
                }
                
                label = label_mapping.get(stance, 2) 
                tweet_text = reason
                
                processed_data.append({
                    'tweet_id': tweet_id,
                    'label': label,
                    'tweet_text': tweet_text
                })
        else:
            processed_data.append({
                'tweet_id': tweet_id,
                'label': 2,  
                'tweet_text': ''
            })
    
    df = pd.DataFrame(processed_data)
    df.to_csv(output_csv_path, index=False, encoding='utf-8')
    
    print(f"Successfully converted {len(processed_data)} records to CSV format")
    print(f"Output saved to: {output_csv_path}")
    
    return df

def merge_with_existing_csv(new_csv_path, existing_csv_path, merged_output_path):
    new_df = pd.read_csv(new_csv_path)
    existing_df = pd.read_csv(existing_csv_path, encoding='unicode_escape')
    
    print(f"New data: {len(new_df)} records")
    print(f"Existing data: {len(existing_df)} records")
    
    merged_df = pd.concat([existing_df, new_df], ignore_index=True)
    merged_df = merged_df.drop_duplicates(subset=['tweet_id'], keep='first')
    
    merged_df.to_csv(merged_output_path, index=False, encoding='utf-8')
    
    print(f"Merged dataset: {len(merged_df)} records")
    print(f"Merged output saved to: {merged_output_path}")
    
    return merged_df

if __name__ == "__main__":
    json_file = "sample_data.json"
    new_csv_file = "sample_data_converted.csv"
    existing_csv_file = "covid-19_vaccine_tweets_with_sentiment.csv"
    merged_csv_file = "merged_covid_vaccine_tweets.csv"
    
    df = parse_json_to_csv(json_file, new_csv_file)
    print(df.head())
    merged_df = merge_with_existing_csv(new_csv_file, existing_csv_file, merged_csv_file)
    
    print(merged_df['label'].value_counts().sort_index())