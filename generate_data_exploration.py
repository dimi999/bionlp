import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load and process data
df = pd.read_csv("merged_covid_vaccine_tweets.csv")
df['stance'] = df['label'].map({1: 'Against', 2: 'Neutral', 3: 'Favor'})
df['word_count'] = df['tweet_text'].str.split().str.len()
df['char_count'] = df['tweet_text'].str.len()

# Create plots
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
colors = {'Against': '#d62728', 'Neutral': '#7f7f7f', 'Favor': '#2ca02c'}

# Stance Distribution
stance_counts = df['stance'].value_counts()
axes[0, 0].bar(stance_counts.index, stance_counts.values, 
               color=[colors[s] for s in stance_counts.index])
axes[0, 0].set_title('Stance Distribution')

# Tweet Length Distribution
axes[0, 1].hist(df['word_count'], bins=30, color='#1f77b4', alpha=0.7)
axes[0, 1].set_title('Tweet Length Distribution')
axes[0, 1].set_xlabel('Word Count')

# Character Count Distribution
axes[0, 2].hist(df['char_count'], bins=30, color='#ff7f0e', alpha=0.7)
axes[0, 2].set_title('Character Count Distribution')
axes[0, 2].set_xlabel('Character Count')

# Word Count by Stance
for stance in ['Against', 'Neutral', 'Favor']:
    data = df[df['stance'] == stance]['word_count']
    axes[1, 0].hist(data, bins=30, alpha=0.6, label=stance, color=colors[stance])
axes[1, 0].set_title('Word Count by Stance')
axes[1, 0].legend()

# Character Count by Stance
for stance in ['Against', 'Neutral', 'Favor']:
    data = df[df['stance'] == stance]['char_count']
    axes[1, 1].hist(data, bins=30, alpha=0.6, label=stance, color=colors[stance])
axes[1, 1].set_title('Character Count by Stance')
axes[1, 1].legend()

# Average Metrics by Stance
avg_metrics = df.groupby('stance')[['word_count', 'char_count']].mean()
x = np.arange(len(avg_metrics.index))
axes[1, 2].bar(x - 0.2, avg_metrics['word_count'], 0.4, label='word_count', color='#1f77b4')
axes[1, 2].bar(x + 0.2, avg_metrics['char_count'], 0.4, label='char_count', color='#ff7f0e')
axes[1, 2].set_title('Average Metrics by Stance')
axes[1, 2].set_xticks(x)
axes[1, 2].set_xticklabels(avg_metrics.index)
axes[1, 2].legend()

plt.tight_layout()
plt.savefig('data_exploration.png', dpi=300, bbox_inches='tight')
plt.show()
