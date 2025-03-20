# Install libraries if needed: pip install nltk matplotlib seaborn
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns

# Download NLTK data (run once)
nltk.download('punkt')
nltk.download('stopwords')


# Step 1: Load the processed text
def load_text(csv_path):
    df = pd.read_csv(csv_path)
    return df['text'][0]  # Assuming one row of text


# Step 2: Analyze the text
def analyze_text(text):
    # Tokenize into words and sentences
    words = word_tokenize(text.lower())
    sentences = sent_tokenize(text)

    # Remove stopwords (e.g., "the", "and") and non-alphabetic tokens
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word.isalpha() and word not in stop_words]

    # Calculate basic stats
    word_count = len(words)
    sentence_count = len(sentences)
    avg_word_per_sentence = word_count / sentence_count if sentence_count > 0 else 0

    # Get top 10 most common words
    word_freq = Counter(words).most_common(10)

    return {
        'word_count': word_count,
        'sentence_count': sentence_count,
        'avg_word_per_sentence': avg_word_per_sentence,
        'word_freq': word_freq
    }


# Step 3: Visualize results
def visualize_word_freq(word_freq):
    words, counts = zip(*word_freq)  # Unzip into two lists
    plt.figure(figsize=(10, 6))
    sns.barplot(x=list(counts), y=list(words))
    plt.title('Top 10 Most Frequent Words in Financial Document')
    plt.xlabel('Frequency')
    plt.ylabel('Words')
    plt.tight_layout()
    plt.savefig('word_freq_plot.png')  # Save the plot
    plt.show()


# Example usage
csv_path = "processed_text.csv"  # Your CSV from Step 1
text = load_text(csv_path)
analysis = analyze_text(text)

# Print stats
print(f"Total Words: {analysis['word_count']}")
print(f"Total Sentences: {analysis['sentence_count']}")
print(f"Avg Words per Sentence: {analysis['avg_word_per_sentence']:.2f}")
print("Top 10 Words:", analysis['word_freq'])

# Visualize
visualize_word_freq(analysis['word_freq'])
