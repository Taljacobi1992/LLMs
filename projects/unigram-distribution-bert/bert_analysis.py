# imports
from collections import Counter
from transformers import AutoTokenizer
import re
import spacy
import matplotlib.pyplot as plt
import seaborn as sns

# Load transformer tokenizer and spacy English model
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
nlp = spacy.load("en_core_web_sm")

# Load and preprocess text
with open("Parasomnia.txt", "r", encoding="utf-8") as file:
    text = file.read().lower()

# Since we are working with one big chunk of text, we want to split it into sentences, because the tranformenr we are using, BERT has a token length limit of 512
sentences = re.split(r'(?<=[.!?])\s+', text)

tokens = []
for sentence in sentences:
    tokens.extend(tokenizer.tokenize(sentence))

# Filter out subword tokens (e.g., ##ing, ##ed) and non-alphabetic tokens
filtered_tokens = [t for t in tokens if not t.startswith("##") and t.isalpha()]

# Process tokens with SpaCy to remove stopwords and non-significant tokens
doc = nlp(" ".join(filtered_tokens))  # Convert filtered tokens back to a string for SpaCy processing
tokens = [token.lemma_ for token in doc if not token.is_stop and token.is_alpha]
# Finally filter out tokens below 2 letters
final_tokens = [t for t in tokens if len(t) > 2]  

# Compute unigram frequency distribution
unigram_counts = Counter(final_tokens)

# Top 10 most common words
most_common = unigram_counts.most_common(10)

# Separate words and their counts for plotting
words, counts = zip(*most_common)

# Set Seaborn style
sns.set(style="whitegrid")

# Plot the unigram counts using Seaborn's barplot
plt.figure(figsize=(10, 6))
sns.barplot(x=words, y=counts, palette="Blues_d")
plt.xlabel('Words')
plt.ylabel('Frequency')
plt.title('Top 10 Most Common words in Parasomnia')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()

# Show the plot
plt.show()




