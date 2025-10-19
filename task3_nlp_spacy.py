# task3_nlp_spacy.py
import spacy
import pandas as pd
import random
from collections import Counter
import matplotlib.pyplot as plt
from textblob import TextBlob  # For more advanced sentiment analysis

# Load spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Downloading spaCy model...")
    from spacy.cli import download
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

# Load real Amazon reviews dataset
reviews_data = []
with open("datasets/test.ft.txt", "r", encoding="utf-8") as file:
    for i, line in enumerate(file):
        if i >= 50:  # Process first 50 reviews for efficient demo
            break

        parts = line.strip().split()
        label = parts[0]  # __label__1 (negative) or __label__2 (positive)
        review_text = " ".join(parts[1:])  # The review text

        # Convert label to rating (1 for negative, 5 for positive)
        rating = 1 if label == "__label__1" else 5

        reviews_data.append({
            "review": review_text,
            "rating": rating
        })

print(f"Loaded {len(reviews_data)} real Amazon reviews from dataset")

def advanced_sentiment_analysis(text):
    """Advanced sentiment analysis using multiple approaches"""
    doc = nlp(text)
    
    # Rule-based sentiment with spaCy
    positive_words = {
        'love', 'amazing', 'excellent', 'perfect', 'good', 'great', 'awesome',
        'outstanding', 'superb', 'impressive', 'best', 'recommended', 'exceptional'
    }
    negative_words = {
        'terrible', 'disappointed', 'bad', 'poor', 'awful', 'horrible',
        'regret', 'cheap', 'bugs', 'overheats', 'loud'
    }
    
    # Count sentiment words
    positive_count = sum(1 for token in doc if token.lemma_.lower() in positive_words)
    negative_count = sum(1 for token in doc if token.lemma_.lower() in negative_words)
    
    # TextBlob sentiment for comparison
    blob = TextBlob(text)
    textblob_polarity = blob.sentiment.polarity
    textblob_subjectivity = blob.sentiment.subjectivity
    
    # Determine final sentiment
    if positive_count > negative_count:
        rule_sentiment = "positive"
    elif negative_count > positive_count:
        rule_sentiment = "negative"
    else:
        rule_sentiment = "neutral"
    
    # Combine approaches
    if textblob_polarity > 0.1:
        final_sentiment = "positive"
    elif textblob_polarity < -0.1:
        final_sentiment = "negative"
    else:
        final_sentiment = rule_sentiment  # Fall back to rule-based
    
    return {
        'rule_sentiment': rule_sentiment,
        'textblob_sentiment': 'positive' if textblob_polarity > 0.1 else 'negative' if textblob_polarity < -0.1 else 'neutral',
        'final_sentiment': final_sentiment,
        'positive_words': positive_count,
        'negative_words': negative_count,
        'textblob_polarity': textblob_polarity,
        'textblob_subjectivity': textblob_subjectivity
    }

def analyze_reviews_with_ner(reviews):
    """Perform comprehensive NLP analysis on reviews"""
    results = []
    
    for review_data in reviews:
        review_text = review_data["review"]
        rating = review_data["rating"]
        doc = nlp(review_text)
        
        # Named Entity Recognition
        entities = [(ent.text, ent.label_, ent.lemma_) for ent in doc.ents]
        
        # Extract product names and brands (custom rules)
        products = []
        brands = []
        
        for ent in doc.ents:
            if ent.label_ in ["ORG", "PRODUCT"]:
                if any(brand in ent.text for brand in ["Apple", "Samsung", "Sony", "Microsoft", "Amazon", "Google", "Dell", "HP"]):
                    brands.append(ent.text)
                else:
                    products.append(ent.text)
        
        # Sentiment analysis
        sentiment_result = advanced_sentiment_analysis(review_text)
        
        # Additional NLP features
        nouns = [token.lemma_ for token in doc if token.pos_ == "NOUN"]
        adjectives = [token.lemma_ for token in doc if token.pos_ == "ADJ"]
        
        results.append({
            'review': review_text,
            'rating': rating,
            'entities': entities,
            'brands': list(set(brands)),
            'products': list(set(products)),
            'sentiment': sentiment_result,
            'nouns': nouns[:5],  # Top 5 nouns
            'adjectives': adjectives[:5],  # Top 5 adjectives
            'length': len(review_text.split())
        })
    
    return results

# Analyze reviews
print("ANALYZING: Analyzing Reviews with spaCy NLP...")
analysis_results = analyze_reviews_with_ner(reviews_data)

# Display results
print("\n" + "="*80)
print("COMPREHENSIVE REVIEW ANALYSIS RESULTS")
print("="*80)

for i, result in enumerate(analysis_results, 1):
    print(f"\nREVIEW Review {i} (Rating: {result['rating']}/5):")
    print(f"   Text: {result['review']}")
    print(f"   Sentiment: {result['sentiment']['final_sentiment'].upper()}")
    print(f"   Brands Detected: {result['brands']}")
    print(f"   Products Mentioned: {result['products']}")
    print(f"   Key Entities: {result['entities']}")
    print(f"   Key Nouns: {result['nouns']}")
    print(f"   Key Adjectives: {result['adjectives']}")
    print(f"   Analysis Details:")
    print(f"     - Rule-based: {result['sentiment']['rule_sentiment']}")
    print(f"     - TextBlob: {result['sentiment']['textblob_sentiment']} (polarity: {result['sentiment']['textblob_polarity']:.2f})")
    print(f"     - Positive words: {result['sentiment']['positive_words']}")
    print(f"     - Negative words: {result['sentiment']['negative_words']}")
    print("-" * 80)

# Summary Statistics
sentiments = [result['sentiment']['final_sentiment'] for result in analysis_results]
sentiment_counts = Counter(sentiments)

# Visualization
plt.figure(figsize=(15, 10))

# Sentiment distribution
plt.subplot(2, 2, 1)
plt.bar(sentiment_counts.keys(), sentiment_counts.values(), color=['red', 'gray', 'green'])
plt.title('Sentiment Distribution Across Reviews')
plt.xlabel('Sentiment')
plt.ylabel('Count')

# Rating vs Sentiment
ratings = [result['rating'] for result in analysis_results]
sentiment_numeric = [1 if s == 'positive' else -1 if s == 'negative' else 0 for s in sentiments]

plt.subplot(2, 2, 2)
plt.scatter(ratings, sentiment_numeric, alpha=0.7, s=100)
plt.xlabel('Rating (1-5)')
plt.ylabel('Sentiment (-1 to 1)')
plt.title('Rating vs Sentiment Analysis')
plt.grid(True, alpha=0.3)

# Entity frequency
all_entities = [ent[0] for result in analysis_results for ent in result['entities']]
entity_counts = Counter(all_entities).most_common(10)

plt.subplot(2, 2, 3)
if entity_counts:
    entities, counts = zip(*entity_counts)
    plt.barh(entities, counts, color='skyblue')
    plt.title('Top 10 Named Entities')
    plt.xlabel('Frequency')

# Review length distribution
lengths = [result['length'] for result in analysis_results]

plt.subplot(2, 2, 4)
plt.hist(lengths, bins=10, alpha=0.7, color='orange')
plt.title('Distribution of Review Lengths')
plt.xlabel('Number of Words')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()

# Advanced NER Analysis
print("\nADVANCED ADVANCED NER INSIGHTS:")
all_brands = [brand for result in analysis_results for brand in result['brands']]
brand_counts = Counter(all_brands)

if brand_counts:
    print("Brand Mentions:")
    for brand, count in brand_counts.most_common():
        print(f"  {brand}: {count} mentions")

# Save results to DataFrame
df_results = pd.DataFrame([{
    'review': r['review'],
    'rating': r['rating'],
    'sentiment': r['sentiment']['final_sentiment'],
    'brands': ', '.join(r['brands']),
    'products': ', '.join(r['products']),
    'entity_count': len(r['entities'])
} for r in analysis_results])

print("\nRESULTS Results DataFrame:")
print(df_results)

print("\nCOMPLETE NLP Analysis Complete! spaCy successfully processed all reviews.")