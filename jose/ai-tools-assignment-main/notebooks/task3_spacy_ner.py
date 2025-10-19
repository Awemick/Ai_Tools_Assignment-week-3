import spacy

# Load spaCy English model
nlp = spacy.load("en_core_web_sm")

# Define simple positive and negative keywords for sentiment analysis
positive_words = {"love", "amazing", "great", "highly recommend", "helpful", "easy to use", "excellent", "perfect"}
negative_words = {"disappointing", "poor", "bad", "drains too fast", "terrible", "worst", "awful"}

def analyze_sentiment(text):
    """Rule-based sentiment analysis based on keyword presence."""
    text_lower = text.lower()
    if any(kw in text_lower for kw in positive_words):
        return "Positive"
    elif any(kw in text_lower for kw in negative_words):
        return "Negative"
    else:
        return "Neutral"

# Path to the plain text file
file_path = "test.ft.txt"

# Open and read the plain text file line by line
with open(file_path, "r", encoding="utf-8") as file:
    # Process first 10 reviews for demonstration
    for i, line in enumerate(file):
        if i >= 10:
            break
        
        # Each line starts with a label like __label__2 followed by the review text
        # Split by whitespace, remove the label part(s), then join back the review text
        parts = line.strip().split()
        # Remove all parts that start with '__label__' prefix
        review_words = [word for word in parts if not word.startswith("__label__")]
        review_text = " ".join(review_words)
        
        doc = nlp(review_text)
        
        # Extract entities labeled as ORG, PRODUCT, or PERSON
        entities = [(ent.text, ent.label_) for ent in doc.ents if ent.label_ in ("ORG", "PRODUCT", "PERSON")]
        
        sentiment = analyze_sentiment(review_text)
        
        print(f"Review {i+1}: {review_text}")
        print("Extracted entities:", entities)
        print("Sentiment:", sentiment)
        print("-" * 60)
