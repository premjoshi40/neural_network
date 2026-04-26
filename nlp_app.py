# =============================================================================
# NLP Application: Text Classification and Sentiment Analysis
# =============================================================================
# This app demonstrates basic NLP tasks including:
# 1. Text preprocessing (cleaning, tokenization, stopword removal)
# 2. Sentiment analysis (positive/negative classification)
# 3. Text classification (categorizing text into predefined categories)
# =============================================================================

# -------------------------------
# IMPORT REQUIRED LIBRARIES
# -------------------------------
import re  # Regular expressions for text pattern matching and cleaning
import string  # String manipulation for punctuation handling
from collections import Counter  # Count word frequencies for analysis

# Try to import transformers for advanced NLP; fall back to simple approach if unavailable
try:
    from transformers import pipeline  # Hugging Face transformers for pre-trained models
    TRANSFORMERS_AVAILABLE = True  # Flag to check if transformers is installed
except ImportError:
    # If transformers is not installed, use simple rule-based approach
    TRANSFORMERS_AVAILABLE = False  # Set flag to False
    print("Note: transformers not installed. Using simple sentiment analysis.")


# -------------------------------
# TEXT PREPROCESSING FUNCTIONS
# -------------------------------

def clean_text(text):
    """
    Clean and normalize input text by removing special characters and extra spaces.
    """
    # Convert all text to lowercase for consistent processing
    text = text.lower()
    
    # Remove URLs starting with http, https, or www
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    
    # Remove mentions (@username) and hashtags (#hashtag)
    text = re.sub(r'@\w+|#\w+', '', text)
    
    # Remove extra whitespace and leading/trailing spaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Return the cleaned text
    return text


def tokenize(text):
    """
    Split text into individual words (tokens).
    """
    # Split the text by spaces to create a list of words
    tokens = text.split()
    
    # Return the list of tokens
    return tokens


def remove_stopwords(tokens):
    """
    Remove common words that don't add meaningful information.
    """
    # Define a set of common English stopwords to filter out
    stopwords = {
        'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you',
        "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself',
        'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her',
        'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them',
        'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom',
        'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are',
        'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had',
        'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and',
        'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at',
        'by', 'for', 'with', 'about', 'against', 'between', 'into',
        'through', 'during', 'before', 'after', 'above', 'below', 'to',
        'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under',
        'again', 'further', 'then', 'once', 'here', 'there', 'when',
        'where', 'why', 'how', 'all', 'each', 'few', 'more', 'most',
        'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same',
        'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just',
        'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm',
        'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't",
        'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn',
        "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn',
        "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't",
        'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't",
        'won', "won't", 'wouldn', "wouldn't"
    }
    
    # Filter out stopwords from the tokens list
    filtered_tokens = [token for token in tokens if token not in stopwords]
    
    # Return the filtered tokens
    return filtered_tokens


def preprocess_text(text):
    """
    Apply all preprocessing steps: clean, tokenize, and remove stopwords.
    """
    # Step 1: Clean the text (remove URLs, mentions, etc.)
    cleaned = clean_text(text)
    
    # Step 2: Tokenize the cleaned text into words
    tokens = tokenize(cleaned)
    
    # Step 3: Remove stopwords to keep meaningful words
    filtered_tokens = remove_stopwords(tokens)
    
    # Return the preprocessed tokens
    return filtered_tokens


# -------------------------------
# SIMPLE SENTIMENT ANALYSIS (Rule-based)
# -------------------------------

def simple_sentiment_analysis(text):
    """
    Perform sentiment analysis using a simple word-matching approach.
    This is a basic rule-based method without machine learning.
    """
    # Define lists of positive words commonly associated with positive sentiment
    positive_words = {
        'good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic',
        'love', 'loved', 'happy', 'joy', 'best', 'beautiful', 'awesome',
        'perfect', 'brilliant', 'outstanding', 'superb', 'positive',
        'pleasant', 'delightful', 'impressive', 'nice', 'lovely'
    }
    
    # Define lists of negative words commonly associated with negative sentiment
    negative_words = {
        'bad', 'terrible', 'awful', 'horrible', 'hate', 'hated', 'sad',
        'worst', 'poor', 'disappointing', 'boring', 'ugly', 'dreadful',
        'negative', 'unpleasant', 'pathetic', 'useless', 'stupid',
        'annoying', 'frustrating', 'mess', 'wrong', 'fail'
    }
    
    # Preprocess the input text to get meaningful tokens
    tokens = preprocess_text(text)
    
    # Count how many positive words appear in the text
    positive_count = sum(1 for word in tokens if word in positive_words)
    
    # Count how many negative words appear in the text
    negative_count = sum(1 for word in tokens if word in negative_words)
    
    # Determine sentiment based on the difference between positive and negative counts
    if positive_count > negative_count:
        # More positive words → positive sentiment
        sentiment = 'positive'
        confidence = positive_count / (positive_count + negative_count + 1)
    elif negative_count > positive_count:
        # More negative words → negative sentiment
        sentiment = 'negative'
        confidence = negative_count / (positive_count + negative_count + 1)
    else:
        # Equal or no sentiment words → neutral
        sentiment = 'neutral'
        confidence = 0.5
    
    # Return the sentiment label and confidence score
    return sentiment, confidence


# -------------------------------
# ADVANCED SENTIMENT ANALYSIS (Using Transformers)
# -------------------------------

def advanced_sentiment_analysis(text):
    """
    Perform sentiment analysis using a pre-trained transformer model.
    This provides more accurate results than the simple approach.
    """
    # Check if transformers library is available
    if not TRANSFORMERS_AVAILABLE:
        # If not available, fall back to simple sentiment analysis
        return simple_sentiment_analysis(text)
    
    # Create a sentiment-analysis pipeline with a pre-trained model
    # This loads a model fine-tuned for sentiment classification
    sentiment_pipeline = pipeline(
        "sentiment-analysis",  # Task type: analyze sentiment
        model="distilbert-base-uncased-finetuned-sst-2-english"  # Pre-trained model
    )
    
    # Run the text through the sentiment analysis model
    result = sentiment_pipeline(text)[0]  # Get the first (and only) result
    
    # Extract the label (POSITIVE or NEGATIVE) and convert to lowercase
    label = result['label'].lower()
    
    # Extract the confidence score (probability)
    score = result['score']
    
    # Return the sentiment and confidence
    return label, score


# -------------------------------
# TEXT CLASSIFICATION
# -------------------------------

def classify_text(text):
    """
    Classify text into predefined categories based on keywords.
    This is a simple rule-based classification approach.
    """
    # Preprocess the text to get meaningful tokens
    tokens = preprocess_text(text)
    
    # Convert tokens to a set for faster keyword matching
    token_set = set(tokens)
    
    # Define category keywords for classification
    categories = {
        'technology': {
            'computer', 'software', 'hardware', 'ai', 'machine', 'learning',
            'data', 'code', 'programming', 'developer', 'tech', 'internet',
            'digital', 'algorithm', 'robot', 'automation', 'app', 'web'
        },
        'sports': {
            'game', 'player', 'team', 'score', 'win', 'match', 'football',
            'basketball', 'soccer', 'baseball', 'tennis', 'olympic', 'sport',
            'championship', 'tournament', 'coach', 'ball', 'goal'
        },
        'business': {
            'company', 'market', 'stock', 'money', 'profit', 'investment',
            'business', 'finance', 'economy', 'trade', 'sale', 'customer',
            'product', 'service', 'brand', 'growth', 'revenue', 'cost'
        },
        'health': {
            'health', 'doctor', 'medicine', 'hospital', 'patient', 'disease',
            'treatment', 'symptom', 'wellness', 'fitness', 'diet', 'exercise',
            'medical', 'healthcare', 'nurse', 'therapy', 'clinic', 'virus'
        },
        'entertainment': {
            'movie', 'music', 'film', 'actor', 'actress', 'song', 'show',
            'tv', 'television', 'celebrity', 'star', 'entertainment', 'drama',
            'comedy', 'video', 'game', 'art', 'theater', 'performance'
        }
    }
    
    # Initialize scores for each category
    category_scores = {category: 0 for category in categories}
    
    # Count matches for each category
    for category, keywords in categories.items():
        # Count how many keywords from this category appear in the text
        category_scores[category] = len(token_set.intersection(keywords))
    
    # Find the category with the highest score
    if max(category_scores.values()) > 0:
        # Get the category with maximum matches
        predicted_category = max(category_scores, key=category_scores.get)
    else:
        # No category matched → classify as 'other'
        predicted_category = 'other'
    
    # Return the predicted category
    return predicted_category


# -------------------------------
# WORD FREQUENCY ANALYSIS
# -------------------------------

def get_word_frequency(text, top_n=10):
    """
    Analyze text to find the most frequently occurring words.
    """
    # Preprocess the text to get meaningful tokens
    tokens = preprocess_text(text)
    
    # Count the frequency of each word
    word_counts = Counter(tokens)
    
    # Get the most common words
    most_common = word_counts.most_common(top_n)
    
    # Return the top N most frequent words
    return most_common


# -------------------------------
# MAIN APPLICATION
# -------------------------------

def main():
    """
    Main function to run the NLP application with example texts.
    """
    # Print a header for the NLP Application
    print("=" * 60)
    print("       NLP Application: Text Analysis Demo")
    print("=" * 60)
    
    # -------------------------------
    # Example 1: Technology-related text
    # -------------------------------
    # Sample text about technology for analysis
    text1 = "The new artificial intelligence software is amazing! " \
            "Machine learning algorithms help developers build " \
            "better computer programs every day."
    
    # Print the example text
    print("\n" + "-" * 60)
    print("Example 1:")
    print(text1)
    print("-" * 60)
    
    # Perform sentiment analysis on example 1
    sentiment, confidence = advanced_sentiment_analysis(text1)
    print(f"Sentiment: {sentiment} (confidence: {confidence:.2f})")
    
    # Classify the text into a category
    category = classify_text(text1)
    print(f"Category: {category}")
    
    # Get word frequency analysis
    word_freq = get_word_frequency(text1)
    print(f"Top words: {word_freq}")
    
    
    # -------------------------------
    # Example 2: Negative sentiment text
    # -------------------------------
    # Sample negative text for sentiment analysis
    text2 = "This product is terrible! The quality is awful and " \
            "the service was disappointing. I hate it."
    
    # Print the example text
    print("\n" + "-" * 60)
    print("Example 2:")
    print(text2)
    print("-" * 60)
    
    # Perform sentiment analysis on example 2
    sentiment, confidence = advanced_sentiment_analysis(text2)
    print(f"Sentiment: {sentiment} (confidence: {confidence:.2f})")
    
    # Classify the text into a category
    category = classify_text(text2)
    print(f"Category: {category}")
    
    
    # -------------------------------
    # Example 3: Business-related text
    # -------------------------------
    # Sample business text for classification
    text3 = "The company reported strong profit growth in the " \
            "first quarter. Stock prices increased as market " \
            "conditions improved for business investment."
    
    # Print the example text
    print("\n" + "-" * 60)
    print("Example 3:")
    print(text3)
    print("-" * 60)
    
    # Perform sentiment analysis on example 3
    sentiment, confidence = advanced_sentiment_analysis(text3)
    print(f"Sentiment: {sentiment} (confidence: {confidence:.2f})")
    
    # Classify the text into a category
    category = classify_text(text3)
    print(f"Category: {category}")
    
    
    # -------------------------------
    # Example 4: Sports-related text
    # -------------------------------
    # Sample sports text for classification
    text4 = "The football team won the championship game! " \
            "The player scored an amazing goal in the final " \
            "minute to secure the victory."
    
    # Print the example text
    print("\n" + "-" * 60)
    print("Example 4:")
    print(text4)
    print("-" * 60)
    
    # Perform sentiment analysis on example 4
    sentiment, confidence = advanced_sentiment_analysis(text4)
    print(f"Sentiment: {sentiment} (confidence: {confidence:.2f})")
    
    # Classify the text into a category
    category = classify_text(text4)
    print(f"Category: {category}")
    
    
    # -------------------------------
    # Example 5: User Input
    # -------------------------------
    # Allow user to enter their own text for analysis
    print("\n" + "=" * 60)
    print("Enter your own text for analysis:")
    print("=" * 60)
    
    # Read input from user
    user_text = input("> ")
    
    # Check if user provided any text
    if user_text.strip():
        # Perform sentiment analysis on user input
        sentiment, confidence = advanced_sentiment_analysis(user_text)
        print(f"\nSentiment: {sentiment} (confidence: {confidence:.2f})")
        
        # Classify the user input
        category = classify_text(user_text)
        print(f"Category: {category}")
        
        # Show word frequency for user input
        word_freq = get_word_frequency(user_text)
        print(f"Top words: {word_freq}")
    else:
        # No input provided
        print("No text entered.")
    
    
    # Print closing message
    print("\n" + "=" * 60)
    print("       NLP Application Demo Complete")
    print("=" * 60)


# -------------------------------
# APPLICATION ENTRY POINT
# -------------------------------
# This block ensures the main function runs only when the script is executed directly
if __name__ == "__main__":
    # Call the main function to start the application
    main()