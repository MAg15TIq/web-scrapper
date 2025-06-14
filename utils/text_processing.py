"""
Advanced text processing utilities for Phase 5 enhancements.
"""
import re
import string
import unicodedata
from typing import Dict, List, Optional, Any, Union, Tuple
from datetime import datetime
import logging

# NLP libraries
try:
    import spacy
    from spacy.lang.en.stop_words import STOP_WORDS
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False

try:
    from textblob import TextBlob
    TEXTBLOB_AVAILABLE = True
except ImportError:
    TEXTBLOB_AVAILABLE = False

try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize, sent_tokenize
    from nltk.stem import PorterStemmer, WordNetLemmatizer
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False

logger = logging.getLogger(__name__)


class AdvancedTextProcessor:
    """Advanced text processing utilities."""
    
    def __init__(self):
        """Initialize the text processor."""
        self.nlp = None
        self.stemmer = None
        self.lemmatizer = None
        self.stop_words = set()
        
        # Initialize NLP tools
        self._initialize_nlp_tools()
        
        # Text patterns
        self.patterns = {
            "email": re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
            "phone": re.compile(r'(\+?1?[-.\s]?)?\(?([0-9]{3})\)?[-.\s]?([0-9]{3})[-.\s]?([0-9]{4})'),
            "url": re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'),
            "social_security": re.compile(r'\b\d{3}-\d{2}-\d{4}\b'),
            "credit_card": re.compile(r'\b(?:\d{4}[-\s]?){3}\d{4}\b'),
            "date": re.compile(r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b'),
            "time": re.compile(r'\b\d{1,2}:\d{2}(?::\d{2})?\s?(?:AM|PM|am|pm)?\b'),
            "currency": re.compile(r'\$\d+(?:,\d{3})*(?:\.\d{2})?'),
            "percentage": re.compile(r'\b\d+(?:\.\d+)?%\b'),
            "hashtag": re.compile(r'#\w+'),
            "mention": re.compile(r'@\w+'),
            "ip_address": re.compile(r'\b(?:\d{1,3}\.){3}\d{1,3}\b')
        }
        
        # Language detection patterns
        self.language_patterns = {
            "english": re.compile(r'\b(?:the|and|or|but|in|on|at|to|for|of|with|by)\b', re.IGNORECASE),
            "spanish": re.compile(r'\b(?:el|la|los|las|y|o|pero|en|de|con|por|para)\b', re.IGNORECASE),
            "french": re.compile(r'\b(?:le|la|les|et|ou|mais|dans|de|avec|par|pour)\b', re.IGNORECASE),
            "german": re.compile(r'\b(?:der|die|das|und|oder|aber|in|von|mit|für)\b', re.IGNORECASE),
            "italian": re.compile(r'\b(?:il|la|lo|gli|le|e|o|ma|in|di|con|per)\b', re.IGNORECASE)
        }
    
    def _initialize_nlp_tools(self):
        """Initialize NLP tools and models."""
        try:
            # Initialize spaCy
            if SPACY_AVAILABLE:
                try:
                    self.nlp = spacy.load("en_core_web_sm")
                    self.stop_words.update(STOP_WORDS)
                    logger.info("Initialized spaCy model")
                except OSError:
                    logger.warning("spaCy model not found. Install with: python -m spacy download en_core_web_sm")
            
            # Initialize NLTK
            if NLTK_AVAILABLE:
                try:
                    self.stemmer = PorterStemmer()
                    self.lemmatizer = WordNetLemmatizer()
                    
                    # Download required NLTK data
                    try:
                        nltk.data.find('tokenizers/punkt')
                        nltk.data.find('corpora/stopwords')
                        nltk.data.find('corpora/wordnet')
                    except LookupError:
                        logger.info("Downloading required NLTK data...")
                        nltk.download('punkt', quiet=True)
                        nltk.download('stopwords', quiet=True)
                        nltk.download('wordnet', quiet=True)
                    
                    self.stop_words.update(stopwords.words('english'))
                    logger.info("Initialized NLTK tools")
                    
                except Exception as e:
                    logger.warning(f"Failed to initialize NLTK: {e}")
            
            # Fallback stop words
            if not self.stop_words:
                self.stop_words = {
                    'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
                    'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
                    'to', 'was', 'will', 'with', 'the', 'this', 'but', 'they', 'have',
                    'had', 'what', 'said', 'each', 'which', 'she', 'do', 'how', 'their'
                }
                
        except Exception as e:
            logger.error(f"Error initializing NLP tools: {e}")
    
    def clean_text(self, text: str, options: Dict[str, bool] = None) -> str:
        """
        Clean and normalize text.
        
        Args:
            text: Input text to clean
            options: Cleaning options
            
        Returns:
            Cleaned text
        """
        if not text:
            return ""
        
        options = options or {}
        
        # Default cleaning options
        default_options = {
            "remove_html": True,
            "remove_urls": True,
            "remove_emails": False,
            "remove_phone_numbers": False,
            "normalize_whitespace": True,
            "normalize_unicode": True,
            "remove_punctuation": False,
            "lowercase": False,
            "remove_numbers": False,
            "remove_stop_words": False
        }
        
        # Merge with provided options
        clean_options = {**default_options, **options}
        
        # Remove HTML tags
        if clean_options["remove_html"]:
            text = re.sub(r'<[^>]+>', '', text)
        
        # Remove URLs
        if clean_options["remove_urls"]:
            text = self.patterns["url"].sub('', text)
        
        # Remove emails
        if clean_options["remove_emails"]:
            text = self.patterns["email"].sub('', text)
        
        # Remove phone numbers
        if clean_options["remove_phone_numbers"]:
            text = self.patterns["phone"].sub('', text)
        
        # Normalize Unicode
        if clean_options["normalize_unicode"]:
            text = unicodedata.normalize('NFKD', text)
        
        # Remove numbers
        if clean_options["remove_numbers"]:
            text = re.sub(r'\d+', '', text)
        
        # Remove punctuation
        if clean_options["remove_punctuation"]:
            text = text.translate(str.maketrans('', '', string.punctuation))
        
        # Convert to lowercase
        if clean_options["lowercase"]:
            text = text.lower()
        
        # Normalize whitespace
        if clean_options["normalize_whitespace"]:
            text = re.sub(r'\s+', ' ', text).strip()
        
        # Remove stop words
        if clean_options["remove_stop_words"]:
            words = text.split()
            text = ' '.join([word for word in words if word.lower() not in self.stop_words])
        
        return text
    
    def extract_entities(self, text: str, entity_types: List[str] = None) -> Dict[str, List[str]]:
        """
        Extract various entities from text using regex patterns.
        
        Args:
            text: Input text
            entity_types: Specific entity types to extract
            
        Returns:
            Dictionary of extracted entities by type
        """
        entities = {}
        
        # Default entity types to extract
        if entity_types is None:
            entity_types = list(self.patterns.keys())
        
        for entity_type in entity_types:
            if entity_type in self.patterns:
                matches = self.patterns[entity_type].findall(text)
                if matches:
                    # Handle different match formats
                    if entity_type == "phone":
                        # Phone regex returns tuples, join them
                        entities[entity_type] = [''.join(match) for match in matches]
                    else:
                        entities[entity_type] = matches
        
        return entities
    
    def tokenize_text(self, text: str, method: str = "spacy") -> List[str]:
        """
        Tokenize text into words.
        
        Args:
            text: Input text
            method: Tokenization method ("spacy", "nltk", "simple")
            
        Returns:
            List of tokens
        """
        if not text:
            return []
        
        if method == "spacy" and self.nlp:
            doc = self.nlp(text)
            return [token.text for token in doc if not token.is_space]
        
        elif method == "nltk" and NLTK_AVAILABLE:
            return word_tokenize(text)
        
        else:
            # Simple tokenization
            return re.findall(r'\b\w+\b', text.lower())
    
    def extract_sentences(self, text: str, method: str = "spacy") -> List[str]:
        """
        Extract sentences from text.
        
        Args:
            text: Input text
            method: Sentence extraction method
            
        Returns:
            List of sentences
        """
        if not text:
            return []
        
        if method == "spacy" and self.nlp:
            doc = self.nlp(text)
            return [sent.text.strip() for sent in doc.sents]
        
        elif method == "nltk" and NLTK_AVAILABLE:
            return sent_tokenize(text)
        
        else:
            # Simple sentence splitting
            sentences = re.split(r'[.!?]+', text)
            return [sent.strip() for sent in sentences if sent.strip()]
    
    def extract_keywords(self, text: str, max_keywords: int = 10, method: str = "frequency") -> List[Tuple[str, float]]:
        """
        Extract keywords from text.
        
        Args:
            text: Input text
            max_keywords: Maximum number of keywords to return
            method: Keyword extraction method
            
        Returns:
            List of (keyword, score) tuples
        """
        if not text:
            return []
        
        # Clean and tokenize text
        cleaned_text = self.clean_text(text, {
            "lowercase": True,
            "remove_punctuation": True,
            "normalize_whitespace": True
        })
        
        tokens = self.tokenize_text(cleaned_text)
        
        # Filter tokens
        filtered_tokens = [
            token for token in tokens 
            if len(token) > 2 and token.lower() not in self.stop_words
        ]
        
        if method == "frequency":
            # Simple frequency-based keyword extraction
            from collections import Counter
            token_counts = Counter(filtered_tokens)
            total_tokens = len(filtered_tokens)
            
            keywords = [
                (token, count / total_tokens) 
                for token, count in token_counts.most_common(max_keywords)
            ]
            
        elif method == "tfidf" and self.nlp:
            # TF-IDF based extraction using spaCy
            doc = self.nlp(text)
            
            # Calculate term frequencies
            token_freq = {}
            for token in doc:
                if (not token.is_stop and not token.is_punct and 
                    not token.is_space and len(token.text) > 2):
                    token_freq[token.lemma_.lower()] = token_freq.get(token.lemma_.lower(), 0) + 1
            
            # Simple TF scoring (without IDF since we have single document)
            total_tokens = sum(token_freq.values())
            keywords = [
                (token, freq / total_tokens) 
                for token, freq in sorted(token_freq.items(), key=lambda x: x[1], reverse=True)[:max_keywords]
            ]
            
        else:
            # Fallback to frequency method
            from collections import Counter
            token_counts = Counter(filtered_tokens)
            total_tokens = len(filtered_tokens)
            
            keywords = [
                (token, count / total_tokens) 
                for token, count in token_counts.most_common(max_keywords)
            ]
        
        return keywords
    
    def detect_language_simple(self, text: str) -> Dict[str, float]:
        """
        Simple language detection using pattern matching.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary of language scores
        """
        if not text:
            return {"english": 0.0}
        
        text_lower = text.lower()
        language_scores = {}
        
        for language, pattern in self.language_patterns.items():
            matches = len(pattern.findall(text_lower))
            words = len(text_lower.split())
            score = matches / words if words > 0 else 0.0
            language_scores[language] = score
        
        # Normalize scores
        total_score = sum(language_scores.values())
        if total_score > 0:
            language_scores = {lang: score / total_score for lang, score in language_scores.items()}
        
        return language_scores
    
    def calculate_readability(self, text: str) -> Dict[str, float]:
        """
        Calculate readability metrics for text.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary of readability metrics
        """
        if not text:
            return {}
        
        sentences = self.extract_sentences(text)
        words = self.tokenize_text(text)
        
        if not sentences or not words:
            return {}
        
        # Basic metrics
        num_sentences = len(sentences)
        num_words = len(words)
        num_chars = len(text)
        
        # Calculate syllables (approximation)
        def count_syllables(word):
            word = word.lower()
            vowels = "aeiouy"
            syllable_count = 0
            prev_was_vowel = False
            
            for char in word:
                is_vowel = char in vowels
                if is_vowel and not prev_was_vowel:
                    syllable_count += 1
                prev_was_vowel = is_vowel
            
            # Handle silent e
            if word.endswith('e') and syllable_count > 1:
                syllable_count -= 1
            
            return max(1, syllable_count)
        
        total_syllables = sum(count_syllables(word) for word in words)
        
        # Flesch Reading Ease Score
        if num_sentences > 0 and num_words > 0:
            avg_sentence_length = num_words / num_sentences
            avg_syllables_per_word = total_syllables / num_words
            
            flesch_score = 206.835 - (1.015 * avg_sentence_length) - (84.6 * avg_syllables_per_word)
            flesch_score = max(0, min(100, flesch_score))  # Clamp between 0-100
        else:
            flesch_score = 0
        
        # Flesch-Kincaid Grade Level
        if num_sentences > 0 and num_words > 0:
            fk_grade = (0.39 * avg_sentence_length) + (11.8 * avg_syllables_per_word) - 15.59
            fk_grade = max(0, fk_grade)
        else:
            fk_grade = 0
        
        return {
            "flesch_reading_ease": flesch_score,
            "flesch_kincaid_grade": fk_grade,
            "avg_sentence_length": avg_sentence_length if num_sentences > 0 else 0,
            "avg_word_length": sum(len(word) for word in words) / num_words if num_words > 0 else 0,
            "total_words": num_words,
            "total_sentences": num_sentences,
            "total_characters": num_chars
        }
    
    def normalize_text(self, text: str, normalization_type: str = "standard") -> str:
        """
        Normalize text according to specified type.
        
        Args:
            text: Input text
            normalization_type: Type of normalization
            
        Returns:
            Normalized text
        """
        if not text:
            return ""
        
        if normalization_type == "standard":
            # Standard normalization
            options = {
                "normalize_whitespace": True,
                "normalize_unicode": True,
                "remove_html": True
            }
            
        elif normalization_type == "aggressive":
            # Aggressive normalization
            options = {
                "normalize_whitespace": True,
                "normalize_unicode": True,
                "remove_html": True,
                "remove_urls": True,
                "remove_punctuation": True,
                "lowercase": True
            }
            
        elif normalization_type == "minimal":
            # Minimal normalization
            options = {
                "normalize_whitespace": True,
                "normalize_unicode": True
            }
            
        else:
            # Default to standard
            options = {
                "normalize_whitespace": True,
                "normalize_unicode": True,
                "remove_html": True
            }
        
        return self.clean_text(text, options)


# Global instance
text_processor = AdvancedTextProcessor()
