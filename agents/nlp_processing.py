"""
NLP Processing agent for the web scraping system.
"""
import logging
import asyncio
import time
import json
import os
import hashlib
from typing import Dict, Any, Optional, List, Union, Tuple
from datetime import datetime, timedelta
import re
import unicodedata
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import html2text

from agents.base import Agent
from models.task import Task, TaskStatus, TaskType
from models.message import Message, TaskMessage, ResultMessage, ErrorMessage, StatusMessage, Priority

# Try to import NLP libraries, but provide fallbacks if not available
try:
    import nltk
    from nltk.tokenize import word_tokenize, sent_tokenize
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer, PorterStemmer
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False

try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    from textblob import TextBlob
    TEXTBLOB_AVAILABLE = True
except ImportError:
    TEXTBLOB_AVAILABLE = False

try:
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


class NLPProcessingAgent(Agent):
    """
    Agent for natural language processing tasks on extracted content.
    """
    def __init__(self, agent_id: Optional[str] = None, coordinator_id: str = "coordinator"):
        """
        Initialize a new NLP processing agent.

        Args:
            agent_id: Unique identifier for the agent. If None, a UUID will be generated.
            coordinator_id: ID of the coordinator agent. Used for message routing.
        """
        super().__init__(agent_id=agent_id, agent_type="nlp_processing", coordinator_id=coordinator_id)

        # Initialize NLP components
        self.nlp_models = {}
        self.initialized = False

        # Cache for processed results
        self.processed_cache: Dict[str, Dict[str, Any]] = {}

        # Register additional message handlers
        self.register_handler("nlp_config", self._handle_nlp_config_message)
        self.register_handler("analyze_text", self._handle_analyze_text)
        self.register_handler("extract_entities", self._handle_extract_entities)
        self.register_handler("analyze_sentiment", self._handle_analyze_sentiment)
        self.register_handler("summarize_text", self._handle_summarize_text)
        self.register_handler("extract_keywords", self._handle_extract_keywords)

        # Start periodic tasks
        self._start_periodic_tasks()

    def _start_periodic_tasks(self) -> None:
        """Start periodic tasks for the NLP processor agent."""
        asyncio.create_task(self._periodic_cache_cleanup())

    async def _periodic_cache_cleanup(self) -> None:
        """Periodically clean up the processed cache."""
        while self.running:
            try:
                await asyncio.sleep(3600)  # Every hour
                if not self.running:
                    break

                self._cleanup_cache()
            except Exception as e:
                self.logger.error(f"Error in cache cleanup: {str(e)}", exc_info=True)

    def _cleanup_cache(self) -> None:
        """Clean up the processed cache."""
        current_time = time.time()
        self.processed_cache = {
            key: value
            for key, value in self.processed_cache.items()
            if current_time - value.get("timestamp", 0) < 86400  # 24 hours
        }

    async def start(self) -> None:
        """Start the agent and initialize NLP components."""
        # Initialize NLP components
        await self._initialize_nlp()

        # Start the agent
        await super().start()

    async def _initialize_nlp(self) -> None:
        """Initialize NLP components."""
        if self.initialized:
            return

        self.logger.info("Initializing NLP components")

        # Initialize NLTK components if available
        if NLTK_AVAILABLE:
            try:
                # Download required NLTK resources
                nltk.download('punkt', quiet=True)
                nltk.download('stopwords', quiet=True)
                nltk.download('wordnet', quiet=True)

                # Initialize components
                self.nlp_models["lemmatizer"] = WordNetLemmatizer()
                self.nlp_models["stemmer"] = PorterStemmer()
                self.nlp_models["stopwords"] = set(stopwords.words('english'))

                self.logger.info("NLTK components initialized")
            except Exception as e:
                self.logger.error(f"Error initializing NLTK: {str(e)}")
        else:
            self.logger.warning("NLTK is not available")

        # Initialize spaCy models if available
        if SPACY_AVAILABLE:
            try:
                # Load spaCy models
                self.nlp_models["spacy_en"] = spacy.load("en_core_web_sm")

                self.logger.info("spaCy models initialized")
            except Exception as e:
                self.logger.error(f"Error initializing spaCy: {str(e)}")
        else:
            self.logger.warning("spaCy is not available")

        # Initialize scikit-learn components if available
        if SKLEARN_AVAILABLE:
            try:
                # Initialize TF-IDF vectorizer
                self.nlp_models["tfidf"] = TfidfVectorizer()

                self.logger.info("scikit-learn components initialized")
            except Exception as e:
                self.logger.error(f"Error initializing scikit-learn: {str(e)}")
        else:
            self.logger.warning("scikit-learn is not available")

        # Initialize TextBlob if available
        if TEXTBLOB_AVAILABLE:
            try:
                # TextBlob doesn't need explicit initialization
                self.logger.info("TextBlob is available")
            except Exception as e:
                self.logger.error(f"Error initializing TextBlob: {str(e)}")
        else:
            self.logger.warning("TextBlob is not available")

        # Initialize transformer models if available
        if TRANSFORMERS_AVAILABLE:
            try:
                # Initialize transformer pipelines
                self.nlp_models["sentiment_analyzer"] = pipeline("sentiment-analysis")
                self.nlp_models["ner_pipeline"] = pipeline("ner")
                self.nlp_models["summarizer"] = pipeline("summarization")

                self.logger.info("Transformer models initialized")
            except Exception as e:
                self.logger.error(f"Error initializing transformer models: {str(e)}")
        else:
            self.logger.warning("Transformer models are not available")

        self.initialized = True

    async def _handle_analyze_text(self, message: Message) -> None:
        """
        Handle a text analysis request.

        Args:
            message: The message containing text analysis parameters.
        """
        if not hasattr(message, "text"):
            self.logger.warning("Received analyze_text message without text")
            return

        try:
            result = await self.analyze_text(
                message.text,
                message.options if hasattr(message, "options") else {}
            )

            # Send result
            response = ResultMessage(
                sender_id=self.agent_id,
                recipient_id=message.sender_id,
                task_id=message.task_id if hasattr(message, "task_id") else None,
                result=result
            )
            self.outbox.put(response)

        except Exception as e:
            self.logger.error(f"Error analyzing text: {str(e)}", exc_info=True)

            # Send error
            error = ErrorMessage(
                sender_id=self.agent_id,
                recipient_id=message.sender_id,
                task_id=message.task_id if hasattr(message, "task_id") else None,
                error_type="analysis_error",
                error_message=str(e)
            )
            self.outbox.put(error)

    async def _handle_extract_entities(self, message: Message) -> None:
        """
        Handle an entity extraction request.

        Args:
            message: The message containing entity extraction parameters.
        """
        if not hasattr(message, "text"):
            self.logger.warning("Received extract_entities message without text")
            return

        try:
            result = await self.extract_entities(
                message.text,
                message.entity_types if hasattr(message, "entity_types") else None
            )

            # Send result
            response = ResultMessage(
                sender_id=self.agent_id,
                recipient_id=message.sender_id,
                task_id=message.task_id if hasattr(message, "task_id") else None,
                result=result
            )
            self.outbox.put(response)

        except Exception as e:
            self.logger.error(f"Error extracting entities: {str(e)}", exc_info=True)

            # Send error
            error = ErrorMessage(
                sender_id=self.agent_id,
                recipient_id=message.sender_id,
                task_id=message.task_id if hasattr(message, "task_id") else None,
                error_type="entity_extraction_error",
                error_message=str(e)
            )
            self.outbox.put(error)

    async def _handle_analyze_sentiment(self, message: Message) -> None:
        """
        Handle a sentiment analysis request.

        Args:
            message: The message containing sentiment analysis parameters.
        """
        if not hasattr(message, "text"):
            self.logger.warning("Received analyze_sentiment message without text")
            return

        try:
            result = await self.analyze_sentiment(
                message.text,
                message.options if hasattr(message, "options") else {}
            )

            # Send result
            response = ResultMessage(
                sender_id=self.agent_id,
                recipient_id=message.sender_id,
                task_id=message.task_id if hasattr(message, "task_id") else None,
                result=result
            )
            self.outbox.put(response)

        except Exception as e:
            self.logger.error(f"Error analyzing sentiment: {str(e)}", exc_info=True)

            # Send error
            error = ErrorMessage(
                sender_id=self.agent_id,
                recipient_id=message.sender_id,
                task_id=message.task_id if hasattr(message, "task_id") else None,
                error_type="sentiment_analysis_error",
                error_message=str(e)
            )
            self.outbox.put(error)

    async def _handle_summarize_text(self, message: Message) -> None:
        """
        Handle a text summarization request.

        Args:
            message: The message containing summarization parameters.
        """
        if not hasattr(message, "text"):
            self.logger.warning("Received summarize_text message without text")
            return

        try:
            result = await self.summarize_text(
                message.text,
                message.options if hasattr(message, "options") else {}
            )

            # Send result
            response = ResultMessage(
                sender_id=self.agent_id,
                recipient_id=message.sender_id,
                task_id=message.task_id if hasattr(message, "task_id") else None,
                result=result
            )
            self.outbox.put(response)

        except Exception as e:
            self.logger.error(f"Error summarizing text: {str(e)}", exc_info=True)

            # Send error
            error = ErrorMessage(
                sender_id=self.agent_id,
                recipient_id=message.sender_id,
                task_id=message.task_id if hasattr(message, "task_id") else None,
                error_type="summarization_error",
                error_message=str(e)
            )
            self.outbox.put(error)

    async def _handle_extract_keywords(self, message: Message) -> None:
        """
        Handle a keyword extraction request.

        Args:
            message: The message containing keyword extraction parameters.
        """
        if not hasattr(message, "text"):
            self.logger.warning("Received extract_keywords message without text")
            return

        try:
            result = await self.extract_keywords(
                message.text,
                message.options if hasattr(message, "options") else {}
            )

            # Send result
            response = ResultMessage(
                sender_id=self.agent_id,
                recipient_id=message.sender_id,
                task_id=message.task_id if hasattr(message, "task_id") else None,
                result=result
            )
            self.outbox.put(response)

        except Exception as e:
            self.logger.error(f"Error extracting keywords: {str(e)}", exc_info=True)

            # Send error
            error = ErrorMessage(
                sender_id=self.agent_id,
                recipient_id=message.sender_id,
                task_id=message.task_id if hasattr(message, "task_id") else None,
                error_type="keyword_extraction_error",
                error_message=str(e)
            )
            self.outbox.put(error)

    async def analyze_text(self, text: str, options: Dict[str, Any] = {}) -> Dict[str, Any]:
        """
        Analyze text using various NLP techniques.

        Args:
            text: The text to analyze.
            options: Analysis options.

        Returns:
            A dictionary containing analysis results.
        """
        # Check cache
        cache_key = hashlib.md5(text.encode()).hexdigest()
        if cache_key in self.processed_cache:
            return self.processed_cache[cache_key]["result"]

        # Ensure NLP components are initialized
        if not self.initialized:
            await self._initialize_nlp()

        # Process text with spaCy if available
        if SPACY_AVAILABLE and "spacy_en" in self.nlp_models:
            doc = self.nlp_models["spacy_en"](text)

            # Basic statistics
            stats = {
                "char_count": len(text),
                "word_count": len(doc),
                "sentence_count": len(list(doc.sents)),
                "avg_word_length": sum(len(token.text) for token in doc) / len(doc) if doc else 0,
                "avg_sentence_length": len(doc) / len(list(doc.sents)) if doc.sents else 0
            }

            # Part-of-speech analysis
            pos_counts = {}
            for token in doc:
                pos_counts[token.pos_] = pos_counts.get(token.pos_, 0) + 1

            # Named entity analysis
            entities = []
            for ent in doc.ents:
                entities.append({
                    "text": ent.text,
                    "label": ent.label_,
                    "start": ent.start_char,
                    "end": ent.end_char
                })

            # Dependency analysis
            dependencies = []
            for token in doc:
                dependencies.append({
                    "text": token.text,
                    "dep": token.dep_,
                    "head": token.head.text,
                    "head_pos": token.head.pos_
                })

            # Store result
            result = {
                "statistics": stats,
                "pos_counts": pos_counts,
                "entities": entities,
                "dependencies": dependencies,
                "method": "spacy"
            }
        else:
            # Fallback to basic analysis
            words = text.split() if text else []
            sentences = re.split(r'[.!?]+', text) if text else []
            sentences = [s.strip() for s in sentences if s.strip()]

            stats = {
                "char_count": len(text),
                "word_count": len(words),
                "sentence_count": len(sentences),
                "avg_word_length": sum(len(word) for word in words) / len(words) if words else 0,
                "avg_sentence_length": len(words) / len(sentences) if sentences else 0
            }

            result = {
                "statistics": stats,
                "method": "basic"
            }

        # Store in cache
        self.processed_cache[cache_key] = {
            "result": result,
            "timestamp": time.time()
        }

        return result

    async def extract_entities(self, text: str, entity_types: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Extract named entities from text.

        Args:
            text: The text to process.
            entity_types: Optional list of entity types to extract.

        Returns:
            A dictionary containing extracted entities.
        """
        # Check cache
        cache_key = hashlib.md5((text + str(entity_types)).encode()).hexdigest()
        if cache_key in self.processed_cache:
            return self.processed_cache[cache_key]["result"]

        # Ensure NLP components are initialized
        if not self.initialized:
            await self._initialize_nlp()

        # Try transformer-based NER if available
        if TRANSFORMERS_AVAILABLE and "ner_pipeline" in self.nlp_models:
            try:
                ner_results = self.nlp_models["ner_pipeline"](text)

                # Process entities
                entities = []
                for entity in ner_results:
                    if entity_types is None or entity["entity_group"] in entity_types:
                        entities.append({
                            "text": entity["word"],
                            "label": entity["entity_group"],
                            "score": entity["score"],
                            "start": entity["start"],
                            "end": entity["end"]
                        })

                # Group entities by type
                grouped_entities = {}
                for entity in entities:
                    label = entity["label"]
                    if label not in grouped_entities:
                        grouped_entities[label] = []
                    grouped_entities[label].append(entity)

                result = {
                    "entities": entities,
                    "grouped_entities": grouped_entities,
                    "entity_types": list(grouped_entities.keys()),
                    "method": "transformers"
                }

                # Store in cache
                self.processed_cache[cache_key] = {
                    "result": result,
                    "timestamp": time.time()
                }

                return result
            except Exception as e:
                self.logger.warning(f"Error using transformer NER: {str(e)}")

        # Try spaCy NER if available
        if SPACY_AVAILABLE and "spacy_en" in self.nlp_models:
            try:
                # Process text
                doc = self.nlp_models["spacy_en"](text)

                # Extract entities
                entities = []
                for ent in doc.ents:
                    if entity_types is None or ent.label_ in entity_types:
                        entities.append({
                            "text": ent.text,
                            "label": ent.label_,
                            "start": ent.start_char,
                            "end": ent.end_char,
                            "description": spacy.explain(ent.label_)
                        })

                # Group entities by type
                grouped_entities = {}
                for entity in entities:
                    label = entity["label"]
                    if label not in grouped_entities:
                        grouped_entities[label] = []
                    grouped_entities[label].append(entity)

                # Store result
                result = {
                    "entities": entities,
                    "grouped_entities": grouped_entities,
                    "entity_types": list(grouped_entities.keys()),
                    "method": "spacy"
                }

                # Store in cache
                self.processed_cache[cache_key] = {
                    "result": result,
                    "timestamp": time.time()
                }

                return result
            except Exception as e:
                self.logger.warning(f"Error using spaCy NER: {str(e)}")

        # Fallback to regex-based extraction
        return await self._extract_entities_regex(text)

    async def analyze_sentiment(self, text: str, options: Dict[str, Any] = {}) -> Dict[str, Any]:
        """
        Analyze sentiment of text.

        Args:
            text: The text to analyze.
            options: Sentiment analysis options.

        Returns:
            A dictionary containing sentiment analysis results.
        """
        # Check cache
        cache_key = hashlib.md5(text.encode()).hexdigest()
        if cache_key in self.processed_cache:
            return self.processed_cache[cache_key]["result"]

        # Ensure NLP components are initialized
        if not self.initialized:
            await self._initialize_nlp()

        # Try transformer-based sentiment analysis if available
        if TRANSFORMERS_AVAILABLE and "sentiment_analyzer" in self.nlp_models:
            try:
                # Process text
                doc = self.nlp_models["spacy_en"](text) if SPACY_AVAILABLE and "spacy_en" in self.nlp_models else None

                # Get sentences
                if doc:
                    sentences = [sent.text for sent in doc.sents]
                else:
                    # Simple sentence splitting
                    sentences = re.split(r'[.!?]+', text)
                    sentences = [s.strip() for s in sentences if s.strip()]

                # Analyze sentiment for each sentence
                sentence_sentiments = []
                for sentence in sentences:
                    result = self.nlp_models["sentiment_analyzer"](sentence)[0]
                    sentence_sentiments.append({
                        "text": sentence,
                        "label": result["label"],
                        "score": result["score"]
                    })

                # Calculate overall sentiment
                positive_score = sum(s["score"] for s in sentence_sentiments if s["label"] == "POSITIVE")
                negative_score = sum(s["score"] for s in sentence_sentiments if s["label"] == "NEGATIVE")

                # Store result
                result = {
                    "overall_sentiment": {
                        "positive_score": positive_score / len(sentence_sentiments) if sentence_sentiments else 0,
                        "negative_score": negative_score / len(sentence_sentiments) if sentence_sentiments else 0,
                        "label": "POSITIVE" if positive_score > negative_score else "NEGATIVE"
                    },
                    "sentence_sentiments": sentence_sentiments,
                    "method": "transformers"
                }

                # Store in cache
                self.processed_cache[cache_key] = {
                    "result": result,
                    "timestamp": time.time()
                }

                return result
            except Exception as e:
                self.logger.warning(f"Error using transformer sentiment analysis: {str(e)}")

        # Try TextBlob sentiment analysis if available
        if TEXTBLOB_AVAILABLE:
            try:
                blob = TextBlob(text)
                polarity = blob.sentiment.polarity
                subjectivity = blob.sentiment.subjectivity

                # Determine sentiment label
                if polarity > 0.1:
                    label = "positive"
                elif polarity < -0.1:
                    label = "negative"
                else:
                    label = "neutral"

                # Store result
                result = {
                    "sentiment": label,
                    "polarity": polarity,
                    "subjectivity": subjectivity,
                    "method": "textblob"
                }

                # Store in cache
                self.processed_cache[cache_key] = {
                    "result": result,
                    "timestamp": time.time()
                }

                return result
            except Exception as e:
                self.logger.warning(f"Error using TextBlob sentiment analysis: {str(e)}")

        # Fallback to lexicon-based sentiment analysis
        return await self._analyze_sentiment_lexicon(text)

    async def summarize_text(self, text: str, options: Dict[str, Any] = {}) -> Dict[str, Any]:
        """
        Summarize text.

        Args:
            text: The text to summarize.
            options: Summarization options.

        Returns:
            A dictionary containing summarization results.
        """
        # Check cache
        cache_key = hashlib.md5((text + json.dumps(options, sort_keys=True)).encode()).hexdigest()
        if cache_key in self.processed_cache:
            return self.processed_cache[cache_key]["result"]

        # Ensure NLP components are initialized
        if not self.initialized:
            await self._initialize_nlp()

        # Get summarization options
        max_length = options.get("max_length", 130)
        min_length = options.get("min_length", 30)

        # Try transformer-based summarization if available
        if TRANSFORMERS_AVAILABLE and "summarizer" in self.nlp_models:
            try:
                # Generate summary
                summary = self.nlp_models["summarizer"](
                    text,
                    max_length=max_length,
                    min_length=min_length,
                    do_sample=False
                )[0]["summary_text"]

                # Calculate compression ratio
                compression_ratio = len(summary) / len(text) if text else 0

                # Store result
                result = {
                    "summary": summary,
                    "compression_ratio": compression_ratio,
                    "original_length": len(text),
                    "summary_length": len(summary),
                    "method": "transformers"
                }

                # Store in cache
                self.processed_cache[cache_key] = {
                    "result": result,
                    "timestamp": time.time()
                }

                return result
            except Exception as e:
                self.logger.warning(f"Error using transformer summarization: {str(e)}")

        # Fallback to extractive summarization
        return await self._summarize_text_extractive(text, max(1, int(max_length / 20)))

    async def extract_keywords(self, text: str, options: Dict[str, Any] = {}) -> Dict[str, Any]:
        """
        Extract keywords from text.

        Args:
            text: The text to process.
            options: Keyword extraction options.

        Returns:
            A dictionary containing extracted keywords.
        """
        # Check cache
        cache_key = hashlib.md5((text + json.dumps(options, sort_keys=True)).encode()).hexdigest()
        if cache_key in self.processed_cache:
            return self.processed_cache[cache_key]["result"]

        # Ensure NLP components are initialized
        if not self.initialized:
            await self._initialize_nlp()

        # Get options
        max_keywords = options.get("max_keywords", 10)
        min_freq = options.get("min_freq", 1)

        # Try TF-IDF based keyword extraction if available
        if SKLEARN_AVAILABLE and "tfidf" in self.nlp_models:
            try:
                # Tokenize and preprocess text
                if NLTK_AVAILABLE:
                    # Tokenize
                    tokens = word_tokenize(text.lower())
                    # Remove stopwords
                    tokens = [token for token in tokens if token not in self.nlp_models["stopwords"]]
                    # Lemmatize
                    tokens = [self.nlp_models["lemmatizer"].lemmatize(token) for token in tokens]
                else:
                    # Simple tokenization
                    tokens = text.lower().split()

                # Join tokens back into text
                processed_text = " ".join(tokens)

                # Create TF-IDF matrix
                tfidf = self.nlp_models["tfidf"]
                tfidf_matrix = tfidf.fit_transform([processed_text])

                # Get feature names
                feature_names = tfidf.get_feature_names_out()

                # Get scores
                scores = tfidf_matrix.toarray()[0]

                # Create keyword-score pairs
                keyword_scores = [(feature_names[i], scores[i]) for i in range(len(feature_names))]

                # Sort by score
                keyword_scores.sort(key=lambda x: x[1], reverse=True)

                # Filter by minimum frequency
                keyword_scores = [(kw, score) for kw, score in keyword_scores if text.lower().count(kw) >= min_freq]

                # Limit to max_keywords
                keyword_scores = keyword_scores[:max_keywords]

                # Format result
                keywords = [{"keyword": kw, "score": float(score)} for kw, score in keyword_scores]

                result = {
                    "keywords": keywords,
                    "method": "tfidf"
                }

                # Store in cache
                self.processed_cache[cache_key] = {
                    "result": result,
                    "timestamp": time.time()
                }

                return result
            except Exception as e:
                self.logger.warning(f"Error using TF-IDF keyword extraction: {str(e)}")

        # Fallback to frequency-based keyword extraction
        return await self._extract_keywords_frequency(text, max_keywords, min_freq)

    async def _extractive_summarize(self, text: str, max_sentences: int = 3) -> Dict[str, Any]:
        """
        Perform extractive summarization on text.

        Args:
            text: The text to summarize.
            max_sentences: Maximum number of sentences in the summary.

        Returns:
            A dictionary containing the summary.
        """
        # Ensure NLP components are initialized
        if not self.initialized:
            await self._initialize_nlp()

        # Split into sentences
        if NLTK_AVAILABLE:
            sentences = sent_tokenize(text)
        else:
            # Simple sentence splitting
            sentences = re.split(r'[.!?]+', text)
            sentences = [s.strip() for s in sentences if s.strip()]

        if not sentences:
            return {"summary": "", "method": "extractive"}

        # Calculate sentence scores
        scores = {}

        # Score based on position
        for i, sentence in enumerate(sentences):
            # First and last sentences are important
            position_score = 1.0 if i == 0 or i == len(sentences) - 1 else 0.5

            # Score based on length (prefer medium-length sentences)
            length = len(sentence.split())
            length_score = 0.5 if 5 <= length <= 20 else 0.0

            # Score based on presence of important words
            word_score = 0.0
            if NLTK_AVAILABLE:
                # Tokenize and remove stopwords
                tokens = word_tokenize(sentence.lower())
                tokens = [token for token in tokens if token not in self.nlp_models["stopwords"]]

                # Count important words
                word_score = min(1.0, len(tokens) / 10)

            # Combine scores
            scores[sentence] = position_score + length_score + word_score

        # Sort sentences by score
        ranked_sentences = sorted([(sentence, scores[sentence]) for sentence in sentences],
                                 key=lambda x: x[1], reverse=True)

        # Select top sentences up to max_sentences
        summary_sentences = []

        for sentence, _ in ranked_sentences[:max_sentences]:
            summary_sentences.append(sentence)

        # Reorder sentences to match original order
        summary_sentences.sort(key=lambda s: sentences.index(s))

        # Join sentences
        summary = " ".join(summary_sentences)

        # Calculate compression ratio
        compression_ratio = len(summary) / len(text) if text else 0

        return {
            "summary": summary,
            "compression_ratio": compression_ratio,
            "original_length": len(text),
            "summary_length": len(summary),
            "method": "extractive"
        }

    async def execute_task(self, task: Task) -> Dict[str, Any]:
        """
        Execute a task assigned to this agent.

        Args:
            task: The task to execute.

        Returns:
            A dictionary containing the result of the task.
        """
        self.logger.info(f"Executing task: {task.type}")

        # Ensure NLP components are initialized
        if not self.initialized:
            await self._initialize_nlp()

        if task.type == TaskType.NLP_ENTITY_EXTRACTION:
            return await self._execute_entity_extraction(task)
        elif task.type == TaskType.NLP_SENTIMENT_ANALYSIS:
            return await self._execute_sentiment_analysis(task)
        elif task.type == TaskType.NLP_TEXT_CLASSIFICATION:
            return await self._execute_text_classification(task)
        elif task.type == TaskType.NLP_KEYWORD_EXTRACTION:
            return await self._execute_keyword_extraction(task)
        elif task.type == TaskType.NLP_TEXT_SUMMARIZATION:
            return await self._execute_text_summarization(task)
        elif task.type == TaskType.NLP_LANGUAGE_DETECTION:
            return await self._execute_language_detection(task)
        else:
            raise ValueError(f"Unsupported task type: {task.type}")

    async def _execute_entity_extraction(self, task: Task) -> Dict[str, Any]:
        """
        Execute an entity extraction task.

        Args:
            task: The task to execute.

        Returns:
            A dictionary containing the extracted entities.
        """
        params = task.parameters

        # Get required parameters
        text = params.get("text", "")
        entity_types = params.get("entity_types", ["PERSON", "ORG", "GPE", "DATE", "TIME", "MONEY"])

        if not text:
            return {"entities": {}}

        # Use spaCy for entity extraction if available
        if SPACY_AVAILABLE and "spacy_en" in self.nlp_models:
            return await self._extract_entities_spacy(text, entity_types)

        # Fallback to regex-based extraction
        return await self._extract_entities_regex(text)

    async def _extract_entities_spacy(self, text: str, entity_types: List[str]) -> Dict[str, Any]:
        """
        Extract entities using spaCy.

        Args:
            text: Text to extract entities from.
            entity_types: Types of entities to extract.

        Returns:
            Dictionary containing extracted entities.
        """
        nlp = self.nlp_models["spacy_en"]
        doc = nlp(text)

        entities = {}
        for ent_type in entity_types:
            entities[ent_type] = []

        for ent in doc.ents:
            if ent.label_ in entity_types:
                if ent.text not in entities[ent.label_]:
                    entities[ent.label_].append(ent.text)

        return {
            "entities": entities,
            "method": "spacy"
        }

    async def _extract_entities_regex(self, text: str) -> Dict[str, Any]:
        """
        Extract entities using regex patterns.

        Args:
            text: Text to extract entities from.

        Returns:
            Dictionary containing extracted entities.
        """
        entities = {
            "EMAIL": [],
            "PHONE": [],
            "URL": [],
            "DATE": [],
            "TIME": [],
            "MONEY": []
        }

        # Email pattern
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        emails = re.findall(email_pattern, text)
        entities["EMAIL"] = list(set(emails))

        # Phone pattern
        phone_pattern = r'\b(?:\+\d{1,3}[- ]?)?\(?\d{3}\)?[- ]?\d{3}[- ]?\d{4}\b'
        phones = re.findall(phone_pattern, text)
        entities["PHONE"] = list(set(phones))

        # URL pattern
        url_pattern = r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+(?:/[^/\s]*)*'
        urls = re.findall(url_pattern, text)
        entities["URL"] = list(set(urls))

        # Date pattern
        date_pattern = r'\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b|\b\d{4}[-/]\d{1,2}[-/]\d{1,2}\b'
        dates = re.findall(date_pattern, text)
        entities["DATE"] = list(set(dates))

        # Time pattern
        time_pattern = r'\b\d{1,2}:\d{2}(?::\d{2})?(?:\s*[aApP][mM])?\b'
        times = re.findall(time_pattern, text)
        entities["TIME"] = list(set(times))

        # Money pattern
        money_pattern = r'\$\d+(?:,\d{3})*(?:\.\d{2})?|\d+(?:,\d{3})*(?:\.\d{2})?\s*(?:dollars|USD|€|£|¥)'
        money = re.findall(money_pattern, text)
        entities["MONEY"] = list(set(money))

        return {
            "entities": entities,
            "method": "regex"
        }

    async def _execute_sentiment_analysis(self, task: Task) -> Dict[str, Any]:
        """
        Execute a sentiment analysis task.

        Args:
            task: The task to execute.

        Returns:
            A dictionary containing the sentiment analysis results.
        """
        params = task.parameters

        # Get required parameters
        text = params.get("text", "")

        if not text:
            return {"sentiment": "neutral", "score": 0.0}

        # Use spaCy for sentiment analysis if available
        if SPACY_AVAILABLE and "spacy_en" in self.nlp_models:
            return await self._analyze_sentiment_spacy(text)

        # Fallback to lexicon-based sentiment analysis
        return await self._analyze_sentiment_lexicon(text)

    async def _analyze_sentiment_spacy(self, text: str) -> Dict[str, Any]:
        """
        Analyze sentiment using spaCy.

        Args:
            text: Text to analyze.

        Returns:
            Dictionary containing sentiment analysis results.
        """
        # Note: spaCy doesn't have built-in sentiment analysis
        # This is a placeholder for a custom sentiment analysis component
        # In a real implementation, you would use a dedicated sentiment analysis library

        # Fallback to lexicon-based analysis
        return await self._analyze_sentiment_lexicon(text)

    async def _analyze_sentiment_lexicon(self, text: str) -> Dict[str, Any]:
        """
        Analyze sentiment using a lexicon-based approach.

        Args:
            text: Text to analyze.

        Returns:
            Dictionary containing sentiment analysis results.
        """
        # Simple lexicon-based sentiment analysis
        positive_words = {
            "good", "great", "excellent", "amazing", "wonderful", "fantastic",
            "terrific", "outstanding", "superb", "brilliant", "awesome",
            "happy", "pleased", "satisfied", "love", "like", "enjoy",
            "positive", "perfect", "best", "better", "recommend"
        }

        negative_words = {
            "bad", "terrible", "horrible", "awful", "poor", "disappointing",
            "mediocre", "subpar", "inferior", "worst", "worse", "hate",
            "dislike", "negative", "angry", "upset", "unhappy", "dissatisfied",
            "problem", "issue", "complaint", "fail", "failure", "broken"
        }

        # Tokenize and normalize text
        if NLTK_AVAILABLE:
            tokens = word_tokenize(text.lower())
            # Remove stopwords
            tokens = [token for token in tokens if token not in self.nlp_models["stopwords"]]
        else:
            # Simple tokenization
            tokens = text.lower().split()

        # Count positive and negative words
        positive_count = sum(1 for token in tokens if token in positive_words)
        negative_count = sum(1 for token in tokens if token in negative_words)

        # Calculate sentiment score
        total_count = positive_count + negative_count
        if total_count == 0:
            sentiment_score = 0.0
            sentiment = "neutral"
        else:
            sentiment_score = (positive_count - negative_count) / total_count
            if sentiment_score > 0.1:
                sentiment = "positive"
            elif sentiment_score < -0.1:
                sentiment = "negative"
            else:
                sentiment = "neutral"

        return {
            "sentiment": sentiment,
            "score": sentiment_score,
            "positive_count": positive_count,
            "negative_count": negative_count,
            "method": "lexicon"
        }

    async def _execute_text_classification(self, task: Task) -> Dict[str, Any]:
        """
        Execute a text classification task.

        Args:
            task: The task to execute.

        Returns:
            A dictionary containing the classification results.
        """
        params = task.parameters

        # Get required parameters
        text = params.get("text", "")
        categories = params.get("categories", {})

        if not text or not categories:
            return {"category": None, "confidence": 0.0}

        # Use scikit-learn for classification if available
        if SKLEARN_AVAILABLE and "tfidf" in self.nlp_models:
            return await self._classify_text_tfidf(text, categories)

        # Fallback to keyword-based classification
        return await self._classify_text_keywords(text, categories)

    async def _classify_text_tfidf(self, text: str, categories: Dict[str, List[str]]) -> Dict[str, Any]:
        """
        Classify text using TF-IDF and cosine similarity.

        Args:
            text: Text to classify.
            categories: Dictionary mapping category names to example texts.

        Returns:
            Dictionary containing classification results.
        """
        # Prepare category examples
        category_texts = []
        category_names = []

        for category, examples in categories.items():
            for example in examples:
                category_texts.append(example)
                category_names.append(category)

        if not category_texts:
            return {"category": None, "confidence": 0.0}

        # Add the input text
        all_texts = category_texts + [text]

        # Vectorize texts
        tfidf = self.nlp_models["tfidf"]
        tfidf_matrix = tfidf.fit_transform(all_texts)

        # Calculate cosine similarity
        text_vector = tfidf_matrix[-1]
        category_vectors = tfidf_matrix[:-1]
        similarities = cosine_similarity(text_vector, category_vectors)[0]

        # Find the most similar category
        max_similarity_index = similarities.argmax()
        max_similarity = similarities[max_similarity_index]
        best_category = category_names[max_similarity_index]

        return {
            "category": best_category,
            "confidence": float(max_similarity),
            "method": "tfidf"
        }

    async def _classify_text_keywords(self, text: str, categories: Dict[str, List[str]]) -> Dict[str, Any]:
        """
        Classify text using keyword matching.

        Args:
            text: Text to classify.
            categories: Dictionary mapping category names to keyword lists.

        Returns:
            Dictionary containing classification results.
        """
        text_lower = text.lower()
        scores = {}

        for category, keywords in categories.items():
            # Count keyword occurrences
            score = 0
            for keyword in keywords:
                score += text_lower.count(keyword.lower())

            scores[category] = score

        # Find the category with the highest score
        if not scores:
            return {"category": None, "confidence": 0.0}

        best_category = max(scores, key=scores.get)
        total_score = sum(scores.values())

        if total_score == 0:
            confidence = 0.0
        else:
            confidence = scores[best_category] / total_score

        return {
            "category": best_category,
            "confidence": confidence,
            "scores": scores,
            "method": "keyword"
        }

    async def _execute_keyword_extraction(self, task: Task) -> Dict[str, Any]:
        """
        Execute a keyword extraction task.

        Args:
            task: The task to execute.

        Returns:
            A dictionary containing the extracted keywords.
        """
        params = task.parameters

        # Get required parameters
        text = params.get("text", "")
        max_keywords = params.get("max_keywords", 10)

        if not text:
            return {"keywords": []}

        # Use scikit-learn for keyword extraction if available
        if SKLEARN_AVAILABLE and "tfidf" in self.nlp_models:
            return await self._extract_keywords_tfidf(text, max_keywords)

        # Fallback to frequency-based keyword extraction
        return await self._extract_keywords_frequency(text, max_keywords)

    async def _extract_keywords_tfidf(self, text: str, max_keywords: int) -> Dict[str, Any]:
        """
        Extract keywords using TF-IDF.

        Args:
            text: Text to extract keywords from.
            max_keywords: Maximum number of keywords to extract.

        Returns:
            Dictionary containing extracted keywords.
        """
        # Split text into sentences
        if NLTK_AVAILABLE:
            sentences = sent_tokenize(text)
        else:
            # Simple sentence splitting
            sentences = re.split(r'[.!?]+', text)
            sentences = [s.strip() for s in sentences if s.strip()]

        if not sentences:
            return {"keywords": [], "method": "tfidf"}

        # Vectorize sentences
        tfidf = self.nlp_models["tfidf"]
        tfidf_matrix = tfidf.fit_transform(sentences)

        # Get feature names
        feature_names = tfidf.get_feature_names_out()

        # Calculate average TF-IDF score for each word
        word_scores = {}
        for i, sentence in enumerate(sentences):
            feature_index = tfidf_matrix[i].indices
            feature_scores = tfidf_matrix[i].data

            for idx, score in zip(feature_index, feature_scores):
                word = feature_names[idx]
                if word not in word_scores:
                    word_scores[word] = []
                word_scores[word].append(score)

        # Calculate average scores
        avg_scores = {word: sum(scores) / len(scores) for word, scores in word_scores.items()}

        # Sort words by score and get top keywords
        keywords = sorted(avg_scores.items(), key=lambda x: x[1], reverse=True)[:max_keywords]

        return {
            "keywords": [{"word": word, "score": float(score)} for word, score in keywords],
            "method": "tfidf"
        }

    async def _extract_keywords_frequency(self, text: str, max_keywords: int) -> Dict[str, Any]:
        """
        Extract keywords using frequency analysis.

        Args:
            text: Text to extract keywords from.
            max_keywords: Maximum number of keywords to extract.

        Returns:
            Dictionary containing extracted keywords.
        """
        # Tokenize and normalize text
        if NLTK_AVAILABLE:
            tokens = word_tokenize(text.lower())
            # Remove stopwords and short words
            tokens = [token for token in tokens if token not in self.nlp_models["stopwords"] and len(token) > 2]
        else:
            # Simple tokenization
            tokens = text.lower().split()
            # Remove short words
            tokens = [token for token in tokens if len(token) > 2]

        # Count word frequencies
        word_freq = {}
        for token in tokens:
            if token in word_freq:
                word_freq[token] += 1
            else:
                word_freq[token] = 1

        # Sort words by frequency and get top keywords
        keywords = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:max_keywords]

        return {
            "keywords": [{"word": word, "frequency": freq} for word, freq in keywords],
            "method": "frequency"
        }

    async def _execute_text_summarization(self, task: Task) -> Dict[str, Any]:
        """
        Execute a text summarization task.

        Args:
            task: The task to execute.

        Returns:
            A dictionary containing the summarized text.
        """
        params = task.parameters

        # Get required parameters
        text = params.get("text", "")
        max_sentences = params.get("max_sentences", 3)

        if not text:
            return {"summary": ""}

        # Use extractive summarization
        return await self._summarize_text_extractive(text, max_sentences)

    async def _summarize_text_extractive(self, text: str, max_sentences: int) -> Dict[str, Any]:
        """
        Summarize text using extractive summarization.

        Args:
            text: Text to summarize.
            max_sentences: Maximum number of sentences in the summary.

        Returns:
            Dictionary containing the summarized text.
        """
        # Split text into sentences
        if NLTK_AVAILABLE:
            sentences = sent_tokenize(text)
        else:
            # Simple sentence splitting
            sentences = re.split(r'[.!?]+', text)
            sentences = [s.strip() for s in sentences if s.strip()]

        if not sentences:
            return {"summary": "", "method": "extractive"}

        # If there are fewer sentences than the maximum, return the original text
        if len(sentences) <= max_sentences:
            return {"summary": text, "method": "extractive"}

        # Use TF-IDF for sentence scoring if available
        if SKLEARN_AVAILABLE and "tfidf" in self.nlp_models:
            # Vectorize sentences
            tfidf = self.nlp_models["tfidf"]
            tfidf_matrix = tfidf.fit_transform(sentences)

            # Calculate sentence scores based on TF-IDF
            sentence_scores = []
            for i in range(len(sentences)):
                score = tfidf_matrix[i].sum()
                sentence_scores.append((i, score))

            # Sort sentences by score
            sentence_scores.sort(key=lambda x: x[1], reverse=True)

            # Select top sentences
            top_sentence_indices = [idx for idx, _ in sentence_scores[:max_sentences]]
            top_sentence_indices.sort()  # Sort by original order

            # Construct summary
            summary_sentences = [sentences[idx] for idx in top_sentence_indices]
            summary = " ".join(summary_sentences)

            return {
                "summary": summary,
                "method": "extractive_tfidf"
            }

        # Fallback to position-based summarization
        # Select first sentence, last sentence, and sentences in between
        selected_sentences = []

        # Always include the first sentence
        selected_sentences.append(sentences[0])

        # Select sentences from the middle if needed
        if max_sentences > 2 and len(sentences) > 2:
            middle_count = max_sentences - 2
            step = (len(sentences) - 2) / (middle_count + 1)

            for i in range(1, middle_count + 1):
                idx = min(int(i * step) + 1, len(sentences) - 2)
                selected_sentences.append(sentences[idx])

        # Include the last sentence if there's room
        if max_sentences >= 2 and len(sentences) >= 2:
            selected_sentences.append(sentences[-1])

        summary = " ".join(selected_sentences)

        return {
            "summary": summary,
            "method": "extractive_position"
        }

    async def _execute_language_detection(self, task: Task) -> Dict[str, Any]:
        """
        Execute a language detection task.

        Args:
            task: The task to execute.

        Returns:
            A dictionary containing the detected language.
        """
        params = task.parameters

        # Get required parameters
        text = params.get("text", "")

        if not text:
            return {"language": "unknown", "confidence": 0.0}

        # Use spaCy for language detection if available
        if SPACY_AVAILABLE and "spacy_en" in self.nlp_models:
            return await self._detect_language_spacy(text)

        # Fallback to n-gram based language detection
        return await self._detect_language_ngram(text)

    async def _detect_language_spacy(self, text: str) -> Dict[str, Any]:
        """
        Detect language using spaCy.

        Args:
            text: Text to detect language for.

        Returns:
            Dictionary containing the detected language.
        """
        # Note: spaCy doesn't have built-in language detection
        # This is a placeholder for a custom language detection component
        # In a real implementation, you would use a dedicated language detection library

        # Fallback to n-gram based detection
        return await self._detect_language_ngram(text)

    async def _detect_language_ngram(self, text: str) -> Dict[str, Any]:
        """
        Detect language using n-gram frequency profiles.

        Args:
            text: Text to detect language for.

        Returns:
            Dictionary containing the detected language.
        """
        # Simple language detection based on common words
        # This is a very basic implementation and should be replaced with a proper language detection library

        text_lower = text.lower()

        # Language profiles (common words)
        language_profiles = {
            "english": ["the", "and", "to", "of", "a", "in", "that", "is", "was", "for"],
            "spanish": ["el", "la", "de", "que", "y", "en", "un", "ser", "se", "no"],
            "french": ["le", "la", "de", "et", "à", "en", "un", "être", "que", "pour"],
            "german": ["der", "die", "und", "in", "den", "von", "zu", "das", "mit", "sich"],
            "italian": ["il", "di", "che", "la", "in", "e", "un", "a", "per", "è"],
            "portuguese": ["de", "a", "o", "que", "e", "do", "da", "em", "um", "para"]
        }

        # Count occurrences of common words
        scores = {}
        for language, words in language_profiles.items():
            score = 0
            for word in words:
                # Count word occurrences (with word boundaries)
                pattern = r'\b' + re.escape(word) + r'\b'
                score += len(re.findall(pattern, text_lower))

            scores[language] = score

        # Find the language with the highest score
        if not scores:
            return {"language": "unknown", "confidence": 0.0}

        best_language = max(scores, key=scores.get)
        total_score = sum(scores.values())

        if total_score == 0:
            confidence = 0.0
        else:
            confidence = scores[best_language] / total_score

        return {
            "language": best_language,
            "confidence": confidence,
            "scores": scores,
            "method": "ngram"
        }

    async def _handle_nlp_config_message(self, message: Message) -> None:
        """
        Handle an NLP configuration message.

        Args:
            message: The message to handle.
        """
        if not hasattr(message, "config") or not message.config:
            self.logger.warning("Received NLP config message without config")
            return

        self.logger.info("Updating NLP configuration")

        # Update configuration
        # In a real implementation, this would update model parameters, etc.

        # Send acknowledgment
        status_message = StatusMessage(
            sender_id=self.agent_id,
            recipient_id=message.sender_id,
            status="nlp_config_updated",
            details={}
        )
        self.outbox.put(status_message)
