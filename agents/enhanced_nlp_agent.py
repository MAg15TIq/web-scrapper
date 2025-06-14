"""
Enhanced Natural Language Processing Agent with advanced capabilities.
"""
import asyncio
import logging
import time
import re
import hashlib
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta

from agents.base import Agent
from models.message import Message, TaskMessage, ResultMessage, ErrorMessage, Priority
from models.task import Task, TaskStatus, TaskType
from models.data_processing import (
    ProcessingType, EntityType, SentimentLabel, LanguageCode,
    ExtractedEntity, SentimentResult, TextSummary, LanguageDetectionResult,
    TranslationResult, ProcessingRequest, ProcessingResponse
)
from config.data_processing_config import data_processing_config

# NLP libraries
try:
    import spacy
    from spacy import displacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    logging.warning("spaCy not available. Install with: pip install spacy")

try:
    from textblob import TextBlob
    TEXTBLOB_AVAILABLE = True
except ImportError:
    TEXTBLOB_AVAILABLE = False
    logging.warning("TextBlob not available. Install with: pip install textblob")

try:
    from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer as AutoTokenizerSeq2Seq
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logging.warning("Transformers not available. Install with: pip install transformers torch")

try:
    from langdetect import detect, detect_langs, DetectorFactory
    DetectorFactory.seed = 0  # For consistent results
    LANGDETECT_AVAILABLE = True
except ImportError:
    LANGDETECT_AVAILABLE = False
    logging.warning("langdetect not available. Install with: pip install langdetect")

try:
    from googletrans import Translator
    GOOGLETRANS_AVAILABLE = True
except ImportError:
    GOOGLETRANS_AVAILABLE = False
    logging.warning("googletrans not available. Install with: pip install googletrans==4.0.0rc1")


class EnhancedNLPAgent(Agent):
    """
    Enhanced Natural Language Processing Agent with advanced capabilities.
    """
    
    def __init__(self, agent_id: Optional[str] = None, coordinator_id: str = "coordinator"):
        """
        Initialize the Enhanced NLP Agent.
        
        Args:
            agent_id: Unique identifier for the agent
            coordinator_id: ID of the coordinator agent
        """
        super().__init__(agent_id=agent_id, agent_type="enhanced_nlp", coordinator_id=coordinator_id)
        
        # Configuration
        self.config = data_processing_config.nlp
        
        # Initialize models
        self.models = {}
        self.tokenizers = {}
        self.pipelines = {}
        
        # Processing cache
        self.cache = {}
        self.cache_timestamps = {}
        
        # Performance metrics
        self.processing_stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "average_processing_time": 0.0,
            "cache_hits": 0
        }
        
        # Initialize capabilities
        self.capabilities = {
            "entity_extraction": SPACY_AVAILABLE,
            "sentiment_analysis": TRANSFORMERS_AVAILABLE or TEXTBLOB_AVAILABLE,
            "text_summarization": TRANSFORMERS_AVAILABLE,
            "language_detection": LANGDETECT_AVAILABLE or TEXTBLOB_AVAILABLE,
            "translation": GOOGLETRANS_AVAILABLE or TRANSFORMERS_AVAILABLE,
            "keyword_extraction": SPACY_AVAILABLE or TEXTBLOB_AVAILABLE,
            "emotion_analysis": TRANSFORMERS_AVAILABLE,
            "text_classification": TRANSFORMERS_AVAILABLE
        }
        
        # Register message handlers
        self.register_handler("extract_entities", self._handle_extract_entities)
        self.register_handler("analyze_sentiment", self._handle_analyze_sentiment)
        self.register_handler("summarize_text", self._handle_summarize_text)
        self.register_handler("detect_language", self._handle_detect_language)
        self.register_handler("translate_text", self._handle_translate_text)
        self.register_handler("extract_keywords", self._handle_extract_keywords)
        self.register_handler("analyze_emotions", self._handle_analyze_emotions)
        self.register_handler("classify_text", self._handle_classify_text)
        self.register_handler("process_nlp_batch", self._handle_process_batch)
        
        # Initialize models asynchronously
        asyncio.create_task(self._initialize_models())
        
        # Start periodic tasks
        asyncio.create_task(self._start_periodic_tasks())
    
    async def _initialize_models(self):
        """Initialize NLP models and pipelines."""
        try:
            self.logger.info("Initializing NLP models...")
            
            # Initialize spaCy model
            if SPACY_AVAILABLE:
                try:
                    self.models["spacy"] = spacy.load(self.config.spacy_model)
                    self.logger.info(f"Loaded spaCy model: {self.config.spacy_model}")
                except OSError:
                    self.logger.warning(f"spaCy model {self.config.spacy_model} not found. Using en_core_web_sm")
                    try:
                        self.models["spacy"] = spacy.load("en_core_web_sm")
                    except OSError:
                        self.logger.error("No spaCy model available. Install with: python -m spacy download en_core_web_sm")
                        self.capabilities["entity_extraction"] = False
            
            # Initialize transformer models
            if TRANSFORMERS_AVAILABLE:
                try:
                    # Sentiment analysis
                    self.pipelines["sentiment"] = pipeline(
                        "sentiment-analysis",
                        model=self.config.sentiment_model,
                        return_all_scores=True
                    )
                    
                    # Text summarization
                    self.pipelines["summarization"] = pipeline(
                        "summarization",
                        model=self.config.summarization_model
                    )
                    
                    # Emotion analysis
                    self.pipelines["emotion"] = pipeline(
                        "text-classification",
                        model="j-hartmann/emotion-english-distilroberta-base",
                        return_all_scores=True
                    )
                    
                    self.logger.info("Initialized transformer pipelines")
                    
                except Exception as e:
                    self.logger.warning(f"Failed to initialize some transformer models: {e}")
            
            # Initialize translation
            if GOOGLETRANS_AVAILABLE:
                self.models["translator"] = Translator()
                self.logger.info("Initialized Google Translator")
            
            self.logger.info("NLP models initialization completed")
            
        except Exception as e:
            self.logger.error(f"Error initializing NLP models: {e}")
    
    async def _start_periodic_tasks(self):
        """Start periodic maintenance tasks."""
        while True:
            try:
                await asyncio.sleep(300)  # Run every 5 minutes
                await self._cleanup_cache()
                await self._update_performance_metrics()
            except Exception as e:
                self.logger.error(f"Error in periodic tasks: {e}")
    
    async def _cleanup_cache(self):
        """Clean up expired cache entries."""
        current_time = time.time()
        expired_keys = []
        
        for key, timestamp in self.cache_timestamps.items():
            if current_time - timestamp > self.config.cache_ttl:
                expired_keys.append(key)
        
        for key in expired_keys:
            self.cache.pop(key, None)
            self.cache_timestamps.pop(key, None)
        
        if expired_keys:
            self.logger.debug(f"Cleaned up {len(expired_keys)} expired cache entries")
    
    async def _update_performance_metrics(self):
        """Update performance metrics."""
        if self.processing_stats["total_requests"] > 0:
            success_rate = (self.processing_stats["successful_requests"] / 
                          self.processing_stats["total_requests"]) * 100
            cache_hit_rate = (self.processing_stats["cache_hits"] / 
                            self.processing_stats["total_requests"]) * 100
            
            self.logger.info(
                f"NLP Agent Performance - Success Rate: {success_rate:.1f}%, "
                f"Cache Hit Rate: {cache_hit_rate:.1f}%, "
                f"Avg Processing Time: {self.processing_stats['average_processing_time']:.2f}s"
            )
    
    def _get_cache_key(self, text: str, operation: str, params: Dict[str, Any] = None) -> str:
        """Generate cache key for text processing."""
        params_str = str(sorted(params.items())) if params else ""
        content = f"{operation}:{text}:{params_str}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def _get_from_cache(self, cache_key: str) -> Optional[Any]:
        """Get result from cache if available and not expired."""
        if not self.config.enable_caching:
            return None
        
        if cache_key in self.cache:
            timestamp = self.cache_timestamps.get(cache_key, 0)
            if time.time() - timestamp < self.config.cache_ttl:
                self.processing_stats["cache_hits"] += 1
                return self.cache[cache_key]
            else:
                # Remove expired entry
                self.cache.pop(cache_key, None)
                self.cache_timestamps.pop(cache_key, None)
        
        return None
    
    def _store_in_cache(self, cache_key: str, result: Any):
        """Store result in cache."""
        if self.config.enable_caching:
            self.cache[cache_key] = result
            self.cache_timestamps[cache_key] = time.time()
    
    async def _handle_extract_entities(self, message: Message):
        """Handle entity extraction request."""
        try:
            text = message.data.get("text", "")
            entity_types = message.data.get("entity_types", None)
            
            result = await self.extract_entities(text, entity_types)
            
            response = ResultMessage(
                sender_id=self.agent_id,
                recipient_id=message.sender_id,
                task_id=getattr(message, "task_id", None),
                result=result
            )
            self.outbox.put(response)
            
        except Exception as e:
            self.logger.error(f"Error in entity extraction: {e}")
            error_response = ErrorMessage(
                sender_id=self.agent_id,
                recipient_id=message.sender_id,
                task_id=getattr(message, "task_id", None),
                error=str(e)
            )
            self.outbox.put(error_response)
    
    async def extract_entities(self, text: str, entity_types: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Extract named entities from text.
        
        Args:
            text: Input text
            entity_types: Specific entity types to extract
            
        Returns:
            Dictionary containing extracted entities
        """
        start_time = time.time()
        self.processing_stats["total_requests"] += 1
        
        try:
            # Check cache
            cache_key = self._get_cache_key(text, "entities", {"types": entity_types})
            cached_result = self._get_from_cache(cache_key)
            if cached_result:
                return cached_result
            
            entities = []
            
            if SPACY_AVAILABLE and "spacy" in self.models:
                doc = self.models["spacy"](text[:self.config.max_text_length])
                
                for ent in doc.ents:
                    if entity_types is None or ent.label_ in entity_types:
                        entity = ExtractedEntity(
                            text=ent.text,
                            label=EntityType(ent.label_) if ent.label_ in EntityType.__members__ else EntityType.PERSON,
                            start=ent.start_char,
                            end=ent.end_char,
                            confidence=0.9,  # spaCy doesn't provide confidence scores
                            description=spacy.explain(ent.label_),
                            metadata={"method": "spacy"}
                        )
                        entities.append(entity.dict())
            
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
                "total_entities": len(entities),
                "entity_types": list(grouped_entities.keys()),
                "processing_time": time.time() - start_time,
                "method": "spacy" if SPACY_AVAILABLE else "fallback"
            }
            
            # Store in cache
            self._store_in_cache(cache_key, result)
            
            self.processing_stats["successful_requests"] += 1
            return result
            
        except Exception as e:
            self.processing_stats["failed_requests"] += 1
            self.logger.error(f"Error extracting entities: {e}")
            raise
        
        finally:
            processing_time = time.time() - start_time
            self.processing_stats["average_processing_time"] = (
                (self.processing_stats["average_processing_time"] *
                 (self.processing_stats["total_requests"] - 1) + processing_time) /
                self.processing_stats["total_requests"]
            )

    async def _handle_analyze_sentiment(self, message: Message):
        """Handle sentiment analysis request."""
        try:
            text = message.data.get("text", "")
            include_emotions = message.data.get("include_emotions", False)

            result = await self.analyze_sentiment(text, include_emotions)

            response = ResultMessage(
                sender_id=self.agent_id,
                recipient_id=message.sender_id,
                task_id=getattr(message, "task_id", None),
                result=result
            )
            self.outbox.put(response)

        except Exception as e:
            self.logger.error(f"Error in sentiment analysis: {e}")
            error_response = ErrorMessage(
                sender_id=self.agent_id,
                recipient_id=message.sender_id,
                task_id=getattr(message, "task_id", None),
                error=str(e)
            )
            self.outbox.put(error_response)

    async def analyze_sentiment(self, text: str, include_emotions: bool = False) -> Dict[str, Any]:
        """
        Analyze sentiment of text with optional emotion analysis.

        Args:
            text: Input text
            include_emotions: Whether to include emotion analysis

        Returns:
            Dictionary containing sentiment analysis results
        """
        start_time = time.time()
        self.processing_stats["total_requests"] += 1

        try:
            # Check cache
            cache_key = self._get_cache_key(text, "sentiment", {"emotions": include_emotions})
            cached_result = self._get_from_cache(cache_key)
            if cached_result:
                return cached_result

            sentiment_result = None
            emotions = {}

            # Use transformer model if available
            if TRANSFORMERS_AVAILABLE and "sentiment" in self.pipelines:
                try:
                    results = self.pipelines["sentiment"](text[:self.config.max_text_length])

                    # Convert to our format
                    scores = {result["label"].lower(): result["score"] for result in results[0]}

                    # Determine primary sentiment
                    primary_sentiment = max(scores.items(), key=lambda x: x[1])

                    sentiment_result = SentimentResult(
                        label=SentimentLabel(primary_sentiment[0].upper()),
                        score=primary_sentiment[1],
                        positive_score=scores.get("positive", 0.0),
                        negative_score=scores.get("negative", 0.0),
                        neutral_score=scores.get("neutral", 0.0)
                    )

                    # Add emotion analysis if requested
                    if include_emotions and "emotion" in self.pipelines:
                        emotion_results = self.pipelines["emotion"](text[:self.config.max_text_length])
                        emotions = {result["label"]: result["score"] for result in emotion_results[0]}
                        sentiment_result.emotions = emotions

                except Exception as e:
                    self.logger.warning(f"Transformer sentiment analysis failed: {e}")

            # Fallback to TextBlob
            if sentiment_result is None and TEXTBLOB_AVAILABLE:
                blob = TextBlob(text)
                polarity = blob.sentiment.polarity

                if polarity > 0.1:
                    label = SentimentLabel.POSITIVE
                elif polarity < -0.1:
                    label = SentimentLabel.NEGATIVE
                else:
                    label = SentimentLabel.NEUTRAL

                sentiment_result = SentimentResult(
                    label=label,
                    score=abs(polarity),
                    positive_score=max(0, polarity),
                    negative_score=max(0, -polarity),
                    neutral_score=1 - abs(polarity)
                )

            if sentiment_result is None:
                raise Exception("No sentiment analysis method available")

            result = {
                "sentiment": sentiment_result.dict(),
                "emotions": emotions,
                "processing_time": time.time() - start_time,
                "method": "transformers" if TRANSFORMERS_AVAILABLE else "textblob"
            }

            # Store in cache
            self._store_in_cache(cache_key, result)

            self.processing_stats["successful_requests"] += 1
            return result

        except Exception as e:
            self.processing_stats["failed_requests"] += 1
            self.logger.error(f"Error analyzing sentiment: {e}")
            raise

    async def _handle_summarize_text(self, message: Message):
        """Handle text summarization request."""
        try:
            text = message.data.get("text", "")
            max_length = message.data.get("max_length", self.config.summary_max_length)
            min_length = message.data.get("min_length", self.config.summary_min_length)

            result = await self.summarize_text(text, max_length, min_length)

            response = ResultMessage(
                sender_id=self.agent_id,
                recipient_id=message.sender_id,
                task_id=getattr(message, "task_id", None),
                result=result
            )
            self.outbox.put(response)

        except Exception as e:
            self.logger.error(f"Error in text summarization: {e}")
            error_response = ErrorMessage(
                sender_id=self.agent_id,
                recipient_id=message.sender_id,
                task_id=getattr(message, "task_id", None),
                error=str(e)
            )
            self.outbox.put(error_response)

    async def summarize_text(self, text: str, max_length: int = None, min_length: int = None) -> Dict[str, Any]:
        """
        Generate text summary using transformer models.

        Args:
            text: Input text to summarize
            max_length: Maximum summary length
            min_length: Minimum summary length

        Returns:
            Dictionary containing summarization results
        """
        start_time = time.time()
        self.processing_stats["total_requests"] += 1

        try:
            max_length = max_length or self.config.summary_max_length
            min_length = min_length or self.config.summary_min_length

            # Check cache
            cache_key = self._get_cache_key(text, "summary", {"max": max_length, "min": min_length})
            cached_result = self._get_from_cache(cache_key)
            if cached_result:
                return cached_result

            if not TRANSFORMERS_AVAILABLE or "summarization" not in self.pipelines:
                raise Exception("Text summarization requires transformers library")

            # Truncate text if too long
            input_text = text[:self.config.max_text_length]

            # Generate summary
            summary_result = self.pipelines["summarization"](
                input_text,
                max_length=max_length,
                min_length=min_length,
                do_sample=False
            )

            summary_text = summary_result[0]["summary_text"]

            # Extract keywords using spaCy if available
            keywords = []
            if SPACY_AVAILABLE and "spacy" in self.models:
                doc = self.models["spacy"](input_text)
                keywords = [token.lemma_.lower() for token in doc
                          if token.is_alpha and not token.is_stop and len(token.text) > 2]
                keywords = list(set(keywords))[:10]  # Top 10 unique keywords

            # Calculate compression ratio
            compression_ratio = len(summary_text) / len(input_text) if input_text else 0

            summary = TextSummary(
                summary=summary_text,
                original_length=len(input_text),
                summary_length=len(summary_text),
                compression_ratio=compression_ratio,
                keywords=keywords
            )

            result = {
                "summary": summary.dict(),
                "processing_time": time.time() - start_time,
                "method": "transformers"
            }

            # Store in cache
            self._store_in_cache(cache_key, result)

            self.processing_stats["successful_requests"] += 1
            return result

        except Exception as e:
            self.processing_stats["failed_requests"] += 1
            self.logger.error(f"Error summarizing text: {e}")
            raise

    async def _handle_detect_language(self, message: Message):
        """Handle language detection request."""
        try:
            text = message.data.get("text", "")

            result = await self.detect_language(text)

            response = ResultMessage(
                sender_id=self.agent_id,
                recipient_id=message.sender_id,
                task_id=getattr(message, "task_id", None),
                result=result
            )
            self.outbox.put(response)

        except Exception as e:
            self.logger.error(f"Error in language detection: {e}")
            error_response = ErrorMessage(
                sender_id=self.agent_id,
                recipient_id=message.sender_id,
                task_id=getattr(message, "task_id", None),
                error=str(e)
            )
            self.outbox.put(error_response)

    async def detect_language(self, text: str) -> Dict[str, Any]:
        """
        Detect the language of input text.

        Args:
            text: Input text

        Returns:
            Dictionary containing language detection results
        """
        start_time = time.time()
        self.processing_stats["total_requests"] += 1

        try:
            # Check cache
            cache_key = self._get_cache_key(text, "language")
            cached_result = self._get_from_cache(cache_key)
            if cached_result:
                return cached_result

            detection_result = None
            alternatives = []

            # Use langdetect if available
            if LANGDETECT_AVAILABLE:
                try:
                    detected_lang = detect(text)
                    lang_probs = detect_langs(text)

                    # Get confidence for detected language
                    confidence = next((lang.prob for lang in lang_probs if lang.lang == detected_lang), 0.0)

                    # Get alternatives
                    alternatives = [{"language": lang.lang, "confidence": lang.prob}
                                  for lang in lang_probs[:3] if lang.lang != detected_lang]

                    detection_result = LanguageDetectionResult(
                        language=LanguageCode(detected_lang) if detected_lang in LanguageCode.__members__ else LanguageCode.EN,
                        confidence=confidence,
                        alternatives=alternatives
                    )

                except Exception as e:
                    self.logger.warning(f"langdetect failed: {e}")

            # Fallback to TextBlob
            if detection_result is None and TEXTBLOB_AVAILABLE:
                try:
                    blob = TextBlob(text)
                    detected_lang = blob.detect_language()

                    detection_result = LanguageDetectionResult(
                        language=LanguageCode(detected_lang) if detected_lang in LanguageCode.__members__ else LanguageCode.EN,
                        confidence=0.8,  # TextBlob doesn't provide confidence
                        alternatives=[]
                    )

                except Exception as e:
                    self.logger.warning(f"TextBlob language detection failed: {e}")

            if detection_result is None:
                # Default to English
                detection_result = LanguageDetectionResult(
                    language=LanguageCode.EN,
                    confidence=0.5,
                    alternatives=[]
                )

            result = {
                "language_detection": detection_result.dict(),
                "processing_time": time.time() - start_time,
                "method": "langdetect" if LANGDETECT_AVAILABLE else "textblob"
            }

            # Store in cache
            self._store_in_cache(cache_key, result)

            self.processing_stats["successful_requests"] += 1
            return result

        except Exception as e:
            self.processing_stats["failed_requests"] += 1
            self.logger.error(f"Error detecting language: {e}")
            raise
