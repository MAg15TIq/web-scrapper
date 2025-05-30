"""
Data Transformation agent for the web scraping system.
"""
import asyncio
import logging
import time
import re
import json
import os
from typing import Dict, Any, Optional, List, Union, Tuple
from datetime import datetime, timedelta
import string
import hashlib
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import html2text
import dateutil.parser
import unicodedata
from jsonschema import validate, ValidationError

from agents.base import Agent
from models.task import Task, TaskStatus, TaskType
from models.message import Message, TaskMessage, ResultMessage, ErrorMessage, StatusMessage

try:
    import nltk
    from nltk.sentiment import SentimentIntensityAnalyzer
    from nltk.tokenize import word_tokenize, sent_tokenize
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.cluster import KMeans
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


class DataCleaner:
    """
    Class for cleaning and normalizing data.
    """
    def __init__(self):
        """Initialize the data cleaner."""
        self.logger = logging.getLogger("data_cleaner")

        # Common patterns for data cleaning
        self.patterns = {
            "price": re.compile(r'[\$£€]?\s*(\d+(?:[.,]\d+)?)', re.IGNORECASE),
            "email": re.compile(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'),
            "phone": re.compile(r'(?:\+\d{1,3}[- ]?)?\(?\d{3}\)?[- ]?\d{3}[- ]?\d{4}'),
            "url": re.compile(r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+(?:/[-\w%!$&\'()*+,;=:]+)*'),
            "whitespace": re.compile(r'\s+'),
            "html_tags": re.compile(r'<[^>]+>'),
            "special_chars": re.compile(r'[^\w\s]'),
            "digits": re.compile(r'\d+'),
        }

    def clean_data(self, data: List[Dict[str, Any]], operations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Clean and normalize data.

        Args:
            data: List of dictionaries containing the data to clean.
            operations: List of cleaning operations to perform.

        Returns:
            The cleaned data.
        """
        if not data:
            return []

        # Convert to DataFrame for easier processing
        df = pd.DataFrame(data)

        # Apply each operation
        for op in operations:
            field = op.get("field", "")
            operation = op.get("operation", "")

            if not field or not operation:
                continue

            # Apply to all fields
            if field == "*":
                for col in df.columns:
                    df = self._apply_operation(df, col, operation, op)
            else:
                # Apply to specific field
                if field in df.columns:
                    df = self._apply_operation(df, field, operation, op)

        # Convert back to list of dictionaries
        return df.to_dict(orient="records")

    def _apply_operation(self, df: pd.DataFrame, field: str, operation: str, op_params: Dict[str, Any]) -> pd.DataFrame:
        """
        Apply a cleaning operation to a field.

        Args:
            df: DataFrame containing the data.
            field: Field to apply the operation to.
            operation: Operation to apply.
            op_params: Additional parameters for the operation.

        Returns:
            The DataFrame with the operation applied.
        """
        # Skip non-string columns for text operations
        if operation in ["strip_whitespace", "lowercase", "uppercase", "remove_html", "remove_special_chars"] and not pd.api.types.is_string_dtype(df[field]):
            return df

        if operation == "strip_whitespace":
            df[field] = df[field].str.strip()

        elif operation == "lowercase":
            df[field] = df[field].str.lower()

        elif operation == "uppercase":
            df[field] = df[field].str.upper()

        elif operation == "remove_html":
            df[field] = df[field].str.replace(self.patterns["html_tags"], "", regex=True)

        elif operation == "remove_special_chars":
            df[field] = df[field].str.replace(self.patterns["special_chars"], "", regex=True)

        elif operation == "extract_number":
            df[field] = df[field].apply(self._extract_number)

        elif operation == "extract_email":
            df[field] = df[field].apply(self._extract_pattern, args=(self.patterns["email"],))

        elif operation == "extract_phone":
            df[field] = df[field].apply(self._extract_pattern, args=(self.patterns["phone"],))

        elif operation == "extract_url":
            df[field] = df[field].apply(self._extract_pattern, args=(self.patterns["url"],))

        elif operation == "format_date":
            input_format = op_params.get("input_format")
            output_format = op_params.get("output_format", "%Y-%m-%d")
            df[field] = df[field].apply(self._format_date, args=(input_format, output_format))

        elif operation == "replace_text":
            old_text = op_params.get("old_text", "")
            new_text = op_params.get("new_text", "")
            if old_text:
                df[field] = df[field].str.replace(old_text, new_text, regex=False)

        elif operation == "remove_empty":
            # For each row, remove the field if it's empty
            df = df.dropna(subset=[field])

        elif operation == "fill_empty":
            fill_value = op_params.get("fill_value", "")
            df[field] = df[field].fillna(fill_value)

        return df

    def _extract_number(self, value: Any) -> Optional[float]:
        """Extract a number from a string."""
        if pd.isna(value):
            return None

        value_str = str(value)
        match = self.patterns["price"].search(value_str)
        if match:
            # Replace comma with dot for proper float parsing
            num_str = match.group(1).replace(",", ".")
            try:
                return float(num_str)
            except ValueError:
                return None
        return None

    def _extract_pattern(self, value: Any, pattern: re.Pattern) -> Optional[str]:
        """Extract a pattern from a string."""
        if pd.isna(value):
            return None

        value_str = str(value)
        match = pattern.search(value_str)
        if match:
            return match.group(0)
        return None

    def _format_date(self, value: Any, input_format: Optional[str], output_format: str) -> Optional[str]:
        """Format a date string."""
        if pd.isna(value):
            return None

        value_str = str(value)
        try:
            if input_format:
                date_obj = datetime.strptime(value_str, input_format)
            else:
                # Try common formats
                for fmt in ["%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%Y/%m/%d", "%d-%m-%Y", "%m-%d-%Y"]:
                    try:
                        date_obj = datetime.strptime(value_str, fmt)
                        break
                    except ValueError:
                        continue
                else:
                    # If no format worked, try pandas to_datetime
                    date_obj = pd.to_datetime(value_str).to_pydatetime()

            return date_obj.strftime(output_format)
        except (ValueError, TypeError):
            return None


class SchemaTransformer:
    """
    Class for transforming data from one schema to another.
    """
    def __init__(self):
        """Initialize the schema transformer."""
        self.logger = logging.getLogger("schema_transformer")

    def transform_schema(self, data: List[Dict[str, Any]], mapping: Dict[str, str],
                         target_schema: Optional[Dict[str, str]] = None) -> List[Dict[str, Any]]:
        """
        Transform data from one schema to another.

        Args:
            data: List of dictionaries containing the data to transform.
            mapping: Mapping from target fields to source fields.
            target_schema: Optional schema for the target data.

        Returns:
            The transformed data.
        """
        if not data:
            return []

        # Convert to DataFrame for easier processing
        df = pd.DataFrame(data)

        # Create a new DataFrame with the target schema
        target_df = pd.DataFrame()

        # Apply the mapping
        for target_field, source_field in mapping.items():
            if source_field in df.columns:
                target_df[target_field] = df[source_field]
            else:
                # If the source field doesn't exist, create an empty column
                target_df[target_field] = None

        # Apply the target schema if provided
        if target_schema:
            for field, data_type in target_schema.items():
                if field in target_df.columns:
                    target_df = self._convert_type(target_df, field, data_type)

        # Convert back to list of dictionaries
        return target_df.to_dict(orient="records")

    def _convert_type(self, df: pd.DataFrame, field: str, data_type: str) -> pd.DataFrame:
        """
        Convert a field to the specified data type.

        Args:
            df: DataFrame containing the data.
            field: Field to convert.
            data_type: Data type to convert to.

        Returns:
            The DataFrame with the field converted.
        """
        try:
            if data_type in ["int", "integer"]:
                df[field] = pd.to_numeric(df[field], errors="coerce").astype("Int64")
            elif data_type == "float":
                df[field] = pd.to_numeric(df[field], errors="coerce")
            elif data_type in ["bool", "boolean"]:
                df[field] = df[field].astype(bool)
            elif data_type == "date":
                df[field] = pd.to_datetime(df[field], errors="coerce").dt.date
            elif data_type == "datetime":
                df[field] = pd.to_datetime(df[field], errors="coerce")
            elif data_type in ["string", "str"]:
                df[field] = df[field].astype(str)
        except Exception as e:
            self.logger.error(f"Error converting {field} to {data_type}: {str(e)}")

        return df


class TextAnalyzer:
    """
    Class for analyzing text data.
    """
    def __init__(self):
        """Initialize the text analyzer."""
        self.logger = logging.getLogger("text_analyzer")
        self.initialized = False

        # Check if NLTK is available
        if not NLTK_AVAILABLE:
            self.logger.warning("NLTK is not available. Text analysis will be limited.")
            return

        try:
            # Download necessary NLTK resources
            nltk.download("punkt", quiet=True)
            nltk.download("stopwords", quiet=True)
            nltk.download("wordnet", quiet=True)
            nltk.download("vader_lexicon", quiet=True)

            # Initialize NLTK components
            self.stop_words = set(stopwords.words("english"))
            self.lemmatizer = WordNetLemmatizer()
            self.sentiment_analyzer = SentimentIntensityAnalyzer()

            self.initialized = True
        except Exception as e:
            self.logger.error(f"Error initializing text analyzer: {str(e)}")

    def analyze_text(self, text: str, analyses: List[str], language: str = "en") -> Dict[str, Any]:
        """
        Analyze text.

        Args:
            text: Text to analyze.
            analyses: List of analyses to perform.
            language: Language of the text.

        Returns:
            A dictionary containing the analysis results.
        """
        if not text:
            return {}

        results = {}

        # Basic text statistics (always included)
        results["statistics"] = self._get_text_statistics(text)

        # Perform requested analyses
        for analysis in analyses:
            if analysis == "sentiment" and self.initialized:
                results["sentiment"] = self._analyze_sentiment(text)
            elif analysis == "entities" and self.initialized:
                results["entities"] = self._extract_entities(text)
            elif analysis == "keywords" and self.initialized:
                results["keywords"] = self._extract_keywords(text)
            elif analysis == "language":
                results["language"] = self._detect_language(text)
            elif analysis == "summary" and self.initialized:
                results["summary"] = self._generate_summary(text)

        return results

    def _get_text_statistics(self, text: str) -> Dict[str, Any]:
        """Get basic statistics about the text."""
        sentences = sent_tokenize(text) if NLTK_AVAILABLE else text.split(".")
        words = word_tokenize(text) if NLTK_AVAILABLE else text.split()

        return {
            "character_count": len(text),
            "word_count": len(words),
            "sentence_count": len(sentences),
            "average_word_length": sum(len(word) for word in words) / len(words) if words else 0,
            "average_sentence_length": len(words) / len(sentences) if sentences else 0,
        }

    def _analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """Analyze sentiment of the text."""
        if not self.initialized:
            return {"error": "NLTK not initialized"}

        sentiment_scores = self.sentiment_analyzer.polarity_scores(text)

        # Determine overall sentiment
        compound = sentiment_scores["compound"]
        if compound >= 0.05:
            sentiment = "positive"
        elif compound <= -0.05:
            sentiment = "negative"
        else:
            sentiment = "neutral"

        return {
            "sentiment": sentiment,
            "compound_score": compound,
            "positive_score": sentiment_scores["pos"],
            "negative_score": sentiment_scores["neg"],
            "neutral_score": sentiment_scores["neu"],
        }

    def _extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """Extract entities from the text."""
        if not self.initialized:
            return [{"error": "NLTK not initialized"}]

        # This is a simplified entity extraction
        # In a real implementation, you would use a proper NER model

        entities = []
        words = word_tokenize(text)

        # Look for capitalized words that aren't at the start of sentences
        for i, word in enumerate(words):
            if (word[0].isupper() and
                i > 0 and
                words[i-1] not in [".", "!", "?"] and
                word.lower() not in self.stop_words):

                entities.append({
                    "text": word,
                    "type": "UNKNOWN",  # Without a proper NER model, we can't determine the type
                    "position": i
                })

        return entities

    def _extract_keywords(self, text: str) -> List[Dict[str, Any]]:
        """Extract keywords from the text."""
        if not self.initialized:
            return [{"error": "NLTK not initialized"}]

        words = word_tokenize(text.lower())

        # Remove stopwords and punctuation
        words = [word for word in words
                if word.lower() not in self.stop_words
                and word not in string.punctuation]

        # Lemmatize words
        words = [self.lemmatizer.lemmatize(word) for word in words]

        # Count word frequencies
        word_freq = {}
        for word in words:
            if word in word_freq:
                word_freq[word] += 1
            else:
                word_freq[word] = 1

        # Sort by frequency
        keywords = [{"word": word, "frequency": freq}
                   for word, freq in sorted(word_freq.items(), key=lambda x: x[1], reverse=True)]

        # Return top 10 keywords
        return keywords[:10]

    def _detect_language(self, text: str) -> Dict[str, Any]:
        """Detect the language of the text."""
        # This is a simplified language detection
        # In a real implementation, you would use a proper language detection library

        # Count common words in different languages
        english_common = ["the", "and", "is", "in", "to", "of", "a", "for", "that", "with"]
        spanish_common = ["el", "la", "de", "en", "y", "a", "que", "los", "se", "del"]
        french_common = ["le", "la", "de", "et", "en", "un", "une", "du", "des", "est"]
        german_common = ["der", "die", "und", "in", "den", "von", "zu", "das", "mit", "sich"]

        text_lower = text.lower()
        words = set(word_tokenize(text_lower) if NLTK_AVAILABLE else text_lower.split())

        scores = {
            "en": sum(1 for word in english_common if word in words),
            "es": sum(1 for word in spanish_common if word in words),
            "fr": sum(1 for word in french_common if word in words),
            "de": sum(1 for word in german_common if word in words),
        }

        # Get the language with the highest score
        detected_lang = max(scores.items(), key=lambda x: x[1])

        # If the score is too low, the language is unknown
        if detected_lang[1] < 2:
            detected_lang = ("unknown", 0)

        return {
            "language_code": detected_lang[0],
            "confidence": min(detected_lang[1] / 10, 1.0),  # Normalize to 0-1
        }

    def _generate_summary(self, text: str) -> Dict[str, Any]:
        """Generate a summary of the text."""
        if not self.initialized:
            return {"error": "NLTK not initialized"}

        sentences = sent_tokenize(text)

        if len(sentences) <= 3:
            # Text is already short, use as is
            return {
                "summary": text,
                "summary_length": len(text),
                "original_length": len(text),
                "compression_ratio": 1.0
            }

        # Extract the first and last sentences, and one from the middle
        summary_sentences = [
            sentences[0],
            sentences[len(sentences) // 2],
            sentences[-1]
        ]

        summary = " ".join(summary_sentences)

        return {
            "summary": summary,
            "summary_length": len(summary),
            "original_length": len(text),
            "compression_ratio": len(summary) / len(text)
        }


class DataEnricher:
    """
    Class for enriching data with additional information.
    """
    def __init__(self):
        """Initialize the data enricher."""
        self.logger = logging.getLogger("data_enricher")
        self.text_analyzer = TextAnalyzer()

    def enrich_data(self, data: List[Dict[str, Any]], operations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Enrich data with additional information.

        Args:
            data: List of dictionaries containing the data to enrich.
            operations: List of enrichment operations to perform.

        Returns:
            The enriched data.
        """
        if not data:
            return []

        # Convert to DataFrame for easier processing
        df = pd.DataFrame(data)

        # Apply each operation
        for op in operations:
            operation_type = op.get("type", "")

            if operation_type == "text_analysis":
                field = op.get("field", "")
                analyses = op.get("analyses", ["sentiment", "keywords"])
                target_field = op.get("target_field", f"{field}_analysis")

                if field in df.columns:
                    df[target_field] = df[field].apply(
                        lambda text: self.text_analyzer.analyze_text(text, analyses) if text else {}
                    )

            elif operation_type == "generate_id":
                fields = op.get("fields", [])
                target_field = op.get("target_field", "id")

                if fields:
                    df[target_field] = df.apply(
                        lambda row: self._generate_id([row.get(field, "") for field in fields]),
                        axis=1
                    )

            elif operation_type == "combine_fields":
                fields = op.get("fields", [])
                target_field = op.get("target_field", "combined")
                separator = op.get("separator", " ")

                if fields:
                    df[target_field] = df.apply(
                        lambda row: separator.join([str(row.get(field, "")) for field in fields if row.get(field)]),
                        axis=1
                    )

            elif operation_type == "categorize":
                field = op.get("field", "")
                categories = op.get("categories", {})
                target_field = op.get("target_field", f"{field}_category")

                if field in df.columns and categories:
                    df[target_field] = df[field].apply(
                        lambda value: self._categorize(value, categories)
                    )

        # Convert back to list of dictionaries
        return df.to_dict(orient="records")

    def _generate_id(self, values: List[Any]) -> str:
        """Generate a unique ID from a list of values."""
        # Combine values into a string
        combined = "_".join([str(value) for value in values if value])

        # Generate a hash
        hash_obj = hashlib.md5(combined.encode())

        return hash_obj.hexdigest()[:12]

    def _categorize(self, value: Any, categories: Dict[str, List[str]]) -> str:
        """Categorize a value based on a mapping."""
        if pd.isna(value):
            return "unknown"

        value_str = str(value).lower()

        for category, keywords in categories.items():
            for keyword in keywords:
                if keyword.lower() in value_str:
                    return category

        return "other"


class DataTransformationAgent(Agent):
    """
    Agent responsible for transforming and enriching data.
    """
    def __init__(self, agent_id: Optional[str] = None, coordinator_id: str = "coordinator"):
        """
        Initialize a new data transformation agent.

        Args:
            agent_id: Unique identifier for the agent. If None, a UUID will be generated.
            coordinator_id: ID of the coordinator agent.
        """
        super().__init__(agent_id=agent_id, agent_type="data_transformation")
        self.coordinator_id = coordinator_id
        self.data_cleaner = DataCleaner()
        self.schema_transformer = SchemaTransformer()
        self.text_analyzer = TextAnalyzer()
        self.data_enricher = DataEnricher()

    async def _send_message(self, message: Message) -> None:
        """
        Send a message to another agent.

        Args:
            message: The message to send.
        """
        # Add message to outbox
        self.outbox.put(message)

    async def execute_task(self, task: Task) -> Dict[str, Any]:
        """
        Execute a task assigned to this agent.

        Args:
            task: The task to execute.

        Returns:
            A dictionary containing the result of the task.
        """
        if task.type == TaskType.CLEAN_DATA:
            return await self._clean_data(task)
        elif task.type == TaskType.TRANSFORM_SCHEMA:
            return await self._transform_schema(task)
        elif task.type == TaskType.ENRICH_DATA:
            return await self._enrich_data(task)
        elif task.type == TaskType.ANALYZE_TEXT:
            return await self._analyze_text(task)
        else:
            raise ValueError(f"Unsupported task type for data transformation agent: {task.type}")

    async def _clean_data(self, task: Task) -> Dict[str, Any]:
        """
        Clean and normalize data.

        Args:
            task: The task containing the data to clean.

        Returns:
            A dictionary containing the cleaned data.
        """
        data = task.parameters.get("data")
        if not data:
            raise ValueError("Data parameter is required for clean_data task")

        operations = task.parameters.get("operations", [])
        add_metadata = task.parameters.get("add_metadata", False)

        self.logger.info(f"Cleaning data with {len(operations)} operations")

        start_time = time.time()

        # Clean the data
        cleaned_data = self.data_cleaner.clean_data(data, operations)

        end_time = time.time()
        execution_time = end_time - start_time

        result = {
            "data": cleaned_data,
            "original_count": len(data),
            "cleaned_count": len(cleaned_data),
            "execution_time": execution_time
        }

        if add_metadata:
            result["operations"] = operations
            result["timestamp"] = end_time

        return result

    async def _transform_schema(self, task: Task) -> Dict[str, Any]:
        """
        Transform data from one schema to another.

        Args:
            task: The task containing the data to transform.

        Returns:
            A dictionary containing the transformed data.
        """
        data = task.parameters.get("data")
        if not data:
            raise ValueError("Data parameter is required for transform_schema task")

        source_schema = task.parameters.get("source_schema")
        target_schema = task.parameters.get("target_schema")
        mapping = task.parameters.get("mapping")

        if not mapping:
            raise ValueError("Mapping parameter is required for transform_schema task")

        self.logger.info(f"Transforming schema with {len(mapping)} field mappings")

        start_time = time.time()

        # Transform the schema
        transformed_data = self.schema_transformer.transform_schema(data, mapping, target_schema)

        end_time = time.time()
        execution_time = end_time - start_time

        return {
            "data": transformed_data,
            "original_schema": list(data[0].keys()) if data else [],
            "transformed_schema": list(transformed_data[0].keys()) if transformed_data else [],
            "mapping": mapping,
            "execution_time": execution_time
        }

    async def _enrich_data(self, task: Task) -> Dict[str, Any]:
        """
        Enrich data with additional information.

        Args:
            task: The task containing the data to enrich.

        Returns:
            A dictionary containing the enriched data.
        """
        data = task.parameters.get("data")
        if not data:
            raise ValueError("Data parameter is required for enrich_data task")

        operations = task.parameters.get("operations", [])

        self.logger.info(f"Enriching data with {len(operations)} operations")

        start_time = time.time()

        # Enrich the data
        enriched_data = self.data_enricher.enrich_data(data, operations)

        end_time = time.time()
        execution_time = end_time - start_time

        return {
            "data": enriched_data,
            "original_fields": list(data[0].keys()) if data else [],
            "enriched_fields": list(enriched_data[0].keys()) if enriched_data else [],
            "operations": operations,
            "execution_time": execution_time
        }

    async def _analyze_text(self, task: Task) -> Dict[str, Any]:
        """
        Analyze text.

        Args:
            task: The task containing the text to analyze.

        Returns:
            A dictionary containing the analysis results.
        """
        text = task.parameters.get("text")
        if not text:
            raise ValueError("Text parameter is required for analyze_text task")

        analyses = task.parameters.get("analyses", ["sentiment", "entities", "keywords"])
        language = task.parameters.get("language", "en")

        self.logger.info(f"Analyzing text with {len(analyses)} analyses")

        start_time = time.time()

        # Analyze the text
        analysis_results = self.text_analyzer.analyze_text(text, analyses, language)

        end_time = time.time()
        execution_time = end_time - start_time

        return {
            "text": text[:100] + "..." if len(text) > 100 else text,  # Truncate for the result
            "analyses": analyses,
            "results": analysis_results,
            "execution_time": execution_time
        }

    async def cleanup(self) -> None:
        """Clean up resources used by the agent."""
        pass

        # Schema cache
        self.schemas: Dict[str, Dict[str, Any]] = {}

        # Transformation rules
        self.transformation_rules: Dict[str, Dict[str, Any]] = {}

        # Validation rules
        self.validation_rules: Dict[str, Dict[str, Any]] = {}

        # Initialize transformation functions
        self.transformers = {
            "clean_text": self._clean_text,
            "extract_numbers": self._extract_numbers,
            "normalize_dates": self._normalize_dates,
            "convert_currency": self._convert_currency,
            "extract_emails": self._extract_emails,
            "extract_phone_numbers": self._extract_phone_numbers,
            "extract_urls": self._extract_urls,
            "normalize_whitespace": self._normalize_whitespace,
            "remove_html": self._remove_html,
            "convert_to_lowercase": self._convert_to_lowercase,
            "convert_to_uppercase": self._convert_to_uppercase,
            "hash_value": self._hash_value
        }

        # Register message handlers
        self.register_handler("transform", self._handle_transform)
        self.register_handler("validate", self._handle_validate)
        self.register_handler("clean", self._handle_clean)
        self.register_handler("normalize", self._handle_normalize)

        # Start periodic tasks
        self._start_periodic_tasks()

    def _start_periodic_tasks(self) -> None:
        """Start periodic tasks for the data transformer agent."""
        asyncio.create_task(self._periodic_schema_cleanup())
        asyncio.create_task(self._periodic_rule_cleanup())

    async def _periodic_schema_cleanup(self) -> None:
        """Periodically clean up unused schemas."""
        while self.running:
            try:
                await asyncio.sleep(3600)  # Every hour
                if not self.running:
                    break

                self._cleanup_schemas()
            except Exception as e:
                self.logger.error(f"Error in schema cleanup: {str(e)}", exc_info=True)

    async def _periodic_rule_cleanup(self) -> None:
        """Periodically clean up unused rules."""
        while self.running:
            try:
                await asyncio.sleep(3600)  # Every hour
                if not self.running:
                    break

                self._cleanup_rules()
            except Exception as e:
                self.logger.error(f"Error in rule cleanup: {str(e)}", exc_info=True)

    def _cleanup_schemas(self) -> None:
        """Clean up unused schemas."""
        current_time = time.time()
        self.schemas = {
            key: schema
            for key, schema in self.schemas.items()
            if current_time - schema.get("last_used", 0) < 86400  # 24 hours
        }

    def _cleanup_rules(self) -> None:
        """Clean up unused rules."""
        current_time = time.time()
        self.transformation_rules = {
            key: rule
            for key, rule in self.transformation_rules.items()
            if current_time - rule.get("last_used", 0) < 86400  # 24 hours
        }
        self.validation_rules = {
            key: rule
            for key, rule in self.validation_rules.items()
            if current_time - rule.get("last_used", 0) < 86400  # 24 hours
        }

    async def _handle_transform(self, message: Message) -> None:
        """
        Handle a data transformation request.

        Args:
            message: The message containing transformation parameters.
        """
        if not hasattr(message, "data") or not hasattr(message, "rules"):
            self.logger.warning("Received transform message without required parameters")
            return

        try:
            result = await self.transform_data(
                message.data,
                message.rules
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
            self.logger.error(f"Error transforming data: {str(e)}", exc_info=True)

            # Send error
            error = ErrorMessage(
                sender_id=self.agent_id,
                recipient_id=message.sender_id,
                task_id=message.task_id if hasattr(message, "task_id") else None,
                error_type="transformation_error",
                error_message=str(e)
            )
            self.outbox.put(error)

    async def _handle_validate(self, message: Message) -> None:
        """
        Handle a data validation request.

        Args:
            message: The message containing validation parameters.
        """
        if not hasattr(message, "data") or not hasattr(message, "schema"):
            self.logger.warning("Received validate message without required parameters")
            return

        try:
            result = await self.validate_data(
                message.data,
                message.schema
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
            self.logger.error(f"Error validating data: {str(e)}", exc_info=True)

            # Send error
            error = ErrorMessage(
                sender_id=self.agent_id,
                recipient_id=message.sender_id,
                task_id=message.task_id if hasattr(message, "task_id") else None,
                error_type="validation_error",
                error_message=str(e)
            )
            self.outbox.put(error)

    async def _handle_clean(self, message: Message) -> None:
        """
        Handle a data cleaning request.

        Args:
            message: The message containing cleaning parameters.
        """
        if not hasattr(message, "data") or not hasattr(message, "rules"):
            self.logger.warning("Received clean message without required parameters")
            return

        try:
            result = await self.clean_data(
                message.data,
                message.rules
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
            self.logger.error(f"Error cleaning data: {str(e)}", exc_info=True)

            # Send error
            error = ErrorMessage(
                sender_id=self.agent_id,
                recipient_id=message.sender_id,
                task_id=message.task_id if hasattr(message, "task_id") else None,
                error_type="cleaning_error",
                error_message=str(e)
            )
            self.outbox.put(error)

    async def _handle_normalize(self, message: Message) -> None:
        """
        Handle a data normalization request.

        Args:
            message: The message containing normalization parameters.
        """
        if not hasattr(message, "data") or not hasattr(message, "rules"):
            self.logger.warning("Received normalize message without required parameters")
            return

        try:
            result = await self.normalize_data(
                message.data,
                message.rules
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
            self.logger.error(f"Error normalizing data: {str(e)}", exc_info=True)

            # Send error
            error = ErrorMessage(
                sender_id=self.agent_id,
                recipient_id=message.sender_id,
                task_id=message.task_id if hasattr(message, "task_id") else None,
                error_type="normalization_error",
                error_message=str(e)
            )
            self.outbox.put(error)

    async def transform_data(self, data: Any, rules: Dict[str, Any]) -> Dict[str, Any]:
        """
        Transform data according to rules.

        Args:
            data: The data to transform.
            rules: The transformation rules to apply.

        Returns:
            A dictionary containing the transformed data.
        """
        # Store rules
        rule_id = hashlib.md5(json.dumps(rules, sort_keys=True).encode()).hexdigest()
        self.transformation_rules[rule_id] = {
            "rules": rules,
            "last_used": time.time()
        }

        # Apply transformations
        if isinstance(data, dict):
            transformed_data = await self._transform_dict(data, rules)
        elif isinstance(data, list):
            transformed_data = await self._transform_list(data, rules)
        else:
            transformed_data = await self._transform_value(data, rules)

        return {
            "transformed_data": transformed_data,
            "rule_id": rule_id
        }

    async def validate_data(self, data: Any, schema: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate data against a schema.

        Args:
            data: The data to validate.
            schema: The JSON schema to validate against.

        Returns:
            A dictionary containing validation results.
        """
        # Store schema
        schema_id = hashlib.md5(json.dumps(schema, sort_keys=True).encode()).hexdigest()
        self.schemas[schema_id] = {
            "schema": schema,
            "last_used": time.time()
        }

        try:
            # Validate data
            validate(instance=data, schema=schema)

            return {
                "valid": True,
                "schema_id": schema_id
            }

        except ValidationError as e:
            return {
                "valid": False,
                "schema_id": schema_id,
                "errors": str(e),
                "path": list(e.path)
            }

    async def clean_data(self, data: Any, rules: Dict[str, Any]) -> Dict[str, Any]:
        """
        Clean data according to rules.

        Args:
            data: The data to clean.
            rules: The cleaning rules to apply.

        Returns:
            A dictionary containing the cleaned data.
        """
        # Convert rules to the format expected by DataCleaner
        operations = []
        for field, field_rules in rules.items():
            for operation, params in field_rules.items():
                op = {
                    "field": field,
                    "operation": operation
                }
                if isinstance(params, dict):
                    op.update(params)
                operations.append(op)

        # Clean data using DataCleaner
        cleaned_data = self.data_cleaner.clean_data(
            data if isinstance(data, list) else [data],
            operations
        )

        # If input was a single item, return a single item
        if not isinstance(data, list):
            cleaned_data = cleaned_data[0] if cleaned_data else {}

        return {
            "cleaned_data": cleaned_data
        }

    async def normalize_data(self, data: Any, rules: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize data according to rules.

        Args:
            data: The data to normalize.
            rules: The normalization rules to apply.

        Returns:
            A dictionary containing the normalized data.
        """
        # Extract mapping and schema from rules
        mapping = rules.get("mapping", {})
        schema = rules.get("schema", {})

        # Transform schema using SchemaTransformer
        normalized_data = self.schema_transformer.transform_schema(
            data if isinstance(data, list) else [data],
            mapping,
            schema
        )

        # If input was a single item, return a single item
        if not isinstance(data, list):
            normalized_data = normalized_data[0] if normalized_data else {}

        return {
            "normalized_data": normalized_data
        }

    async def _transform_dict(self, data: Dict[str, Any], rules: Dict[str, Any]) -> Dict[str, Any]:
        """Transform a dictionary according to rules."""
        result = {}

        for key, value in data.items():
            # Get field rules
            field_rules = rules.get(key, {})

            # Transform value
            if isinstance(value, (dict, list)):
                transformed = await self.transform_data(value, field_rules)
                transformed = transformed.get("transformed_data", value)
            else:
                transformed = await self._transform_value(value, field_rules)

            # Apply field transformations
            if "rename" in field_rules:
                key = field_rules["rename"]

            result[key] = transformed

        return result

    async def _transform_list(self, data: List[Any], rules: Dict[str, Any]) -> List[Any]:
        """Transform a list according to rules."""
        result = []

        for item in data:
            # Transform item
            if isinstance(item, (dict, list)):
                transformed = await self.transform_data(item, rules)
                transformed = transformed.get("transformed_data", item)
            else:
                transformed = await self._transform_value(item, rules)

            result.append(transformed)

        return result

    async def _transform_value(self, value: Any, rules: Dict[str, Any]) -> Any:
        """Transform a value according to rules."""
        if not rules:
            return value

        # Apply type conversion
        if "type" in rules:
            value = self._convert_type(value, rules["type"])

        # Apply format conversion
        if "format" in rules:
            value = self._convert_format(value, rules["format"])

        # Apply value mapping
        if "map" in rules:
            value = rules["map"].get(value, value)

        # Apply default value
        if value is None and "default" in rules:
            value = rules["default"]

        return value

    def _convert_type(self, value: Any, type_name: str) -> Any:
        """Convert a value to the specified type."""
        if value is None:
            return None

        try:
            if type_name == "int" or type_name == "integer":
                return int(value)
            elif type_name == "float":
                return float(value)
            elif type_name == "bool" or type_name == "boolean":
                return bool(value)
            elif type_name == "str" or type_name == "string":
                return str(value)
            elif type_name == "list":
                return list(value)
            elif type_name == "dict":
                return dict(value)
        except (ValueError, TypeError):
            pass

        return value

    def _convert_format(self, value: Any, format_spec: Dict[str, Any]) -> Any:
        """Convert a value to the specified format."""
        if value is None:
            return None

        format_type = format_spec.get("type", "")

        if format_type == "date":
            try:
                input_format = format_spec.get("input_format")
                output_format = format_spec.get("output_format", "%Y-%m-%d")

                if input_format:
                    date_obj = datetime.strptime(str(value), input_format)
                else:
                    date_obj = dateutil.parser.parse(str(value))

                return date_obj.strftime(output_format)
            except (ValueError, TypeError):
                pass

        return value

    def _clean_text(self, text: str) -> str:
        """Clean text by removing special characters and normalizing whitespace."""
        if not text:
            return ""

        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)

        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()

        return text

    def _extract_numbers(self, text: str) -> List[float]:
        """Extract numbers from text."""
        if not text:
            return []

        # Find all numbers in the text
        numbers = re.findall(r'[-+]?\d*\.\d+|\d+', text)

        # Convert to float
        return [float(num) for num in numbers]

    def _normalize_dates(self, text: str, output_format: str = "%Y-%m-%d") -> str:
        """Normalize dates in text to a standard format."""
        if not text:
            return ""

        # Try to parse the date
        try:
            date_obj = dateutil.parser.parse(text)
            return date_obj.strftime(output_format)
        except (ValueError, TypeError):
            return text

    def _convert_currency(self, value: str) -> float:
        """Convert currency string to float."""
        if not value:
            return 0.0

        # Remove currency symbols and commas
        value = re.sub(r'[^\d.-]', '', value)

        # Convert to float
        try:
            return float(value)
        except (ValueError, TypeError):
            return 0.0

    def _extract_emails(self, text: str) -> List[str]:
        """Extract email addresses from text."""
        if not text:
            return []

        # Find all email addresses in the text
        emails = re.findall(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', text)

        return emails

    def _extract_phone_numbers(self, text: str) -> List[str]:
        """Extract phone numbers from text."""
        if not text:
            return []

        # Find all phone numbers in the text
        phones = re.findall(r'(?:\+\d{1,3}[- ]?)?\(?\d{3}\)?[- ]?\d{3}[- ]?\d{4}', text)

        return phones

    def _extract_urls(self, text: str) -> List[str]:
        """Extract URLs from text."""
        if not text:
            return []

        # Find all URLs in the text
        urls = re.findall(r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+(?:/[^/\s]*)*', text)

        return urls

    def _normalize_whitespace(self, text: str) -> str:
        """Normalize whitespace in text."""
        if not text:
            return ""

        # Replace multiple whitespace characters with a single space
        return re.sub(r'\s+', ' ', text).strip()

    def _remove_html(self, text: str) -> str:
        """Remove HTML tags from text."""
        if not text:
            return ""

        # Remove HTML tags
        return re.sub(r'<[^>]+>', '', text)

    def _convert_to_lowercase(self, text: str) -> str:
        """Convert text to lowercase."""
        if not text:
            return ""

        return text.lower()

    def _convert_to_uppercase(self, text: str) -> str:
        """Convert text to uppercase."""
        if not text:
            return ""

        return text.upper()

    def _hash_value(self, value: str, algorithm: str = "md5") -> str:
        """Hash a value using the specified algorithm."""
        if not value:
            return ""

        value_str = str(value)

        if algorithm == "md5":
            return hashlib.md5(value_str.encode()).hexdigest()
        elif algorithm == "sha1":
            return hashlib.sha1(value_str.encode()).hexdigest()
        elif algorithm == "sha256":
            return hashlib.sha256(value_str.encode()).hexdigest()
        else:
            return hashlib.md5(value_str.encode()).hexdigest()
