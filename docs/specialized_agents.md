# Specialized Agents for Web Scraping System

This document describes the specialized agent types available in the web scraping system and how to use them.

## Overview

The web scraping system uses a multi-agent architecture where each agent specializes in a specific task. The system includes the following specialized agent types:

1. **Core Agents**
   - Coordinator Agent: Manages the overall workflow
   - Scraper Agent: Handles web page fetching
   - Parser Agent: Extracts data from HTML content
   - Storage Agent: Stores and manages extracted data

2. **Advanced Agents**
   - JavaScript Agent: Handles JavaScript rendering and interaction
   - Authentication Agent: Manages login and session handling
   - Anti-Detection Agent: Helps avoid detection by websites
   - Data Transformation Agent: Cleans and transforms extracted data
   - API Integration Agent: Interacts with external APIs
   - NLP Processing Agent: Performs natural language processing on extracted text
   - Image Processing Agent: Processes and analyzes images from web pages

## New Specialized Agents

### API Integration Agent

The API Integration Agent provides capabilities for interacting with external APIs as an alternative or complement to web scraping.

#### Features:
- Making API requests with authentication
- Handling pagination in API responses
- Transforming API data to match desired schemas
- Supporting various authentication methods (API key, OAuth, etc.)

#### Example Usage:

```python
# Configure API
api_config_task = Task(
    type=TaskType.API_AUTHENTICATE,
    parameters={
        "api_id": "weather_api",
        "auth_type": "key",
        "api_key": "your_api_key_here"
    }
)
api_config_task_id = await coordinator.submit_task(api_config_task)

# Make API request
api_request_task = Task(
    type=TaskType.API_REQUEST,
    parameters={
        "api_id": "weather_api",
        "endpoint": "https://api.openweathermap.org/data/2.5/weather",
        "method": "GET",
        "query_params": {
            "q": "London,uk",
            "units": "metric"
        }
    },
    dependencies=[api_config_task_id]
)
api_request_task_id = await coordinator.submit_task(api_request_task)
```

### NLP Processing Agent

The NLP Processing Agent provides natural language processing capabilities for analyzing text content extracted from web pages.

#### Features:
- Entity extraction (people, organizations, locations, dates, etc.)
- Sentiment analysis
- Text classification
- Keyword extraction
- Text summarization
- Language detection

#### Example Usage:

```python
# Extract entities from text
entity_task = Task(
    type=TaskType.NLP_ENTITY_EXTRACTION,
    parameters={
        "text": article_text,
        "entity_types": ["PERSON", "ORG", "GPE", "DATE"]
    }
)
entity_task_id = await coordinator.submit_task(entity_task)

# Analyze sentiment
sentiment_task = Task(
    type=TaskType.NLP_SENTIMENT_ANALYSIS,
    parameters={
        "text": article_text
    }
)
sentiment_task_id = await coordinator.submit_task(sentiment_task)
```

### Image Processing Agent

The Image Processing Agent provides capabilities for processing and analyzing images from web pages.

#### Features:
- Downloading images from web pages
- Optical Character Recognition (OCR) for extracting text from images
- Image classification
- Extracting images from HTML content
- Comparing images for similarity

#### Example Usage:

```python
# Extract images from HTML
image_extraction_task = Task(
    type=TaskType.IMAGE_EXTRACTION,
    parameters={
        "html_content": html_content,
        "base_url": "https://example.com/news",
        "download": True
    }
)
image_task_id = await coordinator.submit_task(image_extraction_task)

# Perform OCR on an image
ocr_task = Task(
    type=TaskType.IMAGE_OCR,
    parameters={
        "image_path": "path/to/image.jpg",
        "language": "eng"
    }
)
ocr_task_id = await coordinator.submit_task(ocr_task)
```

## Installation Requirements

To use these specialized agents, you need to install additional dependencies:

```bash
# Install base requirements
pip install -r requirements.txt

# For NLP Processing Agent
python -m spacy download en_core_web_sm

# For Image Processing Agent with OCR
# On Windows:
# Download and install Tesseract OCR from https://github.com/UB-Mannheim/tesseract/wiki
# Add Tesseract to your PATH

# On Linux:
# sudo apt-get install tesseract-ocr
# sudo apt-get install libtesseract-dev
```

## Task Types

Each specialized agent supports specific task types:

### API Integration Agent Tasks
- `API_REQUEST`: Make a request to an external API
- `API_PAGINATE`: Handle pagination in API responses
- `API_AUTHENTICATE`: Authenticate with an API
- `API_TRANSFORM`: Transform API response data

### NLP Processing Agent Tasks
- `NLP_ENTITY_EXTRACTION`: Extract named entities from text
- `NLP_SENTIMENT_ANALYSIS`: Analyze sentiment of text
- `NLP_TEXT_CLASSIFICATION`: Classify text into categories
- `NLP_KEYWORD_EXTRACTION`: Extract keywords from text
- `NLP_TEXT_SUMMARIZATION`: Generate a summary of text
- `NLP_LANGUAGE_DETECTION`: Detect the language of text

### Image Processing Agent Tasks
- `IMAGE_DOWNLOAD`: Download an image from a URL
- `IMAGE_OCR`: Extract text from an image using OCR
- `IMAGE_CLASSIFICATION`: Classify an image
- `IMAGE_EXTRACTION`: Extract images from HTML content
- `IMAGE_COMPARISON`: Compare two images for similarity

## Integration with Existing Agents

The new specialized agents integrate seamlessly with the existing agent architecture. The Coordinator Agent has been updated to route tasks to the appropriate specialized agent based on the task type.

To use these agents in your workflow:

1. Create instances of the specialized agents
2. Register them with the Coordinator Agent
3. Submit tasks of the appropriate types to the Coordinator Agent
4. The Coordinator will route the tasks to the appropriate agents

See the example script in `examples/advanced_agents_example.py` for a complete demonstration of how to use these specialized agents in a workflow.
