# 🚀 Phase 5: Advanced Data Processing - Installation Guide

## 📋 Prerequisites

- Python 3.8 or higher
- Existing web scraper system
- Internet connection for downloading models and dependencies

## 🔧 Installation Steps

### 1. Core Dependencies (Required)

```bash
# Install core dependencies
pip install pydantic pydantic-settings
pip install httpx requests
pip install numpy
```

### 2. Natural Language Processing (Optional but Recommended)

```bash
# Install NLP libraries
pip install spacy transformers torch textblob langdetect

# Download spaCy English model
python -m spacy download en_core_web_sm

# Download NLTK data (if using NLTK features)
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"

# Install translation support
pip install googletrans==4.0.0rc1
```

### 3. Computer Vision (Optional)

```bash
# Install computer vision libraries
pip install opencv-python pillow scikit-image

# Install OCR engines
pip install pytesseract easyocr paddleocr

# Install object detection
pip install ultralytics

# Note: For Tesseract, you may need to install the binary separately:
# Windows: Download from https://github.com/UB-Mannheim/tesseract/wiki
# Ubuntu: sudo apt install tesseract-ocr
# macOS: brew install tesseract
```

### 4. Data Enrichment (Optional)

```bash
# Install geocoding and validation libraries
pip install googlemaps geopy phonenumbers email-validator

# Install financial data support
pip install yfinance

# Install additional validation libraries
pip install geoip2
```

### 5. Configuration Setup

Create a `.env` file in your project root:

```env
# NLP Configuration
SPACY_MODEL=en_core_web_sm
SENTIMENT_MODEL=cardiffnlp/twitter-roberta-base-sentiment-latest
SUMMARIZATION_MODEL=facebook/bart-large-cnn
MAX_TEXT_LENGTH=10000
ENTITY_CONFIDENCE_THRESHOLD=0.7

# Computer Vision Configuration
TESSERACT_PATH=/usr/bin/tesseract  # Adjust path as needed
MAX_IMAGE_SIZE=2048
OBJECT_DETECTION_MODEL=yolov5s
OBJECT_CONFIDENCE_THRESHOLD=0.5

# Data Enrichment APIs (Optional - add your API keys)
GOOGLE_MAPS_API_KEY=your_google_maps_api_key_here
OPENCAGE_API_KEY=your_opencage_api_key_here
MAPBOX_API_KEY=your_mapbox_api_key_here
CLEARBIT_API_KEY=your_clearbit_api_key_here
OPENWEATHER_API_KEY=your_openweather_api_key_here
ALPHA_VANTAGE_API_KEY=your_alpha_vantage_api_key_here

# Rate Limiting
GLOBAL_RATE_LIMIT=100
GEOCODING_RATE_LIMIT=50
NLP_CACHE_TTL=3600
CV_CACHE_TTL=1800
ENRICHMENT_CACHE_TTL=7200
```

## 🧪 Verification

### 1. Run Basic Tests

```bash
# Test basic functionality
python test_phase5_basic.py
```

Expected output:
```
🚀 Phase 5: Advanced Data Processing - Basic Tests
============================================================
Testing Phase 5 imports...
✅ Data processing models imported successfully
✅ Text processing utilities imported successfully
✅ Data validation utilities imported successfully

Testing data models...
✅ Entity model: Apple Inc. (EntityType.ORGANIZATION)
✅ Sentiment model: SentimentLabel.POSITIVE (score: 0.85)
✅ Geocoding model: 37.422, -122.0841

Testing text processing...
✅ Text cleaning: '  Hello World!  This is a TEST.  ' -> 'hello world! this is a test.'
✅ Entity extraction found: ['email', 'phone', 'mention']
✅ Keyword extraction: ['python', 'great', 'programming']

Testing data validation...
✅ Email validation: True (confidence: 0.80)
✅ Phone validation: True (confidence: 0.70)
✅ URL validation: True (confidence: 0.80)
✅ Coordinates validation: True (confidence: 1.00)

============================================================
📊 Test Results: 4/4 tests passed
✅ All basic tests passed! Phase 5 implementation is working correctly.
```

### 2. Test Individual Components

```python
# Test NLP functionality
python -c "
from utils.text_processing import text_processor
result = text_processor.extract_keywords('Python is great for data science')
print('Keywords:', [kw[0] for kw in result])
"

# Test data validation
python -c "
from utils.data_validation import data_validator
result = data_validator.validate_data_type('test@example.com', 'email')
print('Email valid:', result['is_valid'])
"
```

### 3. Run Full Demonstration (Optional)

```bash
# Run comprehensive demonstration
python examples/phase5_advanced_data_processing.py
```

## 🔍 Troubleshooting

### Common Issues and Solutions

#### 1. Import Errors

**Problem**: `ModuleNotFoundError` for specific libraries
**Solution**: Install the missing optional dependencies:

```bash
# For NLP issues
pip install spacy transformers

# For computer vision issues
pip install opencv-python pillow

# For data enrichment issues
pip install geopy googlemaps
```

#### 2. spaCy Model Not Found

**Problem**: `OSError: [E050] Can't find model 'en_core_web_sm'`
**Solution**: Download the spaCy model:

```bash
python -m spacy download en_core_web_sm
```

#### 3. Tesseract Not Found

**Problem**: `TesseractNotFoundError`
**Solution**: Install Tesseract binary and set path:

```bash
# Ubuntu/Debian
sudo apt install tesseract-ocr

# Windows: Download from GitHub and set TESSERACT_PATH in .env
# macOS
brew install tesseract
```

#### 4. CUDA/GPU Issues

**Problem**: PyTorch CUDA errors
**Solution**: Install CPU-only version:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

#### 5. API Rate Limiting

**Problem**: External API rate limit exceeded
**Solution**: Configure rate limits in `.env`:

```env
GLOBAL_RATE_LIMIT=50
GEOCODING_RATE_LIMIT=25
```

## 📊 Performance Optimization

### 1. Memory Usage

For large-scale processing, consider:

```env
# Reduce model sizes
SPACY_MODEL=en_core_web_sm  # Instead of en_core_web_lg
MAX_TEXT_LENGTH=5000        # Reduce for memory efficiency
MAX_IMAGE_SIZE=1024         # Reduce for faster processing
```

### 2. Processing Speed

```env
# Enable caching
NLP_ENABLE_CACHING=true
CV_ENABLE_CACHING=true
ENRICHMENT_ENABLE_CACHING=true

# Optimize cache TTL
NLP_CACHE_TTL=7200      # 2 hours
CV_CACHE_TTL=3600       # 1 hour
```

### 3. API Usage

```env
# Optimize API calls
GEOCODING_PROVIDER=nominatim  # Free alternative to Google Maps
YAHOO_FINANCE_ENABLED=true    # Free financial data
```

## 🔐 Security Considerations

### 1. API Key Management

- Store API keys in `.env` file (not in code)
- Use environment variables in production
- Rotate API keys regularly
- Monitor API usage

### 2. Data Privacy

- Enable data encryption for sensitive information
- Configure data retention policies
- Implement audit logging
- Follow GDPR compliance guidelines

## 🚀 Integration with Existing System

### 1. Add to Existing Workflows

```python
from agents.enhanced_nlp_agent import EnhancedNLPAgent
from agents.data_enrichment_agent import DataEnrichmentAgent

# In your existing scraping workflow
async def enhanced_scraping_workflow():
    # ... existing scraping code ...
    
    # Add NLP processing
    nlp_agent = EnhancedNLPAgent()
    entities = await nlp_agent.extract_entities(scraped_text)
    
    # Add data enrichment
    enrichment_agent = DataEnrichmentAgent()
    geocoding = await enrichment_agent.geocode_address(extracted_address)
```

### 2. Update Task Types

Add new task types to your existing system:

```python
from models.task import TaskType

# New task types available:
# TaskType.NLP_ENTITY_EXTRACTION
# TaskType.NLP_SENTIMENT_ANALYSIS
# TaskType.CV_OCR_EXTRACTION
# TaskType.DATA_GEOCODING
# TaskType.DATA_VALIDATION
```

## ✅ Success Checklist

- [ ] Core dependencies installed
- [ ] Optional dependencies installed (as needed)
- [ ] Configuration file created
- [ ] Basic tests passing
- [ ] API keys configured (if using external services)
- [ ] Integration with existing system completed
- [ ] Performance optimizations applied

## 📞 Support

If you encounter issues:

1. Check the troubleshooting section above
2. Verify all dependencies are installed correctly
3. Check the logs for detailed error messages
4. Ensure API keys are correctly configured
5. Test with minimal examples first

## 🎉 Next Steps

After successful installation:

1. **Explore Examples**: Run the demonstration scripts
2. **Configure APIs**: Set up external service API keys
3. **Integrate**: Add Phase 5 capabilities to your workflows
4. **Optimize**: Tune performance settings for your use case
5. **Monitor**: Set up logging and monitoring for production use

---

**Phase 5: Advanced Data Processing** is now ready to transform your web scraping system with intelligent AI-powered capabilities!
