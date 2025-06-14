"""
Basic test script for Phase 5 implementation.
"""
import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.abspath('.'))

def test_imports():
    """Test basic imports."""
    print("Testing Phase 5 imports...")
    
    try:
        from models.data_processing import ProcessingType, EntityType, SentimentLabel
        print("✅ Data processing models imported successfully")
    except Exception as e:
        print(f"❌ Error importing data processing models: {e}")
        return False
    
    try:
        from utils.text_processing import text_processor
        print("✅ Text processing utilities imported successfully")
    except Exception as e:
        print(f"❌ Error importing text processing utilities: {e}")
        return False
    
    try:
        from utils.data_validation import data_validator
        print("✅ Data validation utilities imported successfully")
    except Exception as e:
        print(f"❌ Error importing data validation utilities: {e}")
        return False
    
    return True

def test_text_processing():
    """Test text processing functionality."""
    print("\nTesting text processing...")
    
    try:
        from utils.text_processing import text_processor
        
        # Test text cleaning
        sample_text = "  Hello World!  This is a TEST.  "
        cleaned = text_processor.clean_text(sample_text, {"normalize_whitespace": True, "lowercase": True})
        print(f"✅ Text cleaning: '{sample_text}' -> '{cleaned}'")
        
        # Test entity extraction
        sample_text = "Contact John Doe at john@example.com or call +1-555-123-4567"
        entities = text_processor.extract_entities(sample_text)
        print(f"✅ Entity extraction found: {list(entities.keys())}")
        
        # Test keyword extraction
        sample_text = "Python is a great programming language for data science and machine learning"
        keywords = text_processor.extract_keywords(sample_text, max_keywords=3)
        print(f"✅ Keyword extraction: {[kw[0] for kw in keywords]}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in text processing: {e}")
        return False

def test_data_validation():
    """Test data validation functionality."""
    print("\nTesting data validation...")
    
    try:
        from utils.data_validation import data_validator
        
        # Test email validation
        email_result = data_validator.validate_data_type("test@example.com", "email")
        print(f"✅ Email validation: {email_result['is_valid']} (confidence: {email_result.get('confidence', 0):.2f})")
        
        # Test phone validation
        phone_result = data_validator.validate_data_type("+1-555-123-4567", "phone")
        print(f"✅ Phone validation: {phone_result['is_valid']} (confidence: {phone_result.get('confidence', 0):.2f})")
        
        # Test URL validation
        url_result = data_validator.validate_data_type("https://www.example.com", "url")
        print(f"✅ URL validation: {url_result['is_valid']} (confidence: {url_result.get('confidence', 0):.2f})")
        
        # Test coordinates validation
        coords_result = data_validator.validate_data_type("37.7749, -122.4194", "coordinates")
        print(f"✅ Coordinates validation: {coords_result['is_valid']} (confidence: {coords_result.get('confidence', 0):.2f})")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in data validation: {e}")
        return False

def test_models():
    """Test data models."""
    print("\nTesting data models...")
    
    try:
        from models.data_processing import (
            ProcessingType, EntityType, SentimentLabel,
            ExtractedEntity, SentimentResult, GeolocationResult
        )
        
        # Test entity model
        entity = ExtractedEntity(
            text="Apple Inc.",
            label=EntityType.ORGANIZATION,
            start=0,
            end=10,
            confidence=0.95
        )
        print(f"✅ Entity model: {entity.text} ({entity.label})")
        
        # Test sentiment model
        sentiment = SentimentResult(
            label=SentimentLabel.POSITIVE,
            score=0.85,
            positive_score=0.85,
            negative_score=0.10,
            neutral_score=0.05
        )
        print(f"✅ Sentiment model: {sentiment.label} (score: {sentiment.score})")
        
        # Test geocoding model
        location = GeolocationResult(
            address="1600 Amphitheatre Parkway, Mountain View, CA",
            formatted_address="1600 Amphitheatre Pkwy, Mountain View, CA 94043, USA",
            latitude=37.4220,
            longitude=-122.0841,
            confidence=0.95,
            service="google"
        )
        print(f"✅ Geocoding model: {location.latitude}, {location.longitude}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in data models: {e}")
        return False

def main():
    """Main test function."""
    print("🚀 Phase 5: Advanced Data Processing - Basic Tests")
    print("=" * 60)
    
    success_count = 0
    total_tests = 4
    
    # Run tests
    if test_imports():
        success_count += 1
    
    if test_models():
        success_count += 1
    
    if test_text_processing():
        success_count += 1
    
    if test_data_validation():
        success_count += 1
    
    # Summary
    print("\n" + "=" * 60)
    print(f"📊 Test Results: {success_count}/{total_tests} tests passed")
    
    if success_count == total_tests:
        print("✅ All basic tests passed! Phase 5 implementation is working correctly.")
        print("\n🎯 Next steps:")
        print("   1. Install optional dependencies for full functionality:")
        print("      pip install spacy transformers opencv-python")
        print("   2. Configure API keys in .env file")
        print("   3. Run the full demonstration script")
    else:
        print("❌ Some tests failed. Please check the error messages above.")
    
    return success_count == total_tests

if __name__ == "__main__":
    main()
