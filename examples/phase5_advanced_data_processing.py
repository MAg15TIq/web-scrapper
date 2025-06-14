"""
Example script demonstrating Phase 5: Advanced Data Processing capabilities.

This script showcases the enhanced NLP, computer vision, and data enrichment features.
"""
import asyncio
import logging
import sys
import os
import json
import base64
from typing import Dict, Any, List

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agents.enhanced_nlp_agent import EnhancedNLPAgent
from agents.enhanced_image_processing import EnhancedImageProcessingAgent
from agents.data_enrichment_agent import DataEnrichmentAgent
from agents.geocoding_agent import GeocodingAgent
from models.data_processing import ProcessingType, EntityType, SentimentLabel
from utils.text_processing import text_processor
from utils.image_analysis import image_analyzer
from utils.data_validation import data_validator


async def demonstrate_nlp_capabilities():
    """Demonstrate enhanced NLP processing capabilities."""
    print("\n" + "="*60)
    print("🧠 ENHANCED NLP PROCESSING DEMONSTRATION")
    print("="*60)
    
    # Initialize NLP agent
    nlp_agent = EnhancedNLPAgent()
    
    # Sample text for processing
    sample_text = """
    Apple Inc. is an American multinational technology company headquartered in Cupertino, California.
    The company was founded by Steve Jobs, Steve Wozniak, and Ronald Wayne in April 1976.
    Apple is known for its innovative products including the iPhone, iPad, Mac computers, and Apple Watch.
    The company's stock price has performed exceptionally well over the past decade, making it one of the
    most valuable companies in the world. Many customers love Apple products for their design and user experience,
    though some critics argue they are overpriced. Apple's headquarters, Apple Park, opened in 2017 and
    can accommodate over 12,000 employees. You can visit their website at https://www.apple.com or
    contact them at info@apple.com. Their customer service number is +1-800-275-2273.
    """
    
    print(f"📝 Sample Text (first 200 chars): {sample_text[:200]}...")
    
    # 1. Entity Extraction
    print("\n1️⃣ Entity Extraction:")
    entities_result = await nlp_agent.extract_entities(sample_text)
    if entities_result:
        print(f"   Found {entities_result.get('total_entities', 0)} entities:")
        for entity_type, entities in entities_result.get('grouped_entities', {}).items():
            print(f"   - {entity_type}: {[e['text'] for e in entities[:3]]}")  # Show first 3
    
    # 2. Sentiment Analysis
    print("\n2️⃣ Sentiment Analysis:")
    sentiment_result = await nlp_agent.analyze_sentiment(sample_text, include_emotions=True)
    if sentiment_result:
        sentiment = sentiment_result.get('sentiment', {})
        print(f"   Sentiment: {sentiment.get('label', 'Unknown')} (confidence: {sentiment.get('score', 0):.2f})")
        emotions = sentiment_result.get('emotions', {})
        if emotions:
            top_emotions = sorted(emotions.items(), key=lambda x: x[1], reverse=True)[:3]
            print(f"   Top emotions: {', '.join([f'{emotion}: {score:.2f}' for emotion, score in top_emotions])}")
    
    # 3. Text Summarization
    print("\n3️⃣ Text Summarization:")
    summary_result = await nlp_agent.summarize_text(sample_text, max_length=100)
    if summary_result:
        summary = summary_result.get('summary', {})
        print(f"   Summary: {summary.get('summary', 'N/A')}")
        print(f"   Compression ratio: {summary.get('compression_ratio', 0):.2f}")
    
    # 4. Language Detection
    print("\n4️⃣ Language Detection:")
    language_result = await nlp_agent.detect_language(sample_text)
    if language_result:
        lang_detection = language_result.get('language_detection', {})
        print(f"   Detected language: {lang_detection.get('language', 'Unknown')} (confidence: {lang_detection.get('confidence', 0):.2f})")
    
    # 5. Advanced Text Processing
    print("\n5️⃣ Advanced Text Processing:")
    
    # Extract entities using utility
    extracted_entities = text_processor.extract_entities(sample_text)
    print(f"   Extracted entities: {list(extracted_entities.keys())}")
    
    # Extract keywords
    keywords = text_processor.extract_keywords(sample_text, max_keywords=5)
    print(f"   Top keywords: {[kw[0] for kw in keywords]}")
    
    # Calculate readability
    readability = text_processor.calculate_readability(sample_text)
    print(f"   Readability score: {readability.get('flesch_reading_ease', 0):.1f}")
    print(f"   Grade level: {readability.get('flesch_kincaid_grade', 0):.1f}")


async def demonstrate_computer_vision_capabilities():
    """Demonstrate enhanced computer vision capabilities."""
    print("\n" + "="*60)
    print("👁️ ENHANCED COMPUTER VISION DEMONSTRATION")
    print("="*60)
    
    # Initialize image processing agent
    image_agent = EnhancedImageProcessingAgent()
    
    # Create a sample image (placeholder - in real use, you'd have actual image data)
    print("📸 Creating sample image for demonstration...")
    
    # For demonstration, we'll create a simple test image
    import numpy as np
    try:
        import cv2
        
        # Create a sample image with some text and shapes
        img = np.ones((400, 600, 3), dtype=np.uint8) * 255  # White background
        
        # Add some colored rectangles (simulating buttons)
        cv2.rectangle(img, (50, 50), (200, 100), (100, 150, 200), -1)  # Button 1
        cv2.rectangle(img, (250, 50), (400, 100), (150, 200, 100), -1)  # Button 2
        
        # Add some text
        cv2.putText(img, "Sample Button 1", (60, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(img, "Sample Button 2", (260, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(img, "This is sample text for OCR", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        
        # Add input field simulation
        cv2.rectangle(img, (50, 250), (400, 290), (240, 240, 240), -1)
        cv2.rectangle(img, (50, 250), (400, 290), (100, 100, 100), 2)
        cv2.putText(img, "Enter your email here", (60, 275), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
        
        # Encode image to base64
        _, buffer = cv2.imencode('.jpg', img)
        image_data = base64.b64encode(buffer).decode('utf-8')
        
        print("✅ Sample image created successfully")
        
        # 1. Visual Element Detection
        print("\n1️⃣ Visual Element Detection:")
        visual_elements = image_analyzer.detect_visual_elements(img, ["button", "input_field", "text"])
        for element_type, elements in visual_elements.items():
            print(f"   - {element_type.title()}s found: {len(elements)}")
            for i, element in enumerate(elements[:2]):  # Show first 2
                coords = element.get('coordinates', {})
                print(f"     Element {i+1}: ({coords.get('x', 0)}, {coords.get('y', 0)}) "
                      f"{coords.get('width', 0)}x{coords.get('height', 0)}")
        
        # 2. Image Enhancement
        print("\n2️⃣ Image Enhancement:")
        enhanced_img = image_analyzer.enhance_image(img, "auto")
        print("   ✅ Applied automatic image enhancement")
        
        # 3. Image Feature Extraction
        print("\n3️⃣ Image Feature Extraction:")
        features = image_analyzer.extract_image_features(img)
        print(f"   Dimensions: {features.get('dimensions', {})}")
        print(f"   Channels: {features.get('channels', 0)}")
        
        color_analysis = features.get('color_analysis', {})
        if color_analysis:
            avg_color = color_analysis.get('average_color', [])
            print(f"   Average color (BGR): {[int(c) for c in avg_color]}")
        
        # 4. Screenshot Comparison (using same image for demo)
        print("\n4️⃣ Screenshot Comparison:")
        # Create a slightly modified version
        img2 = img.copy()
        cv2.rectangle(img2, (450, 50), (550, 100), (200, 100, 150), -1)  # Add new button
        
        comparison = image_analyzer.compare_screenshots(img, img2)
        print(f"   Similarity score: {comparison.get('similarity_score', 0):.3f}")
        print(f"   Changes detected: {comparison.get('total_changes', 0)}")
        print(f"   Change percentage: {comparison.get('change_percentage', 0):.2f}%")
        
    except ImportError:
        print("⚠️ OpenCV not available - skipping computer vision demonstration")
    except Exception as e:
        print(f"❌ Error in computer vision demonstration: {e}")


async def demonstrate_data_enrichment_capabilities():
    """Demonstrate data enrichment capabilities."""
    print("\n" + "="*60)
    print("🔍 DATA ENRICHMENT DEMONSTRATION")
    print("="*60)
    
    # Initialize data enrichment agent
    enrichment_agent = DataEnrichmentAgent()
    
    # Sample data for enrichment
    sample_data = {
        "company": "Apple Inc.",
        "email": "john.doe@apple.com",
        "phone": "+1-800-275-2273",
        "address": "1 Apple Park Way, Cupertino, CA 95014",
        "website": "https://www.apple.com",
        "coordinates": "37.3349, -122.0090"
    }
    
    print("📊 Sample data for enrichment:")
    for key, value in sample_data.items():
        print(f"   {key}: {value}")
    
    # 1. Data Validation
    print("\n1️⃣ Data Validation:")
    
    # Validate email
    email_validation = data_validator.validate_data_type(sample_data["email"], "email")
    print(f"   Email validation: {'✅ Valid' if email_validation['is_valid'] else '❌ Invalid'} "
          f"(confidence: {email_validation.get('confidence', 0):.2f})")
    
    # Validate phone
    phone_validation = data_validator.validate_data_type(sample_data["phone"], "phone")
    print(f"   Phone validation: {'✅ Valid' if phone_validation['is_valid'] else '❌ Invalid'} "
          f"(confidence: {phone_validation.get('confidence', 0):.2f})")
    
    # Validate URL
    url_validation = data_validator.validate_data_type(sample_data["website"], "url")
    print(f"   URL validation: {'✅ Valid' if url_validation['is_valid'] else '❌ Invalid'} "
          f"(confidence: {url_validation.get('confidence', 0):.2f})")
    
    # Validate coordinates
    coords_validation = data_validator.validate_data_type(sample_data["coordinates"], "coordinates")
    print(f"   Coordinates validation: {'✅ Valid' if coords_validation['is_valid'] else '❌ Invalid'} "
          f"(confidence: {coords_validation.get('confidence', 0):.2f})")
    
    # 2. Geocoding
    print("\n2️⃣ Geocoding:")
    try:
        geocoding_result = await enrichment_agent.geocode_address(sample_data["address"])
        if geocoding_result:
            geocoding = geocoding_result.get('geocoding', {})
            print(f"   Address: {geocoding.get('formatted_address', 'N/A')}")
            print(f"   Coordinates: {geocoding.get('latitude', 0):.4f}, {geocoding.get('longitude', 0):.4f}")
            print(f"   Country: {geocoding.get('country', 'N/A')}")
            print(f"   City: {geocoding.get('city', 'N/A')}")
    except Exception as e:
        print(f"   ⚠️ Geocoding not available: {e}")
    
    # 3. Data Normalization
    print("\n3️⃣ Data Normalization:")
    
    # Normalize text
    normalized_company = text_processor.normalize_text(sample_data["company"], "standard")
    print(f"   Company name normalized: {normalized_company}")
    
    # Clean and normalize email
    normalized_email = text_processor.clean_text(sample_data["email"], {"lowercase": True})
    print(f"   Email normalized: {normalized_email}")
    
    # 4. Entity Extraction from Address
    print("\n4️⃣ Entity Extraction from Address:")
    address_entities = text_processor.extract_entities(sample_data["address"])
    for entity_type, entities in address_entities.items():
        if entities:
            print(f"   {entity_type}: {entities}")


async def demonstrate_batch_processing():
    """Demonstrate batch processing capabilities."""
    print("\n" + "="*60)
    print("⚡ BATCH PROCESSING DEMONSTRATION")
    print("="*60)
    
    # Sample batch data
    batch_texts = [
        "Apple Inc. is a great technology company with innovative products.",
        "I'm not happy with the customer service. The support was terrible.",
        "The new iPhone features are amazing! Best phone ever made.",
        "Microsoft Corporation develops software and cloud services.",
        "Google's search engine is the most popular in the world."
    ]
    
    batch_emails = [
        "john.doe@apple.com",
        "invalid-email",
        "jane.smith@microsoft.com",
        "test@google.com",
        "bad@email@format.com"
    ]
    
    print(f"📦 Processing {len(batch_texts)} texts and {len(batch_emails)} emails...")
    
    # 1. Batch Sentiment Analysis
    print("\n1️⃣ Batch Sentiment Analysis:")
    nlp_agent = EnhancedNLPAgent()
    
    sentiment_results = []
    for i, text in enumerate(batch_texts):
        try:
            result = await nlp_agent.analyze_sentiment(text)
            sentiment = result.get('sentiment', {})
            sentiment_results.append({
                "text_id": i + 1,
                "sentiment": sentiment.get('label', 'Unknown'),
                "confidence": sentiment.get('score', 0)
            })
        except Exception as e:
            print(f"   Error processing text {i+1}: {e}")
    
    for result in sentiment_results:
        print(f"   Text {result['text_id']}: {result['sentiment']} (confidence: {result['confidence']:.2f})")
    
    # 2. Batch Email Validation
    print("\n2️⃣ Batch Email Validation:")
    
    email_results = []
    for i, email in enumerate(batch_emails):
        validation = data_validator.validate_data_type(email, "email")
        email_results.append({
            "email_id": i + 1,
            "email": email,
            "is_valid": validation['is_valid'],
            "confidence": validation.get('confidence', 0)
        })
    
    for result in email_results:
        status = "✅ Valid" if result['is_valid'] else "❌ Invalid"
        print(f"   Email {result['email_id']}: {status} - {result['email']} "
              f"(confidence: {result['confidence']:.2f})")
    
    # 3. Batch Statistics
    print("\n3️⃣ Batch Processing Statistics:")
    
    # Sentiment statistics
    positive_count = sum(1 for r in sentiment_results if r['sentiment'] == 'POSITIVE')
    negative_count = sum(1 for r in sentiment_results if r['sentiment'] == 'NEGATIVE')
    neutral_count = len(sentiment_results) - positive_count - negative_count
    
    print(f"   Sentiment distribution:")
    print(f"   - Positive: {positive_count}/{len(sentiment_results)} ({positive_count/len(sentiment_results)*100:.1f}%)")
    print(f"   - Negative: {negative_count}/{len(sentiment_results)} ({negative_count/len(sentiment_results)*100:.1f}%)")
    print(f"   - Neutral: {neutral_count}/{len(sentiment_results)} ({neutral_count/len(sentiment_results)*100:.1f}%)")
    
    # Email validation statistics
    valid_emails = sum(1 for r in email_results if r['is_valid'])
    print(f"   Email validation:")
    print(f"   - Valid emails: {valid_emails}/{len(email_results)} ({valid_emails/len(email_results)*100:.1f}%)")


async def main():
    """Main demonstration function."""
    print("🚀 PHASE 5: ADVANCED DATA PROCESSING DEMONSTRATION")
    print("This script demonstrates the enhanced capabilities of Phase 5 implementation.")
    
    try:
        # Set up logging
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        
        # Run demonstrations
        await demonstrate_nlp_capabilities()
        await demonstrate_computer_vision_capabilities()
        await demonstrate_data_enrichment_capabilities()
        await demonstrate_batch_processing()
        
        print("\n" + "="*60)
        print("✅ PHASE 5 DEMONSTRATION COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("\n📋 Summary of demonstrated capabilities:")
        print("   🧠 Enhanced NLP: Entity extraction, sentiment analysis, summarization, language detection")
        print("   👁️ Computer Vision: Visual element detection, image enhancement, feature extraction")
        print("   🔍 Data Enrichment: Geocoding, validation, normalization, external API integration")
        print("   ⚡ Batch Processing: Efficient processing of multiple items")
        print("\n🎯 These capabilities can be integrated into your web scraping workflows")
        print("   to provide intelligent data processing and enrichment.")
        
    except Exception as e:
        print(f"\n❌ Error during demonstration: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
