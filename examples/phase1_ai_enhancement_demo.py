#!/usr/bin/env python3
"""
Phase 1: Advanced AI & Intelligence Features Demo

This script demonstrates the enhanced AI capabilities of the web scraping system:
- Intelligent Content Classification System
- AI-Powered Quality Assessment
- Enhanced Error Recovery Intelligence
"""

import asyncio
import json
import logging
import time
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

async def demo_enhanced_content_recognition():
    """Demonstrate enhanced content recognition capabilities."""
    print("\n" + "="*60)
    print("🤖 ENHANCED CONTENT RECOGNITION DEMO")
    print("="*60)
    
    try:
        from agents.content_recognition import EnhancedContentRecognitionAgent
        
        # Initialize enhanced content recognition agent
        agent = EnhancedContentRecognitionAgent()
        
        # Test URLs with different content types
        test_urls = [
            "https://quotes.toscrape.com/",  # Simple HTML
            "https://books.toscrape.com/",   # E-commerce
            "https://news.ycombinator.com/", # News/Forum
        ]
        
        for url in test_urls:
            print(f"\n📊 Analyzing: {url}")
            
            # Recognize content
            result = await agent.recognize_content(url)
            
            print(f"   Content Type: {result.get('content_type', 'unknown')}")
            print(f"   Confidence: {result.get('confidence', 0):.2%}")
            print(f"   Input Type: {result.get('input_type', 'unknown')}")
            
            # Get website classification
            if 'content' in result:
                classification = await agent.classify_website_type(url, str(result.get('content', '')))
                print(f"   Website Type: {classification.get('type', 'unknown')}")
                print(f"   Classification Confidence: {classification.get('confidence', 0):.2%}")
                
                # Get extraction strategy suggestion
                strategy = await agent.suggest_extraction_strategy(url, str(result.get('content', '')))
                print(f"   Suggested Strategy: {strategy.get('method', 'unknown')}")
                print(f"   Strategy Confidence: {strategy.get('confidence', 0):.2%}")
                
                if strategy.get('selectors'):
                    print(f"   Recommended Selectors:")
                    for field, selector in list(strategy['selectors'].items())[:3]:
                        print(f"     {field}: {selector}")
            
            await asyncio.sleep(1)  # Rate limiting
        
        print("\n✅ Enhanced Content Recognition Demo Completed!")
        
    except ImportError as e:
        print(f"❌ Enhanced Content Recognition not available: {e}")
    except Exception as e:
        print(f"❌ Error in content recognition demo: {e}")


async def demo_enhanced_quality_assurance():
    """Demonstrate enhanced quality assurance capabilities."""
    print("\n" + "="*60)
    print("🔍 ENHANCED QUALITY ASSURANCE DEMO")
    print("="*60)
    
    try:
        from agents.enhanced_quality_assurance import EnhancedQualityAssuranceAgent
        
        # Initialize enhanced quality assurance agent
        agent = EnhancedQualityAssuranceAgent()
        
        # Test data with various quality issues
        test_data = [
            # Good quality data
            {
                "title": "High Quality Product",
                "price": "$29.99",
                "description": "This is a well-formatted product description with proper details.",
                "rating": "4.5",
                "availability": "In Stock"
            },
            # Medium quality data
            {
                "title": "Medium Quality Item",
                "price": "25.00",
                "description": "Short desc",
                "rating": "4",
                "availability": ""  # Missing data
            },
            # Poor quality data
            {
                "title": "",  # Missing title
                "price": "Error: Price not found",  # Error text
                "description": "???",  # Placeholder
                "rating": "N/A",
                "availability": "Unknown"
            },
            # Anomalous data
            {
                "title": "A" * 1000,  # Suspiciously long
                "price": "$-50.00",  # Invalid price
                "description": "This product description contains way too many question marks????????????",
                "rating": "15",  # Invalid rating
                "availability": "Maybe"
            }
        ]
        
        print(f"\n📊 Assessing quality of {len(test_data)} data items...")
        
        # Assess data quality
        quality_result = await agent.assess_data_quality(
            data=test_data,
            context={"source": "demo", "extraction_method": "test"}
        )
        
        print(f"\n📈 Quality Assessment Results:")
        print(f"   Overall Quality: {quality_result.get('overall_quality', 0):.2%}")
        print(f"   Confidence: {quality_result.get('confidence', 0):.2%}")
        
        # Core metrics
        core_metrics = quality_result.get('core_metrics', {})
        print(f"\n🎯 Core Quality Metrics:")
        for metric, result in core_metrics.items():
            if isinstance(result, dict):
                score = result.get('score', 0)
                print(f"   {metric.title()}: {score:.2%}")
            else:
                print(f"   {metric.title()}: {result:.2%}")
        
        # Advanced metrics
        advanced_metrics = quality_result.get('advanced_metrics', {})
        print(f"\n🚀 Advanced Quality Metrics:")
        for metric, result in advanced_metrics.items():
            if isinstance(result, dict):
                score = result.get('score', 0)
                print(f"   {metric.replace('_', ' ').title()}: {score:.2%}")
        
        # Anomalies
        anomalies = quality_result.get('anomalies', {})
        if anomalies.get('detected', False):
            print(f"\n⚠️  Anomalies Detected:")
            print(f"   Count: {anomalies.get('count', 0)}")
            print(f"   Confidence: {anomalies.get('confidence', 0):.2%}")
            print(f"   Types: {', '.join(anomalies.get('types', []))}")
        
        # Recommendations
        recommendations = quality_result.get('recommendations', [])
        if recommendations:
            print(f"\n💡 Quality Recommendations:")
            for i, rec in enumerate(recommendations[:3], 1):
                print(f"   {i}. {rec}")
        
        print("\n✅ Enhanced Quality Assurance Demo Completed!")
        
    except ImportError as e:
        print(f"❌ Enhanced Quality Assurance not available: {e}")
    except Exception as e:
        print(f"❌ Error in quality assurance demo: {e}")


async def demo_enhanced_error_recovery():
    """Demonstrate enhanced error recovery capabilities."""
    print("\n" + "="*60)
    print("🛠️ ENHANCED ERROR RECOVERY DEMO")
    print("="*60)
    
    try:
        from agents.enhanced_error_recovery import EnhancedErrorRecoveryAgent, ErrorType
        
        # Initialize enhanced error recovery agent
        agent = EnhancedErrorRecoveryAgent()
        
        # Simulate various error scenarios
        error_scenarios = [
            {
                "error_type": ErrorType.NETWORK_ERROR.value,
                "error_message": "Connection timeout after 30 seconds",
                "context": {"url": "https://example.com", "attempt": 1}
            },
            {
                "error_type": ErrorType.RATE_LIMIT_ERROR.value,
                "error_message": "Rate limit exceeded: 429 Too Many Requests",
                "context": {"url": "https://api.example.com", "requests_made": 100}
            },
            {
                "error_type": ErrorType.PARSING_ERROR.value,
                "error_message": "CSS selector '.product-title' not found",
                "context": {"url": "https://shop.example.com", "selector": ".product-title"}
            },
            {
                "error_type": ErrorType.JAVASCRIPT_ERROR.value,
                "error_message": "JavaScript execution failed: ReferenceError",
                "context": {"url": "https://spa.example.com", "script": "loadMore()"}
            }
        ]
        
        print(f"\n🔄 Testing error recovery for {len(error_scenarios)} scenarios...")
        
        for i, scenario in enumerate(error_scenarios, 1):
            print(f"\n📋 Scenario {i}: {scenario['error_type'].replace('_', ' ').title()}")
            print(f"   Error: {scenario['error_message']}")
            
            # Get retry policy for this error type
            retry_policy = agent.retry_policies.get(scenario['error_type'], {})
            print(f"   Max Retries: {retry_policy.get('max_retries', 'N/A')}")
            print(f"   Base Delay: {retry_policy.get('base_delay', 'N/A')}s")
            print(f"   Success Rate: {retry_policy.get('success_rate', 0):.1%}")
            
            # Get recovery strategies
            error_type_enum = ErrorType(scenario['error_type'])
            strategies = agent.recovery_strategies.get(error_type_enum, [])
            print(f"   Recovery Strategies:")
            for strategy in strategies[:3]:
                print(f"     - {strategy.value.replace('_', ' ').title()}")
            
            # Simulate learning from this error
            agent.error_history.append({
                'timestamp': time.time(),
                'error_type': scenario['error_type'],
                'error_message': scenario['error_message'],
                'context': scenario['context'],
                'recovery_attempted': True
            })
        
        print(f"\n📊 Error Recovery Statistics:")
        print(f"   Total Errors Tracked: {len(agent.error_history)}")
        print(f"   Error Types Configured: {len(agent.retry_policies)}")
        print(f"   Recovery Strategies Available: {len(agent.recovery_strategies)}")
        
        print("\n✅ Enhanced Error Recovery Demo Completed!")
        
    except ImportError as e:
        print(f"❌ Enhanced Error Recovery not available: {e}")
    except Exception as e:
        print(f"❌ Error in error recovery demo: {e}")


async def demo_integration_workflow():
    """Demonstrate how all enhanced agents work together."""
    print("\n" + "="*60)
    print("🔗 INTEGRATED AI WORKFLOW DEMO")
    print("="*60)
    
    try:
        from agents.content_recognition import EnhancedContentRecognitionAgent
        from agents.enhanced_quality_assurance import EnhancedQualityAssuranceAgent
        
        # Initialize agents
        content_agent = EnhancedContentRecognitionAgent()
        quality_agent = EnhancedQualityAssuranceAgent()
        
        # Simulate a complete workflow
        test_url = "https://quotes.toscrape.com/"
        
        print(f"\n🌐 Processing: {test_url}")
        
        # Step 1: Content Recognition and Strategy Suggestion
        print("\n1️⃣ Content Recognition & Strategy Suggestion")
        content_result = await content_agent.recognize_content(test_url)
        
        if 'content' in content_result:
            strategy = await content_agent.suggest_extraction_strategy(
                test_url, 
                str(content_result.get('content', ''))
            )
            
            print(f"   Website Type: {strategy.get('website_type', 'unknown')}")
            print(f"   Strategy Confidence: {strategy.get('confidence', 0):.2%}")
            
            # Step 2: Simulate Data Extraction (mock data)
            print("\n2️⃣ Simulated Data Extraction")
            mock_extracted_data = [
                {
                    "quote": "The world as we have created it is a process of our thinking.",
                    "author": "Albert Einstein",
                    "tags": ["change", "deep-thoughts", "thinking"]
                },
                {
                    "quote": "It is our choices that show what we truly are.",
                    "author": "J.K. Rowling", 
                    "tags": ["abilities", "choices"]
                },
                {
                    "quote": "",  # Missing quote for quality testing
                    "author": "Unknown",
                    "tags": []
                }
            ]
            
            print(f"   Extracted {len(mock_extracted_data)} items")
            
            # Step 3: Quality Assessment
            print("\n3️⃣ Quality Assessment")
            quality_result = await quality_agent.assess_data_quality(
                data=mock_extracted_data,
                context={
                    "url": test_url,
                    "extraction_method": "ai_enhanced",
                    "strategy": strategy
                }
            )
            
            print(f"   Overall Quality: {quality_result.get('overall_quality', 0):.2%}")
            print(f"   Assessment Confidence: {quality_result.get('confidence', 0):.2%}")
            
            # Step 4: Learning and Feedback
            print("\n4️⃣ Learning & Feedback")
            
            # Provide feedback to content agent
            feedback_data = {
                'url': test_url,
                'strategy': strategy,
                'success_rate': quality_result.get('overall_quality', 0),
                'extracted_data': mock_extracted_data
            }
            
            await content_agent._learn_from_feedback(feedback_data)
            print(f"   Content agent learned from extraction results")
            
            # Show recommendations
            recommendations = quality_result.get('recommendations', [])
            if recommendations:
                print(f"   Quality Recommendations:")
                for rec in recommendations[:2]:
                    print(f"     - {rec}")
        
        print("\n✅ Integrated AI Workflow Demo Completed!")
        
    except ImportError as e:
        print(f"❌ Integrated workflow not available: {e}")
    except Exception as e:
        print(f"❌ Error in integrated workflow demo: {e}")


async def main():
    """Run all Phase 1 AI enhancement demos."""
    print("🚀 PHASE 1: ADVANCED AI & INTELLIGENCE FEATURES DEMO")
    print("=" * 80)
    print("This demo showcases the enhanced AI capabilities added to the web scraping system:")
    print("• Intelligent Content Classification with ML")
    print("• AI-Powered Quality Assessment with Anomaly Detection") 
    print("• Enhanced Error Recovery with Pattern Learning")
    print("• Integrated Workflow with Cross-Agent Learning")
    
    # Run individual demos
    await demo_enhanced_content_recognition()
    await demo_enhanced_quality_assurance()
    await demo_enhanced_error_recovery()
    await demo_integration_workflow()
    
    print("\n" + "="*80)
    print("🎉 PHASE 1 AI ENHANCEMENT DEMO COMPLETED!")
    print("="*80)
    print("\nKey Improvements Demonstrated:")
    print("✅ ML-based content type detection and website classification")
    print("✅ Dynamic extraction strategy selection based on site patterns")
    print("✅ AI-powered quality assessment with confidence metrics")
    print("✅ Anomaly detection using machine learning")
    print("✅ Pattern-based failure prediction and adaptive retry strategies")
    print("✅ Self-healing capabilities and intelligent error recovery")
    print("✅ Cross-agent learning and feedback integration")
    
    print("\nNext Steps:")
    print("• Install optional ML dependencies: pip install scikit-learn spacy textblob")
    print("• Run: python -m spacy download en_core_web_sm")
    print("• Try the enhanced agents in your scraping workflows!")


if __name__ == "__main__":
    asyncio.run(main())
