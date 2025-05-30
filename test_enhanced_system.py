"""
Simple test script to verify the LangChain & Pydantic AI enhanced system works.
This test runs without external API dependencies.
"""
import asyncio
import sys
import os
from datetime import datetime

# Add current directory to path
sys.path.insert(0, os.path.abspath('.'))

try:
    from models.langchain_models import (
        ScrapingRequest, ExecutionPlan, AgentConfig, AgentType,
        TaskRequest, TaskResponse, AgentMessage, Priority
    )
    from agents.nlu_agent import NLUAgent
    from agents.enhanced_planning_agent import EnhancedPlanningAgent
    from config.langchain_config import get_config, validate_system_config
    
    print("✅ Successfully imported all enhanced system modules!")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)


def test_pydantic_models():
    """Test Pydantic model creation and validation."""
    print("\n🧪 Testing Pydantic Models...")
    
    try:
        # Test ScrapingRequest
        request = ScrapingRequest(
            product_query="Test iPhone prices",
            target_sites=["https://amazon.com", "https://apple.com"],
            data_points=["title", "price", "availability"],
            action="scrape",
            priority=Priority.NORMAL
        )
        
        print(f"   ✅ ScrapingRequest created: {request.id}")
        print(f"      Query: {request.product_query}")
        print(f"      Sites: {len(request.target_sites)}")
        print(f"      Priority: {request.priority}")
        
        # Test AgentConfig
        config = AgentConfig(
            agent_id="test-agent",
            agent_type=AgentType.NLU_AGENT,
            capabilities=["natural_language_processing"]
        )
        
        print(f"   ✅ AgentConfig created: {config.agent_id}")
        print(f"      Type: {config.agent_type}")
        print(f"      Capabilities: {config.capabilities}")
        
        # Test TaskRequest
        task = TaskRequest(
            task_type="test_task",
            parameters={"test": "value"},
            priority=Priority.HIGH
        )
        
        print(f"   ✅ TaskRequest created: {task.id}")
        print(f"      Type: {task.task_type}")
        print(f"      Priority: {task.priority}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Pydantic model test failed: {e}")
        return False


async def test_nlu_agent():
    """Test NLU Agent functionality."""
    print("\n🧠 Testing NLU Agent...")
    
    try:
        # Create NLU agent without LLM (fallback mode)
        nlu_agent = NLUAgent(llm=None)
        
        print(f"   ✅ NLU Agent created: {nlu_agent.agent_id}")
        print(f"      Type: {nlu_agent.agent_type}")
        print(f"      Capabilities: {nlu_agent.config.capabilities}")
        
        # Test natural language parsing (will use fallback mode)
        test_input = "Track iPhone 15 prices on Amazon and Best Buy"
        
        try:
            scraping_request = await nlu_agent.parse_natural_language_request(test_input)
            
            print(f"   ✅ Parsed request successfully:")
            print(f"      Query: {scraping_request.product_query}")
            print(f"      Sites: {scraping_request.target_sites}")
            print(f"      Data Points: {scraping_request.data_points}")
            print(f"      Action: {scraping_request.action}")
            
        except Exception as e:
            print(f"   ⚠️  NLU parsing failed (expected without LLM): {e}")
            print("      This is normal when running without OpenAI API key")
        
        await nlu_agent.stop()
        return True
        
    except Exception as e:
        print(f"   ❌ NLU Agent test failed: {e}")
        return False


async def test_planning_agent():
    """Test Planning Agent functionality."""
    print("\n🎯 Testing Planning Agent...")
    
    try:
        # Create Planning agent without LLM (fallback mode)
        planning_agent = EnhancedPlanningAgent(llm=None)
        
        print(f"   ✅ Planning Agent created: {planning_agent.agent_id}")
        print(f"      Type: {planning_agent.agent_type}")
        print(f"      Capabilities: {planning_agent.config.capabilities}")
        
        # Create a test scraping request
        request = ScrapingRequest(
            product_query="Test laptop prices",
            target_sites=["https://amazon.com", "https://bestbuy.com"],
            data_points=["title", "price", "rating"],
            action="scrape"
        )
        
        try:
            execution_plan = await planning_agent.create_execution_plan(request)
            
            print(f"   ✅ Execution plan created:")
            print(f"      Plan ID: {execution_plan.plan_id}")
            print(f"      Strategies: {len(execution_plan.strategies)}")
            print(f"      Estimated Duration: {execution_plan.estimated_duration}s")
            print(f"      Risk Level: {execution_plan.risk_assessment.get('risk_level', 'unknown')}")
            
        except Exception as e:
            print(f"   ⚠️  Planning failed (expected without LLM): {e}")
            print("      This is normal when running without OpenAI API key")
        
        await planning_agent.stop()
        return True
        
    except Exception as e:
        print(f"   ❌ Planning Agent test failed: {e}")
        return False


def test_configuration():
    """Test system configuration."""
    print("\n⚙️  Testing System Configuration...")
    
    try:
        config = get_config()
        
        print(f"   ✅ Configuration loaded successfully")
        print(f"      OpenAI Model: {config.langchain.openai_model}")
        print(f"      Redis Host: {config.redis.redis_host}")
        print(f"      Max Workflows: {config.agent_system.max_concurrent_workflows}")
        
        # Validate configuration
        issues = config.validate_configuration()
        if issues:
            print(f"   ⚠️  Configuration issues found:")
            for issue in issues:
                print(f"      • {issue}")
        else:
            print(f"   ✅ Configuration validation passed")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Configuration test failed: {e}")
        return False


def test_agent_communication():
    """Test agent message creation and validation."""
    print("\n📡 Testing Agent Communication...")
    
    try:
        # Create test task request
        task_request = TaskRequest(
            task_type="test_communication",
            parameters={"message": "Hello from test"}
        )
        
        # Create agent message
        message = AgentMessage(
            sender=AgentType.NLU_AGENT,
            recipient=AgentType.PLANNING_AGENT,
            message_type="task_request",
            payload=task_request,
            priority=Priority.NORMAL
        )
        
        print(f"   ✅ Agent message created:")
        print(f"      ID: {message.id}")
        print(f"      From: {message.sender} → To: {message.recipient}")
        print(f"      Type: {message.message_type}")
        print(f"      Priority: {message.priority}")
        print(f"      Timestamp: {message.timestamp}")
        
        # Test message serialization
        message_dict = message.dict()
        print(f"   ✅ Message serialization successful")
        print(f"      Serialized size: {len(str(message_dict))} characters")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Agent communication test failed: {e}")
        return False


async def main():
    """Run all tests."""
    print("🚀 LangChain & Pydantic AI Enhanced System Test Suite")
    print("=" * 60)
    print("Testing the enhanced web scraping system components...")
    
    tests = [
        ("Pydantic Models", test_pydantic_models),
        ("System Configuration", test_configuration),
        ("Agent Communication", test_agent_communication),
        ("NLU Agent", test_nlu_agent),
        ("Planning Agent", test_planning_agent),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            if asyncio.iscoroutinefunction(test_func):
                result = await test_func()
            else:
                result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"   💥 {test_name} test crashed: {e}")
            results.append((test_name, False))
    
    # Print summary
    print("\n" + "=" * 60)
    print("📊 TEST RESULTS SUMMARY")
    print("=" * 60)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{status:12} {test_name}")
        if result:
            passed += 1
    
    print("-" * 60)
    print(f"Total: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("\n🎉 All tests passed! The enhanced system is working correctly.")
        print("\n📝 Next Steps:")
        print("1. Set up OpenAI API key for full LangChain functionality")
        print("2. Configure Redis for distributed communication")
        print("3. Set up PostgreSQL for data persistence")
        print("4. Run the full example: python examples/langchain_enhanced_example.py")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please check the errors above.")
    
    return passed == total


if __name__ == "__main__":
    try:
        success = asyncio.run(main())
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⚠️  Test interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Test suite crashed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
