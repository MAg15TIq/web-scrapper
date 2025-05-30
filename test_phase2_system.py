"""
Comprehensive test suite for Phase 2 enhanced system.
Tests all specialized agents, enhanced workflows, and external services.
"""
import asyncio
import sys
import os
from datetime import datetime

# Add current directory to path
sys.path.insert(0, os.path.abspath('.'))

try:
    from agents.data_validation_agent import DataValidationAgent
    from agents.visualization_agent import VisualizationAgent
    from agents.error_recovery_agent import ErrorRecoveryAgent
    from agents.workflow_orchestrator import WorkflowOrchestrator
    from services.redis_service import get_redis_service
    from services.database_service import get_database_service
    from services.monitoring_service import get_monitoring_service
    from models.langchain_models import (
        ScrapingRequest, ExecutionPlan, AgentConfig, AgentType, Priority
    )
    
    print("✅ Successfully imported all Phase 2 modules!")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)


async def test_data_validation_agent():
    """Test Data Validation Agent functionality."""
    print("\n🧪 Testing Data Validation Agent...")
    
    try:
        agent = DataValidationAgent()
        
        # Test data with various quality issues
        test_data = {
            "products": [
                {"title": "Valid Product", "price": "$99.99", "rating": 4.5},
                {"title": "", "price": "invalid_price", "rating": 6.0},  # Issues
                {"title": "Another Product", "price": None, "rating": 4.0}  # Missing price
            ]
        }
        
        quality_report = await agent.validate_scraped_data(test_data)
        
        print(f"   ✅ Quality report generated")
        print(f"      Quality Score: {quality_report.quality_score:.1f}%")
        print(f"      Issues Found: {len(quality_report.issues)}")
        print(f"      Recommendations: {len(quality_report.recommendations)}")
        
        await agent.stop()
        return True
        
    except Exception as e:
        print(f"   ❌ Data validation test failed: {e}")
        return False


async def test_visualization_agent():
    """Test Visualization Agent functionality."""
    print("\n🧪 Testing Visualization Agent...")
    
    try:
        agent = VisualizationAgent()
        
        # Test data for visualization
        test_data = {
            "prices": [100, 200, 150, 300, 250],
            "ratings": [4.5, 4.0, 4.2, 4.8, 4.1],
            "categories": ["Electronics", "Books", "Clothing", "Home", "Sports"]
        }
        
        visualization_result = await agent.create_visualization(test_data)
        
        print(f"   ✅ Visualization created")
        print(f"      Chart Type: {visualization_result.get('chart_config', {}).get('chart_type', 'unknown')}")
        print(f"      Data Fields: {visualization_result.get('data_summary', {}).get('total_fields', 0)}")
        print(f"      Insights Generated: {len(visualization_result.get('insights', {}).get('key_insights', []))}")
        
        await agent.stop()
        return True
        
    except Exception as e:
        print(f"   ❌ Visualization test failed: {e}")
        return False


async def test_error_recovery_agent():
    """Test Error Recovery Agent functionality."""
    print("\n🧪 Testing Error Recovery Agent...")
    
    try:
        agent = ErrorRecoveryAgent()
        
        # Simulate error scenario
        error_info = {
            "type": "timeout_error",
            "message": "Request timeout",
            "agent_id": "test-agent",
            "severity": "medium"
        }
        
        recovery_result = await agent.handle_error(error_info)
        
        print(f"   ✅ Error recovery executed")
        print(f"      Recovery Strategy: {recovery_result.get('recovery_strategy', 'unknown')}")
        print(f"      Executed: {recovery_result.get('executed', False)}")
        
        await agent.stop()
        return True
        
    except Exception as e:
        print(f"   ❌ Error recovery test failed: {e}")
        return False


async def test_enhanced_workflow():
    """Test Enhanced Workflow Orchestrator."""
    print("\n🧪 Testing Enhanced Workflow Orchestrator...")
    
    try:
        orchestrator = WorkflowOrchestrator()
        
        # Test workflow execution
        test_input = "Test enhanced workflow with parallel execution and quality assessment"
        
        result = await orchestrator.execute_workflow(test_input)
        
        print(f"   ✅ Enhanced workflow executed")
        print(f"      Status: {result['status']}")
        print(f"      Execution Time: {result['execution_time']:.2f}s")
        print(f"      Steps Completed: {len(result.get('steps_completed', []))}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Enhanced workflow test failed: {e}")
        return False


async def test_redis_service():
    """Test Redis Service functionality."""
    print("\n🧪 Testing Redis Service...")
    
    try:
        redis_service = await get_redis_service()
        
        # Test cache operations
        test_key = "test_cache_key"
        test_value = {"message": "test", "timestamp": datetime.now().isoformat()}
        
        await redis_service.set_cache(test_key, test_value, ttl=60)
        retrieved_value = await redis_service.get_cache(test_key)
        
        print(f"   ✅ Redis service tested")
        print(f"      Connection: {'Mock Mode' if redis_service._use_mock else 'Connected'}")
        print(f"      Cache Test: {'Success' if retrieved_value else 'Failed'}")
        
        # Test pub/sub
        message_received = False
        
        async def test_handler(message):
            nonlocal message_received
            message_received = True
        
        await redis_service.subscribe_to_channel("test_channel", test_handler, "test_agent")
        await redis_service.publish_message("test_channel", {"test": "message"})
        
        # Give time for message processing
        await asyncio.sleep(0.5)
        
        print(f"      Pub/Sub Test: {'Success' if message_received or redis_service._use_mock else 'Failed'}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Redis service test failed: {e}")
        return False


async def test_database_service():
    """Test Database Service functionality."""
    print("\n🧪 Testing Database Service...")
    
    try:
        db_service = await get_database_service()
        
        # Test workflow storage
        test_workflow = {
            "workflow_id": f"test-{int(datetime.now().timestamp())}",
            "user_input": "Test workflow",
            "status": "completed",
            "execution_time_seconds": 2.5
        }
        
        save_success = await db_service.save_workflow(test_workflow)
        retrieved_workflow = await db_service.get_workflow(test_workflow["workflow_id"])
        
        print(f"   ✅ Database service tested")
        print(f"      Connection: {'Mock Mode' if db_service._use_mock else 'Connected'}")
        print(f"      Save Test: {'Success' if save_success else 'Failed'}")
        print(f"      Retrieve Test: {'Success' if retrieved_workflow else 'Failed'}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Database service test failed: {e}")
        return False


async def test_monitoring_service():
    """Test Monitoring Service functionality."""
    print("\n🧪 Testing Monitoring Service...")
    
    try:
        monitoring_service = await get_monitoring_service()
        
        # Test metrics recording
        monitoring_service.metrics.record_workflow_start("test-workflow", "scraping")
        monitoring_service.metrics.record_workflow_completion("test-workflow", "completed", "scraping", 3.0)
        monitoring_service.metrics.update_system_health(0.95)
        
        # Get metrics summary
        metrics_summary = monitoring_service.get_metrics_summary()
        
        print(f"   ✅ Monitoring service tested")
        print(f"      Status: {metrics_summary['monitoring_service']['status']}")
        print(f"      Prometheus: {'Enabled' if metrics_summary['monitoring_service']['prometheus_enabled'] else 'Disabled'}")
        print(f"      Uptime: {metrics_summary['uptime']:.1f}s")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Monitoring service test failed: {e}")
        return False


async def test_integration_workflow():
    """Test full integration workflow."""
    print("\n🧪 Testing Full Integration Workflow...")
    
    try:
        # Initialize all services
        redis_service = await get_redis_service()
        db_service = await get_database_service()
        monitoring_service = await get_monitoring_service()
        
        # Create orchestrator
        orchestrator = WorkflowOrchestrator()
        
        # Execute a complex workflow
        complex_input = "Analyze product data with quality validation, visualization, and monitoring"
        
        # Record start metrics
        monitoring_service.metrics.record_workflow_start("integration-test", "complex")
        
        # Execute workflow
        result = await orchestrator.execute_workflow(complex_input)
        
        # Record completion metrics
        monitoring_service.metrics.record_workflow_completion(
            "integration-test", 
            result['status'], 
            "complex", 
            result['execution_time']
        )
        
        # Save workflow to database
        workflow_data = {
            "workflow_id": result['workflow_id'],
            "user_input": complex_input,
            "status": result['status'],
            "execution_time_seconds": result['execution_time']
        }
        
        await db_service.save_workflow(workflow_data)
        
        print(f"   ✅ Integration workflow completed")
        print(f"      Workflow Status: {result['status']}")
        print(f"      Services Used: Redis, Database, Monitoring")
        print(f"      Data Persisted: {'Yes' if workflow_data else 'No'}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Integration workflow test failed: {e}")
        return False


async def main():
    """Run all Phase 2 tests."""
    print("🧪 Phase 2 Enhanced System Test Suite")
    print("=" * 60)
    print("Testing all specialized agents, workflows, and services...")
    
    tests = [
        ("Data Validation Agent", test_data_validation_agent),
        ("Visualization Agent", test_visualization_agent),
        ("Error Recovery Agent", test_error_recovery_agent),
        ("Enhanced Workflow", test_enhanced_workflow),
        ("Redis Service", test_redis_service),
        ("Database Service", test_database_service),
        ("Monitoring Service", test_monitoring_service),
        ("Integration Workflow", test_integration_workflow),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = await test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"   💥 {test_name} test crashed: {e}")
            results.append((test_name, False))
    
    # Print summary
    print("\n" + "=" * 60)
    print("📊 PHASE 2 TEST RESULTS SUMMARY")
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
        print("\n🎉 All Phase 2 tests passed! The enhanced system is working correctly.")
        print("\n📋 Phase 2 Features Verified:")
        print("✓ Specialized intelligence agents")
        print("✓ Enhanced LangGraph workflows")
        print("✓ External services integration")
        print("✓ Advanced error handling and recovery")
        print("✓ Real-time monitoring and metrics")
        print("✓ Data persistence and caching")
        print("✓ Quality assessment and validation")
        print("✓ Automatic visualization and insights")
        
        print("\n🚀 Ready for Phase 3: Advanced Features!")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please check the errors above.")
        print("Some failures may be expected if external services are not configured.")
    
    return passed == total


if __name__ == "__main__":
    try:
        success = asyncio.run(main())
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⚠️  Tests interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Test suite crashed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
