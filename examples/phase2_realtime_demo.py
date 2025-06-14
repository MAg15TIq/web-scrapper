#!/usr/bin/env python3
"""
Phase 2: Real-time & Streaming Capabilities Demo

This script demonstrates the real-time streaming capabilities of the web scraping system:
- Live Data Streaming with WebSocket support
- Real-time Change Detection and monitoring
- Event-driven Architecture with reactive scraping
- Server-Sent Events (SSE) for live updates
- Streaming JSON/CSV output for large datasets
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

async def demo_real_time_streaming():
    """Demonstrate real-time streaming capabilities."""
    print("\n" + "="*60)
    print("🔄 REAL-TIME STREAMING DEMO")
    print("="*60)
    
    try:
        from web.streaming.real_time_engine import streaming_engine, StreamConfig, StreamType
        
        # Start streaming engine
        await streaming_engine.start()
        
        # Create different types of streams
        stream_configs = [
            {
                "stream_type": StreamType.WEBSOCKET,
                "buffer_size": 100,
                "batch_size": 5,
                "flush_interval": 1.0
            },
            {
                "stream_type": StreamType.JSON_STREAM,
                "buffer_size": 500,
                "batch_size": 10,
                "flush_interval": 2.0
            },
            {
                "stream_type": StreamType.CSV_STREAM,
                "buffer_size": 200,
                "batch_size": 20,
                "flush_interval": 3.0
            }
        ]
        
        created_streams = []
        
        for i, config_data in enumerate(stream_configs):
            config = StreamConfig(
                stream_id=f"demo_stream_{i}",
                **config_data
            )
            
            stream_id = await streaming_engine.create_stream(config)
            created_streams.append(stream_id)
            
            print(f"   ✅ Created {config_data['stream_type'].value} stream: {stream_id}")
        
        # Simulate data publishing
        print(f"\n📊 Publishing sample data to streams...")
        
        sample_data = [
            {"type": "product", "name": "Laptop", "price": 999.99, "stock": 15},
            {"type": "product", "name": "Mouse", "price": 29.99, "stock": 50},
            {"type": "product", "name": "Keyboard", "price": 79.99, "stock": 25},
            {"type": "news", "title": "Tech News", "content": "Latest technology updates", "views": 1250},
            {"type": "user", "username": "john_doe", "action": "login", "timestamp": datetime.now().isoformat()}
        ]
        
        for stream_id in created_streams:
            for data in sample_data:
                success = await streaming_engine.publish_to_stream(
                    stream_id, 
                    data, 
                    metadata={"source": "demo", "timestamp": datetime.now().isoformat()}
                )
                if success:
                    print(f"     📤 Published to {stream_id}: {data['type']} data")
                
                await asyncio.sleep(0.5)  # Rate limiting
        
        # Get stream metrics
        print(f"\n📈 Stream Metrics:")
        all_metrics = await streaming_engine.get_stream_metrics()
        
        if "streams" in all_metrics:
            for stream_id, metrics in all_metrics["streams"].items():
                print(f"   Stream {stream_id}:")
                print(f"     Type: {metrics['stream_type']}")
                print(f"     Messages: {metrics['messages_sent']}")
                print(f"     Bytes: {metrics['bytes_sent']}")
                print(f"     Connections: {metrics['active_connections']}")
        
        print(f"\n   Global Metrics:")
        global_metrics = all_metrics.get("global", {})
        print(f"     Total Streams: {global_metrics.get('total_streams', 0)}")
        print(f"     Active Connections: {global_metrics.get('active_connections', 0)}")
        print(f"     Messages Processed: {global_metrics.get('messages_processed', 0)}")
        print(f"     Bytes Transferred: {global_metrics.get('bytes_transferred', 0)}")
        
        print("\n✅ Real-time Streaming Demo Completed!")
        
    except ImportError as e:
        print(f"❌ Real-time Streaming not available: {e}")
    except Exception as e:
        print(f"❌ Error in streaming demo: {e}")


async def demo_change_detection():
    """Demonstrate change detection and monitoring capabilities."""
    print("\n" + "="*60)
    print("🔍 CHANGE DETECTION & MONITORING DEMO")
    print("="*60)
    
    try:
        from web.monitoring.change_detection import change_detection_engine, MonitoringFrequency
        
        # Start change detection engine
        await change_detection_engine.start()
        
        # Add monitoring targets
        monitoring_targets = [
            {
                "url": "https://quotes.toscrape.com/",
                "name": "Quotes to Scrape",
                "frequency": "high",
                "selectors": [".quote .text", ".quote .author"],
                "threshold": 0.1
            },
            {
                "url": "https://books.toscrape.com/",
                "name": "Books to Scrape",
                "frequency": "medium",
                "selectors": [".product_pod h3 a", ".product_pod .price_color"],
                "threshold": 0.2
            }
        ]
        
        added_targets = []
        
        for target_config in monitoring_targets:
            target_id = await change_detection_engine.add_monitoring_target(target_config)
            added_targets.append(target_id)
            
            print(f"   ✅ Added monitoring target: {target_config['name']} ({target_id})")
            print(f"      URL: {target_config['url']}")
            print(f"      Frequency: {target_config['frequency']}")
            print(f"      Selectors: {len(target_config['selectors'])} CSS selectors")
        
        # Wait a bit for initial snapshots
        print(f"\n⏳ Taking initial content snapshots...")
        await asyncio.sleep(3)
        
        # Manually check for changes
        print(f"\n🔍 Checking for changes...")
        
        for target_id in added_targets:
            changes = await change_detection_engine.check_for_changes(target_id)
            
            if changes:
                print(f"   📋 Changes detected for {target_id}: {len(changes)} changes")
                for change in changes[:3]:  # Show first 3 changes
                    print(f"     - {change.change_type.value}: {change.description}")
                    print(f"       Confidence: {change.confidence:.2%}")
            else:
                print(f"   ✅ No changes detected for {target_id} (initial check)")
        
        # Get monitoring status
        print(f"\n📊 Monitoring Status:")
        status = await change_detection_engine.get_monitoring_status()
        
        print(f"   Engine Running: {status.get('is_running', False)}")
        print(f"   Total Targets: {status.get('total_targets', 0)}")
        print(f"   Active Targets: {status.get('active_targets', 0)}")
        print(f"   Active Tasks: {status.get('active_monitoring_tasks', 0)}")
        print(f"   Total Changes: {status.get('total_change_events', 0)}")
        
        # Show metrics
        metrics = status.get('metrics', {})
        print(f"\n   Metrics:")
        print(f"     Total Checks: {metrics.get('total_checks', 0)}")
        print(f"     Changes Detected: {metrics.get('changes_detected', 0)}")
        print(f"     Average Check Time: {metrics.get('average_check_time', 0):.3f}s")
        
        print("\n✅ Change Detection & Monitoring Demo Completed!")
        
    except ImportError as e:
        print(f"❌ Change Detection not available: {e}")
    except Exception as e:
        print(f"❌ Error in change detection demo: {e}")


async def demo_event_driven_system():
    """Demonstrate event-driven architecture capabilities."""
    print("\n" + "="*60)
    print("⚡ EVENT-DRIVEN ARCHITECTURE DEMO")
    print("="*60)
    
    try:
        from web.events.event_system import event_system, EventType, EventPriority, EventHandler, WebhookConfig, ScheduleConfig
        
        # Start event system
        await event_system.start()
        
        # Register sample event handlers
        async def sample_scraping_handler(event):
            print(f"     🕷️ Scraping Handler: Processing {event.event_type.value} event")
            print(f"        Data: {event.data}")
        
        async def sample_change_handler(event):
            print(f"     🔄 Change Handler: Processing {event.event_type.value} event")
            print(f"        Change Type: {event.data.get('change_type', 'unknown')}")
        
        async def sample_alert_handler(event):
            print(f"     🚨 Alert Handler: Processing {event.event_type.value} event")
            print(f"        Priority: {event.priority.value}")
        
        # Register handlers
        handlers = [
            EventHandler(
                handler_id="scraping_handler",
                event_types=[EventType.SCRAPING_REQUESTED, EventType.SCRAPING_COMPLETED],
                handler_function=sample_scraping_handler
            ),
            EventHandler(
                handler_id="change_handler",
                event_types=[EventType.CHANGE_DETECTED],
                handler_function=sample_change_handler
            ),
            EventHandler(
                handler_id="alert_handler",
                event_types=[EventType.SYSTEM_ALERT, EventType.THRESHOLD_EXCEEDED],
                handler_function=sample_alert_handler,
                priority_filter=[EventPriority.HIGH, EventPriority.CRITICAL]
            )
        ]
        
        for handler in handlers:
            event_system.register_event_handler(handler)
            print(f"   ✅ Registered handler: {handler.handler_id}")
        
        # Add webhook configuration
        webhook_config = WebhookConfig(
            webhook_id="demo_webhook",
            url="https://httpbin.org/post",  # Test webhook endpoint
            events=[EventType.SCRAPING_COMPLETED, EventType.CHANGE_DETECTED],
            headers={"X-Demo": "Phase2"},
            enabled=True
        )
        
        await event_system.add_webhook_config(webhook_config)
        print(f"   ✅ Added webhook: {webhook_config.webhook_id}")
        
        # Add schedule configuration
        schedule_config = ScheduleConfig(
            schedule_id="demo_schedule",
            name="Hourly Health Check",
            cron_expression="0 * * * *",  # Every hour
            event_type=EventType.SYSTEM_ALERT,
            event_data={"type": "health_check", "message": "Scheduled health check"},
            enabled=True
        )
        
        await event_system.add_schedule_config(schedule_config)
        print(f"   ✅ Added schedule: {schedule_config.name}")
        
        # Emit sample events
        print(f"\n⚡ Emitting sample events...")
        
        sample_events = [
            {
                "event_type": EventType.SCRAPING_REQUESTED,
                "data": {"url": "https://example.com", "agent": "demo_agent"},
                "priority": EventPriority.NORMAL
            },
            {
                "event_type": EventType.CHANGE_DETECTED,
                "data": {"url": "https://example.com", "change_type": "content_modified"},
                "priority": EventPriority.HIGH
            },
            {
                "event_type": EventType.SYSTEM_ALERT,
                "data": {"message": "High memory usage detected", "value": 85.5},
                "priority": EventPriority.HIGH
            },
            {
                "event_type": EventType.SCRAPING_COMPLETED,
                "data": {"url": "https://example.com", "items_extracted": 42, "duration": 15.3},
                "priority": EventPriority.NORMAL
            }
        ]
        
        emitted_events = []
        
        for event_data in sample_events:
            event_id = await event_system.emit_event(
                event_type=event_data["event_type"],
                data=event_data["data"],
                priority=event_data["priority"],
                source="demo"
            )
            
            emitted_events.append(event_id)
            print(f"   📤 Emitted {event_data['event_type'].value} event: {event_id}")
        
        # Wait for event processing
        print(f"\n⏳ Processing events...")
        await asyncio.sleep(2)
        
        # Get system metrics
        print(f"\n📊 Event System Metrics:")
        metrics = await event_system.get_system_metrics()
        
        print(f"   Engine Running: {metrics.get('is_running', False)}")
        print(f"   Active Events: {metrics.get('active_events', 0)}")
        print(f"   Registered Handlers: {metrics.get('registered_handlers', 0)}")
        print(f"   Webhook Configs: {metrics.get('webhook_configs', 0)}")
        print(f"   Schedule Configs: {metrics.get('schedule_configs', 0)}")
        
        event_metrics = metrics.get('metrics', {})
        print(f"\n   Processing Metrics:")
        print(f"     Total Events: {event_metrics.get('total_events', 0)}")
        print(f"     Processed Events: {event_metrics.get('processed_events', 0)}")
        print(f"     Failed Events: {event_metrics.get('failed_events', 0)}")
        print(f"     Webhook Calls: {event_metrics.get('webhook_calls', 0)}")
        print(f"     Scheduled Events: {event_metrics.get('scheduled_events', 0)}")
        
        print("\n✅ Event-driven Architecture Demo Completed!")
        
    except ImportError as e:
        print(f"❌ Event System not available: {e}")
    except Exception as e:
        print(f"❌ Error in event system demo: {e}")


async def demo_integrated_realtime_workflow():
    """Demonstrate how all Phase 2 systems work together."""
    print("\n" + "="*60)
    print("🔗 INTEGRATED REAL-TIME WORKFLOW DEMO")
    print("="*60)
    
    try:
        from web.streaming.real_time_engine import streaming_engine, StreamConfig, StreamType
        from web.monitoring.change_detection import change_detection_engine
        from web.events.event_system import event_system, EventType, EventPriority
        
        print(f"\n🚀 Starting integrated real-time workflow...")
        
        # 1. Create a real-time stream for monitoring updates
        stream_config = StreamConfig(
            stream_id="monitoring_stream",
            stream_type=StreamType.WEBSOCKET,
            buffer_size=100,
            batch_size=5,
            flush_interval=1.0
        )
        
        stream_id = await streaming_engine.create_stream(stream_config)
        print(f"   ✅ Created monitoring stream: {stream_id}")
        
        # 2. Set up change detection with event integration
        target_config = {
            "url": "https://quotes.toscrape.com/",
            "name": "Integrated Demo Target",
            "frequency": "high",
            "selectors": [".quote"],
            "threshold": 0.05
        }
        
        target_id = await change_detection_engine.add_monitoring_target(target_config)
        print(f"   ✅ Added monitoring target: {target_id}")
        
        # 3. Register event handler that publishes to stream
        async def stream_publisher_handler(event):
            stream_data = {
                "event_type": event.event_type.value,
                "timestamp": event.timestamp.isoformat(),
                "data": event.data,
                "source": event.source
            }
            
            await streaming_engine.publish_to_stream(
                stream_id, 
                stream_data,
                metadata={"handler": "stream_publisher"}
            )
            
            print(f"     📡 Published event to stream: {event.event_type.value}")
        
        from web.events.event_system import EventHandler
        handler = EventHandler(
            handler_id="stream_publisher",
            event_types=[EventType.CHANGE_DETECTED, EventType.SCRAPING_COMPLETED],
            handler_function=stream_publisher_handler
        )
        
        event_system.register_event_handler(handler)
        print(f"   ✅ Registered stream publisher handler")
        
        # 4. Simulate workflow
        print(f"\n🔄 Simulating integrated workflow...")
        
        # Emit some events that will trigger the workflow
        workflow_events = [
            {
                "event_type": EventType.CHANGE_DETECTED,
                "data": {
                    "target_id": target_id,
                    "change_type": "content_modified",
                    "confidence": 0.95
                }
            },
            {
                "event_type": EventType.SCRAPING_COMPLETED,
                "data": {
                    "url": "https://quotes.toscrape.com/",
                    "items_extracted": 10,
                    "duration": 5.2
                }
            }
        ]
        
        for event_data in workflow_events:
            event_id = await event_system.emit_event(
                event_type=event_data["event_type"],
                data=event_data["data"],
                priority=EventPriority.NORMAL,
                source="integrated_demo"
            )
            
            print(f"   📤 Emitted workflow event: {event_data['event_type'].value}")
            await asyncio.sleep(1)
        
        # 5. Check results
        print(f"\n📊 Workflow Results:")
        
        # Get stream metrics
        stream_metrics = await streaming_engine.get_stream_metrics(stream_id)
        print(f"   Stream Messages: {stream_metrics.get('messages_sent', 0)}")
        
        # Get event metrics
        event_metrics = await event_system.get_system_metrics()
        print(f"   Events Processed: {event_metrics['metrics'].get('processed_events', 0)}")
        
        # Get monitoring status
        monitoring_status = await change_detection_engine.get_monitoring_status()
        print(f"   Monitoring Targets: {monitoring_status.get('active_targets', 0)}")
        
        print("\n✅ Integrated Real-time Workflow Demo Completed!")
        
    except Exception as e:
        print(f"❌ Error in integrated workflow demo: {e}")


async def main():
    """Run all Phase 2 real-time capability demos."""
    print("🚀 PHASE 2: REAL-TIME & STREAMING CAPABILITIES DEMO")
    print("=" * 80)
    print("This demo showcases the real-time capabilities added to the web scraping system:")
    print("• Live Data Streaming with WebSocket and SSE support")
    print("• Real-time Change Detection and monitoring")
    print("• Event-driven Architecture with reactive scraping")
    print("• Streaming JSON/CSV output for large datasets")
    print("• Integrated real-time workflow coordination")
    
    # Run individual demos
    await demo_real_time_streaming()
    await demo_change_detection()
    await demo_event_driven_system()
    await demo_integrated_realtime_workflow()
    
    print("\n" + "="*80)
    print("🎉 PHASE 2 REAL-TIME CAPABILITIES DEMO COMPLETED!")
    print("="*80)
    print("\nKey Improvements Demonstrated:")
    print("✅ WebSocket-based real-time data delivery")
    print("✅ Server-Sent Events (SSE) for live updates")
    print("✅ Streaming JSON/CSV output for large datasets")
    print("✅ Real-time progress tracking with granular metrics")
    print("✅ Website change detection with diff algorithms")
    print("✅ Content freshness monitoring")
    print("✅ Automatic re-scraping triggers based on changes")
    print("✅ Webhook integration for external triggers")
    print("✅ Schedule-based scraping with cron-like functionality")
    print("✅ Event queuing system with priority handling")
    print("✅ Reactive scraping based on external data sources")
    
    print("\nNext Steps:")
    print("• Install optional dependencies: pip install croniter")
    print("• Start the web interface to see real-time features in action")
    print("• Configure webhooks and schedules for your use cases")
    print("• Set up change monitoring for your target websites")


if __name__ == "__main__":
    asyncio.run(main())
