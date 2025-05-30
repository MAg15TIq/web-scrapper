"""
Example demonstrating the new advanced features of the web scraping system:
1. Real-Time Monitoring Dashboard
2. Advanced Pagination Handling Engine
3. Distributed Result Aggregation with Apache Arrow
"""
import asyncio
import os
import sys
import time
from urllib.parse import urlparse

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agents.coordinator import CoordinatorAgent
from agents.scraper import ScraperAgent
from agents.parser import ParserAgent
from agents.storage import StorageAgent
from agents.javascript import JavaScriptAgent
from agents.anti_detection import AntiDetectionAgent
from models.task import Task, TaskType
from monitoring.dashboard import dashboard_manager
from utils.pagination_engine import pagination_engine
from utils.result_aggregator import result_aggregator


async def wait_for_task(coordinator, task_id, timeout=60.0):
    """Wait for a task to complete."""
    start_time = time.time()
    while time.time() - start_time < timeout:
        task_status = coordinator.get_task_status(task_id)
        if task_status and task_status["status"] in ["completed", "failed"]:
            if task_status["status"] == "completed":
                return task_status["result"]
            else:
                print(f"Task failed: {task_status['error']}")
                return None
        await asyncio.sleep(0.1)
    
    print(f"Timeout waiting for task {task_id}")
    return None


async def main():
    """Run the advanced features example."""
    print("Starting advanced features example...")
    
    # Create agents
    coordinator = CoordinatorAgent()
    scraper = ScraperAgent(coordinator_id=coordinator.agent_id)
    parser = ParserAgent(coordinator_id=coordinator.agent_id)
    storage = StorageAgent(coordinator_id=coordinator.agent_id)
    javascript = JavaScriptAgent(coordinator_id=coordinator.agent_id)
    anti_detection = AntiDetectionAgent(coordinator_id=coordinator.agent_id)
    
    # Register agents with coordinator
    coordinator.register_agent(scraper)
    coordinator.register_agent(parser)
    coordinator.register_agent(storage)
    coordinator.register_agent(javascript)
    coordinator.register_agent(anti_detection)
    
    # Start the monitoring dashboard
    await dashboard_manager.start()
    
    # URL to scrape
    url = "https://quotes.toscrape.com/"
    domain = urlparse(url).netloc
    
    print(f"Scraping {url} with advanced features...")
    
    # Update dashboard with initial stats
    dashboard_manager.update_stats({
        "scraping_progress": 0.0,
        "active_agents": len(coordinator.agents),
        "failed_tasks": 0,
        "data_collected": 0,
        "agent_stats": coordinator.agents,
        "domain_stats": {
            domain: {
                "request_count": 0,
                "current_rate": 1.0,
                "success_rate": 1.0
            }
        }
    })
    
    # Step 1: Use the advanced pagination engine
    print("\n1. Using advanced pagination engine...")
    pagination_task = Task(
        type=TaskType.FOLLOW_PAGINATION,
        parameters={
            "url": url,
            "max_pages": 3,
            "wait_between_requests": 1.0,
            "use_browser": False
        }
    )
    
    pagination_task_id = await coordinator.submit_task(pagination_task)
    
    # Update dashboard while waiting for the task
    for progress in range(1, 4):
        # Update dashboard
        dashboard_manager.update_stats({
            "scraping_progress": progress / 3.0,
            "active_agents": len(coordinator.agents),
            "failed_tasks": 0,
            "data_collected": progress * 10,
            "domain_stats": {
                domain: {
                    "request_count": progress,
                    "current_rate": 1.0,
                    "success_rate": 1.0
                }
            }
        })
        await asyncio.sleep(2)
    
    pagination_result = await wait_for_task(coordinator, pagination_task_id)
    
    if not pagination_result:
        print("Failed to follow pagination. Exiting.")
        await dashboard_manager.stop()
        return
    
    print(f"Successfully scraped {pagination_result['pages_fetched']} pages")
    
    # Step 2: Parse the content from each page
    print("\n2. Parsing content from each page...")
    all_quotes = []
    
    for page_result in pagination_result["results"]:
        parse_task = Task(
            type=TaskType.PARSE_CONTENT,
            parameters={
                "content": page_result["content"],
                "selectors": {
                    "quotes": ".quote",
                    "text": ".quote .text",
                    "author": ".quote .author",
                    "tags": ".quote .tags .tag"
                },
                "base_url": page_result["url"]
            }
        )
        
        parse_task_id = await coordinator.submit_task(parse_task)
        parse_result = await wait_for_task(coordinator, parse_task_id)
        
        if not parse_result:
            print(f"Failed to parse content for page {page_result.get('page_number', 'unknown')}. Skipping.")
            continue
        
        # Extract quotes
        quotes = []
        for i, _ in enumerate(parse_result["data"].get("text", [])):
            quote = {
                "text": parse_result["data"]["text"][i] if i < len(parse_result["data"]["text"]) else "",
                "author": parse_result["data"]["author"][i] if i < len(parse_result["data"]["author"]) else "",
                "tags": parse_result["data"]["tags"][i] if i < len(parse_result["data"]["tags"]) else [],
                "page": page_result.get("page_number", 1),
                "source": domain,
                "timestamp": time.time()
            }
            quotes.append(quote)
        
        all_quotes.extend(quotes)
        
        # Update dashboard
        dashboard_manager.update_stats({
            "scraping_progress": 0.5 + (page_result.get("page_number", 1) / pagination_result["pages_fetched"] / 2),
            "data_collected": len(all_quotes),
            "domain_stats": {
                domain: {
                    "request_count": pagination_result["pages_fetched"],
                    "current_rate": 1.0,
                    "success_rate": 1.0
                }
            }
        })
    
    print(f"Extracted {len(all_quotes)} quotes from {pagination_result['pages_fetched']} pages")
    
    # Step 3: Use the distributed result aggregator
    print("\n3. Aggregating results with Apache Arrow...")
    
    # Split the quotes into chunks to simulate distributed scraping
    chunk_size = max(1, len(all_quotes) // 3)
    chunks = [all_quotes[i:i + chunk_size] for i in range(0, len(all_quotes), chunk_size)]
    
    # Add some duplicate quotes to demonstrate deduplication
    if len(chunks) > 1 and len(chunks[0]) > 0:
        chunks[1].append(chunks[0][0])
    
    # Add a quote with a different timestamp to demonstrate conflict resolution
    if len(chunks) > 2 and len(chunks[0]) > 0:
        duplicate = chunks[0][0].copy()
        duplicate["timestamp"] = time.time() + 100  # Newer timestamp
        duplicate["text"] = duplicate["text"] + " (updated)"
        chunks[2].append(duplicate)
    
    aggregate_task = Task(
        type=TaskType.AGGREGATE_RESULTS,
        parameters={
            "chunks": chunks,
            "id_columns": ["text", "author"],
            "timestamp_column": "timestamp",
            "source_column": "source",
            "source_authority": {domain: 1.0},
            "similarity_threshold": 0.8,
            "text_columns": ["text"],
            "output_format": "records",
            "path": "output/aggregated_quotes.json",
            "format": "json"
        }
    )
    
    aggregate_task_id = await coordinator.submit_task(aggregate_task)
    
    # Update dashboard while waiting for the task
    dashboard_manager.update_stats({
        "scraping_progress": 0.9,
        "data_collected": len(all_quotes),
    })
    
    aggregate_result = await wait_for_task(coordinator, aggregate_task_id)
    
    if not aggregate_result:
        print("Failed to aggregate results. Exiting.")
        await dashboard_manager.stop()
        return
    
    print(f"Successfully aggregated {aggregate_result['aggregated_records']} quotes")
    print(f"Saved to {aggregate_result['storage_result']['path']}")
    
    # Final dashboard update
    dashboard_manager.update_stats({
        "scraping_progress": 1.0,
        "data_collected": aggregate_result['aggregated_records'],
        "throughput": pagination_result["pages_fetched"] / (time.time() - dashboard_manager.dashboard.stats["start_time"])
    })
    
    print("\nAdvanced features example completed!")
    print("Press Ctrl+C to exit the dashboard and end the program.")
    
    # Keep the dashboard running until user interrupts
    try:
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        print("\nExiting...")
    finally:
        await dashboard_manager.stop()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nExiting...")
