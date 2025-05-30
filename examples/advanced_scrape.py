"""
Advanced example of using the web scraping system with specialized agents.
"""
import asyncio
import os
import sys
from urllib.parse import urlparse

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agents.coordinator import CoordinatorAgent
from agents.scraper import ScraperAgent
from agents.parser import ParserAgent
from agents.storage import StorageAgent
from agents.javascript import JavaScriptAgent
from agents.authentication import AuthenticationAgent
from agents.anti_detection import AntiDetectionAgent
from agents.data_transformation import DataTransformationAgent
from models.task import Task, TaskType


async def main():
    """Run an advanced scraping example."""
    print("Starting advanced scraping example...")
    
    # Create agents
    coordinator = CoordinatorAgent()
    scraper = ScraperAgent(coordinator_id=coordinator.agent_id)
    parser = ParserAgent(coordinator_id=coordinator.agent_id)
    storage = StorageAgent(coordinator_id=coordinator.agent_id)
    javascript = JavaScriptAgent(coordinator_id=coordinator.agent_id)
    authentication = AuthenticationAgent(coordinator_id=coordinator.agent_id)
    anti_detection = AntiDetectionAgent(coordinator_id=coordinator.agent_id)
    data_transformation = DataTransformationAgent(coordinator_id=coordinator.agent_id)
    
    # Register agents with coordinator
    coordinator.register_agent(scraper)
    coordinator.register_agent(parser)
    coordinator.register_agent(storage)
    coordinator.register_agent(javascript)
    coordinator.register_agent(authentication)
    coordinator.register_agent(anti_detection)
    coordinator.register_agent(data_transformation)
    
    # URL to scrape
    url = "https://quotes.toscrape.com/"
    domain = urlparse(url).netloc
    
    print(f"Scraping {url} with advanced features...")
    
    # Step 1: Generate a browser fingerprint
    print("\n1. Generating browser fingerprint...")
    fingerprint_task = Task(
        type=TaskType.GENERATE_FINGERPRINT,
        parameters={
            "domain": domain,
            "consistent": True
        }
    )
    
    fingerprint_task_id = await coordinator.submit_task(fingerprint_task)
    fingerprint_result = await wait_for_task(coordinator, fingerprint_task_id)
    
    if not fingerprint_result:
        print("Failed to generate fingerprint. Exiting.")
        return
    
    print(f"Generated fingerprint for {domain}")
    print(f"User Agent: {fingerprint_result['fingerprint']['user_agent']}")
    
    # Step 2: Check if the site is blocking scraping
    print("\n2. Checking if site is blocking scraping...")
    blocking_task = Task(
        type=TaskType.CHECK_BLOCKING,
        parameters={
            "url": url,
            "check_methods": ["status_code", "content_analysis", "redirect"]
        }
    )
    
    blocking_task_id = await coordinator.submit_task(blocking_task)
    blocking_result = await wait_for_task(coordinator, blocking_task_id)
    
    if not blocking_result:
        print("Failed to check blocking. Exiting.")
        return
    
    if blocking_result["is_blocked"]:
        print(f"Site is blocking scraping: {blocking_result['reason']}")
        print("Continuing anyway for demonstration purposes...")
    else:
        print("Site is not blocking scraping")
    
    # Step 3: Optimize request pattern
    print("\n3. Optimizing request pattern...")
    optimize_task = Task(
        type=TaskType.OPTIMIZE_REQUEST_PATTERN,
        parameters={
            "domain": domain,
            "aggressive": False
        }
    )
    
    optimize_task_id = await coordinator.submit_task(optimize_task)
    optimize_result = await wait_for_task(coordinator, optimize_task_id)
    
    if not optimize_result:
        print("Failed to optimize request pattern. Exiting.")
        return
    
    print(f"Optimal request frequency: {optimize_result['requests_per_hour']} requests per hour")
    print(f"Recommended delay between requests: {optimize_result['delay']:.2f} seconds")
    
    # Step 4: Scrape the content
    print("\n4. Scraping content...")
    scrape_task = Task(
        type=TaskType.FETCH_URL,
        parameters={"url": url}
    )
    
    scrape_task_id = await coordinator.submit_task(scrape_task)
    scrape_result = await wait_for_task(coordinator, scrape_task_id)
    
    if not scrape_result:
        print("Failed to scrape content. Exiting.")
        return
    
    print(f"Successfully scraped {len(scrape_result['content'])} bytes of content")
    
    # Step 5: Parse the content
    print("\n5. Parsing content...")
    parse_task = Task(
        type=TaskType.PARSE_CONTENT,
        parameters={
            "content": scrape_result["content"],
            "selectors": {
                "quote": ".quote .text",
                "author": ".quote .author",
                "tags": ".quote .tags .tag"
            },
            "normalize": True
        }
    )
    
    parse_task_id = await coordinator.submit_task(parse_task)
    parse_result = await wait_for_task(coordinator, parse_task_id)
    
    if not parse_result:
        print("Failed to parse content. Exiting.")
        return
    
    extracted_data = parse_result["extracted_data"]
    print(f"Extracted {len(extracted_data['quote'])} quotes")
    
    # Step 6: Clean and normalize the data
    print("\n6. Cleaning and normalizing data...")
    clean_task = Task(
        type=TaskType.CLEAN_DATA,
        parameters={
            "data": [
                {"quote": quote, "author": author, "tags": tags}
                for quote, author, tags in zip(
                    extracted_data["quote"],
                    extracted_data["author"],
                    extracted_data["tags"]
                )
            ],
            "operations": [
                {"field": "quote", "operation": "strip_whitespace"},
                {"field": "author", "operation": "strip_whitespace"},
                {"field": "*", "operation": "remove_empty"}
            ],
            "add_metadata": True
        }
    )
    
    clean_task_id = await coordinator.submit_task(clean_task)
    clean_result = await wait_for_task(coordinator, clean_task_id)
    
    if not clean_result:
        print("Failed to clean data. Exiting.")
        return
    
    cleaned_data = clean_result["data"]
    print(f"Cleaned {len(cleaned_data)} items")
    
    # Step 7: Analyze text content
    print("\n7. Analyzing text content...")
    if cleaned_data:
        # Analyze the first quote
        first_quote = cleaned_data[0]["quote"]
        analyze_task = Task(
            type=TaskType.ANALYZE_TEXT,
            parameters={
                "text": first_quote,
                "analyses": ["sentiment", "keywords", "language"],
                "language": "en"
            }
        )
        
        analyze_task_id = await coordinator.submit_task(analyze_task)
        analyze_result = await wait_for_task(coordinator, analyze_task_id)
        
        if not analyze_result:
            print("Failed to analyze text. Skipping this step.")
        else:
            results = analyze_result["results"]
            if "sentiment" in results:
                sentiment = results["sentiment"]
                print(f"Quote sentiment: {sentiment['sentiment']} (score: {sentiment['compound_score']:.2f})")
            
            if "keywords" in results and isinstance(results["keywords"], list) and results["keywords"]:
                keywords = [kw["word"] for kw in results["keywords"][:5]]
                print(f"Top keywords: {', '.join(keywords)}")
    
    # Step 8: Store the data
    print("\n8. Storing data...")
    output_path = "output/advanced_scrape_results.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    store_task = Task(
        type=TaskType.STORE_DATA,
        parameters={
            "data": cleaned_data,
            "format": "json",
            "path": output_path
        }
    )
    
    store_task_id = await coordinator.submit_task(store_task)
    store_result = await wait_for_task(coordinator, store_task_id)
    
    if not store_result:
        print("Failed to store data. Exiting.")
        return
    
    print(f"Successfully stored {len(cleaned_data)} items to {output_path}")
    print("\nAdvanced scraping example completed!")


async def wait_for_task(coordinator, task_id):
    """Wait for a task to complete and return its result."""
    while True:
        task_status = coordinator.get_task_status(task_id)
        if task_status and task_status["status"] in ["completed", "failed"]:
            break
        await asyncio.sleep(0.1)
    
    if task_status["status"] == "failed":
        print(f"Task failed: {task_status['error']['message']}")
        return None
    
    return task_status["result"]


if __name__ == "__main__":
    asyncio.run(main())
