"""
Simple example of using the web scraping system.
"""
import asyncio
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agents.coordinator import CoordinatorAgent
from agents.scraper import ScraperAgent
from agents.parser import ParserAgent
from agents.storage import StorageAgent
from models.task import Task, TaskType


async def main():
    """Run a simple scraping example."""
    print("Starting simple scraping example...")

    # Create agents
    coordinator = CoordinatorAgent()
    scraper = ScraperAgent(coordinator_id=coordinator.agent_id)
    parser = ParserAgent(coordinator_id=coordinator.agent_id)
    storage = StorageAgent(coordinator_id=coordinator.agent_id)

    # Register agents with coordinator
    coordinator.register_agent(scraper)
    coordinator.register_agent(parser)
    coordinator.register_agent(storage)

    # URL to scrape
    url = "https://quotes.toscrape.com/"

    print(f"Scraping {url}...")

    # Create task for scraping
    scrape_task = Task(
        type=TaskType.FETCH_URL,
        parameters={"url": url}
    )

    # Submit task to coordinator
    task_id = await coordinator.submit_task(scrape_task)

    # Wait for task to complete
    while True:
        task_status = coordinator.get_task_status(task_id)
        if task_status and task_status["status"] in ["completed", "failed"]:
            break
        await asyncio.sleep(0.1)

    # Check if task succeeded
    if task_status["status"] == "failed":
        print(f"Error scraping {url}: {task_status['error']['message']}")
        return

    # Get the content
    content = task_status["result"]["content"]

    print("Parsing content...")

    # Create task for parsing
    parse_task = Task(
        type=TaskType.PARSE_CONTENT,
        parameters={
            "content": content,
            "selectors": {
                "quotes": ".quote",
                "quote_texts": ".quote .text",
                "authors": ".quote .author",
                "tags": ".quote .tags .tag"
            }
        }
    )

    # Submit task to coordinator
    task_id = await coordinator.submit_task(parse_task)

    # Wait for task to complete
    while True:
        task_status = coordinator.get_task_status(task_id)
        if task_status and task_status["status"] in ["completed", "failed"]:
            break
        await asyncio.sleep(0.1)

    # Check if task succeeded
    if task_status["status"] == "failed":
        print(f"Error parsing content: {task_status['error']['message']}")
        return

    # Get the extracted data
    extracted_data = task_status["result"]["extracted_data"]

    # Process the data
    quotes = []
    for i, text in enumerate(extracted_data["quote_texts"]):
        quotes.append({
            "text": text.strip('"').strip('"'),
            "author": extracted_data["authors"][i] if i < len(extracted_data["authors"]) else "Unknown",
            "tags": extracted_data["tags"][i].split() if i < len(extracted_data["tags"]) else []
        })

    print(f"Extracted {len(quotes)} quotes")

    # Create output directory if it doesn't exist
    os.makedirs("output", exist_ok=True)

    print("Storing data...")

    # Create task for storing
    store_task = Task(
        type=TaskType.STORE_DATA,
        parameters={
            "data": quotes,
            "format": "json",
            "path": "output/quotes.json"
        }
    )

    # Submit task to coordinator
    task_id = await coordinator.submit_task(store_task)

    # Wait for task to complete
    while True:
        task_status = coordinator.get_task_status(task_id)
        if task_status and task_status["status"] in ["completed", "failed"]:
            break
        await asyncio.sleep(0.1)

    # Check if task succeeded
    if task_status["status"] == "failed":
        print(f"Error storing data: {task_status['error']['message']}")
        return

    print("Data stored successfully!")
    print("Example completed.")


if __name__ == "__main__":
    asyncio.run(main())
