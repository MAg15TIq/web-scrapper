"""
Example of using the JavaScript agent to render a page with dynamic content.
"""
import asyncio
import os
import sys
import logging

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agents.coordinator import CoordinatorAgent
from agents.javascript import JavaScriptAgent
from agents.parser import ParserAgent
from agents.storage import StorageAgent
from models.task import Task, TaskType


async def main():
    """Run a JavaScript rendering example."""
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    print("Starting JavaScript rendering example...")
    
    # Create agents
    coordinator = CoordinatorAgent()
    javascript_agent = JavaScriptAgent(coordinator_id=coordinator.agent_id)
    parser = ParserAgent(coordinator_id=coordinator.agent_id)
    storage = StorageAgent(coordinator_id=coordinator.agent_id)
    
    # Register agents with coordinator
    coordinator.register_agent(javascript_agent)
    coordinator.register_agent(parser)
    coordinator.register_agent(storage)
    
    # URL to a page with dynamic content
    url = "https://quotes.toscrape.com/js/"
    
    print(f"Rendering page with JavaScript: {url}")
    
    # Create task for rendering the page
    render_task = Task(
        type=TaskType.RENDER_PAGE,
        parameters={
            "url": url,
            "wait_for": ".quote",  # Wait for quotes to be loaded
            "timeout": 30,
            "viewport": {"width": 1280, "height": 800}
        }
    )
    
    # Submit task to coordinator
    task_id = await coordinator.submit_task(render_task)
    
    # Wait for task to complete
    while True:
        task_status = coordinator.get_task_status(task_id)
        if task_status and task_status["status"] in ["completed", "failed"]:
            break
        await asyncio.sleep(0.1)
    
    # Check if task succeeded
    if task_status["status"] == "failed":
        print(f"Error rendering page: {task_status['error']['message']}")
        return
    
    # Get the rendered content and page ID
    content = task_status["result"]["content"]
    page_id = task_status["result"]["page_id"]
    screenshot_path = task_status["result"]["screenshot_path"]
    
    print(f"Page rendered successfully. Screenshot saved to: {screenshot_path}")
    
    # Scroll the page to load more content
    print("Scrolling page to load more content...")
    
    scroll_task = Task(
        type=TaskType.SCROLL_PAGE,
        parameters={
            "page_id": page_id,
            "max_scrolls": 3,
            "scroll_delay": 1.0
        }
    )
    
    # Submit task to coordinator
    task_id = await coordinator.submit_task(scroll_task)
    
    # Wait for task to complete
    while True:
        task_status = coordinator.get_task_status(task_id)
        if task_status and task_status["status"] in ["completed", "failed"]:
            break
        await asyncio.sleep(0.1)
    
    # Check if task succeeded
    if task_status["status"] == "failed":
        print(f"Error scrolling page: {task_status['error']['message']}")
        return
    
    # Get the updated content
    content = task_status["result"]["content"]
    screenshot_path = task_status["result"]["screenshot_path"]
    
    print(f"Page scrolled successfully. Screenshot saved to: {screenshot_path}")
    
    # Parse the content
    print("Parsing content...")
    
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
    
    # Store the data
    print("Storing data...")
    
    store_task = Task(
        type=TaskType.STORE_DATA,
        parameters={
            "data": quotes,
            "format": "json",
            "path": "output/js_quotes.json"
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
    
    print("Data stored successfully to output/js_quotes.json")
    
    # Take a screenshot of a specific element
    print("Taking a screenshot of a specific element...")
    
    # Execute a script to highlight the first quote
    execute_script_task = Task(
        type=TaskType.EXECUTE_SCRIPT,
        parameters={
            "page_id": page_id,
            "script": """
                const firstQuote = document.querySelector('.quote');
                if (firstQuote) {
                    firstQuote.style.border = '3px solid red';
                    firstQuote.style.backgroundColor = 'lightyellow';
                    return true;
                }
                return false;
            """
        }
    )
    
    # Submit task to coordinator
    task_id = await coordinator.submit_task(execute_script_task)
    
    # Wait for task to complete
    while True:
        task_status = coordinator.get_task_status(task_id)
        if task_status and task_status["status"] in ["completed", "failed"]:
            break
        await asyncio.sleep(0.1)
    
    # Check if task succeeded
    if task_status["status"] == "failed" or not task_status["result"]["result"]:
        print("Error highlighting element")
    else:
        # Take a screenshot
        screenshot_task = Task(
            type=TaskType.TAKE_SCREENSHOT,
            parameters={
                "page_id": page_id,
                "output_path": "output/highlighted_quote.png",
                "full_page": False
            }
        )
        
        # Submit task to coordinator
        task_id = await coordinator.submit_task(screenshot_task)
        
        # Wait for task to complete
        while True:
            task_status = coordinator.get_task_status(task_id)
            if task_status and task_status["status"] in ["completed", "failed"]:
                break
            await asyncio.sleep(0.1)
        
        # Check if task succeeded
        if task_status["status"] == "failed":
            print(f"Error taking screenshot: {task_status['error']['message']}")
        else:
            print(f"Screenshot saved to: {task_status['result']['screenshot_path']}")
    
    # Clean up resources
    await javascript_agent.cleanup()
    
    print("Example completed.")


if __name__ == "__main__":
    asyncio.run(main())
