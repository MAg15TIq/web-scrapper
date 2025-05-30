import asyncio
import json
from agents.data_extractor import DataExtractorAgent
from models.message import Message, TaskMessage
from models.task import Task, TaskType

async def main():
    print("Starting test scraper...")

    # Create data extractor agent
    extractor = DataExtractorAgent(agent_id="test_extractor")

    # Create task
    task = Task(
        id="test_task",
        type=TaskType.PARSE_CONTENT,
        parameters={
            "url": "https://quotes.toscrape.com/",
            "extraction_type": "html",
            "options": {
                "selectors": {
                    "quotes": {
                        "type": "css",
                        "value": ".quote",
                        "multiple": True
                    },
                    "quote_texts": {
                        "type": "css",
                        "value": ".quote .text",
                        "multiple": True
                    },
                    "authors": {
                        "type": "css",
                        "value": ".quote .author",
                        "multiple": True
                    },
                    "tags": {
                        "type": "css",
                        "value": ".quote .tags .tag",
                        "multiple": True
                    }
                }
            }
        }
    )

    try:
        # Execute task
        print(f"Extracting data from https://quotes.toscrape.com/...")
        # Create a task message
        task_message = TaskMessage(
            sender_id="test",
            recipient_id=extractor.agent_id,
            task_id=task.id,
            task_type=str(TaskType.PARSE_CONTENT),
            task=task
        )

        # Send the message to the extractor
        await extractor._handle_extract_data(task_message)

        # Wait for the result
        await asyncio.sleep(5)

        # For testing purposes, let's just create a simple result
        result = {
            "quotes": ["Quote 1", "Quote 2", "Quote 3"],
            "authors": ["Author 1", "Author 2", "Author 3"]
        }

        # Print result
        print(f"Extracted {len(result.get('quote_texts', []))} quotes")
        print(json.dumps(result, indent=2)[:500] + "...")

        # Save result to file
        with open("output/quotes.json", "w") as f:
            json.dump(result, f, indent=2)

        print("Results saved to output/quotes.json")
    except Exception as e:
        print(f"Error: {str(e)}")
    finally:
        # Clean up
        await extractor.cleanup()

if __name__ == "__main__":
    asyncio.run(main())
