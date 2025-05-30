#!/usr/bin/env python
"""
Example script demonstrating the use of advanced specialized agents.
"""
import os
import asyncio
import logging
from typing import Dict, Any, List

from agents.coordinator import CoordinatorAgent
from agents.scraper import ScraperAgent
from agents.parser import ParserAgent
from agents.storage import StorageAgent
from agents.api_integration import APIIntegrationAgent
from agents.nlp_processing import NLPProcessingAgent
from agents.image_processing import ImageProcessingAgent
from models.task import Task, TaskType


async def run_example():
    """Run the example workflow."""
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Create output directory
    os.makedirs("output", exist_ok=True)
    
    # Create agents
    coordinator = CoordinatorAgent()
    scraper = ScraperAgent()
    parser = ParserAgent()
    storage = StorageAgent()
    api_agent = APIIntegrationAgent()
    nlp_agent = NLPProcessingAgent()
    image_agent = ImageProcessingAgent()
    
    # Register agents with coordinator
    coordinator.register_agent(scraper)
    coordinator.register_agent(parser)
    coordinator.register_agent(storage)
    coordinator.register_agent(api_agent)
    coordinator.register_agent(nlp_agent)
    coordinator.register_agent(image_agent)
    
    # Start agents
    agent_tasks = [
        asyncio.create_task(coordinator.start()),
        asyncio.create_task(scraper.start()),
        asyncio.create_task(parser.start()),
        asyncio.create_task(storage.start()),
        asyncio.create_task(api_agent.start()),
        asyncio.create_task(nlp_agent.start()),
        asyncio.create_task(image_agent.start())
    ]
    
    try:
        # Example 1: Web scraping with NLP analysis
        logging.info("Starting Example 1: Web scraping with NLP analysis")
        
        # Step 1: Fetch a web page
        fetch_task = Task(
            type=TaskType.FETCH_URL,
            parameters={
                "url": "https://example.com/news",
                "method": "GET",
                "headers": {"User-Agent": "Mozilla/5.0"}
            }
        )
        fetch_task_id = await coordinator.submit_task(fetch_task)
        
        # Wait for the task to complete
        while True:
            status = coordinator.get_task_status(fetch_task_id)
            if status["status"] in ["completed", "failed"]:
                break
            await asyncio.sleep(0.5)
        
        if status["status"] == "failed":
            logging.error(f"Fetch task failed: {status['error']}")
            return
        
        # Step 2: Parse the content
        html_content = status["result"]["content"]
        parse_task = Task(
            type=TaskType.PARSE_CONTENT,
            parameters={
                "content": html_content,
                "selectors": {
                    "title": "h1",
                    "article": "article p",
                    "date": ".published-date"
                }
            },
            dependencies=[fetch_task_id]
        )
        parse_task_id = await coordinator.submit_task(parse_task)
        
        # Wait for the task to complete
        while True:
            status = coordinator.get_task_status(parse_task_id)
            if status["status"] in ["completed", "failed"]:
                break
            await asyncio.sleep(0.5)
        
        if status["status"] == "failed":
            logging.error(f"Parse task failed: {status['error']}")
            return
        
        # Step 3: Perform NLP analysis on the article text
        article_text = " ".join(status["result"]["data"]["article"])
        
        # Entity extraction
        entity_task = Task(
            type=TaskType.NLP_ENTITY_EXTRACTION,
            parameters={
                "text": article_text,
                "entity_types": ["PERSON", "ORG", "GPE", "DATE"]
            },
            dependencies=[parse_task_id]
        )
        entity_task_id = await coordinator.submit_task(entity_task)
        
        # Sentiment analysis
        sentiment_task = Task(
            type=TaskType.NLP_SENTIMENT_ANALYSIS,
            parameters={
                "text": article_text
            },
            dependencies=[parse_task_id]
        )
        sentiment_task_id = await coordinator.submit_task(sentiment_task)
        
        # Keyword extraction
        keyword_task = Task(
            type=TaskType.NLP_KEYWORD_EXTRACTION,
            parameters={
                "text": article_text,
                "max_keywords": 10
            },
            dependencies=[parse_task_id]
        )
        keyword_task_id = await coordinator.submit_task(keyword_task)
        
        # Wait for all NLP tasks to complete
        nlp_task_ids = [entity_task_id, sentiment_task_id, keyword_task_id]
        for task_id in nlp_task_ids:
            while True:
                status = coordinator.get_task_status(task_id)
                if status["status"] in ["completed", "failed"]:
                    break
                await asyncio.sleep(0.5)
        
        # Step 4: Extract images from the page
        image_extraction_task = Task(
            type=TaskType.IMAGE_EXTRACTION,
            parameters={
                "html_content": html_content,
                "base_url": "https://example.com/news",
                "download": True
            },
            dependencies=[fetch_task_id]
        )
        image_task_id = await coordinator.submit_task(image_extraction_task)
        
        # Wait for image extraction to complete
        while True:
            status = coordinator.get_task_status(image_task_id)
            if status["status"] in ["completed", "failed"]:
                break
            await asyncio.sleep(0.5)
        
        # Step 5: Store the combined results
        entity_status = coordinator.get_task_status(entity_task_id)
        sentiment_status = coordinator.get_task_status(sentiment_task_id)
        keyword_status = coordinator.get_task_status(keyword_task_id)
        image_status = coordinator.get_task_status(image_task_id)
        
        # Combine all results
        combined_data = {
            "article": {
                "title": status["result"]["data"]["title"],
                "text": article_text,
                "date": status["result"]["data"].get("date")
            },
            "nlp_analysis": {
                "entities": entity_status["result"]["entities"] if entity_status["status"] == "completed" else {},
                "sentiment": sentiment_status["result"] if sentiment_status["status"] == "completed" else {},
                "keywords": keyword_status["result"]["keywords"] if keyword_status["status"] == "completed" else []
            },
            "images": image_status["result"]["image_urls"] if image_status["status"] == "completed" else []
        }
        
        store_task = Task(
            type=TaskType.STORE_DATA,
            parameters={
                "data": combined_data,
                "format": "json",
                "path": "output/article_analysis.json"
            },
            dependencies=[entity_task_id, sentiment_task_id, keyword_task_id, image_task_id]
        )
        store_task_id = await coordinator.submit_task(store_task)
        
        # Wait for storage to complete
        while True:
            status = coordinator.get_task_status(store_task_id)
            if status["status"] in ["completed", "failed"]:
                break
            await asyncio.sleep(0.5)
        
        logging.info("Example 1 completed")
        
        # Example 2: API Integration with data transformation
        logging.info("Starting Example 2: API Integration with data transformation")
        
        # Step 1: Configure API
        api_config_task = Task(
            type=TaskType.API_AUTHENTICATE,
            parameters={
                "api_id": "weather_api",
                "auth_type": "key",
                "api_key": "your_api_key_here"  # In a real app, use environment variables
            }
        )
        api_config_task_id = await coordinator.submit_task(api_config_task)
        
        # Wait for API configuration to complete
        while True:
            status = coordinator.get_task_status(api_config_task_id)
            if status["status"] in ["completed", "failed"]:
                break
            await asyncio.sleep(0.5)
        
        # Step 2: Make API request
        api_request_task = Task(
            type=TaskType.API_REQUEST,
            parameters={
                "api_id": "weather_api",
                "endpoint": "https://api.openweathermap.org/data/2.5/weather",
                "method": "GET",
                "query_params": {
                    "q": "London,uk",
                    "units": "metric"
                }
            },
            dependencies=[api_config_task_id]
        )
        api_request_task_id = await coordinator.submit_task(api_request_task)
        
        # Wait for API request to complete
        while True:
            status = coordinator.get_task_status(api_request_task_id)
            if status["status"] in ["completed", "failed"]:
                break
            await asyncio.sleep(0.5)
        
        if status["status"] == "failed":
            logging.error(f"API request failed: {status['error']}")
        else:
            # Step 3: Transform API data
            api_transform_task = Task(
                type=TaskType.API_TRANSFORM,
                parameters={
                    "data": status["result"]["data"],
                    "transformations": [
                        {
                            "type": "map",
                            "mapping": {
                                "city": "name",
                                "temperature": "main.temp",
                                "weather_description": "weather.0.description",
                                "humidity": "main.humidity",
                                "wind_speed": "wind.speed"
                            }
                        }
                    ]
                },
                dependencies=[api_request_task_id]
            )
            api_transform_task_id = await coordinator.submit_task(api_transform_task)
            
            # Wait for transformation to complete
            while True:
                status = coordinator.get_task_status(api_transform_task_id)
                if status["status"] in ["completed", "failed"]:
                    break
                await asyncio.sleep(0.5)
            
            # Step 4: Store the transformed data
            if status["status"] == "completed":
                store_api_task = Task(
                    type=TaskType.STORE_DATA,
                    parameters={
                        "data": status["result"]["data"],
                        "format": "json",
                        "path": "output/weather_data.json"
                    },
                    dependencies=[api_transform_task_id]
                )
                store_api_task_id = await coordinator.submit_task(store_api_task)
                
                # Wait for storage to complete
                while True:
                    status = coordinator.get_task_status(store_api_task_id)
                    if status["status"] in ["completed", "failed"]:
                        break
                    await asyncio.sleep(0.5)
        
        logging.info("Example 2 completed")
        
        # Allow some time for all tasks to complete
        await asyncio.sleep(2)
        
    finally:
        # Stop all agents
        await coordinator.stop()
        await scraper.stop()
        await parser.stop()
        await storage.stop()
        await api_agent.stop()
        await nlp_agent.stop()
        await image_agent.stop()
        
        # Cancel all tasks
        for task in agent_tasks:
            task.cancel()
        
        try:
            await asyncio.gather(*agent_tasks, return_exceptions=True)
        except asyncio.CancelledError:
            pass


if __name__ == "__main__":
    asyncio.run(run_example())
