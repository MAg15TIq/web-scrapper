"""
Example script demonstrating the self-aware intelligent web scraping system.
"""
import asyncio
import logging
import sys
import json
from typing import Dict, Any, List, Optional

from agents.master_intelligence import MasterIntelligenceAgent
from agents.url_intelligence import URLIntelligenceAgent
from agents.content_recognition import ContentRecognitionAgent
from agents.document_intelligence import DocumentIntelligenceAgent
from agents.performance_optimization import PerformanceOptimizationAgent
from agents.quality_assurance import QualityAssuranceAgent
from models.intelligence import ContentType, InputType, WebsiteType, ContentAnalysisResult
from models.task import Task, TaskType, TaskStatus


async def analyze_url(url: str) -> Dict[str, Any]:
    """
    Analyze a URL using the URL Intelligence Agent.

    Args:
        url: The URL to analyze.

    Returns:
        A dictionary containing the analysis results.
    """
    # Create URL Intelligence Agent
    url_agent = URLIntelligenceAgent(agent_id="url_intelligence")

    # Start the agent
    await url_agent.start()

    try:
        # Create a task for URL analysis
        task = Task(
            type=TaskType.ANALYZE_URL,
            parameters={"url": url}
        )

        # Execute the task
        result = await url_agent.execute_task(task)

        return result
    finally:
        # Stop the agent
        await url_agent.stop()


async def recognize_content(content: str) -> Dict[str, Any]:
    """
    Recognize and analyze content using the Content Recognition Agent.

    Args:
        content: The content to analyze.

    Returns:
        A dictionary containing the analysis results.
    """
    # Create Content Recognition Agent
    content_agent = ContentRecognitionAgent(agent_id="content_recognition")

    # Start the agent
    await content_agent.start()

    try:
        # Create a task for content recognition
        task = Task(
            type=TaskType.RECOGNIZE_CONTENT,
            parameters={"content": content}
        )

        # Execute the task
        result = await content_agent.execute_task(task)

        return result
    finally:
        # Stop the agent
        await content_agent.stop()


async def process_document(document_path: str, document_type: Optional[str] = None) -> Dict[str, Any]:
    """
    Process a document using the Document Intelligence Agent.

    Args:
        document_path: The path to the document.
        document_type: The document type if known. If None, it will be detected.

    Returns:
        A dictionary containing the processing results.
    """
    # Create Document Intelligence Agent
    document_agent = DocumentIntelligenceAgent(agent_id="document_intelligence")

    # Start the agent
    await document_agent.start()

    try:
        # Create a task for document processing
        task = Task(
            type=TaskType.PROCESS_DOCUMENT,
            parameters={
                "document_data": document_path,
                "document_type": document_type
            }
        )

        # Execute the task
        result = await document_agent.execute_task(task)

        return result
    finally:
        # Stop the agent
        await document_agent.stop()


async def intelligent_scraping(input_data: str) -> Dict[str, Any]:
    """
    Perform intelligent scraping using the Master Intelligence Agent.

    Args:
        input_data: The input data to analyze. Can be a URL, file path, or raw content.

    Returns:
        A dictionary containing the scraping results.
    """
    # Create Master Intelligence Agent
    master_agent = MasterIntelligenceAgent(agent_id="master_intelligence")

    # Start the agent
    await master_agent.start()

    try:
        # Analyze the input
        analysis_result = await master_agent.analyze_input(input_data)
        print(f"Input analysis result: {analysis_result}")

        # Select appropriate agents
        agent_assignments = await master_agent.select_agents(analysis_result)
        print(f"Selected agents: {agent_assignments}")

        # Process the input based on its type
        if analysis_result.input_type == InputType.URL:
            # Analyze URL
            url_result = await analyze_url(input_data)
            print(f"URL analysis result: {url_result}")

            # Process based on content type
            if analysis_result.content_type == ContentType.HTML:
                # Fetch the content
                # In a real implementation, you would use a scraper agent to fetch the content
                # For this example, we'll just use a placeholder
                content = "<html><body><h1>Example Content</h1></body></html>"

                # Recognize content
                content_result = await recognize_content(content)
                print(f"Content recognition result: {content_result}")

                return {
                    "analysis": analysis_result.dict(),
                    "url_analysis": url_result,
                    "content_analysis": content_result
                }

            elif analysis_result.content_type in [ContentType.PDF, ContentType.DOC, ContentType.DOCX]:
                # Process document
                document_result = await process_document(input_data, analysis_result.content_type.value)
                print(f"Document processing result: {document_result}")

                return {
                    "analysis": analysis_result.dict(),
                    "url_analysis": url_result,
                    "document_analysis": document_result
                }

            else:
                return {
                    "analysis": analysis_result.dict(),
                    "url_analysis": url_result,
                    "error": f"Unsupported content type: {analysis_result.content_type}"
                }

        elif analysis_result.input_type == InputType.FILE:
            # Process document
            document_result = await process_document(input_data)
            print(f"Document processing result: {document_result}")

            return {
                "analysis": analysis_result.dict(),
                "document_analysis": document_result
            }

        else:
            # Recognize content
            content_result = await recognize_content(input_data)
            print(f"Content recognition result: {content_result}")

            return {
                "analysis": analysis_result.dict(),
                "content_analysis": content_result
            }

    finally:
        # Stop the agent
        await master_agent.stop()


async def optimize_performance() -> Dict[str, Any]:
    """
    Optimize system performance using the Performance Optimization Agent.

    Returns:
        A dictionary containing optimization results.
    """
    # Create Performance Optimization Agent
    performance_agent = PerformanceOptimizationAgent(agent_id="performance_optimization")

    # Start the agent
    await performance_agent.start()

    try:
        # Create a task for performance optimization
        task = Task(
            type=TaskType.OPTIMIZE_PERFORMANCE,
            parameters={
                "target_components": ["system", "agents", "tasks"],
                "optimization_goals": ["speed", "memory", "accuracy"],
                "max_resource_usage": {
                    "cpu": 0.8,  # 80% CPU usage
                    "memory": 0.7  # 70% memory usage
                }
            }
        )

        # Execute the task
        result = await performance_agent.execute_task(task)

        return result
    finally:
        # Stop the agent
        await performance_agent.stop()


async def validate_data(data: List[Dict[str, Any]], schema: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate data using the Quality Assurance Agent.

    Args:
        data: The data to validate.
        schema: The schema to validate against.

    Returns:
        A dictionary containing validation results.
    """
    # Create Quality Assurance Agent
    quality_agent = QualityAssuranceAgent(agent_id="quality_assurance")

    # Start the agent
    await quality_agent.start()

    try:
        # Create a task for data validation
        task = Task(
            type=TaskType.VALIDATE_DATA,
            parameters={
                "data": data,
                "schema": schema
            }
        )

        # Execute the task
        result = await quality_agent.execute_task(task)

        return result
    finally:
        # Stop the agent
        await quality_agent.stop()


async def score_data_quality(data: List[Dict[str, Any]], schema: Dict[str, Any]) -> Dict[str, Any]:
    """
    Score data quality using the Quality Assurance Agent.

    Args:
        data: The data to score.
        schema: The schema to validate against.

    Returns:
        A dictionary containing quality scores.
    """
    # Create Quality Assurance Agent
    quality_agent = QualityAssuranceAgent(agent_id="quality_assurance")

    # Start the agent
    await quality_agent.start()

    try:
        # Create a task for quality scoring
        task = Task(
            type=TaskType.SCORE_DATA_QUALITY,
            parameters={
                "data": data,
                "schema": schema,
                "quality_thresholds": {
                    "completeness": 0.9,
                    "accuracy": 0.95,
                    "consistency": 0.9
                }
            }
        )

        # Execute the task
        result = await quality_agent.execute_task(task)

        return result
    finally:
        # Stop the agent
        await quality_agent.stop()


async def main():
    """Main function."""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)]
    )

    # Check command line arguments
    if len(sys.argv) < 2:
        print("Usage: python intelligent_scraping.py <command> [arguments]")
        print("Commands:")
        print("  scrape <input_data>   - Perform intelligent scraping")
        print("  optimize              - Optimize system performance")
        print("  validate              - Validate sample data")
        print("  quality               - Score sample data quality")
        return

    command = sys.argv[1]

    if command == "scrape":
        if len(sys.argv) < 3:
            print("Usage: python intelligent_scraping.py scrape <input_data>")
            return

        input_data = sys.argv[2]
        print(f"Processing input: {input_data}")

        # Perform intelligent scraping
        result = await intelligent_scraping(input_data)

        # Print result
        print(json.dumps(result, indent=2))

    elif command == "optimize":
        print("Optimizing system performance...")

        # Optimize performance
        result = await optimize_performance()

        # Print result
        print(json.dumps(result, indent=2))

    elif command == "validate":
        print("Validating sample data...")

        # Sample data
        data = [
            {"title": "Product 1", "price": 10.99, "description": "This is product 1"},
            {"title": "Product 2", "price": 20.99, "description": None},
            {"title": None, "price": 30.99, "description": "This is product 3"}
        ]

        # Sample schema
        schema = {
            "title": {"type": "string", "required": True},
            "price": {"type": "number", "required": True, "minimum": 0},
            "description": {"type": "string", "required": False}
        }

        # Validate data
        result = await validate_data(data, schema)

        # Print result
        print(json.dumps(result, indent=2))

    elif command == "quality":
        print("Scoring sample data quality...")

        # Sample data
        data = [
            {"title": "Product 1", "price": 10.99, "description": "This is product 1"},
            {"title": "Product 2", "price": 20.99, "description": None},
            {"title": None, "price": 30.99, "description": "This is product 3"}
        ]

        # Sample schema
        schema = {
            "title": {"type": "string", "required": True},
            "price": {"type": "number", "required": True, "minimum": 0},
            "description": {"type": "string", "required": False}
        }

        # Score data quality
        result = await score_data_quality(data, schema)

        # Print result
        print(json.dumps(result, indent=2))

    else:
        print(f"Unknown command: {command}")
        print("Usage: python intelligent_scraping.py <command> [arguments]")
        print("Commands:")
        print("  scrape <input_data>   - Perform intelligent scraping")
        print("  optimize              - Optimize system performance")
        print("  validate              - Validate sample data")
        print("  quality               - Score sample data quality")


if __name__ == "__main__":
    asyncio.run(main())
