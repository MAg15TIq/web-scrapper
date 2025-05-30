"""
Command-line interface for the self-aware intelligent web scraping system.
"""
import asyncio
import logging
import sys
import json
import os
import argparse
from typing import Dict, Any, List, Optional, Union

from agents.master_intelligence import MasterIntelligenceAgent
from agents.url_intelligence import URLIntelligenceAgent
from agents.content_recognition import ContentRecognitionAgent
from agents.document_intelligence import DocumentIntelligenceAgent
from agents.performance_optimization import PerformanceOptimizationAgent
from agents.quality_assurance import QualityAssuranceAgent
from models.intelligence import ContentType, InputType, WebsiteType, ContentAnalysisResult
from models.task import Task, TaskType, TaskStatus


async def analyze_input(input_data: str, verbose: bool = False) -> Dict[str, Any]:
    """
    Analyze input data using the Master Intelligence Agent.

    Args:
        input_data: The input data to analyze.
        verbose: Whether to print verbose output.

    Returns:
        A dictionary containing the analysis results.
    """
    # Create Master Intelligence Agent
    master_agent = MasterIntelligenceAgent(agent_id="master_intelligence")

    # Start the agent
    await master_agent.start()

    try:
        # Analyze the input
        analysis_result = await master_agent.analyze_input(input_data)

        if verbose:
            print(f"Input analysis result: {analysis_result}")

        return analysis_result.dict()
    finally:
        # Stop the agent
        await master_agent.stop()


async def analyze_url(url: str, verbose: bool = False) -> Dict[str, Any]:
    """
    Analyze a URL using the URL Intelligence Agent.

    Args:
        url: The URL to analyze.
        verbose: Whether to print verbose output.

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

        if verbose:
            print(f"URL analysis result: {result}")

        return result
    finally:
        # Stop the agent
        await url_agent.stop()


async def recognize_content(content: Union[str, bytes], verbose: bool = False) -> Dict[str, Any]:
    """
    Recognize and analyze content using the Content Recognition Agent.

    Args:
        content: The content to analyze.
        verbose: Whether to print verbose output.

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

        if verbose:
            print(f"Content recognition result: {result}")

        return result
    finally:
        # Stop the agent
        await content_agent.stop()


async def process_document(document_path: str, document_type: Optional[str] = None, verbose: bool = False) -> Dict[str, Any]:
    """
    Process a document using the Document Intelligence Agent.

    Args:
        document_path: The path to the document.
        document_type: The document type if known. If None, it will be detected.
        verbose: Whether to print verbose output.

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

        if verbose:
            print(f"Document processing result: {result}")

        return result
    finally:
        # Stop the agent
        await document_agent.stop()


async def extract_text_from_document(document_path: str, document_type: Optional[str] = None, verbose: bool = False) -> Dict[str, Any]:
    """
    Extract text from a document using the Document Intelligence Agent.

    Args:
        document_path: The path to the document.
        document_type: The document type if known. If None, it will be detected.
        verbose: Whether to print verbose output.

    Returns:
        A dictionary containing the extracted text.
    """
    # Create Document Intelligence Agent
    document_agent = DocumentIntelligenceAgent(agent_id="document_intelligence")

    # Start the agent
    await document_agent.start()

    try:
        # Create a task for text extraction
        task = Task(
            type=TaskType.EXTRACT_TEXT,
            parameters={
                "document_data": document_path,
                "document_type": document_type
            }
        )

        # Execute the task
        result = await document_agent.execute_task(task)

        if verbose:
            print(f"Text extraction result: {result}")

        return result
    finally:
        # Stop the agent
        await document_agent.stop()


async def extract_tables_from_document(document_path: str, document_type: Optional[str] = None, verbose: bool = False) -> Dict[str, Any]:
    """
    Extract tables from a document using the Document Intelligence Agent.

    Args:
        document_path: The path to the document.
        document_type: The document type if known. If None, it will be detected.
        verbose: Whether to print verbose output.

    Returns:
        A dictionary containing the extracted tables.
    """
    # Create Document Intelligence Agent
    document_agent = DocumentIntelligenceAgent(agent_id="document_intelligence")

    # Start the agent
    await document_agent.start()

    try:
        # Create a task for table extraction
        task = Task(
            type=TaskType.EXTRACT_TABLES,
            parameters={
                "document_data": document_path,
                "document_type": document_type
            }
        )

        # Execute the task
        result = await document_agent.execute_task(task)

        if verbose:
            print(f"Table extraction result: {result}")

        return result
    finally:
        # Stop the agent
        await document_agent.stop()


async def optimize_performance(verbose: bool = False) -> Dict[str, Any]:
    """
    Optimize system performance using the Performance Optimization Agent.

    Args:
        verbose: Whether to print verbose output.

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

        if verbose:
            print(f"Performance optimization result: {result}")

        return result
    finally:
        # Stop the agent
        await performance_agent.stop()


async def validate_data(data: Union[Dict[str, Any], List[Dict[str, Any]]], schema: Dict[str, Any], verbose: bool = False) -> Dict[str, Any]:
    """
    Validate data using the Quality Assurance Agent.

    Args:
        data: The data to validate. Can be a single object or a list of objects.
        schema: The schema to validate against.
        verbose: Whether to print verbose output.

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

        if verbose:
            print(f"Data validation result: {result}")

        return result
    finally:
        # Stop the agent
        await quality_agent.stop()


async def score_data_quality(data: Union[Dict[str, Any], List[Dict[str, Any]]], schema: Dict[str, Any], quality_thresholds: Optional[Dict[str, float]] = None, verbose: bool = False) -> Dict[str, Any]:
    """
    Score data quality using the Quality Assurance Agent.

    Args:
        data: The data to score. Can be a single object or a list of objects.
        schema: The schema to validate against.
        quality_thresholds: Optional quality thresholds.
        verbose: Whether to print verbose output.

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
                "quality_thresholds": quality_thresholds
            }
        )

        # Execute the task
        result = await quality_agent.execute_task(task)

        if verbose:
            print(f"Data quality score result: {result}")

        return result
    finally:
        # Stop the agent
        await quality_agent.stop()


async def intelligent_scraping(input_data: str, verbose: bool = False) -> Dict[str, Any]:
    """
    Perform intelligent scraping using the Master Intelligence Agent.

    Args:
        input_data: The input data to analyze. Can be a URL, file path, or raw content.
        verbose: Whether to print verbose output.

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

        if verbose:
            print(f"Input analysis result: {analysis_result}")

        # Select appropriate agents
        agent_assignments = await master_agent.select_agents(analysis_result)

        if verbose:
            print(f"Selected agents: {agent_assignments}")

        # Process the input based on its type
        if analysis_result.input_type == InputType.URL:
            # Analyze URL
            url_result = await analyze_url(input_data, verbose)

            # Process based on content type
            if analysis_result.content_type == ContentType.HTML:
                # Fetch the content
                # In a real implementation, you would use a scraper agent to fetch the content
                # For this example, we'll just use a placeholder
                content = "<html><body><h1>Example Content</h1></body></html>"

                # Recognize content
                content_result = await recognize_content(content, verbose)

                return {
                    "analysis": analysis_result.dict(),
                    "url_analysis": url_result,
                    "content_analysis": content_result
                }

            elif analysis_result.content_type in [ContentType.PDF, ContentType.DOC, ContentType.DOCX]:
                # Process document
                document_result = await process_document(input_data, analysis_result.content_type.value, verbose)

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
            document_result = await process_document(input_data, None, verbose)

            return {
                "analysis": analysis_result.dict(),
                "document_analysis": document_result
            }

        else:
            # Recognize content
            content_result = await recognize_content(input_data, verbose)

            return {
                "analysis": analysis_result.dict(),
                "content_analysis": content_result
            }

    finally:
        # Stop the agent
        await master_agent.stop()


async def main():
    """Main function."""
    # Configure argument parser
    parser = argparse.ArgumentParser(description="Self-aware intelligent web scraping system")

    # Add subparsers for different commands
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # Analyze input command
    analyze_parser = subparsers.add_parser("analyze", help="Analyze input data")
    analyze_parser.add_argument("input", help="Input data to analyze")

    # Analyze URL command
    url_parser = subparsers.add_parser("url", help="Analyze a URL")
    url_parser.add_argument("url", help="URL to analyze")

    # Process document command
    document_parser = subparsers.add_parser("document", help="Process a document")
    document_parser.add_argument("path", help="Path to the document")
    document_parser.add_argument("--type", help="Document type (pdf, docx, xlsx, etc.)")

    # Extract text command
    text_parser = subparsers.add_parser("text", help="Extract text from a document")
    text_parser.add_argument("path", help="Path to the document")
    text_parser.add_argument("--type", help="Document type (pdf, docx, xlsx, etc.)")

    # Extract tables command
    tables_parser = subparsers.add_parser("tables", help="Extract tables from a document")
    tables_parser.add_argument("path", help="Path to the document")
    tables_parser.add_argument("--type", help="Document type (pdf, docx, xlsx, etc.)")

    # Intelligent scraping command
    scrape_parser = subparsers.add_parser("scrape", help="Perform intelligent scraping")
    scrape_parser.add_argument("input", help="Input data to scrape")

    # Optimize performance command
    optimize_parser = subparsers.add_parser("optimize", help="Optimize system performance")

    # Validate data command
    validate_parser = subparsers.add_parser("validate", help="Validate data against a schema")
    validate_parser.add_argument("--data", help="JSON data to validate (as string or file path)")
    validate_parser.add_argument("--schema", help="JSON schema to validate against (as string or file path)")

    # Score data quality command
    quality_parser = subparsers.add_parser("quality", help="Score data quality")
    quality_parser.add_argument("--data", help="JSON data to score (as string or file path)")
    quality_parser.add_argument("--schema", help="JSON schema to validate against (as string or file path)")
    quality_parser.add_argument("--thresholds", help="JSON quality thresholds (as string or file path)")

    # Add common arguments
    for p in [analyze_parser, url_parser, document_parser, text_parser, tables_parser, scrape_parser, optimize_parser, validate_parser, quality_parser]:
        p.add_argument("-v", "--verbose", action="store_true", help="Print verbose output")
        p.add_argument("-o", "--output", help="Output file path")

    # Parse arguments
    args = parser.parse_args()

    # Configure logging
    log_level = logging.INFO if args.verbose else logging.WARNING
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)]
    )

    # Execute command
    result = None

    if args.command == "analyze":
        result = await analyze_input(args.input, args.verbose)

    elif args.command == "url":
        result = await analyze_url(args.url, args.verbose)

    elif args.command == "document":
        result = await process_document(args.path, args.type, args.verbose)

    elif args.command == "text":
        result = await extract_text_from_document(args.path, args.type, args.verbose)

    elif args.command == "tables":
        result = await extract_tables_from_document(args.path, args.type, args.verbose)

    elif args.command == "scrape":
        result = await intelligent_scraping(args.input, args.verbose)

    elif args.command == "optimize":
        result = await optimize_performance(args.verbose)

    elif args.command == "validate":
        # Load data and schema
        data = None
        schema = None

        if args.data:
            # Check if it's a file path or a JSON string
            if os.path.exists(args.data):
                with open(args.data, "r") as f:
                    data = json.load(f)
            else:
                try:
                    data = json.loads(args.data)
                except json.JSONDecodeError:
                    print(f"Error: Invalid JSON data: {args.data}")
                    return
        else:
            # Use sample data
            data = [
                {"title": "Product 1", "price": 10.99, "description": "This is product 1"},
                {"title": "Product 2", "price": 20.99, "description": None},
                {"title": None, "price": 30.99, "description": "This is product 3"}
            ]

        if args.schema:
            # Check if it's a file path or a JSON string
            if os.path.exists(args.schema):
                with open(args.schema, "r") as f:
                    schema = json.load(f)
            else:
                try:
                    schema = json.loads(args.schema)
                except json.JSONDecodeError:
                    print(f"Error: Invalid JSON schema: {args.schema}")
                    return
        else:
            # Use sample schema
            schema = {
                "title": {"type": "string", "required": True},
                "price": {"type": "number", "required": True, "minimum": 0},
                "description": {"type": "string", "required": False}
            }

        result = await validate_data(data, schema, args.verbose)

    elif args.command == "quality":
        # Load data, schema, and thresholds
        data = None
        schema = None
        thresholds = None

        if args.data:
            # Check if it's a file path or a JSON string
            if os.path.exists(args.data):
                with open(args.data, "r") as f:
                    data = json.load(f)
            else:
                try:
                    data = json.loads(args.data)
                except json.JSONDecodeError:
                    print(f"Error: Invalid JSON data: {args.data}")
                    return
        else:
            # Use sample data
            data = [
                {"title": "Product 1", "price": 10.99, "description": "This is product 1"},
                {"title": "Product 2", "price": 20.99, "description": None},
                {"title": None, "price": 30.99, "description": "This is product 3"}
            ]

        if args.schema:
            # Check if it's a file path or a JSON string
            if os.path.exists(args.schema):
                with open(args.schema, "r") as f:
                    schema = json.load(f)
            else:
                try:
                    schema = json.loads(args.schema)
                except json.JSONDecodeError:
                    print(f"Error: Invalid JSON schema: {args.schema}")
                    return
        else:
            # Use sample schema
            schema = {
                "title": {"type": "string", "required": True},
                "price": {"type": "number", "required": True, "minimum": 0},
                "description": {"type": "string", "required": False}
            }

        if args.thresholds:
            # Check if it's a file path or a JSON string
            if os.path.exists(args.thresholds):
                with open(args.thresholds, "r") as f:
                    thresholds = json.load(f)
            else:
                try:
                    thresholds = json.loads(args.thresholds)
                except json.JSONDecodeError:
                    print(f"Error: Invalid JSON thresholds: {args.thresholds}")
                    return
        else:
            # Use default thresholds
            thresholds = {
                "completeness": 0.9,
                "accuracy": 0.95,
                "consistency": 0.9
            }

        result = await score_data_quality(data, schema, thresholds, args.verbose)

    else:
        parser.print_help()
        return

    # Print or save result
    if result:
        if args.output:
            with open(args.output, "w") as f:
                json.dump(result, f, indent=2)
            print(f"Result saved to {args.output}")
        else:
            print(json.dumps(result, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
