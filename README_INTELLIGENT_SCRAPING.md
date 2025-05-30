# Self-Aware Intelligent Web Scraping System

This is a self-aware intelligent web scraping system that can automatically analyze inputs, detect content types, and select the most appropriate agents for each task.

## Overview

The system uses a Master Intelligence Agent (MIA) that orchestrates all operations with self-awareness capabilities. It analyzes inputs (URLs, documents, raw content) and selects the most appropriate specialized agents for each task.

### Key Components

1. **Master Intelligence Agent (MIA)**
   - Central decision-making unit
   - Analyzes inputs and selects appropriate agents
   - Orchestrates the entire scraping process

2. **URL Intelligence Agent**
   - Analyzes URLs and websites
   - Detects technologies used on websites
   - Checks robots.txt and site structure

3. **Content Recognition Agent**
   - Identifies and categorizes content types
   - Analyzes content structure
   - Extracts metadata from content

4. **Document Intelligence Agent**
   - Handles all document types (PDF, DOC, DOCX, XLS, etc.)
   - Extracts text, tables, and metadata from documents
   - Processes different document formats

5. **Performance Optimization Agent**
   - Continuously monitors system performance
   - Detects bottlenecks and optimizes resource usage
   - Provides recommendations for performance improvements

6. **Quality Assurance Agent**
   - Validates data against schemas
   - Checks data completeness and consistency
   - Scores data quality and provides improvement recommendations

## Installation

### Prerequisites

- Python 3.8 or higher
- Required Python packages:
  - httpx
  - beautifulsoup4
  - PyPDF2 (for PDF processing)
  - python-docx (for DOCX processing)
  - openpyxl (for Excel processing)
  - pandas (for data processing)

### Install Dependencies

```bash
pip install httpx beautifulsoup4 PyPDF2 python-docx openpyxl pandas
```

## Usage

### Command-Line Interface

The system provides a command-line interface for easy interaction:

```bash
# Analyze input data
python cli/intelligent_cli.py analyze "https://example.com"

# Analyze a URL
python cli/intelligent_cli.py url "https://example.com"

# Process a document
python cli/intelligent_cli.py document "path/to/document.pdf" --type pdf

# Extract text from a document
python cli/intelligent_cli.py text "path/to/document.pdf" --type pdf

# Extract tables from a document
python cli/intelligent_cli.py tables "path/to/document.xlsx" --type xlsx

# Perform intelligent scraping
python cli/intelligent_cli.py scrape "https://example.com"

# Optimize system performance
python cli/intelligent_cli.py optimize

# Validate data against a schema
python cli/intelligent_cli.py validate --data "path/to/data.json" --schema "path/to/schema.json"

# Score data quality
python cli/intelligent_cli.py quality --data "path/to/data.json" --schema "path/to/schema.json"
```

### Example Script

You can also use the example script to see how the system works:

```bash
# Perform intelligent scraping
python examples/intelligent_scraping.py scrape "https://example.com"

# Optimize system performance
python examples/intelligent_scraping.py optimize

# Validate sample data
python examples/intelligent_scraping.py validate

# Score sample data quality
python examples/intelligent_scraping.py quality
```

### Python API

You can use the system programmatically in your Python code:

```python
import asyncio
from examples.intelligent_scraping import (
    intelligent_scraping,
    optimize_performance,
    validate_data,
    score_data_quality
)

async def main():
    # Perform intelligent scraping
    scraping_result = await intelligent_scraping("https://example.com")
    print("Scraping result:", scraping_result)

    # Optimize system performance
    optimization_result = await optimize_performance()
    print("Optimization result:", optimization_result)

    # Sample data and schema
    data = [
        {"title": "Product 1", "price": 10.99, "description": "This is product 1"},
        {"title": "Product 2", "price": 20.99, "description": None},
        {"title": None, "price": 30.99, "description": "This is product 3"}
    ]

    schema = {
        "title": {"type": "string", "required": True},
        "price": {"type": "number", "required": True, "minimum": 0},
        "description": {"type": "string", "required": False}
    }

    # Validate data
    validation_result = await validate_data(data, schema)
    print("Validation result:", validation_result)

    # Score data quality
    quality_result = await score_data_quality(data, schema)
    print("Quality result:", quality_result)

asyncio.run(main())
```

## Features

### Input Analysis

The system can analyze various types of inputs:

- URLs
- File paths
- Raw content (HTML, JSON, XML, etc.)
- Document files (PDF, DOC, DOCX, XLS, etc.)

### Content Recognition

The system can recognize and categorize content types:

- HTML
- JSON
- XML
- PDF
- DOC/DOCX
- XLS/XLSX
- CSV
- TXT
- Images
- Videos
- Audio

### URL Intelligence

The system can analyze URLs and websites:

- Detect website type (e-commerce, news, blog, etc.)
- Detect technologies used (WordPress, React, Angular, etc.)
- Check for JavaScript requirements
- Check for authentication requirements
- Check for anti-bot measures
- Detect pagination
- Analyze robots.txt

### Document Processing

The system can process various document types:

- Extract text from PDF, DOC, DOCX
- Extract tables from PDF, XLSX, CSV
- Extract metadata from documents
- Analyze document structure

### Agent Selection

The system can select the most appropriate agents for each task based on:

- Content type
- Input type
- Website type
- Complexity
- Required capabilities

### Performance Optimization

The system can optimize performance:

- Monitor system resource usage (CPU, memory, disk)
- Detect bottlenecks and performance issues
- Provide recommendations for performance improvements
- Automatically adjust resource allocation
- Track agent and task performance metrics

### Quality Assurance

The system can ensure high-quality output:

- Validate data against schemas
- Check data completeness and consistency
- Score data quality on multiple dimensions
- Detect and report data issues
- Provide recommendations for data quality improvements

## Architecture

The system uses a multi-agent architecture with specialized agents for different tasks:

1. **Master Intelligence Agent (MIA)**
   - Analyzes inputs
   - Selects appropriate agents
   - Orchestrates the scraping process

2. **Specialized Agents**
   - URL Intelligence Agent
   - Content Recognition Agent
   - Document Intelligence Agent
   - (More agents can be added as needed)

3. **Core Agents**
   - Coordinator Agent
   - Scraper Agent
   - Parser Agent
   - Storage Agent

## Extending the System

You can extend the system by adding new specialized agents or enhancing existing ones:

1. Create a new agent class that inherits from `Agent`
2. Implement the required methods (`execute_task`, message handlers, etc.)
3. Register the agent with the Master Intelligence Agent

## License

This project is licensed under the MIT License - see the LICENSE file for details.
