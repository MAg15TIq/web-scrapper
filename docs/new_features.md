# New Features in the Web Scraping System

This document describes the new features added to the web scraping system.

## New Agent Types

### Anti-Detection Agent

The Anti-Detection Agent focuses on evading anti-scraping measures implemented by websites. It provides the following capabilities:

- **Browser Fingerprint Randomization**: Generates realistic browser fingerprints to avoid detection.
- **Request Pattern Variation**: Adjusts request patterns to appear more human-like.
- **Blocking Detection**: Identifies when a website is blocking scraping attempts.
- **Adaptive Request Optimization**: Dynamically adjusts request frequency based on website responses.

#### Usage Example

```python
from agents.anti_detection import AntiDetectionAgent
from models.task import Task, TaskType

# Create and register the agent
anti_detection = AntiDetectionAgent(coordinator_id=coordinator.agent_id)
coordinator.register_agent(anti_detection)

# Generate a browser fingerprint
fingerprint_task = Task(
    type=TaskType.GENERATE_FINGERPRINT,
    parameters={
        "domain": "example.com",
        "consistent": True
    }
)
fingerprint_task_id = await coordinator.submit_task(fingerprint_task)

# Check if a site is blocking scraping
blocking_task = Task(
    type=TaskType.CHECK_BLOCKING,
    parameters={
        "url": "https://example.com",
        "check_methods": ["status_code", "content_analysis", "redirect"]
    }
)
blocking_task_id = await coordinator.submit_task(blocking_task)

# Optimize request pattern
optimize_task = Task(
    type=TaskType.OPTIMIZE_REQUEST_PATTERN,
    parameters={
        "domain": "example.com",
        "aggressive": False
    }
)
optimize_task_id = await coordinator.submit_task(optimize_task)
```

### Data Transformation Agent

The Data Transformation Agent handles post-processing of scraped data. It provides the following capabilities:

- **Data Cleaning and Normalization**: Cleans and normalizes scraped data.
- **Schema Transformation**: Transforms data from one schema to another.
- **Entity Extraction and Linking**: Extracts entities from text data.
- **Text Analysis**: Performs sentiment analysis, keyword extraction, and more.

#### Usage Example

```python
from agents.data_transformation import DataTransformationAgent
from models.task import Task, TaskType

# Create and register the agent
data_transformation = DataTransformationAgent(coordinator_id=coordinator.agent_id)
coordinator.register_agent(data_transformation)

# Clean and normalize data
clean_task = Task(
    type=TaskType.CLEAN_DATA,
    parameters={
        "data": [{"title": " Example Title ", "price": "$10.99"}],
        "operations": [
            {"field": "title", "operation": "strip_whitespace"},
            {"field": "price", "operation": "extract_number"}
        ],
        "add_metadata": True
    }
)
clean_task_id = await coordinator.submit_task(clean_task)

# Transform data schema
transform_task = Task(
    type=TaskType.TRANSFORM_SCHEMA,
    parameters={
        "data": [{"title": "Example", "price": 10.99}],
        "mapping": {
            "name": "title",
            "cost": "price"
        }
    }
)
transform_task_id = await coordinator.submit_task(transform_task)

# Analyze text
analyze_task = Task(
    type=TaskType.ANALYZE_TEXT,
    parameters={
        "text": "This is a sample product description. It's very good quality.",
        "analyses": ["sentiment", "entities", "keywords", "language"]
    }
)
analyze_task_id = await coordinator.submit_task(analyze_task)
```

## Enhanced CLI Interface

The CLI interface has been enhanced with new commands for the new agent types:

- `fingerprint`: Generate a browser fingerprint for a domain.
- `check-blocking`: Check if a site is blocking scraping.
- `analyze-text`: Analyze text content.
- `clean-data`: Clean and normalize data from a file.

The `scrape` command has also been enhanced with new options:

- `--anti-detection` / `-a`: Use anti-detection measures.
- `--clean-data` / `-d`: Clean and normalize data before storing.

## Improved Error Handling and Resilience

### Circuit Breaker

The circuit breaker pattern has been implemented to prevent repeated failures. It automatically stops requests to failing websites and allows them to recover.

```python
from utils.circuit_breaker import registry as circuit_breaker_registry

# Get or create a circuit breaker
circuit_breaker = circuit_breaker_registry.get_or_create(
    name="example.com",
    failure_threshold=5,
    recovery_timeout=60.0
)

# Execute a function with circuit breaker protection
try:
    result = await circuit_breaker.execute(some_async_function, arg1, arg2)
except Exception as e:
    print(f"Operation failed: {str(e)}")
```

### Adaptive Rate Limiter

The adaptive rate limiter dynamically adjusts request rates based on website responses.

```python
from utils.adaptive_rate_limiter import registry as rate_limiter_registry

# Get or create a rate limiter
rate_limiter = rate_limiter_registry.get_or_create(
    domain="example.com",
    initial_rate=1.0,  # 1 request per second
    min_rate=0.1,      # 1 request per 10 seconds
    max_rate=5.0       # 5 requests per second
)

# Wait for permission to make a request
await rate_limiter.acquire()

# Make the request
try:
    start_time = time.time()
    response = await make_request()
    response_time = time.time() - start_time

    # Update the rate limiter with the result
    rate_limiter.update(success=True, response_time=response_time)
except Exception as e:
    # Update the rate limiter with the failure
    rate_limiter.update(success=False)
    raise
```

## Real-Time Monitoring System

The system now includes a real-time monitoring dashboard that displays:

- Scraping progress with visual progress bars
- Active agent count and failed task tracking
- Data collection statistics
- Throughput metrics (requests per second)

The dashboard is built using the `rich` library for advanced terminal visualizations.

```python
from monitoring.dashboard import dashboard_manager

# Start the dashboard
await dashboard_manager.start()

# Update dashboard with statistics
dashboard_manager.update_stats({
    "scraping_progress": 0.75,  # 75% complete
    "active_agents": 142,
    "failed_tasks": 3,
    "data_collected": 1200000,  # 1.2M rows
    "throughput": 45.0,         # 45 reqs/sec
    "agent_stats": {
        "agent-1": {"status": "idle", "tasks": 0},
        "agent-2": {"status": "busy", "tasks": 2}
    },
    "domain_stats": {
        "example.com": {
            "request_count": 150,
            "current_rate": 2.5,
            "success_rate": 0.98
        }
    }
})

# Stop the dashboard when done
await dashboard_manager.stop()
```

## Advanced Pagination Handling Engine

The system now includes a hybrid pagination detection engine that combines:

- URL pattern analysis (detecting `page={n}` patterns)
- DOM button text classification (detecting "Next ›" buttons)
- JavaScript event listener inspection

```python
from utils.pagination_engine import pagination_engine

# Detect pagination method
pagination_info = await pagination_engine.detect_pagination(url, html_content)

# Get the next URL
next_url = await pagination_engine.get_next_url(url, html_content)

# Handle pagination with browser support if needed
pagination_result = await pagination_engine.handle_pagination(
    url,
    html_content,
    browser_available=True
)

# Use the pagination engine through the scraper agent
pagination_task = Task(
    type=TaskType.FOLLOW_PAGINATION,
    parameters={
        "url": "https://example.com/products",
        "max_pages": 5,
        "wait_between_requests": 2.0,
        "use_browser": True  # Enable browser support for JavaScript pagination
    }
)
```

## Distributed Result Aggregation

The system now includes a distributed result aggregation system using Apache Arrow for efficient data merging with:

- Timestamp-based versioning
- Source authority scoring
- Semantic similarity analysis

```python
from utils.result_aggregator import result_aggregator

# Merge multiple result chunks
merged_df = result_aggregator.merge_results(chunks)

# Resolve conflicts in the merged results
resolved_df = result_aggregator.resolve_conflicts(
    df=merged_df,
    id_columns=["id", "url"],
    timestamp_column="scraped_at",
    source_column="source",
    source_authority={"primary": 1.0, "secondary": 0.5}
)

# Deduplicate by similarity
deduplicated_df = result_aggregator.deduplicate_by_similarity(
    df=resolved_df,
    threshold=0.8,
    text_columns=["title", "description"]
)

# Use the result aggregator through the storage agent
aggregate_task = Task(
    type=TaskType.AGGREGATE_RESULTS,
    parameters={
        "chunks": chunks,
        "id_columns": ["id", "url"],
        "timestamp_column": "scraped_at",
        "source_column": "source",
        "source_authority": {"primary": 1.0, "secondary": 0.5},
        "similarity_threshold": 0.8,
        "text_columns": ["title", "description"],
        "output_format": "records",
        "path": "output/aggregated_data.json",
        "format": "json"
    }
)
```

## Example Scripts

New example scripts have been added to demonstrate the new features:

- `examples/advanced_scrape.py`: Demonstrates the use of the Anti-Detection Agent and Data Transformation Agent.
- `examples/advanced_features.py`: Demonstrates the Real-Time Monitoring Dashboard, Advanced Pagination Handling Engine, and Distributed Result Aggregation.

Run the examples:

```bash
python examples/advanced_scrape.py
python examples/advanced_features.py
```
