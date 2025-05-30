"""
Tests for the parser agent.
"""
import pytest
import asyncio
from unittest.mock import patch, MagicMock

from agents.parser import ParserAgent
from models.task import Task, TaskType


@pytest.mark.asyncio
async def test_parser_agent_initialization():
    """Test that the parser agent initializes correctly."""
    agent = ParserAgent(agent_id="test-parser", coordinator_id="test-coordinator")

    assert agent.agent_id == "test-parser"
    assert agent.agent_type == "parser"
    assert agent.coordinator_id == "test-coordinator"


@pytest.mark.asyncio
async def test_parse_content_task():
    """Test that the parser agent can execute a parse_content task."""
    agent = ParserAgent(agent_id="test-parser", coordinator_id="test-coordinator")

    # Sample HTML content
    html_content = """
    <html>
        <body>
            <h1 class="title">Test Page</h1>
            <div class="content">
                <p>This is a test paragraph.</p>
                <span class="price">$10.99</span>
            </div>
        </body>
    </html>
    """

    # Create selectors
    selectors = {
        "title": "h1.title",
        "price": "span.price",
        "content": "div.content p"
    }

    task = Task(
        type=TaskType.PARSE_CONTENT,
        parameters={
            "content": html_content,
            "selectors": selectors,
            "normalize": True
        }
    )

    result = await agent.execute_task(task)

    assert "extracted_data" in result
    assert result["extracted_data"]["title"] == "Test Page"
    assert result["extracted_data"]["price"] == "$10.99"
    assert result["extracted_data"]["content"] == "This is a test paragraph."
    assert result["num_fields"] == 3
    assert result["selectors_used"] == selectors


@pytest.mark.asyncio
async def test_extract_links_task():
    """Test that the parser agent can execute an extract_links task."""
    agent = ParserAgent(agent_id="test-parser", coordinator_id="test-coordinator")

    # Sample HTML content with links
    html_content = """
    <html>
        <body>
            <a href="https://example.com/page1">Page 1</a>
            <a href="https://example.com/page2">Page 2</a>
            <a href="/relative-page">Relative Page</a>
            <a href="https://other-domain.com/page">Other Domain</a>
        </body>
    </html>
    """

    task = Task(
        type=TaskType.EXTRACT_LINKS,
        parameters={
            "content": html_content,
            "base_url": "https://example.com",
            "selector": "a",
            "attribute": "href"
        }
    )

    result = await agent.execute_task(task)

    assert "links" in result
    assert len(result["links"]) == 4
    assert "https://example.com/page1" in result["links"]
    assert "https://example.com/page2" in result["links"]
    assert "https://example.com/relative-page" in result["links"]
    assert "https://other-domain.com/page" in result["links"]


@pytest.mark.asyncio
async def test_unsupported_task_type():
    """Test that the parser agent raises an error for unsupported task types."""
    agent = ParserAgent(agent_id="test-parser", coordinator_id="test-coordinator")

    # Use a valid TaskType that's not supported by the parser agent
    task = Task(
        type=TaskType.STORE_DATA,
        parameters={"data": {"test": "data"}}
    )

    with pytest.raises(ValueError, match="Unsupported task type"):
        await agent.execute_task(task)
