"""
Tests for the scraper agent.
"""
import pytest
import asyncio
from unittest.mock import patch, MagicMock

from agents.scraper import ScraperAgent
from models.task import Task, TaskType


@pytest.mark.asyncio
async def test_scraper_agent_initialization():
    """Test that the scraper agent initializes correctly."""
    agent = ScraperAgent(agent_id="test-scraper", coordinator_id="test-coordinator")

    assert agent.agent_id == "test-scraper"
    assert agent.agent_type == "scraper"
    assert agent.coordinator_id == "test-coordinator"
    assert len(agent.user_agents) > 0


@pytest.mark.asyncio
async def test_fetch_url_task():
    """Test that the scraper agent can execute a fetch_url task."""
    agent = ScraperAgent(agent_id="test-scraper", coordinator_id="test-coordinator")

    # Create a mock response
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.text = "<html><body><h1>Test Page</h1></body></html>"
    mock_response.headers = {"content-type": "text/html"}
    mock_response.encoding = "utf-8"
    mock_response.cookies = {}

    # Mock the HTTP client request method
    with patch.object(agent.client, "request", return_value=mock_response):
        task = Task(
            type=TaskType.FETCH_URL,
            parameters={"url": "https://example.com"}
        )

        result = await agent.execute_task(task)

        assert result["url"] == "https://example.com"
        assert result["status_code"] == 200
        assert result["content"] == "<html><body><h1>Test Page</h1></body></html>"
        assert result["content_type"] == "text/html"


@pytest.mark.asyncio
async def test_unsupported_task_type():
    """Test that the scraper agent raises an error for unsupported task types."""
    agent = ScraperAgent(agent_id="test-scraper", coordinator_id="test-coordinator")

    # Use a valid TaskType that's not supported by the scraper agent
    task = Task(
        type=TaskType.NLP_SENTIMENT_ANALYSIS,
        parameters={"text": "This is a test"}
    )

    with pytest.raises(ValueError, match="Unsupported task type"):
        await agent.execute_task(task)
