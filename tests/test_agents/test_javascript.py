"""
Tests for the JavaScript agent.
"""
import pytest
import asyncio
from unittest.mock import patch, MagicMock, AsyncMock

from agents.javascript import JavaScriptAgent, BrowserPool
from models.task import Task, TaskType


@pytest.mark.asyncio
async def test_javascript_agent_initialization():
    """Test that the JavaScript agent initializes correctly."""
    agent = JavaScriptAgent(agent_id="test-javascript", coordinator_id="test-coordinator")

    assert agent.agent_id == "test-javascript"
    assert agent.agent_type == "javascript"
    assert agent.coordinator_id == "test-coordinator"
    assert isinstance(agent.browser_pool, BrowserPool)


@pytest.mark.asyncio
async def test_render_page_task():
    """Test that the JavaScript agent can execute a render_page task."""
    agent = JavaScriptAgent(agent_id="test-javascript", coordinator_id="test-coordinator")

    # Mock the browser pool
    mock_browser_pool = MagicMock()
    mock_browser_pool.get_browser = AsyncMock(return_value=("browser-1", MagicMock()))
    mock_browser_pool.create_context = AsyncMock(return_value="context-1")
    mock_browser_pool.create_page = AsyncMock(return_value="page-1")

    # Mock the page
    mock_page = AsyncMock()
    mock_page.goto = AsyncMock()
    mock_page.wait_for_selector = AsyncMock()
    mock_page.content = AsyncMock(return_value="<html><body><h1>Test Page</h1></body></html>")
    mock_page.title = AsyncMock(return_value="Test Page")
    mock_page.screenshot = AsyncMock()
    mock_page.url = "https://example.com"

    # Mock the response
    mock_response = MagicMock()
    mock_response.status = 200
    mock_response.headers = {"content-type": "text/html"}
    mock_page.goto.return_value = mock_response

    mock_browser_pool.get_page = AsyncMock(return_value=mock_page)

    # Replace the agent's browser pool with our mock
    agent.browser_pool = mock_browser_pool

    # Create a render_page task
    task = Task(
        type=TaskType.RENDER_PAGE,
        parameters={
            "url": "https://example.com",
            "wait_for": "h1",
            "timeout": 30
        }
    )

    # Execute the task
    result = await agent.execute_task(task)

    # Check that the browser pool methods were called
    mock_browser_pool.get_browser.assert_called_once()
    mock_browser_pool.create_context.assert_called_once()
    mock_browser_pool.create_page.assert_called_once()
    mock_browser_pool.get_page.assert_called_once_with("page-1")

    # Check that the page methods were called
    mock_page.goto.assert_called_once_with("https://example.com", wait_until="networkidle", timeout=30000)
    mock_page.wait_for_selector.assert_called_once_with("h1", timeout=30000)
    mock_page.content.assert_called_once()
    mock_page.title.assert_called_once()
    mock_page.screenshot.assert_called_once()

    # Check the result
    assert result["page_id"] == "page-1"
    assert result["context_id"] == "context-1"
    assert result["browser_id"] == "browser-1"
    assert result["url"] == "https://example.com"
    assert result["title"] == "Test Page"
    assert result["status_code"] == 200
    assert result["content"] == "<html><body><h1>Test Page</h1></body></html>"
    assert "screenshot_path" in result


@pytest.mark.asyncio
async def test_execute_script_task():
    """Test that the JavaScript agent can execute a execute_script task."""
    agent = JavaScriptAgent(agent_id="test-javascript", coordinator_id="test-coordinator")

    # Mock the browser pool
    mock_browser_pool = MagicMock()

    # Mock the page
    mock_page = AsyncMock()
    mock_page.evaluate = AsyncMock(return_value="Test Result")

    mock_browser_pool.get_page = AsyncMock(return_value=mock_page)

    # Replace the agent's browser pool with our mock
    agent.browser_pool = mock_browser_pool

    # Create an execute_script task
    task = Task(
        type=TaskType.EXECUTE_SCRIPT,
        parameters={
            "page_id": "page-1",
            "script": "return document.title;",
            "args": []
        }
    )

    # Execute the task
    result = await agent.execute_task(task)

    # Check that the browser pool methods were called
    mock_browser_pool.get_page.assert_called_once_with("page-1")

    # Check that the page methods were called
    mock_page.evaluate.assert_called_once_with("return document.title;", timeout=30000)

    # Check the result
    assert result["page_id"] == "page-1"
    assert result["result"] == "Test Result"
    assert result["script_length"] == len("return document.title;")


@pytest.mark.asyncio
async def test_unsupported_task_type():
    """Test that the JavaScript agent raises an error for unsupported task types."""
    agent = JavaScriptAgent(agent_id="test-javascript", coordinator_id="test-coordinator")

    # Use a valid TaskType that's not supported by the JavaScript agent
    task = Task(
        type=TaskType.STORE_DATA,
        parameters={"data": {"test": "data"}}
    )

    with pytest.raises(ValueError, match="Unsupported task type"):
        await agent.execute_task(task)


@pytest.mark.asyncio
async def test_cleanup():
    """Test that the JavaScript agent can clean up resources."""
    agent = JavaScriptAgent(agent_id="test-javascript", coordinator_id="test-coordinator")

    # Mock the browser pool
    mock_browser_pool = MagicMock()
    mock_browser_pool.close_all = AsyncMock()

    # Replace the agent's browser pool with our mock
    agent.browser_pool = mock_browser_pool

    # Call cleanup
    await agent.cleanup()

    # Check that close_all was called
    mock_browser_pool.close_all.assert_called_once()
