"""
Tests for the storage agent.
"""
import os
import json
import pytest
import asyncio
import tempfile
from unittest.mock import patch, MagicMock

from agents.storage import StorageAgent
from models.task import Task, TaskType


@pytest.mark.asyncio
async def test_storage_agent_initialization():
    """Test that the storage agent initializes correctly."""
    agent = StorageAgent(agent_id="test-storage", coordinator_id="test-coordinator")

    assert agent.agent_id == "test-storage"
    assert agent.agent_type == "storage"
    assert agent.coordinator_id == "test-coordinator"
    assert "json" in agent.supported_formats
    assert "csv" in agent.supported_formats
    assert "excel" in agent.supported_formats
    assert "sqlite" in agent.supported_formats


@pytest.mark.asyncio
async def test_store_data_json():
    """Test that the storage agent can store data in JSON format."""
    agent = StorageAgent(agent_id="test-storage", coordinator_id="test-coordinator")

    # Sample data
    data = [
        {"id": 1, "name": "Item 1", "price": 10.99},
        {"id": 2, "name": "Item 2", "price": 20.49},
        {"id": 3, "name": "Item 3", "price": 5.99}
    ]

    # Create a temporary file
    temp_file = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
    temp_path = temp_file.name
    temp_file.close()

    try:
        # Create task
        task = Task(
            type=TaskType.STORE_DATA,
            parameters={
                "data": data,
                "format": "json",
                "path": temp_path
            }
        )

        # Execute task
        result = await agent.execute_task(task)

        # Check result
        assert result["format"] == "json"
        assert result["path"] == temp_path
        assert result["records"] == 3
        assert not result["append"]

        # Check file content
        with open(temp_path, "r") as f:
            stored_data = json.load(f)

        assert len(stored_data) == 3
        assert stored_data[0]["id"] == 1
        assert stored_data[1]["name"] == "Item 2"
        assert stored_data[2]["price"] == 5.99

    finally:
        # Clean up
        if os.path.exists(temp_path):
            os.unlink(temp_path)


@pytest.mark.asyncio
async def test_store_data_append():
    """Test that the storage agent can append data to an existing file."""
    agent = StorageAgent(agent_id="test-storage", coordinator_id="test-coordinator")

    # Sample data
    initial_data = [
        {"id": 1, "name": "Item 1", "price": 10.99}
    ]

    append_data = [
        {"id": 2, "name": "Item 2", "price": 20.49},
        {"id": 3, "name": "Item 3", "price": 5.99}
    ]

    # Create a temporary file
    temp_file = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
    temp_path = temp_file.name
    temp_file.close()

    # Write initial data
    with open(temp_path, 'w') as f:
        json.dump(initial_data, f)

    try:
        # Create task to append data
        task = Task(
            type=TaskType.STORE_DATA,
            parameters={
                "data": append_data,
                "format": "json",
                "path": temp_path,
                "append": True
            }
        )

        # Execute task
        result = await agent.execute_task(task)

        # Check result
        assert result["format"] == "json"
        assert result["path"] == temp_path
        assert result["records"] == 2
        assert result["append"]
        assert result["total_records"] == 3

        # Check file content
        with open(temp_path, "r") as f:
            stored_data = json.load(f)

        assert len(stored_data) == 3
        assert stored_data[0]["id"] == 1
        assert stored_data[1]["id"] == 2
        assert stored_data[2]["id"] == 3

    finally:
        # Clean up
        if os.path.exists(temp_path):
            os.unlink(temp_path)


@pytest.mark.asyncio
async def test_unsupported_format():
    """Test that the storage agent raises an error for unsupported formats."""
    agent = StorageAgent(agent_id="test-storage", coordinator_id="test-coordinator")

    task = Task(
        type=TaskType.STORE_DATA,
        parameters={
            "data": [{"id": 1}],
            "format": "unsupported_format",
            "path": "test.dat"
        }
    )

    with pytest.raises(ValueError, match="Unsupported format"):
        await agent.execute_task(task)


@pytest.mark.asyncio
async def test_unsupported_task_type():
    """Test that the storage agent raises an error for unsupported task types."""
    agent = StorageAgent(agent_id="test-storage", coordinator_id="test-coordinator")

    # Use a valid TaskType that's not supported by the storage agent
    task = Task(
        type=TaskType.FETCH_URL,
        parameters={"url": "https://example.com"}
    )

    with pytest.raises(ValueError, match="Unsupported task type"):
        await agent.execute_task(task)
