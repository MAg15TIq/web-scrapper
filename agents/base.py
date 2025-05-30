"""
Base agent class for the web scraping system.
"""
import uuid
import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Callable, Awaitable, Union
from queue import Queue, PriorityQueue

from models.message import Message, TaskMessage, ResultMessage, ErrorMessage, StatusMessage, Priority
from models.task import Task, TaskStatus


class Agent(ABC):
    """
    Base class for all agents in the system.
    """
    def __init__(self, agent_id: Optional[str] = None, agent_type: str = "base", coordinator_id: str = "coordinator"):
        """
        Initialize a new agent.

        Args:
            agent_id: Unique identifier for the agent. If None, a UUID will be generated.
            agent_type: Type of the agent (e.g., "scraper", "parser", "storage").
            coordinator_id: ID of the coordinator agent. Used for message routing.
        """
        self.agent_id = agent_id or f"{agent_type}-{str(uuid.uuid4())[:8]}"
        self.agent_type = agent_type
        self.coordinator_id = coordinator_id
        self.inbox = PriorityQueue()  # Messages received from other agents
        self.outbox = Queue()  # Messages to be sent to other agents
        self.tasks = {}  # Dictionary of tasks assigned to this agent
        self.running = False
        self.logger = logging.getLogger(f"agent.{self.agent_type}.{self.agent_id}")
        self.message_handlers = {}  # Mapping of message types to handler functions
        self._register_default_handlers()

    def _register_default_handlers(self) -> None:
        """Register default message handlers."""
        self.register_handler("task", self._handle_task_message)

    def register_handler(self, message_type: str, handler: Callable[[Message], Awaitable[None]]) -> None:
        """
        Register a handler function for a specific message type.

        Args:
            message_type: Type of message to handle.
            handler: Async function that processes messages of the given type.
        """
        self.message_handlers[message_type] = handler

    async def start(self) -> None:
        """Start the agent's processing loop."""
        self.running = True
        self.logger.info(f"Agent {self.agent_id} started")

        # Create tasks for processing inbox and outbox
        inbox_task = asyncio.create_task(self._process_inbox())
        outbox_task = asyncio.create_task(self._process_outbox())

        # Wait for both tasks to complete (which should only happen when running is False)
        await asyncio.gather(inbox_task, outbox_task)

        self.logger.info(f"Agent {self.agent_id} stopped")

    async def stop(self) -> None:
        """Stop the agent's processing loop."""
        self.running = False
        self.logger.info(f"Agent {self.agent_id} stopping...")

    async def _process_inbox(self) -> None:
        """Process incoming messages from the inbox."""
        while self.running:
            try:
                # Get the next message with the highest priority
                if not self.inbox.empty():
                    _, message = self.inbox.get_nowait()
                    await self._handle_message(message)
                    self.inbox.task_done()
                else:
                    # If no messages, sleep briefly to avoid busy waiting
                    await asyncio.sleep(0.01)
            except Exception as e:
                self.logger.error(f"Error processing inbox: {str(e)}", exc_info=True)

    async def _process_outbox(self) -> None:
        """Process outgoing messages from the outbox."""
        while self.running:
            try:
                # Get the next message to send
                if not self.outbox.empty():
                    message = self.outbox.get_nowait()
                    await self._send_message(message)
                    self.outbox.task_done()
                else:
                    # If no messages, sleep briefly to avoid busy waiting
                    await asyncio.sleep(0.01)
            except Exception as e:
                self.logger.error(f"Error processing outbox: {str(e)}", exc_info=True)

    async def _handle_message(self, message: Message) -> None:
        """
        Handle an incoming message.

        Args:
            message: The message to handle.
        """
        self.logger.debug(f"Handling message: {message.type} from {message.sender_id}")

        # Call the appropriate handler based on message type
        if message.type in self.message_handlers:
            await self.message_handlers[message.type](message)
        else:
            self.logger.warning(f"No handler registered for message type: {message.type}")

    async def _send_message(self, message: Message) -> None:
        """
        Send a message to another agent.

        Args:
            message: The message to send.
        """
        if not message.recipient_id:
            self.logger.warning(f"Cannot send message without recipient_id: {message}")
            return

        # Default implementation logs the message
        # In a real implementation, this would be overridden to use a message broker
        self.logger.debug(f"Sending message to {message.recipient_id}: {message.type}")

        # If coordinator_id is defined, we can route through the coordinator
        if hasattr(self, 'coordinator_id') and self.coordinator_id:
            # This is a placeholder for actual message sending
            # In a real implementation, you would send the message to the coordinator
            # which would then route it to the appropriate agent
            self.logger.debug(f"Routing message through coordinator {self.coordinator_id}")
        else:
            self.logger.warning(f"No coordinator_id defined, cannot route message: {message}")

        # Note: Subclasses should override this method to implement actual message sending

    async def _handle_task_message(self, message: TaskMessage) -> None:
        """
        Handle a task message.

        Args:
            message: The task message to handle.
        """
        self.logger.info(f"Received task: {message.task_type}")

        # Create a task from the message
        task = Task(
            id=message.task_id,
            type=message.task_type,
            status=TaskStatus.ASSIGNED,
            assigned_to=self.agent_id,
            parameters=message.parameters,
            dependencies=message.dependencies
        )

        # Store the task
        self.tasks[task.id] = task

        try:
            # Execute the task
            task.update_status(TaskStatus.RUNNING)
            result = await self.execute_task(task)
            task.complete(result)

            # Send result message
            result_message = ResultMessage(
                sender_id=self.agent_id,
                recipient_id=message.sender_id,
                task_id=task.id,
                result=result
            )
            self.outbox.put(result_message)

        except Exception as e:
            # Handle task execution error
            self.logger.error(f"Error executing task {task.id}: {str(e)}", exc_info=True)
            task.fail(
                error_type=type(e).__name__,
                error_message=str(e),
                traceback=str(e.__traceback__)
            )

            # Send error message
            error_message = ErrorMessage(
                sender_id=self.agent_id,
                recipient_id=message.sender_id,
                task_id=task.id,
                error_type=type(e).__name__,
                error_message=str(e),
                traceback=str(e.__traceback__)
            )
            self.outbox.put(error_message)

    @abstractmethod
    async def execute_task(self, task: Task) -> Dict[str, Any]:
        """
        Execute a task assigned to this agent.

        Args:
            task: The task to execute.

        Returns:
            A dictionary containing the result of the task.
        """
        pass

    def send_status(self, recipient_id: str, status: str, details: Dict[str, Any] = None) -> None:
        """
        Send a status update to another agent.

        Args:
            recipient_id: ID of the agent to send the status to.
            status: Current status of this agent.
            details: Additional details about the status.
        """
        status_message = StatusMessage(
            sender_id=self.agent_id,
            recipient_id=recipient_id,
            status=status,
            details=details or {}
        )
        self.outbox.put(status_message)

    def receive_message(self, message: Message, priority: int = Priority.NORMAL) -> None:
        """
        Receive a message from another agent.

        Args:
            message: The message to receive.
            priority: Priority of the message (lower number = higher priority).
        """
        self.inbox.put((priority, message))
