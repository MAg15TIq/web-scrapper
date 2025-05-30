"""
Coordinator agent for the web scraping system.
"""
import asyncio
import logging
from typing import Dict, Any, List, Optional, Set, Tuple
from queue import PriorityQueue

from agents.base import Agent
from models.message import Message, TaskMessage, ResultMessage, ErrorMessage, StatusMessage, ControlMessage, Priority
from models.task import Task, TaskStatus, TaskType


class CoordinatorAgent(Agent):
    """
    Coordinator agent that manages all other agents in the system.
    """
    def __init__(self, agent_id: str = "coordinator"):
        """
        Initialize a new coordinator agent.

        Args:
            agent_id: Unique identifier for the agent.
        """
        # Coordinator doesn't need a coordinator_id since it is the coordinator
        super().__init__(agent_id=agent_id, agent_type="coordinator", coordinator_id=None)

        # Dictionary of registered agents by ID
        self.agents: Dict[str, Dict[str, Any]] = {}

        # Dictionary of registered agents by type
        self.agents_by_type: Dict[str, List[str]] = {}

        # Task queue for pending tasks
        self.task_queue = PriorityQueue()

        # Dictionary of all tasks in the system
        self.all_tasks: Dict[str, Task] = {}

        # Set of task IDs that are waiting for dependencies
        self.waiting_tasks: Set[str] = set()

        # Register additional message handlers
        self.register_handler("result", self._handle_result_message)
        self.register_handler("error", self._handle_error_message)
        self.register_handler("status", self._handle_status_message)

    def register_agent(self, agent: Agent) -> None:
        """
        Register an agent with the coordinator.

        Args:
            agent: The agent to register.
        """
        self.logger.info(f"Registering agent: {agent.agent_id} (type: {agent.agent_type})")

        # Store agent information
        self.agents[agent.agent_id] = {
            "id": agent.agent_id,
            "type": agent.agent_type,
            "status": "idle",
            "tasks": 0,
            "last_seen": 0.0
        }

        # Add agent to type dictionary
        if agent.agent_type not in self.agents_by_type:
            self.agents_by_type[agent.agent_type] = []
        self.agents_by_type[agent.agent_type].append(agent.agent_id)

    async def _send_message(self, message: Message) -> None:
        """
        Send a message to another agent.

        Args:
            message: The message to send.
        """
        if not message.recipient_id:
            self.logger.warning(f"Cannot send message without recipient_id: {message}")
            return

        if message.recipient_id not in self.agents and message.recipient_id != self.agent_id:
            self.logger.warning(f"Unknown recipient: {message.recipient_id}")
            return

        # In a real implementation, this would use a message broker or direct connection
        self.logger.debug(f"Sending message to {message.recipient_id}: {message.type}")

        # If the message is for the coordinator itself, handle it directly
        if message.recipient_id == self.agent_id:
            self.receive_message(message)
            return

        # For registered agents, route the message
        # In a real implementation, you would use a message broker
        # For now, we'll use a placeholder that simulates message delivery
        # In a production system, this would be replaced with actual message delivery

        # This is where you would integrate with a message broker like RabbitMQ, Kafka, etc.
        # For example: await self.message_broker.publish(message)

        # For demonstration purposes, we'll log that the message was sent
        self.logger.info(f"Message {message.type} from {message.sender_id} routed to {message.recipient_id}")

    async def execute_task(self, task: Task) -> Dict[str, Any]:
        """
        Execute a task assigned to the coordinator.

        Args:
            task: The task to execute.

        Returns:
            A dictionary containing the result of the task.
        """
        # The coordinator doesn't execute tasks directly, it delegates them
        # This method should not be called on the coordinator
        raise NotImplementedError("Coordinator does not execute tasks directly")

    async def submit_task(self, task: Task) -> str:
        """
        Submit a new task to the system.

        Args:
            task: The task to submit.

        Returns:
            The ID of the submitted task.
        """
        self.logger.info(f"Submitting task: {task.id} (type: {task.type})")

        # Store the task
        self.all_tasks[task.id] = task

        # Check if the task has dependencies
        if task.dependencies:
            # Check if all dependencies are completed
            all_completed = True
            for dep_id in task.dependencies:
                if dep_id not in self.all_tasks or self.all_tasks[dep_id].status != TaskStatus.COMPLETED:
                    all_completed = False
                    break

            if not all_completed:
                # Add to waiting tasks
                self.waiting_tasks.add(task.id)
                self.logger.info(f"Task {task.id} waiting for dependencies")
                return task.id

        # Add to task queue
        self.task_queue.put((task.priority, task))

        # Trigger task assignment
        asyncio.create_task(self._assign_tasks())

        return task.id

    async def _assign_tasks(self) -> None:
        """Assign pending tasks to available agents."""
        while not self.task_queue.empty():
            # Get the next task with the highest priority
            _, task = self.task_queue.get_nowait()

            # Find an appropriate agent for the task
            agent_id = await self._find_agent_for_task(task)

            if agent_id:
                # Assign the task to the agent
                await self._assign_task_to_agent(task, agent_id)
            else:
                # No suitable agent found, put the task back in the queue
                self.task_queue.put((task.priority, task))
                self.logger.warning(f"No suitable agent found for task {task.id}, retrying later")
                break

    async def _find_agent_for_task(self, task: Task) -> Optional[str]:
        """
        Find an appropriate agent for a task.

        Args:
            task: The task to assign.

        Returns:
            The ID of the selected agent, or None if no suitable agent is found.
        """
        # Determine which agent types can handle this task
        agent_types = self._get_agent_types_for_task(task.type)

        if not agent_types:
            self.logger.warning(f"No agent type can handle task type: {task.type}")
            return None

        # Find the least busy agent of the appropriate type
        selected_agent_id = None
        min_tasks = float('inf')

        for agent_type in agent_types:
            if agent_type not in self.agents_by_type:
                continue

            for agent_id in self.agents_by_type[agent_type]:
                agent_info = self.agents[agent_id]

                # Skip agents that are not idle
                if agent_info["status"] != "idle":
                    continue

                # Select the agent with the fewest tasks
                if agent_info["tasks"] < min_tasks:
                    min_tasks = agent_info["tasks"]
                    selected_agent_id = agent_id

        return selected_agent_id

    def _get_agent_types_for_task(self, task_type: str) -> List[str]:
        """
        Get the types of agents that can handle a specific task type.

        Args:
            task_type: The type of task.

        Returns:
            A list of agent types that can handle the task.
        """
        # Mapping of task types to agent types
        task_to_agent_mapping = {
            # Basic tasks
            TaskType.FETCH_URL: ["scraper"],
            TaskType.PARSE_CONTENT: ["parser"],
            TaskType.STORE_DATA: ["storage"],
            TaskType.RENDER_JS: ["scraper", "javascript"],
            TaskType.EXTRACT_LINKS: ["parser"],
            TaskType.FOLLOW_PAGINATION: ["scraper"],
            TaskType.AGGREGATE_RESULTS: ["storage"],

            # JavaScript rendering tasks
            TaskType.RENDER_PAGE: ["javascript"],
            TaskType.INTERACT_WITH_PAGE: ["javascript"],
            TaskType.SCROLL_PAGE: ["javascript"],
            TaskType.TAKE_SCREENSHOT: ["javascript"],
            TaskType.EXECUTE_SCRIPT: ["javascript"],

            # Authentication tasks
            TaskType.AUTHENTICATE: ["authentication"],
            TaskType.REFRESH_SESSION: ["authentication"],
            TaskType.SOLVE_CAPTCHA: ["authentication"],
            TaskType.MFA_AUTHENTICATE: ["authentication"],
            TaskType.VERIFY_SESSION: ["authentication"],
            TaskType.ROTATE_CREDENTIALS: ["authentication"],
            TaskType.MAP_AUTH_FLOW: ["authentication"],

            # Anti-detection tasks
            TaskType.GENERATE_FINGERPRINT: ["anti_detection"],
            TaskType.CHECK_BLOCKING: ["anti_detection"],
            TaskType.OPTIMIZE_REQUEST_PATTERN: ["anti_detection"],
            TaskType.SIMULATE_HUMAN_BEHAVIOR: ["anti_detection"],
            TaskType.DETECT_HONEYPOT: ["anti_detection"],
            TaskType.CUSTOMIZE_HEADERS: ["anti_detection"],
            TaskType.ANALYZE_TRAFFIC_PATTERN: ["anti_detection"],

            # Data transformation tasks
            TaskType.CLEAN_DATA: ["data_transformation"],
            TaskType.TRANSFORM_SCHEMA: ["data_transformation"],
            TaskType.ENRICH_DATA: ["data_transformation"],
            TaskType.ANALYZE_TEXT: ["data_transformation"],
            TaskType.DETECT_ANOMALIES: ["data_transformation"],
            TaskType.INFER_FIELD_TYPES: ["data_transformation"],
            TaskType.RECONCILE_ENTITIES: ["data_transformation"],
            TaskType.IMPUTE_MISSING_DATA: ["data_transformation"],
            TaskType.NORMALIZE_DATA: ["data_transformation"],

            # API Integration tasks
            TaskType.API_REQUEST: ["api_integration"],
            TaskType.API_PAGINATE: ["api_integration"],
            TaskType.API_AUTHENTICATE: ["api_integration"],
            TaskType.API_TRANSFORM: ["api_integration"],
            TaskType.API_LEARN_SCHEMA: ["api_integration"],
            TaskType.API_MANAGE_QUOTA: ["api_integration"],
            TaskType.API_CACHE_RESPONSE: ["api_integration"],
            TaskType.API_FALLBACK: ["api_integration"],

            # NLP Processing tasks
            TaskType.NLP_ENTITY_EXTRACTION: ["nlp_processing"],
            TaskType.NLP_SENTIMENT_ANALYSIS: ["nlp_processing"],
            TaskType.NLP_TEXT_CLASSIFICATION: ["nlp_processing"],
            TaskType.NLP_KEYWORD_EXTRACTION: ["nlp_processing"],
            TaskType.NLP_TEXT_SUMMARIZATION: ["nlp_processing"],
            TaskType.NLP_LANGUAGE_DETECTION: ["nlp_processing"],
            TaskType.NLP_TRAIN_DOMAIN_MODEL: ["nlp_processing"],
            TaskType.NLP_CROSS_LANGUAGE_PROCESS: ["nlp_processing"],
            TaskType.NLP_CONTEXT_ANALYSIS: ["nlp_processing"],
            TaskType.NLP_CONTENT_DEDUPLICATION: ["nlp_processing"],

            # Image Processing tasks
            TaskType.IMAGE_DOWNLOAD: ["image_processing"],
            TaskType.IMAGE_OCR: ["image_processing"],
            TaskType.IMAGE_CLASSIFICATION: ["image_processing"],
            TaskType.IMAGE_EXTRACTION: ["image_processing"],
            TaskType.IMAGE_COMPARISON: ["image_processing"],
            TaskType.IMAGE_SOLVE_CAPTCHA: ["image_processing", "authentication"],
            TaskType.IMAGE_CONTENT_CROP: ["image_processing"],
            TaskType.IMAGE_TO_DATA: ["image_processing"],
            TaskType.IMAGE_SIMILARITY_DETECTION: ["image_processing"],
            TaskType.IMAGE_WATERMARK_REMOVAL: ["image_processing"],

            # Monitoring & Alerting tasks
            TaskType.MONITOR_SYSTEM_HEALTH: ["monitoring"],
            TaskType.TRACK_PERFORMANCE: ["monitoring"],
            TaskType.GENERATE_ALERT: ["monitoring"],
            TaskType.MONITOR_RESOURCES: ["monitoring"],
            TaskType.GENERATE_REPORT: ["monitoring"],

            # Compliance tasks
            TaskType.PARSE_ROBOTS_TXT: ["compliance"],
            TaskType.CHECK_RATE_LIMITS: ["compliance"],
            TaskType.MONITOR_TOS: ["compliance"],
            TaskType.CHECK_LEGAL_COMPLIANCE: ["compliance"],
            TaskType.ENFORCE_ETHICAL_SCRAPING: ["compliance"],

            # Data Quality tasks
            TaskType.VALIDATE_DATA: ["data_quality"],
            TaskType.DETECT_DATA_ANOMALIES: ["data_quality"],
            TaskType.CHECK_COMPLETENESS: ["data_quality"],
            TaskType.VERIFY_CONSISTENCY: ["data_quality"],
            TaskType.SCORE_DATA_QUALITY: ["data_quality"],

            # Self-Learning tasks
            TaskType.ANALYZE_PATTERNS: ["self_learning"],
            TaskType.OPTIMIZE_STRATEGY: ["self_learning"],
            TaskType.TUNE_PARAMETERS: ["self_learning"],
            TaskType.DETECT_SITE_CHANGES: ["self_learning"],
            TaskType.SUGGEST_IMPROVEMENTS: ["self_learning"],

            # Site-Specific Specialist tasks
            TaskType.APPLY_SITE_RULES: ["site_specialist"],
            TaskType.NAVIGATE_SITE: ["site_specialist"],
            TaskType.HANDLE_SITE_AUTH: ["site_specialist", "authentication"],
            TaskType.APPLY_SITE_ANTI_DETECTION: ["site_specialist", "anti_detection"],
            TaskType.OPTIMIZE_SITE_SCRAPING: ["site_specialist"],

            # Coordinator Enhancement tasks
            TaskType.ADAPTIVE_SCHEDULE: ["coordinator"],
            TaskType.RECOVER_FROM_FAILURE: ["coordinator", "error_recovery"],
            TaskType.ANALYZE_AGENT_PERFORMANCE: ["coordinator"],
            TaskType.BALANCE_LOAD: ["coordinator"],
            TaskType.CREATE_WORKFLOW_TEMPLATE: ["coordinator"],

            # Scraper Enhancement tasks
            TaskType.SMART_RATE_LIMIT: ["scraper"],
            TaskType.DIFF_CONTENT: ["scraper"],
            TaskType.PROGRESSIVE_RENDER: ["scraper", "javascript"],
            TaskType.MANAGE_PROXIES: ["scraper"],
            TaskType.STREAM_EXTRACT: ["scraper", "data_extractor"],

            # Storage Enhancement tasks
            TaskType.VERSION_DATA: ["storage"],
            TaskType.VALIDATE_SCHEMA: ["storage"],
            TaskType.UPDATE_INCREMENTALLY: ["storage"],
            TaskType.COMPRESS_DATA: ["storage"],
            TaskType.OPTIMIZE_QUERY: ["storage"]
        }

        return task_to_agent_mapping.get(task_type, [])

    async def _assign_task_to_agent(self, task: Task, agent_id: str) -> None:
        """
        Assign a task to a specific agent.

        Args:
            task: The task to assign.
            agent_id: The ID of the agent to assign the task to.
        """
        self.logger.info(f"Assigning task {task.id} to agent {agent_id}")

        # Update task status
        task.assign(agent_id)

        # Update agent information
        self.agents[agent_id]["tasks"] += 1
        self.agents[agent_id]["status"] = "busy"

        # Create task message
        task_message = TaskMessage(
            sender_id=self.agent_id,
            recipient_id=agent_id,
            task_id=task.id,
            task_type=task.type,
            parameters=task.parameters,
            dependencies=task.dependencies
        )

        # Send the message
        self.outbox.put(task_message)

    async def _handle_result_message(self, message: ResultMessage) -> None:
        """
        Handle a result message from an agent.

        Args:
            message: The result message to handle.
        """
        self.logger.info(f"Received result for task {message.task_id} from {message.sender_id}")

        # Update task status
        if message.task_id in self.all_tasks:
            task = self.all_tasks[message.task_id]
            task.complete(message.result)

            # Update agent information
            if task.assigned_to in self.agents:
                self.agents[task.assigned_to]["tasks"] -= 1
                self.agents[task.assigned_to]["status"] = "idle"

            # Check if any waiting tasks can now be scheduled
            await self._check_waiting_tasks()
        else:
            self.logger.warning(f"Received result for unknown task: {message.task_id}")

    async def _handle_error_message(self, message: ErrorMessage) -> None:
        """
        Handle an error message from an agent.

        Args:
            message: The error message to handle.
        """
        self.logger.warning(f"Received error for task {message.task_id} from {message.sender_id}: {message.error_type} - {message.error_message}")

        # Update task status
        if message.task_id in self.all_tasks:
            task = self.all_tasks[message.task_id]
            task.fail(message.error_type, message.error_message, message.traceback)

            # Update agent information
            if task.assigned_to in self.agents:
                self.agents[task.assigned_to]["tasks"] -= 1
                self.agents[task.assigned_to]["status"] = "idle"
        else:
            self.logger.warning(f"Received error for unknown task: {message.task_id}")

    async def _handle_status_message(self, message: StatusMessage) -> None:
        """
        Handle a status message from an agent.

        Args:
            message: The status message to handle.
        """
        self.logger.debug(f"Received status update from {message.sender_id}: {message.status}")

        # Update agent information
        if message.sender_id in self.agents:
            self.agents[message.sender_id]["status"] = message.status
            self.agents[message.sender_id].update(message.details)
        else:
            self.logger.warning(f"Received status from unknown agent: {message.sender_id}")

    async def _check_waiting_tasks(self) -> None:
        """Check if any waiting tasks can now be scheduled."""
        tasks_to_schedule = []

        # Check each waiting task
        for task_id in list(self.waiting_tasks):
            task = self.all_tasks[task_id]

            # Check if all dependencies are completed
            all_completed = True
            for dep_id in task.dependencies:
                if dep_id not in self.all_tasks or self.all_tasks[dep_id].status != TaskStatus.COMPLETED:
                    all_completed = False
                    break

            if all_completed:
                # Remove from waiting tasks
                self.waiting_tasks.remove(task_id)
                tasks_to_schedule.append(task)

        # Schedule the tasks
        for task in tasks_to_schedule:
            self.task_queue.put((task.priority, task))

        # Trigger task assignment if there are tasks to schedule
        if tasks_to_schedule:
            asyncio.create_task(self._assign_tasks())

    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """
        Get the status of a task.

        Args:
            task_id: The ID of the task.

        Returns:
            A dictionary containing the task status, or None if the task is not found.
        """
        if task_id not in self.all_tasks:
            return None

        task = self.all_tasks[task_id]

        return {
            "id": task.id,
            "type": task.type,
            "status": task.status,
            "created_at": task.created_at,
            "updated_at": task.updated_at,
            "assigned_to": task.assigned_to,
            "result": task.result,
            "error": task.error
        }

    def get_agent_status(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """
        Get the status of an agent.

        Args:
            agent_id: The ID of the agent.

        Returns:
            A dictionary containing the agent status, or None if the agent is not found.
        """
        if agent_id not in self.agents:
            return None

        return self.agents[agent_id]
