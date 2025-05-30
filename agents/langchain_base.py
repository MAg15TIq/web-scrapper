"""
Enhanced base agent class with LangChain integration.
This provides the foundation for AI-powered agent communication and reasoning.
"""
import asyncio
import logging
import uuid
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Callable, Awaitable, Union, Type
from datetime import datetime

from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.language_models import BaseLanguageModel
from langchain_core.memory import BaseMemory
from langchain_core.prompts import BasePromptTemplate
from langchain_core.tools import BaseTool
from langchain.agents import AgentExecutor, create_react_agent
from langchain.memory import ConversationBufferWindowMemory
from pydantic import BaseModel, Field

from models.langchain_models import (
    AgentType, AgentConfig, TaskRequest, TaskResponse, 
    AgentMessage, Priority, WorkflowState, AgentPerformanceMetrics
)
from models.message import Message, TaskMessage, ResultMessage, ErrorMessage


class LangChainCallbackHandler(BaseCallbackHandler):
    """Custom callback handler for LangChain operations."""
    
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.logger = logging.getLogger(f"langchain.{agent_id}")
    
    def on_agent_action(self, action: AgentAction, **kwargs) -> None:
        """Called when agent takes an action."""
        self.logger.debug(f"Agent action: {action.tool} - {action.tool_input}")
    
    def on_agent_finish(self, finish: AgentFinish, **kwargs) -> None:
        """Called when agent finishes."""
        self.logger.debug(f"Agent finished: {finish.return_values}")
    
    def on_tool_start(self, serialized: Dict[str, Any], input_str: str, **kwargs) -> None:
        """Called when tool starts."""
        self.logger.debug(f"Tool started: {serialized.get('name', 'unknown')} - {input_str}")
    
    def on_tool_end(self, output: str, **kwargs) -> None:
        """Called when tool ends."""
        self.logger.debug(f"Tool output: {output}")
    
    def on_tool_error(self, error: Union[Exception, KeyboardInterrupt], **kwargs) -> None:
        """Called when tool encounters an error."""
        self.logger.error(f"Tool error: {error}")


class EnhancedAgent(ABC):
    """
    Enhanced base agent class with LangChain integration.
    Provides AI-powered reasoning, memory, and tool usage capabilities.
    """
    
    def __init__(
        self,
        config: AgentConfig,
        llm: Optional[BaseLanguageModel] = None,
        tools: Optional[List[BaseTool]] = None,
        memory: Optional[BaseMemory] = None,
        prompt_template: Optional[BasePromptTemplate] = None
    ):
        """
        Initialize the enhanced agent.
        
        Args:
            config: Agent configuration
            llm: Language model for reasoning
            tools: List of tools available to the agent
            memory: Memory system for context retention
            prompt_template: Custom prompt template
        """
        self.config = config
        self.agent_id = config.agent_id
        self.agent_type = config.agent_type
        self.logger = logging.getLogger(f"agent.{self.agent_type}.{self.agent_id}")
        
        # LangChain components
        self.llm = llm
        self.tools = tools or []
        self.memory = memory or ConversationBufferWindowMemory(
            k=10,  # Remember last 10 interactions
            return_messages=True
        )
        self.prompt_template = prompt_template
        
        # Agent executor for LangChain operations
        self.agent_executor: Optional[AgentExecutor] = None
        if self.llm and self.tools:
            self._initialize_agent_executor()
        
        # Performance tracking
        self.metrics = AgentPerformanceMetrics(
            agent_id=self.agent_id,
            agent_type=self.agent_type
        )
        
        # State management
        self.current_tasks: Dict[str, TaskRequest] = {}
        self.workflow_states: Dict[str, WorkflowState] = {}
        
        # Callback handler
        self.callback_handler = LangChainCallbackHandler(self.agent_id)
        
        # Running state
        self.running = False
        
        self.logger.info(f"Enhanced agent {self.agent_id} initialized with LangChain integration")
    
    def _initialize_agent_executor(self) -> None:
        """Initialize the LangChain agent executor."""
        try:
            if self.prompt_template:
                agent = create_react_agent(
                    llm=self.llm,
                    tools=self.tools,
                    prompt=self.prompt_template
                )
            else:
                # Use default ReAct prompt
                from langchain import hub
                prompt = hub.pull("hwchase17/react")
                agent = create_react_agent(
                    llm=self.llm,
                    tools=self.tools,
                    prompt=prompt
                )
            
            self.agent_executor = AgentExecutor(
                agent=agent,
                tools=self.tools,
                memory=self.memory,
                verbose=True,
                handle_parsing_errors=True,
                max_iterations=10,
                callbacks=[self.callback_handler]
            )
            
            self.logger.info("LangChain agent executor initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize agent executor: {e}")
            self.agent_executor = None
    
    async def start(self) -> None:
        """Start the enhanced agent."""
        self.running = True
        self.logger.info(f"Enhanced agent {self.agent_id} started")
        
        # Start the main processing loop
        await self._main_loop()
    
    async def stop(self) -> None:
        """Stop the enhanced agent."""
        self.running = False
        self.logger.info(f"Enhanced agent {self.agent_id} stopping...")
    
    async def _main_loop(self) -> None:
        """Main processing loop for the agent."""
        while self.running:
            try:
                # Process pending tasks
                await self._process_tasks()
                
                # Update metrics
                self._update_metrics()
                
                # Brief sleep to prevent busy waiting
                await asyncio.sleep(0.1)
                
            except Exception as e:
                self.logger.error(f"Error in main loop: {e}", exc_info=True)
                await asyncio.sleep(1)  # Longer sleep on error
    
    async def _process_tasks(self) -> None:
        """Process pending tasks."""
        # This is a placeholder - subclasses should implement specific task processing
        pass
    
    def _update_metrics(self) -> None:
        """Update agent performance metrics."""
        self.metrics.last_activity = datetime.now()
        # Additional metrics updates can be added here
    
    async def execute_with_reasoning(
        self,
        task: TaskRequest,
        context: Optional[Dict[str, Any]] = None
    ) -> TaskResponse:
        """
        Execute a task using LangChain reasoning capabilities.
        
        Args:
            task: The task to execute
            context: Additional context for the task
            
        Returns:
            Task response with results
        """
        start_time = datetime.now()
        
        try:
            if not self.agent_executor:
                # Fallback to basic execution
                return await self.execute_task_basic(task)
            
            # Prepare input for LangChain agent
            agent_input = self._prepare_agent_input(task, context)
            
            # Execute using LangChain agent
            result = await self.agent_executor.ainvoke(
                agent_input,
                callbacks=[self.callback_handler]
            )
            
            # Process the result
            processed_result = self._process_agent_result(result, task)
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            # Update metrics
            self.metrics.tasks_completed += 1
            self.metrics.average_execution_time = (
                (self.metrics.average_execution_time * (self.metrics.tasks_completed - 1) + execution_time)
                / self.metrics.tasks_completed
            )
            
            return TaskResponse(
                task_id=task.id,
                status="completed",
                result=processed_result,
                execution_time=execution_time,
                agent_id=self.agent_id,
                metadata={
                    "reasoning_used": True,
                    "tools_used": [tool.name for tool in self.tools],
                    "memory_context": len(self.memory.buffer) if hasattr(self.memory, 'buffer') else 0
                }
            )
            
        except Exception as e:
            self.logger.error(f"Error executing task with reasoning: {e}", exc_info=True)
            
            # Update error metrics
            self.metrics.tasks_failed += 1
            
            return TaskResponse(
                task_id=task.id,
                status="failed",
                error={
                    "type": type(e).__name__,
                    "message": str(e),
                    "agent_id": self.agent_id
                },
                execution_time=(datetime.now() - start_time).total_seconds(),
                agent_id=self.agent_id
            )
    
    def _prepare_agent_input(
        self,
        task: TaskRequest,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Prepare input for the LangChain agent.
        
        Args:
            task: The task to execute
            context: Additional context
            
        Returns:
            Formatted input for the agent
        """
        input_data = {
            "input": f"Execute task: {task.task_type}",
            "task_parameters": task.parameters,
            "task_id": task.id,
            "agent_capabilities": self.config.capabilities
        }
        
        if context:
            input_data["context"] = context
        
        return input_data
    
    def _process_agent_result(
        self,
        result: Dict[str, Any],
        task: TaskRequest
    ) -> Dict[str, Any]:
        """
        Process the result from LangChain agent execution.
        
        Args:
            result: Raw result from agent
            task: Original task
            
        Returns:
            Processed result
        """
        # Extract the actual output from LangChain result
        if "output" in result:
            return {"data": result["output"], "task_type": task.task_type}
        else:
            return {"data": result, "task_type": task.task_type}
    
    @abstractmethod
    async def execute_task_basic(self, task: TaskRequest) -> TaskResponse:
        """
        Basic task execution without LangChain reasoning.
        Subclasses must implement this as a fallback.
        
        Args:
            task: The task to execute
            
        Returns:
            Task response
        """
        pass
    
    def add_tool(self, tool: BaseTool) -> None:
        """Add a tool to the agent's toolkit."""
        self.tools.append(tool)
        if self.agent_executor:
            # Reinitialize agent executor with new tools
            self._initialize_agent_executor()
    
    def get_metrics(self) -> AgentPerformanceMetrics:
        """Get current performance metrics."""
        return self.metrics
    
    def save_context(self, key: str, value: Any) -> None:
        """Save context information for future use."""
        if not hasattr(self, '_context'):
            self._context = {}
        self._context[key] = value
    
    def get_context(self, key: str, default: Any = None) -> Any:
        """Retrieve saved context information."""
        if not hasattr(self, '_context'):
            return default
        return self._context.get(key, default)
