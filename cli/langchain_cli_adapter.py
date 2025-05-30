"""
LangChain CLI Adapter
Integrates LangChain capabilities with the CLI for natural language processing.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Union
from datetime import datetime

try:
    from langchain_core.language_models import BaseLanguageModel
    from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
    from langchain_core.output_parsers import JsonOutputParser, PydanticOutputParser
    from langchain_openai import ChatOpenAI
    from langchain.memory import ConversationBufferWindowMemory
    from langchain.schema import HumanMessage, AIMessage, SystemMessage
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False

from cli.command_parser import CommandParser, ParsedCommand
from models.langchain_models import ScrapingRequest, ActionType, Priority


class MockLLM:
    """Mock LLM for when LangChain is not available."""
    
    async def ainvoke(self, messages: List[Dict[str, str]]) -> str:
        """Mock LLM response."""
        user_input = messages[-1].get('content', '').lower()
        
        # Simple pattern matching for demo
        if 'scrape' in user_input or 'extract' in user_input:
            return '''
            {
                "action": "scrape",
                "target": "example.com",
                "parameters": {
                    "data_type": "products",
                    "max_pages": 5
                },
                "agents": ["Scraper Agent", "Parser Agent", "Storage Agent"],
                "confidence": 0.8
            }
            '''
        elif 'analyze' in user_input:
            return '''
            {
                "action": "analyze",
                "target": "data.json",
                "parameters": {
                    "analysis_type": "sentiment"
                },
                "agents": ["NLP Processing Agent", "Data Transformation Agent"],
                "confidence": 0.7
            }
            '''
        else:
            return '''
            {
                "action": "unknown",
                "target": null,
                "parameters": {},
                "agents": [],
                "confidence": 0.1
            }
            '''


class LangChainCLIAdapter:
    """Adapter for integrating LangChain with CLI operations."""
    
    def __init__(self):
        """Initialize the LangChain CLI adapter."""
        self.logger = logging.getLogger("langchain_cli_adapter")
        self.command_parser = CommandParser()
        
        # Initialize LLM
        self.llm = self._initialize_llm()
        
        # Initialize memory for conversation context
        self.memory = ConversationBufferWindowMemory(
            k=10,  # Keep last 10 exchanges
            return_messages=True
        ) if LANGCHAIN_AVAILABLE else None
        
        # Command parsing prompts
        self.command_prompt = self._create_command_prompt()
        self.clarification_prompt = self._create_clarification_prompt()
        
        # Context tracking
        self.conversation_context: List[Dict[str, Any]] = []
        
        self.logger.info(f"LangChain CLI adapter initialized (LangChain available: {LANGCHAIN_AVAILABLE})")
    
    def _initialize_llm(self) -> Union[BaseLanguageModel, MockLLM]:
        """Initialize the language model."""
        if not LANGCHAIN_AVAILABLE:
            self.logger.warning("LangChain not available, using mock LLM")
            return MockLLM()
        
        try:
            # Try to initialize OpenAI model
            import os
            api_key = os.getenv('OPENAI_API_KEY')
            
            if api_key:
                return ChatOpenAI(
                    model="gpt-3.5-turbo",
                    temperature=0.1,
                    max_tokens=1000,
                    api_key=api_key
                )
            else:
                self.logger.warning("No OpenAI API key found, using mock LLM")
                return MockLLM()
                
        except Exception as e:
            self.logger.error(f"Error initializing LLM: {e}")
            return MockLLM()
    
    def _create_command_prompt(self) -> str:
        """Create the command parsing prompt template."""
        return """
You are an AI assistant for a multi-agent web scraping system. Your job is to parse natural language commands into structured actions.

Available Actions:
- SCRAPE: Extract data from websites
- ANALYZE: Analyze existing data
- MONITOR: Monitor websites for changes
- CONFIGURE: Configure system settings
- EXPORT: Export data in various formats
- SCHEDULE: Schedule recurring tasks

Available Agents:
- Scraper Agent: Web scraping and data extraction
- Parser Agent: HTML/XML parsing and data extraction
- Storage Agent: Data storage and management
- JavaScript Agent: JavaScript rendering and execution
- Anti-Detection Agent: Stealth and proxy management
- Data Transformation Agent: Data cleaning and transformation
- NLP Processing Agent: Natural language processing
- Image Processing Agent: Image analysis and OCR
- Master Intelligence Agent: AI coordination and planning

Parse this command: "{user_input}"

Respond with a JSON object containing:
- action: The primary action (SCRAPE, ANALYZE, etc.)
- target: The target URL, file, or resource
- parameters: Dictionary of parameters for the action
- agents: List of required agents
- confidence: Confidence score (0.0-1.0)
- reasoning: Brief explanation of your parsing

Example:
{{
    "action": "scrape",
    "target": "https://example.com",
    "parameters": {{
        "data_type": "products",
        "max_pages": 5,
        "selectors": ["title", "price"]
    }},
    "agents": ["Scraper Agent", "Parser Agent", "Storage Agent"],
    "confidence": 0.9,
    "reasoning": "User wants to scrape product data from a website"
}}
"""
    
    def _create_clarification_prompt(self) -> str:
        """Create the clarification prompt template."""
        return """
The user's command was unclear or incomplete. Based on the context and partial understanding, ask a clarifying question to help complete the task.

Original command: "{user_input}"
Parsed so far: {partial_parse}

Ask a specific, helpful question to clarify what the user wants to do. Keep it concise and focused.
"""
    
    async def parse_natural_language(self, user_input: str) -> Optional[Dict[str, Any]]:
        """Parse natural language input into a structured command."""
        try:
            self.logger.debug(f"Parsing natural language input: {user_input}")
            
            # First try rule-based parsing
            rule_based_result = self.command_parser.parse_command(user_input)
            
            if rule_based_result and self.command_parser.validate_command(rule_based_result):
                self.logger.info("Successfully parsed using rule-based parser")
                return self._convert_parsed_command_to_dict(rule_based_result)
            
            # Fall back to LLM parsing
            llm_result = await self._parse_with_llm(user_input)
            
            if llm_result and llm_result.get('confidence', 0) > 0.5:
                self.logger.info("Successfully parsed using LLM")
                return llm_result
            
            # If both fail, ask for clarification
            clarification = await self._generate_clarification(user_input, llm_result)
            
            return {
                'action': 'clarification',
                'message': clarification,
                'original_input': user_input
            }
            
        except Exception as e:
            self.logger.error(f"Error parsing natural language: {e}")
            return None
    
    async def _parse_with_llm(self, user_input: str) -> Optional[Dict[str, Any]]:
        """Parse input using the language model."""
        try:
            # Prepare the prompt
            prompt = self.command_prompt.format(user_input=user_input)
            
            # Add conversation context
            messages = []
            
            if LANGCHAIN_AVAILABLE and self.memory:
                # Add conversation history
                history = self.memory.chat_memory.messages
                for msg in history[-6:]:  # Last 3 exchanges
                    if isinstance(msg, HumanMessage):
                        messages.append({"role": "user", "content": msg.content})
                    elif isinstance(msg, AIMessage):
                        messages.append({"role": "assistant", "content": msg.content})
            
            # Add current prompt
            messages.append({"role": "user", "content": prompt})
            
            # Get LLM response
            if hasattr(self.llm, 'ainvoke'):
                response = await self.llm.ainvoke(messages)
            else:
                response = await self.llm.ainvoke(messages)
            
            # Parse JSON response
            import json
            if isinstance(response, str):
                # Extract JSON from response
                start_idx = response.find('{')
                end_idx = response.rfind('}') + 1
                if start_idx >= 0 and end_idx > start_idx:
                    json_str = response[start_idx:end_idx]
                    result = json.loads(json_str)
                else:
                    raise ValueError("No valid JSON found in response")
            else:
                result = response
            
            # Validate and enhance result
            return self._validate_llm_result(result)
            
        except Exception as e:
            self.logger.error(f"Error in LLM parsing: {e}")
            return None
    
    def _validate_llm_result(self, result: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Validate and enhance LLM parsing result."""
        try:
            # Ensure required fields
            if 'action' not in result:
                return None
            
            # Normalize action
            action = result['action'].upper()
            if action not in ['SCRAPE', 'ANALYZE', 'MONITOR', 'CONFIGURE', 'EXPORT', 'SCHEDULE']:
                return None
            
            # Ensure other fields exist
            result.setdefault('target', None)
            result.setdefault('parameters', {})
            result.setdefault('agents', [])
            result.setdefault('confidence', 0.5)
            result.setdefault('reasoning', '')
            
            # Validate agents
            valid_agents = [
                'Scraper Agent', 'Parser Agent', 'Storage Agent', 'JavaScript Agent',
                'Anti-Detection Agent', 'Data Transformation Agent', 'NLP Processing Agent',
                'Image Processing Agent', 'Master Intelligence Agent', 'Coordinator Agent'
            ]
            
            result['agents'] = [agent for agent in result['agents'] if agent in valid_agents]
            
            # Add default agents if none specified
            if not result['agents']:
                if action == 'SCRAPE':
                    result['agents'] = ['Scraper Agent', 'Parser Agent', 'Storage Agent']
                elif action == 'ANALYZE':
                    result['agents'] = ['NLP Processing Agent', 'Data Transformation Agent']
                else:
                    result['agents'] = ['Coordinator Agent']
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error validating LLM result: {e}")
            return None
    
    async def _generate_clarification(self, user_input: str, partial_parse: Optional[Dict[str, Any]]) -> str:
        """Generate a clarification question."""
        try:
            prompt = self.clarification_prompt.format(
                user_input=user_input,
                partial_parse=partial_parse or "No clear action identified"
            )
            
            if hasattr(self.llm, 'ainvoke'):
                response = await self.llm.ainvoke([{"role": "user", "content": prompt}])
            else:
                response = await self.llm.ainvoke([{"role": "user", "content": prompt}])
            
            if isinstance(response, str):
                return response.strip()
            else:
                return str(response).strip()
                
        except Exception as e:
            self.logger.error(f"Error generating clarification: {e}")
            return "I didn't understand that command. Could you please rephrase it or be more specific about what you'd like to do?"
    
    def _convert_parsed_command_to_dict(self, parsed_command: ParsedCommand) -> Dict[str, Any]:
        """Convert ParsedCommand to dictionary format."""
        return {
            'action': parsed_command.action.value,
            'target': parsed_command.target,
            'parameters': parsed_command.parameters,
            'agents': parsed_command.agents,
            'confidence': 0.9,  # High confidence for rule-based parsing
            'reasoning': 'Parsed using rule-based parser',
            'priority': parsed_command.priority.value,
            'output_format': parsed_command.output_format.value,
            'filters': parsed_command.filters,
            'options': parsed_command.options
        }
    
    def add_to_conversation_context(self, user_input: str, parsed_result: Dict[str, Any]):
        """Add interaction to conversation context."""
        context_entry = {
            'timestamp': datetime.now().isoformat(),
            'user_input': user_input,
            'parsed_result': parsed_result
        }
        
        self.conversation_context.append(context_entry)
        
        # Keep only last 20 interactions
        if len(self.conversation_context) > 20:
            self.conversation_context = self.conversation_context[-20:]
        
        # Add to LangChain memory if available
        if LANGCHAIN_AVAILABLE and self.memory:
            self.memory.chat_memory.add_user_message(user_input)
            self.memory.chat_memory.add_ai_message(f"Parsed as: {parsed_result.get('action', 'unknown')}")
    
    def get_command_suggestions(self, partial_input: str) -> List[str]:
        """Get command suggestions based on partial input and context."""
        # Get rule-based suggestions
        rule_suggestions = self.command_parser.get_command_suggestions(partial_input)
        
        # Add context-aware suggestions
        context_suggestions = []
        
        # Look at recent commands for patterns
        recent_actions = [ctx['parsed_result'].get('action') for ctx in self.conversation_context[-5:]]
        
        if 'scrape' in recent_actions:
            context_suggestions.extend([
                "analyze the scraped data",
                "export results as CSV",
                "configure anti-detection settings"
            ])
        
        if 'analyze' in recent_actions:
            context_suggestions.extend([
                "export analysis results",
                "visualize the data",
                "schedule regular analysis"
            ])
        
        # Combine and deduplicate
        all_suggestions = rule_suggestions + context_suggestions
        return list(dict.fromkeys(all_suggestions))  # Remove duplicates while preserving order
    
    def clear_conversation_context(self):
        """Clear conversation context."""
        self.conversation_context.clear()
        
        if LANGCHAIN_AVAILABLE and self.memory:
            self.memory.clear()
        
        self.logger.info("Conversation context cleared")
    
    def get_conversation_summary(self) -> Dict[str, Any]:
        """Get summary of conversation context."""
        if not self.conversation_context:
            return {'total_interactions': 0, 'recent_actions': [], 'common_targets': []}
        
        actions = [ctx['parsed_result'].get('action') for ctx in self.conversation_context]
        targets = [ctx['parsed_result'].get('target') for ctx in self.conversation_context if ctx['parsed_result'].get('target')]
        
        from collections import Counter
        action_counts = Counter(actions)
        target_counts = Counter(targets)
        
        return {
            'total_interactions': len(self.conversation_context),
            'recent_actions': actions[-5:],
            'common_actions': action_counts.most_common(3),
            'common_targets': target_counts.most_common(3),
            'last_interaction': self.conversation_context[-1]['timestamp'] if self.conversation_context else None
        }
