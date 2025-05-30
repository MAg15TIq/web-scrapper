"""
Parser agent for the web scraping system.
"""
import logging
import re
from typing import Dict, Any, Optional, List, Union
from bs4 import BeautifulSoup
from parsel import Selector
from urllib.parse import urljoin

from agents.base import Agent
from models.task import Task, TaskType


class ParserAgent(Agent):
    """
    Agent responsible for parsing content and extracting structured data.
    """
    def __init__(self, agent_id: Optional[str] = None, coordinator_id: str = "coordinator"):
        """
        Initialize a new parser agent.
        
        Args:
            agent_id: Unique identifier for the agent. If None, a UUID will be generated.
            coordinator_id: ID of the coordinator agent.
        """
        super().__init__(agent_id=agent_id, agent_type="parser")
        self.coordinator_id = coordinator_id
    
    async def _send_message(self, message: Any) -> None:
        """
        Send a message to another agent.
        
        Args:
            message: The message to send.
        """
        # In a real implementation, this would use a message broker or direct connection
        # For now, we'll just log the message
        self.logger.debug(f"Sending message to {message.recipient_id}: {message.type}")
        
        # This is a placeholder for actual message sending
        # In a real implementation, you would send the message to the recipient agent
        # For example: await self.message_broker.send(message.recipient_id, message)
    
    async def execute_task(self, task: Task) -> Dict[str, Any]:
        """
        Execute a task assigned to this agent.
        
        Args:
            task: The task to execute.
            
        Returns:
            A dictionary containing the result of the task.
        """
        if task.type == TaskType.PARSE_CONTENT:
            return await self._parse_content(task)
        elif task.type == TaskType.EXTRACT_LINKS:
            return await self._extract_links(task)
        else:
            raise ValueError(f"Unsupported task type for parser agent: {task.type}")
    
    async def _parse_content(self, task: Task) -> Dict[str, Any]:
        """
        Parse content and extract structured data.
        
        Args:
            task: The task containing the content to parse.
            
        Returns:
            A dictionary containing the extracted data.
        """
        content = task.parameters.get("content")
        if not content:
            raise ValueError("Content parameter is required for parse_content task")
        
        selectors = task.parameters.get("selectors", {})
        use_xpath = task.parameters.get("use_xpath", False)
        normalize = task.parameters.get("normalize", True)
        
        self.logger.info(f"Parsing content with {len(selectors)} selectors (use_xpath: {use_xpath})")
        
        # Create parser based on the content
        if use_xpath:
            # Use Parsel for XPath selectors
            parser = Selector(text=content)
            extracted_data = {}
            
            for field_name, selector in selectors.items():
                try:
                    # Extract data using XPath
                    results = parser.xpath(selector).getall()
                    
                    if results:
                        if len(results) == 1:
                            extracted_data[field_name] = results[0].strip() if normalize else results[0]
                        else:
                            extracted_data[field_name] = [r.strip() if normalize else r for r in results]
                    else:
                        extracted_data[field_name] = None
                except Exception as e:
                    self.logger.error(f"Error extracting {field_name} with XPath selector {selector}: {str(e)}")
                    extracted_data[field_name] = None
        else:
            # Use BeautifulSoup for CSS selectors
            soup = BeautifulSoup(content, "lxml")
            extracted_data = {}
            
            for field_name, selector in selectors.items():
                try:
                    # Extract data using CSS selector
                    elements = soup.select(selector)
                    
                    if elements:
                        if len(elements) == 1:
                            # Single element
                            element = elements[0]
                            
                            # Check if we should extract an attribute or text
                            if "::attr(" in selector:
                                # Extract attribute
                                attr_match = re.search(r"::attr\(([^)]+)\)$", selector)
                                if attr_match:
                                    attr_name = attr_match.group(1)
                                    # Remove the ::attr part from the selector
                                    clean_selector = selector.split("::attr")[0]
                                    elements = soup.select(clean_selector)
                                    if elements:
                                        extracted_data[field_name] = elements[0].get(attr_name, "")
                                    else:
                                        extracted_data[field_name] = None
                                else:
                                    extracted_data[field_name] = element.text.strip() if normalize else element.text
                            else:
                                # Extract text
                                extracted_data[field_name] = element.text.strip() if normalize else element.text
                        else:
                            # Multiple elements
                            if "::attr(" in selector:
                                # Extract attribute from multiple elements
                                attr_match = re.search(r"::attr\(([^)]+)\)$", selector)
                                if attr_match:
                                    attr_name = attr_match.group(1)
                                    # Remove the ::attr part from the selector
                                    clean_selector = selector.split("::attr")[0]
                                    elements = soup.select(clean_selector)
                                    extracted_data[field_name] = [e.get(attr_name, "") for e in elements]
                                else:
                                    extracted_data[field_name] = [e.text.strip() if normalize else e.text for e in elements]
                            else:
                                # Extract text from multiple elements
                                extracted_data[field_name] = [e.text.strip() if normalize else e.text for e in elements]
                    else:
                        extracted_data[field_name] = None
                except Exception as e:
                    self.logger.error(f"Error extracting {field_name} with CSS selector {selector}: {str(e)}")
                    extracted_data[field_name] = None
        
        return {
            "extracted_data": extracted_data,
            "num_fields": len(extracted_data),
            "selectors_used": selectors,
            "use_xpath": use_xpath
        }
    
    async def _extract_links(self, task: Task) -> Dict[str, Any]:
        """
        Extract links from content.
        
        Args:
            task: The task containing the content to extract links from.
            
        Returns:
            A dictionary containing the extracted links.
        """
        content = task.parameters.get("content")
        if not content:
            raise ValueError("Content parameter is required for extract_links task")
        
        base_url = task.parameters.get("base_url", "")
        selector = task.parameters.get("selector", "a")
        attribute = task.parameters.get("attribute", "href")
        filter_pattern = task.parameters.get("filter_pattern")
        limit = task.parameters.get("limit")
        
        self.logger.info(f"Extracting links from content (selector: {selector}, attribute: {attribute})")
        
        # Parse the content
        soup = BeautifulSoup(content, "lxml")
        
        # Extract links
        elements = soup.select(selector)
        links = []
        
        for element in elements:
            # Get the attribute value
            link = element.get(attribute)
            
            if link:
                # Join with base URL if provided
                if base_url and not link.startswith(("http://", "https://")):
                    link = urljoin(base_url, link)
                
                # Apply filter if provided
                if filter_pattern:
                    if re.search(filter_pattern, link):
                        links.append(link)
                else:
                    links.append(link)
        
        # Apply limit if provided
        if limit and len(links) > limit:
            links = links[:limit]
        
        return {
            "links": links,
            "count": len(links),
            "base_url": base_url,
            "selector": selector,
            "attribute": attribute,
            "filter_pattern": filter_pattern
        }
