"""
API Integration agent for the web scraping system.
"""
import json
import logging
import asyncio
import httpx
from typing import Dict, Any, Optional, List, Union
from urllib.parse import urljoin

from agents.base import Agent
from models.task import Task, TaskStatus, TaskType
from models.message import Message, TaskMessage, ResultMessage, ErrorMessage, StatusMessage, Priority


class APIIntegrationAgent(Agent):
    """
    Agent for integrating with external APIs as an alternative or complement to web scraping.
    """
    def __init__(self, agent_id: Optional[str] = None):
        """
        Initialize a new API integration agent.
        
        Args:
            agent_id: Unique identifier for the agent. If None, a UUID will be generated.
        """
        super().__init__(agent_id=agent_id, agent_type="api_integration")
        
        # Initialize HTTP client
        self.client = httpx.AsyncClient(
            timeout=60.0,
            follow_redirects=True,
            http2=True
        )
        
        # API key storage (in a real implementation, use a secure storage solution)
        self.api_keys = {}
        
        # API endpoint configurations
        self.api_configs = {}
        
        # Register additional message handlers
        self.register_handler("api_config", self._handle_api_config_message)
    
    async def _send_message(self, message: Message) -> None:
        """
        Send a message to another agent.
        
        Args:
            message: The message to send.
        """
        # In a real implementation, this would use a message broker
        self.logger.debug(f"Sending message to {message.recipient_id}: {message.type}")
    
    async def execute_task(self, task: Task) -> Dict[str, Any]:
        """
        Execute a task assigned to this agent.
        
        Args:
            task: The task to execute.
            
        Returns:
            A dictionary containing the result of the task.
        """
        self.logger.info(f"Executing task: {task.type}")
        
        if task.type == TaskType.API_REQUEST:
            return await self._execute_api_request(task)
        elif task.type == TaskType.API_PAGINATE:
            return await self._execute_api_paginate(task)
        elif task.type == TaskType.API_AUTHENTICATE:
            return await self._execute_api_authenticate(task)
        elif task.type == TaskType.API_TRANSFORM:
            return await self._execute_api_transform(task)
        else:
            raise ValueError(f"Unsupported task type: {task.type}")
    
    async def _execute_api_request(self, task: Task) -> Dict[str, Any]:
        """
        Execute an API request task.
        
        Args:
            task: The task to execute.
            
        Returns:
            A dictionary containing the result of the API request.
        """
        params = task.parameters
        
        # Get required parameters
        endpoint = params.get("endpoint")
        method = params.get("method", "GET")
        api_id = params.get("api_id")
        
        if not endpoint:
            raise ValueError("Missing required parameter: endpoint")
        
        # Get API configuration
        api_config = self.api_configs.get(api_id, {})
        base_url = api_config.get("base_url", "")
        
        # Construct full URL
        url = urljoin(base_url, endpoint) if base_url else endpoint
        
        # Prepare request parameters
        request_params = {
            "url": url,
            "method": method,
            "headers": params.get("headers", api_config.get("headers", {})),
            "params": params.get("query_params", {}),
            "json": params.get("body", None) if method in ["POST", "PUT", "PATCH"] else None,
            "timeout": params.get("timeout", 60.0)
        }
        
        # Add API key if available
        if api_id in self.api_keys:
            auth_type = api_config.get("auth_type", "header")
            auth_name = api_config.get("auth_name", "Authorization")
            auth_value = self.api_keys[api_id]
            
            if auth_type == "header":
                request_params["headers"][auth_name] = auth_value
            elif auth_type == "query":
                request_params["params"][auth_name] = auth_value
            elif auth_type == "bearer":
                request_params["headers"]["Authorization"] = f"Bearer {auth_value}"
        
        # Execute request
        self.logger.info(f"Making {method} request to {url}")
        
        try:
            response = await self.client.request(**request_params)
            response.raise_for_status()
            
            # Parse response
            content_type = response.headers.get("content-type", "")
            
            if "application/json" in content_type:
                data = response.json()
            elif "text/csv" in content_type:
                data = {"text": response.text, "format": "csv"}
            elif "text/xml" in content_type or "application/xml" in content_type:
                data = {"text": response.text, "format": "xml"}
            else:
                data = {"text": response.text, "format": "text"}
            
            # Extract pagination information if available
            pagination = self._extract_pagination_info(data, api_config.get("pagination", {}))
            
            return {
                "status_code": response.status_code,
                "headers": dict(response.headers),
                "data": data,
                "pagination": pagination
            }
            
        except httpx.HTTPStatusError as e:
            self.logger.error(f"HTTP error: {e}")
            return {
                "error": True,
                "status_code": e.response.status_code,
                "message": str(e),
                "response": e.response.text
            }
        except httpx.RequestError as e:
            self.logger.error(f"Request error: {e}")
            return {
                "error": True,
                "message": str(e)
            }
    
    def _extract_pagination_info(self, data: Any, pagination_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract pagination information from API response.
        
        Args:
            data: API response data.
            pagination_config: Pagination configuration for the API.
            
        Returns:
            Dictionary containing pagination information.
        """
        if not pagination_config:
            return {}
        
        pagination_type = pagination_config.get("type", "")
        
        if pagination_type == "link_header":
            # Extract from Link header
            links = pagination_config.get("links", {})
            return {
                "has_next": links.get("next") is not None,
                "next_url": links.get("next"),
                "prev_url": links.get("prev"),
                "first_url": links.get("first"),
                "last_url": links.get("last")
            }
        
        elif pagination_type == "json_path":
            # Extract from JSON response using paths
            try:
                paths = pagination_config.get("paths", {})
                result = {}
                
                for key, path in paths.items():
                    value = self._get_nested_value(data, path.split("."))
                    result[key] = value
                
                return result
            except Exception as e:
                self.logger.warning(f"Error extracting pagination info: {e}")
                return {}
        
        return {}
    
    def _get_nested_value(self, data: Any, path: List[str]) -> Any:
        """
        Get a nested value from a dictionary using a path.
        
        Args:
            data: Dictionary to extract value from.
            path: List of keys representing the path to the value.
            
        Returns:
            The value at the specified path, or None if not found.
        """
        current = data
        for key in path:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return None
        return current
    
    async def _execute_api_paginate(self, task: Task) -> Dict[str, Any]:
        """
        Execute an API pagination task.
        
        Args:
            task: The task to execute.
            
        Returns:
            A dictionary containing the aggregated results from all pages.
        """
        params = task.parameters
        
        # Get required parameters
        initial_request = params.get("initial_request", {})
        max_pages = params.get("max_pages", 10)
        data_path = params.get("data_path", "")
        
        # Execute initial request
        initial_task = Task(
            type=TaskType.API_REQUEST,
            parameters=initial_request
        )
        
        result = await self._execute_api_request(initial_task)
        
        # Check for errors
        if result.get("error", False):
            return result
        
        # Extract data from the first page
        all_data = []
        page_data = self._extract_data(result["data"], data_path)
        if page_data:
            all_data.extend(page_data)
        
        # Get pagination information
        pagination = result.get("pagination", {})
        next_url = pagination.get("next_url")
        current_page = 1
        
        # Fetch additional pages
        while next_url and current_page < max_pages:
            # Create a new request for the next page
            next_request = {
                "endpoint": next_url,
                "method": "GET"
            }
            
            # If the next URL is relative, use the same API configuration
            if not next_url.startswith(("http://", "https://")):
                next_request["api_id"] = initial_request.get("api_id")
            
            next_task = Task(
                type=TaskType.API_REQUEST,
                parameters=next_request
            )
            
            # Execute the request
            page_result = await self._execute_api_request(next_task)
            
            # Check for errors
            if page_result.get("error", False):
                break
            
            # Extract data from this page
            page_data = self._extract_data(page_result["data"], data_path)
            if page_data:
                all_data.extend(page_data)
            
            # Update pagination information
            pagination = page_result.get("pagination", {})
            next_url = pagination.get("next_url")
            current_page += 1
            
            # Add a small delay to avoid rate limiting
            await asyncio.sleep(params.get("delay", 1.0))
        
        return {
            "data": all_data,
            "pages_fetched": current_page,
            "has_more": next_url is not None
        }
    
    def _extract_data(self, data: Any, path: str) -> List[Any]:
        """
        Extract data from API response using a path.
        
        Args:
            data: API response data.
            path: Dot-separated path to the data array.
            
        Returns:
            List of extracted data items.
        """
        if not path:
            return data if isinstance(data, list) else [data]
        
        try:
            current = data
            for key in path.split("."):
                current = current[key]
            
            return current if isinstance(current, list) else [current]
        except (KeyError, TypeError):
            self.logger.warning(f"Could not extract data using path: {path}")
            return []
    
    async def _execute_api_authenticate(self, task: Task) -> Dict[str, Any]:
        """
        Execute an API authentication task.
        
        Args:
            task: The task to execute.
            
        Returns:
            A dictionary containing the authentication result.
        """
        params = task.parameters
        
        # Get required parameters
        api_id = params.get("api_id")
        auth_type = params.get("auth_type", "key")
        
        if not api_id:
            raise ValueError("Missing required parameter: api_id")
        
        if auth_type == "key":
            # Simple API key authentication
            api_key = params.get("api_key")
            if not api_key:
                raise ValueError("Missing required parameter: api_key")
            
            # Store the API key
            self.api_keys[api_id] = api_key
            
            return {
                "success": True,
                "api_id": api_id,
                "auth_type": auth_type
            }
            
        elif auth_type == "oauth":
            # OAuth authentication
            token_url = params.get("token_url")
            client_id = params.get("client_id")
            client_secret = params.get("client_secret")
            scope = params.get("scope", "")
            
            if not token_url or not client_id or not client_secret:
                raise ValueError("Missing required OAuth parameters")
            
            # Prepare token request
            token_request = {
                "endpoint": token_url,
                "method": "POST",
                "headers": {
                    "Content-Type": "application/x-www-form-urlencoded"
                },
                "body": {
                    "grant_type": "client_credentials",
                    "client_id": client_id,
                    "client_secret": client_secret,
                    "scope": scope
                }
            }
            
            # Execute token request
            token_task = Task(
                type=TaskType.API_REQUEST,
                parameters=token_request
            )
            
            result = await self._execute_api_request(token_task)
            
            # Check for errors
            if result.get("error", False):
                return result
            
            # Extract access token
            token_data = result.get("data", {})
            access_token = token_data.get("access_token")
            
            if not access_token:
                return {
                    "error": True,
                    "message": "No access token in response",
                    "response": token_data
                }
            
            # Store the access token
            self.api_keys[api_id] = access_token
            
            return {
                "success": True,
                "api_id": api_id,
                "auth_type": auth_type,
                "expires_in": token_data.get("expires_in"),
                "token_type": token_data.get("token_type")
            }
            
        else:
            raise ValueError(f"Unsupported auth type: {auth_type}")
    
    async def _execute_api_transform(self, task: Task) -> Dict[str, Any]:
        """
        Execute an API data transformation task.
        
        Args:
            task: The task to execute.
            
        Returns:
            A dictionary containing the transformed data.
        """
        params = task.parameters
        
        # Get required parameters
        data = params.get("data", [])
        transformations = params.get("transformations", [])
        
        if not data:
            return {"data": []}
        
        # Apply transformations
        transformed_data = data
        
        for transform in transformations:
            transform_type = transform.get("type")
            
            if transform_type == "map":
                # Map fields from one schema to another
                mapping = transform.get("mapping", {})
                transformed_data = self._apply_mapping(transformed_data, mapping)
                
            elif transform_type == "filter":
                # Filter data based on conditions
                conditions = transform.get("conditions", [])
                transformed_data = self._apply_filter(transformed_data, conditions)
                
            elif transform_type == "flatten":
                # Flatten nested structures
                path = transform.get("path", "")
                transformed_data = self._apply_flatten(transformed_data, path)
                
            elif transform_type == "aggregate":
                # Aggregate data
                group_by = transform.get("group_by", [])
                aggregations = transform.get("aggregations", {})
                transformed_data = self._apply_aggregation(transformed_data, group_by, aggregations)
        
        return {
            "data": transformed_data
        }
    
    def _apply_mapping(self, data: List[Dict[str, Any]], mapping: Dict[str, str]) -> List[Dict[str, Any]]:
        """
        Apply field mapping to data.
        
        Args:
            data: List of data items.
            mapping: Dictionary mapping source fields to target fields.
            
        Returns:
            List of transformed data items.
        """
        result = []
        
        for item in data:
            transformed = {}
            
            for target_field, source_field in mapping.items():
                # Handle nested fields with dot notation
                if "." in source_field:
                    parts = source_field.split(".")
                    value = item
                    for part in parts:
                        if isinstance(value, dict) and part in value:
                            value = value[part]
                        else:
                            value = None
                            break
                    transformed[target_field] = value
                else:
                    transformed[target_field] = item.get(source_field)
            
            result.append(transformed)
        
        return result
    
    def _apply_filter(self, data: List[Dict[str, Any]], conditions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Filter data based on conditions.
        
        Args:
            data: List of data items.
            conditions: List of filter conditions.
            
        Returns:
            Filtered list of data items.
        """
        result = []
        
        for item in data:
            matches = True
            
            for condition in conditions:
                field = condition.get("field", "")
                operator = condition.get("operator", "eq")
                value = condition.get("value")
                
                # Get field value (support for nested fields)
                field_value = item
                for part in field.split("."):
                    if isinstance(field_value, dict) and part in field_value:
                        field_value = field_value[part]
                    else:
                        field_value = None
                        break
                
                # Apply operator
                if operator == "eq" and field_value != value:
                    matches = False
                    break
                elif operator == "ne" and field_value == value:
                    matches = False
                    break
                elif operator == "gt" and (field_value is None or field_value <= value):
                    matches = False
                    break
                elif operator == "lt" and (field_value is None or field_value >= value):
                    matches = False
                    break
                elif operator == "contains" and (not isinstance(field_value, str) or value not in field_value):
                    matches = False
                    break
                elif operator == "in" and field_value not in value:
                    matches = False
                    break
            
            if matches:
                result.append(item)
        
        return result
    
    def _apply_flatten(self, data: List[Dict[str, Any]], path: str) -> List[Dict[str, Any]]:
        """
        Flatten nested arrays in data.
        
        Args:
            data: List of data items.
            path: Path to the nested array.
            
        Returns:
            Flattened list of data items.
        """
        if not path:
            return data
        
        result = []
        
        for item in data:
            # Get nested array
            nested = item
            for part in path.split("."):
                if isinstance(nested, dict) and part in nested:
                    nested = nested[part]
                else:
                    nested = []
                    break
            
            if not isinstance(nested, list):
                # If not a list, just add the original item
                result.append(item)
                continue
            
            # Create a new item for each element in the nested array
            for nested_item in nested:
                new_item = item.copy()
                
                # Remove the nested array
                current = new_item
                parts = path.split(".")
                for i, part in enumerate(parts[:-1]):
                    if isinstance(current, dict) and part in current:
                        current = current[part]
                    else:
                        break
                
                if isinstance(current, dict):
                    current.pop(parts[-1], None)
                
                # Add the nested item fields
                if isinstance(nested_item, dict):
                    for k, v in nested_item.items():
                        new_item[k] = v
                
                result.append(new_item)
        
        return result
    
    def _apply_aggregation(self, data: List[Dict[str, Any]], group_by: List[str], 
                          aggregations: Dict[str, Dict[str, str]]) -> List[Dict[str, Any]]:
        """
        Apply aggregation to data.
        
        Args:
            data: List of data items.
            group_by: List of fields to group by.
            aggregations: Dictionary of aggregation operations.
            
        Returns:
            Aggregated list of data items.
        """
        if not group_by:
            return data
        
        # Group data
        groups = {}
        
        for item in data:
            # Create group key
            key_parts = []
            for field in group_by:
                value = item.get(field, "")
                key_parts.append(str(value))
            
            group_key = "|".join(key_parts)
            
            # Add to group
            if group_key not in groups:
                groups[group_key] = {
                    "items": [],
                    "group_values": {field: item.get(field) for field in group_by}
                }
            
            groups[group_key]["items"].append(item)
        
        # Apply aggregations
        result = []
        
        for group_data in groups.values():
            aggregated = group_data["group_values"].copy()
            
            for target_field, agg_config in aggregations.items():
                agg_type = agg_config.get("type", "count")
                source_field = agg_config.get("field", "")
                
                if agg_type == "count":
                    aggregated[target_field] = len(group_data["items"])
                    
                elif agg_type == "sum":
                    total = 0
                    for item in group_data["items"]:
                        value = item.get(source_field, 0)
                        if isinstance(value, (int, float)):
                            total += value
                    aggregated[target_field] = total
                    
                elif agg_type == "avg":
                    values = []
                    for item in group_data["items"]:
                        value = item.get(source_field)
                        if isinstance(value, (int, float)):
                            values.append(value)
                    aggregated[target_field] = sum(values) / len(values) if values else 0
                    
                elif agg_type == "min":
                    values = []
                    for item in group_data["items"]:
                        value = item.get(source_field)
                        if isinstance(value, (int, float)):
                            values.append(value)
                    aggregated[target_field] = min(values) if values else None
                    
                elif agg_type == "max":
                    values = []
                    for item in group_data["items"]:
                        value = item.get(source_field)
                        if isinstance(value, (int, float)):
                            values.append(value)
                    aggregated[target_field] = max(values) if values else None
                    
                elif agg_type == "concat":
                    separator = agg_config.get("separator", ", ")
                    values = []
                    for item in group_data["items"]:
                        value = item.get(source_field)
                        if value is not None:
                            values.append(str(value))
                    aggregated[target_field] = separator.join(values)
            
            result.append(aggregated)
        
        return result
    
    async def _handle_api_config_message(self, message: Message) -> None:
        """
        Handle an API configuration message.
        
        Args:
            message: The message to handle.
        """
        if not hasattr(message, "config") or not message.config:
            self.logger.warning("Received API config message without config")
            return
        
        api_id = message.config.get("api_id")
        if not api_id:
            self.logger.warning("Received API config without api_id")
            return
        
        self.logger.info(f"Updating API configuration for {api_id}")
        self.api_configs[api_id] = message.config
        
        # Send acknowledgment
        status_message = StatusMessage(
            sender_id=self.agent_id,
            recipient_id=message.sender_id,
            status="api_config_updated",
            details={"api_id": api_id}
        )
        self.outbox.put(status_message)
