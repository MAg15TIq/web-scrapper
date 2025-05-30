"""
Quality Assurance Agent for the self-aware web scraping system.

This agent ensures high-quality output.
"""
import asyncio
import logging
import time
import json
import re
from typing import Dict, Any, List, Optional, Set, Tuple, Union
import uuid

from agents.base import Agent
from models.message import Message, TaskMessage, ResultMessage, ErrorMessage, StatusMessage, Priority
from models.task import Task, TaskStatus, TaskType
from models.intelligence import AgentCapability


class QualityAssuranceAgent(Agent):
    """
    Quality Assurance Agent that ensures high-quality output.
    """
    def __init__(self, agent_id: Optional[str] = None, coordinator_id: str = "coordinator"):
        """
        Initialize a new Quality Assurance Agent.

        Args:
            agent_id: Unique identifier for the agent. If None, a UUID will be generated.
            coordinator_id: ID of the coordinator agent. Used for message routing.
        """
        super().__init__(agent_id=agent_id, agent_type="quality_assurance", coordinator_id=coordinator_id)
        
        # Quality metrics
        self.metrics: Dict[str, Dict[str, Any]] = {}
        
        # Quality thresholds
        self.thresholds: Dict[str, float] = {
            "completeness": 0.9,  # 90% completeness
            "accuracy": 0.95,  # 95% accuracy
            "consistency": 0.9,  # 90% consistency
            "timeliness": 0.8,  # 80% timeliness
            "relevance": 0.8  # 80% relevance
        }
        
        # Register message handlers
        self.register_handler("validate_data", self._handle_validate_data)
        self.register_handler("check_completeness", self._handle_check_completeness)
        self.register_handler("check_consistency", self._handle_check_consistency)
        self.register_handler("score_quality", self._handle_score_quality)
    
    async def validate_data(self, data: Union[Dict[str, Any], List[Dict[str, Any]]], schema: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate data against a schema.

        Args:
            data: The data to validate. Can be a single object or a list of objects.
            schema: The schema to validate against.

        Returns:
            A dictionary containing validation results.
        """
        self.logger.info(f"Validating data against schema")
        
        # Convert single object to list for consistent processing
        data_list = data if isinstance(data, list) else [data]
        
        # Initialize validation results
        validation_results = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "item_results": []
        }
        
        # Validate each item
        for i, item in enumerate(data_list):
            item_result = self._validate_item(item, schema)
            
            # Add item result
            validation_results["item_results"].append({
                "index": i,
                "valid": item_result["valid"],
                "errors": item_result["errors"],
                "warnings": item_result["warnings"]
            })
            
            # Update overall validity
            if not item_result["valid"]:
                validation_results["valid"] = False
            
            # Add errors and warnings to overall results
            for error in item_result["errors"]:
                validation_results["errors"].append(f"Item {i}: {error}")
            
            for warning in item_result["warnings"]:
                validation_results["warnings"].append(f"Item {i}: {warning}")
        
        # Add summary
        validation_results["summary"] = {
            "total_items": len(data_list),
            "valid_items": sum(1 for r in validation_results["item_results"] if r["valid"]),
            "error_count": len(validation_results["errors"]),
            "warning_count": len(validation_results["warnings"])
        }
        
        return validation_results
    
    def _validate_item(self, item: Dict[str, Any], schema: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate a single item against a schema.

        Args:
            item: The item to validate.
            schema: The schema to validate against.

        Returns:
            A dictionary containing validation results.
        """
        # Initialize validation results
        validation_result = {
            "valid": True,
            "errors": [],
            "warnings": []
        }
        
        # Check each field in the schema
        for field_name, field_schema in schema.items():
            # Check if field is required
            required = field_schema.get("required", False)
            
            # Check if field exists
            if field_name not in item:
                if required:
                    validation_result["valid"] = False
                    validation_result["errors"].append(f"Required field '{field_name}' is missing")
                continue
            
            # Get field value
            value = item[field_name]
            
            # Check if value is None
            if value is None:
                if required:
                    validation_result["valid"] = False
                    validation_result["errors"].append(f"Required field '{field_name}' is None")
                continue
            
            # Check field type
            expected_type = field_schema.get("type")
            if expected_type:
                valid_type = self._check_type(value, expected_type)
                if not valid_type:
                    validation_result["valid"] = False
                    validation_result["errors"].append(f"Field '{field_name}' has invalid type. Expected {expected_type}, got {type(value).__name__}")
            
            # Check field pattern
            pattern = field_schema.get("pattern")
            if pattern and isinstance(value, str):
                if not re.match(pattern, value):
                    validation_result["valid"] = False
                    validation_result["errors"].append(f"Field '{field_name}' does not match pattern '{pattern}'")
            
            # Check minimum length
            min_length = field_schema.get("min_length")
            if min_length is not None and hasattr(value, "__len__"):
                if len(value) < min_length:
                    validation_result["valid"] = False
                    validation_result["errors"].append(f"Field '{field_name}' is too short. Minimum length is {min_length}, got {len(value)}")
            
            # Check maximum length
            max_length = field_schema.get("max_length")
            if max_length is not None and hasattr(value, "__len__"):
                if len(value) > max_length:
                    validation_result["valid"] = False
                    validation_result["errors"].append(f"Field '{field_name}' is too long. Maximum length is {max_length}, got {len(value)}")
            
            # Check minimum value
            minimum = field_schema.get("minimum")
            if minimum is not None and isinstance(value, (int, float)):
                if value < minimum:
                    validation_result["valid"] = False
                    validation_result["errors"].append(f"Field '{field_name}' is too small. Minimum value is {minimum}, got {value}")
            
            # Check maximum value
            maximum = field_schema.get("maximum")
            if maximum is not None and isinstance(value, (int, float)):
                if value > maximum:
                    validation_result["valid"] = False
                    validation_result["errors"].append(f"Field '{field_name}' is too large. Maximum value is {maximum}, got {value}")
            
            # Check enum values
            enum_values = field_schema.get("enum")
            if enum_values is not None:
                if value not in enum_values:
                    validation_result["valid"] = False
                    validation_result["errors"].append(f"Field '{field_name}' has invalid value. Expected one of {enum_values}, got {value}")
            
            # Check custom validation
            custom_validation = field_schema.get("validation")
            if custom_validation:
                # In a real implementation, you would evaluate the custom validation
                # For now, we'll just add a warning
                validation_result["warnings"].append(f"Custom validation for field '{field_name}' not implemented")
        
        return validation_result
    
    def _check_type(self, value: Any, expected_type: str) -> bool:
        """
        Check if a value has the expected type.

        Args:
            value: The value to check.
            expected_type: The expected type.

        Returns:
            True if the value has the expected type, False otherwise.
        """
        if expected_type == "string":
            return isinstance(value, str)
        elif expected_type == "number":
            return isinstance(value, (int, float))
        elif expected_type == "integer":
            return isinstance(value, int)
        elif expected_type == "boolean":
            return isinstance(value, bool)
        elif expected_type == "array":
            return isinstance(value, list)
        elif expected_type == "object":
            return isinstance(value, dict)
        elif expected_type == "null":
            return value is None
        else:
            return True  # Unknown type, assume valid
    
    async def check_completeness(self, data: Union[Dict[str, Any], List[Dict[str, Any]]], required_fields: List[str], field_weights: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """
        Check the completeness of data.

        Args:
            data: The data to check. Can be a single object or a list of objects.
            required_fields: The list of required fields.
            field_weights: Optional weights for each field.

        Returns:
            A dictionary containing completeness results.
        """
        self.logger.info(f"Checking completeness of data")
        
        # Convert single object to list for consistent processing
        data_list = data if isinstance(data, list) else [data]
        
        # Initialize completeness results
        completeness_results = {
            "overall_completeness": 0.0,
            "item_results": []
        }
        
        # Use equal weights if not provided
        if field_weights is None:
            field_weights = {field: 1.0 for field in required_fields}
        
        # Calculate total weight
        total_weight = sum(field_weights.get(field, 1.0) for field in required_fields)
        
        # Check completeness of each item
        total_completeness = 0.0
        
        for i, item in enumerate(data_list):
            # Calculate item completeness
            item_completeness = 0.0
            missing_fields = []
            
            for field in required_fields:
                field_weight = field_weights.get(field, 1.0)
                
                if field in item and item[field] is not None:
                    item_completeness += field_weight
                else:
                    missing_fields.append(field)
            
            # Normalize completeness
            if total_weight > 0:
                item_completeness /= total_weight
            
            # Add item result
            completeness_results["item_results"].append({
                "index": i,
                "completeness": item_completeness,
                "missing_fields": missing_fields
            })
            
            # Update total completeness
            total_completeness += item_completeness
        
        # Calculate overall completeness
        if data_list:
            completeness_results["overall_completeness"] = total_completeness / len(data_list)
        
        # Add threshold comparison
        completeness_threshold = self.thresholds["completeness"]
        completeness_results["meets_threshold"] = completeness_results["overall_completeness"] >= completeness_threshold
        completeness_results["threshold"] = completeness_threshold
        
        return completeness_results
    
    async def check_consistency(self, data: List[Dict[str, Any]], consistency_rules: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Check the consistency of data.

        Args:
            data: The data to check.
            consistency_rules: The consistency rules to apply.

        Returns:
            A dictionary containing consistency results.
        """
        self.logger.info(f"Checking consistency of data")
        
        # Initialize consistency results
        consistency_results = {
            "overall_consistency": 0.0,
            "rule_results": []
        }
        
        # Check each consistency rule
        total_consistency = 0.0
        
        for rule in consistency_rules:
            rule_type = rule.get("type")
            rule_fields = rule.get("fields", [])
            
            # Initialize rule result
            rule_result = {
                "rule": rule,
                "consistency": 0.0,
                "violations": []
            }
            
            # Apply rule based on type
            if rule_type == "unique":
                # Check uniqueness of fields
                values_seen = set()
                violations = 0
                
                for i, item in enumerate(data):
                    # Create a tuple of field values
                    field_values = tuple(item.get(field) for field in rule_fields)
                    
                    # Check if we've seen these values before
                    if field_values in values_seen:
                        violations += 1
                        rule_result["violations"].append({
                            "index": i,
                            "fields": rule_fields,
                            "values": field_values
                        })
                    else:
                        values_seen.add(field_values)
                
                # Calculate consistency
                if data:
                    rule_result["consistency"] = 1.0 - (violations / len(data))
            
            elif rule_type == "dependency":
                # Check field dependencies
                dependent_field = rule.get("dependent_field")
                dependency_field = rule.get("dependency_field")
                
                if dependent_field and dependency_field:
                    violations = 0
                    
                    for i, item in enumerate(data):
                        # If dependent field has a value, dependency field must also have a value
                        if dependent_field in item and item[dependent_field] is not None:
                            if dependency_field not in item or item[dependency_field] is None:
                                violations += 1
                                rule_result["violations"].append({
                                    "index": i,
                                    "dependent_field": dependent_field,
                                    "dependency_field": dependency_field
                                })
                    
                    # Calculate consistency
                    if data:
                        rule_result["consistency"] = 1.0 - (violations / len(data))
            
            elif rule_type == "format":
                # Check field format
                field = rule.get("field")
                pattern = rule.get("pattern")
                
                if field and pattern:
                    violations = 0
                    
                    for i, item in enumerate(data):
                        if field in item and isinstance(item[field], str):
                            if not re.match(pattern, item[field]):
                                violations += 1
                                rule_result["violations"].append({
                                    "index": i,
                                    "field": field,
                                    "value": item[field],
                                    "pattern": pattern
                                })
                    
                    # Calculate consistency
                    if data:
                        rule_result["consistency"] = 1.0 - (violations / len(data))
            
            # Add rule result
            consistency_results["rule_results"].append(rule_result)
            
            # Update total consistency
            total_consistency += rule_result["consistency"]
        
        # Calculate overall consistency
        if consistency_rules:
            consistency_results["overall_consistency"] = total_consistency / len(consistency_rules)
        
        # Add threshold comparison
        consistency_threshold = self.thresholds["consistency"]
        consistency_results["meets_threshold"] = consistency_results["overall_consistency"] >= consistency_threshold
        consistency_results["threshold"] = consistency_threshold
        
        return consistency_results
    
    async def score_quality(self, data: Union[Dict[str, Any], List[Dict[str, Any]]], schema: Dict[str, Any], quality_thresholds: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """
        Score the quality of data.

        Args:
            data: The data to score. Can be a single object or a list of objects.
            schema: The schema to validate against.
            quality_thresholds: Optional quality thresholds.

        Returns:
            A dictionary containing quality scores.
        """
        self.logger.info(f"Scoring quality of data")
        
        # Convert single object to list for consistent processing
        data_list = data if isinstance(data, list) else [data]
        
        # Update thresholds if provided
        if quality_thresholds:
            self.thresholds.update(quality_thresholds)
        
        # Initialize quality scores
        quality_scores = {
            "overall_quality": 0.0,
            "completeness": 0.0,
            "accuracy": 0.0,
            "consistency": 0.0,
            "timeliness": 0.0,
            "relevance": 0.0,
            "details": {}
        }
        
        # Check completeness
        required_fields = [field for field, field_schema in schema.items() if field_schema.get("required", False)]
        field_weights = {field: schema[field].get("weight", 1.0) for field in required_fields if field in schema}
        
        completeness_results = await self.check_completeness(data_list, required_fields, field_weights)
        quality_scores["completeness"] = completeness_results["overall_completeness"]
        quality_scores["details"]["completeness"] = completeness_results
        
        # Check accuracy (via validation)
        validation_results = await self.validate_data(data_list, schema)
        
        # Calculate accuracy score
        if data_list:
            valid_items = validation_results["summary"]["valid_items"]
            accuracy = valid_items / len(data_list)
        else:
            accuracy = 0.0
        
        quality_scores["accuracy"] = accuracy
        quality_scores["details"]["accuracy"] = validation_results
        
        # Check consistency
        # Create basic consistency rules from schema
        consistency_rules = []
        
        # Add uniqueness rules for fields with unique=True
        unique_fields = [field for field, field_schema in schema.items() if field_schema.get("unique", False)]
        if unique_fields:
            consistency_rules.append({
                "type": "unique",
                "fields": unique_fields
            })
        
        # Add dependency rules for fields with dependencies
        for field, field_schema in schema.items():
            if "depends_on" in field_schema:
                consistency_rules.append({
                    "type": "dependency",
                    "dependent_field": field,
                    "dependency_field": field_schema["depends_on"]
                })
        
        # Add format rules for fields with patterns
        for field, field_schema in schema.items():
            if "pattern" in field_schema:
                consistency_rules.append({
                    "type": "format",
                    "field": field,
                    "pattern": field_schema["pattern"]
                })
        
        consistency_results = await self.check_consistency(data_list, consistency_rules)
        quality_scores["consistency"] = consistency_results["overall_consistency"]
        quality_scores["details"]["consistency"] = consistency_results
        
        # Timeliness and relevance are more subjective
        # For now, we'll use placeholder values
        quality_scores["timeliness"] = 1.0  # Assume data is fresh
        quality_scores["relevance"] = 1.0  # Assume data is relevant
        
        # Calculate overall quality
        # Use weighted average of individual scores
        weights = {
            "completeness": 0.3,
            "accuracy": 0.3,
            "consistency": 0.2,
            "timeliness": 0.1,
            "relevance": 0.1
        }
        
        overall_quality = sum(quality_scores[metric] * weight for metric, weight in weights.items())
        quality_scores["overall_quality"] = overall_quality
        
        # Add threshold comparisons
        quality_scores["meets_thresholds"] = {
            "completeness": quality_scores["completeness"] >= self.thresholds["completeness"],
            "accuracy": quality_scores["accuracy"] >= self.thresholds["accuracy"],
            "consistency": quality_scores["consistency"] >= self.thresholds["consistency"],
            "timeliness": quality_scores["timeliness"] >= self.thresholds["timeliness"],
            "relevance": quality_scores["relevance"] >= self.thresholds["relevance"]
        }
        
        quality_scores["thresholds"] = self.thresholds.copy()
        
        return quality_scores
    
    async def _handle_validate_data(self, message: Message) -> None:
        """
        Handle a validate data message.

        Args:
            message: The message to handle.
        """
        if not hasattr(message, "data") or not hasattr(message, "schema"):
            self.logger.warning("Received validate_data message without data or schema")
            return
        
        try:
            # Validate data
            validation_results = await self.validate_data(message.data, message.schema)
            
            # Send result
            response = ResultMessage(
                sender_id=self.agent_id,
                recipient_id=message.sender_id,
                task_id=message.task_id if hasattr(message, "task_id") else None,
                result=validation_results
            )
            self.outbox.put(response)
        except Exception as e:
            self.logger.error(f"Error validating data: {str(e)}")
            
            # Send error
            error_message = ErrorMessage(
                sender_id=self.agent_id,
                recipient_id=message.sender_id,
                task_id=message.task_id if hasattr(message, "task_id") else None,
                error=str(e)
            )
            self.outbox.put(error_message)
    
    async def _handle_check_completeness(self, message: Message) -> None:
        """
        Handle a check completeness message.

        Args:
            message: The message to handle.
        """
        if not hasattr(message, "data") or not hasattr(message, "required_fields"):
            self.logger.warning("Received check_completeness message without data or required_fields")
            return
        
        try:
            # Get field weights if provided
            field_weights = message.field_weights if hasattr(message, "field_weights") else None
            
            # Check completeness
            completeness_results = await self.check_completeness(message.data, message.required_fields, field_weights)
            
            # Send result
            response = ResultMessage(
                sender_id=self.agent_id,
                recipient_id=message.sender_id,
                task_id=message.task_id if hasattr(message, "task_id") else None,
                result=completeness_results
            )
            self.outbox.put(response)
        except Exception as e:
            self.logger.error(f"Error checking completeness: {str(e)}")
            
            # Send error
            error_message = ErrorMessage(
                sender_id=self.agent_id,
                recipient_id=message.sender_id,
                task_id=message.task_id if hasattr(message, "task_id") else None,
                error=str(e)
            )
            self.outbox.put(error_message)
    
    async def _handle_check_consistency(self, message: Message) -> None:
        """
        Handle a check consistency message.

        Args:
            message: The message to handle.
        """
        if not hasattr(message, "data") or not hasattr(message, "consistency_rules"):
            self.logger.warning("Received check_consistency message without data or consistency_rules")
            return
        
        try:
            # Check consistency
            consistency_results = await self.check_consistency(message.data, message.consistency_rules)
            
            # Send result
            response = ResultMessage(
                sender_id=self.agent_id,
                recipient_id=message.sender_id,
                task_id=message.task_id if hasattr(message, "task_id") else None,
                result=consistency_results
            )
            self.outbox.put(response)
        except Exception as e:
            self.logger.error(f"Error checking consistency: {str(e)}")
            
            # Send error
            error_message = ErrorMessage(
                sender_id=self.agent_id,
                recipient_id=message.sender_id,
                task_id=message.task_id if hasattr(message, "task_id") else None,
                error=str(e)
            )
            self.outbox.put(error_message)
    
    async def _handle_score_quality(self, message: Message) -> None:
        """
        Handle a score quality message.

        Args:
            message: The message to handle.
        """
        if not hasattr(message, "data") or not hasattr(message, "schema"):
            self.logger.warning("Received score_quality message without data or schema")
            return
        
        try:
            # Get quality thresholds if provided
            quality_thresholds = message.quality_thresholds if hasattr(message, "quality_thresholds") else None
            
            # Score quality
            quality_scores = await self.score_quality(message.data, message.schema, quality_thresholds)
            
            # Send result
            response = ResultMessage(
                sender_id=self.agent_id,
                recipient_id=message.sender_id,
                task_id=message.task_id if hasattr(message, "task_id") else None,
                result=quality_scores
            )
            self.outbox.put(response)
        except Exception as e:
            self.logger.error(f"Error scoring quality: {str(e)}")
            
            # Send error
            error_message = ErrorMessage(
                sender_id=self.agent_id,
                recipient_id=message.sender_id,
                task_id=message.task_id if hasattr(message, "task_id") else None,
                error=str(e)
            )
            self.outbox.put(error_message)
    
    async def execute_task(self, task: Task) -> Dict[str, Any]:
        """
        Execute a task assigned to this agent.

        Args:
            task: The task to execute.

        Returns:
            A dictionary containing the result of the task.
        """
        self.logger.info(f"Executing task: {task.type}")
        
        if task.type == TaskType.VALIDATE_DATA:
            # Validate data
            data = task.parameters.get("data")
            schema = task.parameters.get("schema")
            
            if not data or not schema:
                raise ValueError("Missing data or schema parameters")
            
            return await self.validate_data(data, schema)
        
        elif task.type == TaskType.CHECK_COMPLETENESS:
            # Check completeness
            data = task.parameters.get("data")
            required_fields = task.parameters.get("required_fields")
            field_weights = task.parameters.get("field_weights")
            
            if not data or not required_fields:
                raise ValueError("Missing data or required_fields parameters")
            
            return await self.check_completeness(data, required_fields, field_weights)
        
        elif task.type == TaskType.CHECK_CONSISTENCY:
            # Check consistency
            data = task.parameters.get("data")
            consistency_rules = task.parameters.get("consistency_rules")
            
            if not data or not consistency_rules:
                raise ValueError("Missing data or consistency_rules parameters")
            
            return await self.check_consistency(data, consistency_rules)
        
        elif task.type == TaskType.SCORE_DATA_QUALITY:
            # Score quality
            data = task.parameters.get("data")
            schema = task.parameters.get("schema")
            quality_thresholds = task.parameters.get("quality_thresholds")
            
            if not data or not schema:
                raise ValueError("Missing data or schema parameters")
            
            return await self.score_quality(data, schema, quality_thresholds)
        
        else:
            raise ValueError(f"Unsupported task type: {task.type}")
