"""
Data Quality agent for the web scraping system.
"""
import logging
import asyncio
import time
import os
import json
import re
import statistics
from typing import Dict, Any, Optional, List, Union
from datetime import datetime, timedelta
import jsonschema

from agents.base import Agent
from models.task import Task, TaskStatus, TaskType
from models.message import Message, TaskMessage, ResultMessage, ErrorMessage, StatusMessage, Priority


class DataQualityAgent(Agent):
    """
    Agent responsible for ensuring the quality of scraped data,
    including validation, anomaly detection, and completeness checking.
    """
    def __init__(self, agent_id: Optional[str] = None):
        """
        Initialize a new data quality agent.
        
        Args:
            agent_id: Unique identifier for the agent. If None, a UUID will be generated.
        """
        super().__init__(agent_id=agent_id, agent_type="data_quality")
        
        # Directory for storing data quality information
        self.data_dir = "data/quality"
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Schema registry
        self.schemas = {}
        
        # Reference data for anomaly detection
        self.reference_data = {}
        
        # Data quality metrics history
        self.quality_metrics = {}
        
        # Register additional message handlers
        self.register_handler("data_quality_config", self._handle_data_quality_config_message)
    
    async def start(self) -> None:
        """Start the agent and initialize data quality components."""
        # Load schemas if available
        schema_file = os.path.join(self.data_dir, "schemas.json")
        if os.path.exists(schema_file):
            with open(schema_file, "r") as f:
                self.schemas = json.load(f)
        
        # Load reference data if available
        reference_file = os.path.join(self.data_dir, "reference_data.json")
        if os.path.exists(reference_file):
            with open(reference_file, "r") as f:
                self.reference_data = json.load(f)
        
        # Start the agent
        await super().start()
    
    async def stop(self) -> None:
        """Stop the agent and clean up resources."""
        # Save schemas
        schema_file = os.path.join(self.data_dir, "schemas.json")
        with open(schema_file, "w") as f:
            json.dump(self.schemas, f, indent=2)
        
        # Save reference data
        reference_file = os.path.join(self.data_dir, "reference_data.json")
        with open(reference_file, "w") as f:
            json.dump(self.reference_data, f, indent=2)
        
        # Stop the agent
        await super().stop()
    
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
        
        if task.type == TaskType.VALIDATE_DATA:
            return await self._execute_validate_data(task)
        elif task.type == TaskType.DETECT_DATA_ANOMALIES:
            return await self._execute_detect_data_anomalies(task)
        elif task.type == TaskType.CHECK_COMPLETENESS:
            return await self._execute_check_completeness(task)
        elif task.type == TaskType.VERIFY_CONSISTENCY:
            return await self._execute_verify_consistency(task)
        elif task.type == TaskType.SCORE_DATA_QUALITY:
            return await self._execute_score_data_quality(task)
        else:
            raise ValueError(f"Unsupported task type: {task.type}")
    
    async def _execute_validate_data(self, task: Task) -> Dict[str, Any]:
        """
        Execute a data validation task.
        
        Args:
            task: The task to execute.
            
        Returns:
            A dictionary containing the validation results.
        """
        params = task.parameters
        
        # Get required parameters
        data = params.get("data", [])
        schema = params.get("schema")
        
        if not data:
            return {"valid": True, "errors": [], "message": "No data to validate"}
        
        if not schema:
            return {"valid": False, "errors": [{"message": "No schema provided"}], "message": "No schema provided"}
        
        # Get optional parameters
        strict = params.get("strict", True)
        report_all_errors = params.get("report_all_errors", True)
        coerce_types = params.get("coerce_types", False)
        
        # Convert schema to JSON Schema format if it's not already
        json_schema = self._convert_to_json_schema(schema)
        
        # Validate each item in the data
        validation_results = []
        all_valid = True
        
        for i, item in enumerate(data):
            try:
                # Coerce types if requested
                if coerce_types:
                    item = self._coerce_types(item, schema)
                
                # Validate against schema
                jsonschema.validate(item, json_schema)
                validation_results.append({"index": i, "valid": True})
            except jsonschema.exceptions.ValidationError as e:
                all_valid = False
                validation_results.append({
                    "index": i,
                    "valid": False,
                    "error": str(e),
                    "path": ".".join(str(p) for p in e.path)
                })
                
                if strict and not report_all_errors:
                    break
        
        return {
            "valid": all_valid,
            "total_items": len(data),
            "valid_items": sum(1 for r in validation_results if r["valid"]),
            "invalid_items": sum(1 for r in validation_results if not r["valid"]),
            "results": validation_results if report_all_errors else [r for r in validation_results if not r["valid"]],
            "timestamp": datetime.now().isoformat()
        }
    
    def _convert_to_json_schema(self, schema: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert a simplified schema to JSON Schema format.
        
        Args:
            schema: A simplified schema definition.
            
        Returns:
            A JSON Schema compatible schema.
        """
        # Check if it's already a JSON Schema
        if "type" in schema and schema["type"] == "object" and "properties" in schema:
            return schema
        
        # Convert simplified schema to JSON Schema
        properties = {}
        required = []
        
        for field_name, field_def in schema.items():
            if isinstance(field_def, str):
                # Simple type definition
                properties[field_name] = {"type": field_def}
            elif isinstance(field_def, dict):
                # Complex type definition
                field_schema = {}
                
                if "type" in field_def:
                    field_schema["type"] = field_def["type"]
                
                if "required" in field_def and field_def["required"]:
                    required.append(field_name)
                
                if "min_length" in field_def:
                    field_schema["minLength"] = field_def["min_length"]
                
                if "max_length" in field_def:
                    field_schema["maxLength"] = field_def["max_length"]
                
                if "min" in field_def:
                    field_schema["minimum"] = field_def["min"]
                
                if "max" in field_def:
                    field_schema["maximum"] = field_def["max"]
                
                if "pattern" in field_def:
                    field_schema["pattern"] = field_def["pattern"]
                
                if "enum" in field_def:
                    field_schema["enum"] = field_def["enum"]
                
                properties[field_name] = field_schema
        
        return {
            "type": "object",
            "properties": properties,
            "required": required,
            "additionalProperties": False
        }
    
    def _coerce_types(self, item: Dict[str, Any], schema: Dict[str, Any]) -> Dict[str, Any]:
        """
        Coerce item values to the types specified in the schema.
        
        Args:
            item: The data item to coerce.
            schema: The schema defining the expected types.
            
        Returns:
            The item with coerced types.
        """
        coerced_item = {}
        
        for field_name, value in item.items():
            if field_name not in schema:
                coerced_item[field_name] = value
                continue
            
            field_def = schema[field_name]
            field_type = field_def if isinstance(field_def, str) else field_def.get("type")
            
            if value is None:
                coerced_item[field_name] = None
                continue
            
            try:
                if field_type == "string":
                    coerced_item[field_name] = str(value)
                elif field_type == "number":
                    coerced_item[field_name] = float(value)
                elif field_type == "integer":
                    coerced_item[field_name] = int(value)
                elif field_type == "boolean":
                    if isinstance(value, str):
                        coerced_item[field_name] = value.lower() in ("true", "yes", "1")
                    else:
                        coerced_item[field_name] = bool(value)
                else:
                    coerced_item[field_name] = value
            except (ValueError, TypeError):
                # If coercion fails, keep the original value
                coerced_item[field_name] = value
        
        return coerced_item
    
    async def _execute_detect_data_anomalies(self, task: Task) -> Dict[str, Any]:
        """
        Execute a data anomaly detection task.
        
        Args:
            task: The task to execute.
            
        Returns:
            A dictionary containing the anomaly detection results.
        """
        params = task.parameters
        
        # Get required parameters
        data = params.get("data", [])
        
        if not data:
            return {"anomalies_detected": False, "anomalies": [], "message": "No data to analyze"}
        
        # Get optional parameters
        fields_to_check = params.get("fields_to_check", [])
        methods = params.get("methods", ["statistical"])
        reference_data_path = params.get("reference_data")
        sensitivity = params.get("sensitivity", 0.8)  # 0.0 to 1.0
        include_explanation = params.get("include_explanation", True)
        
        # If no fields specified, check all numeric fields
        if not fields_to_check:
            # Identify numeric fields from the first item
            if data:
                for field, value in data[0].items():
                    if isinstance(value, (int, float)):
                        fields_to_check.append(field)
        
        # Load reference data if specified
        reference_values = {}
        if reference_data_path:
            if reference_data_path in self.reference_data:
                reference_values = self.reference_data[reference_data_path]
            elif os.path.exists(reference_data_path):
                with open(reference_data_path, "r") as f:
                    reference_data = json.load(f)
                    
                    # Extract values for each field
                    for field in fields_to_check:
                        reference_values[field] = [item.get(field) for item in reference_data if field in item and item[field] is not None]
                    
                    # Cache the reference values
                    self.reference_data[reference_data_path] = reference_values
        
        # Detect anomalies
        anomalies = []
        
        for field in fields_to_check:
            # Extract values for the field
            values = [item.get(field) for item in data if field in item and item[field] is not None]
            
            if not values:
                continue
            
            # Skip non-numeric fields
            if not all(isinstance(v, (int, float)) for v in values):
                continue
            
            field_anomalies = []
            
            # Statistical method (z-score)
            if "statistical" in methods:
                try:
                    mean = statistics.mean(values)
                    stdev = statistics.stdev(values) if len(values) > 1 else 0
                    
                    if stdev > 0:
                        z_threshold = 3.0 * (1.0 - sensitivity)  # Adjust threshold based on sensitivity
                        
                        for i, item in enumerate(data):
                            if field in item and item[field] is not None and isinstance(item[field], (int, float)):
                                z_score = abs((item[field] - mean) / stdev)
                                
                                if z_score > z_threshold:
                                    anomaly = {
                                        "index": i,
                                        "field": field,
                                        "value": item[field],
                                        "method": "statistical",
                                        "score": z_score
                                    }
                                    
                                    if include_explanation:
                                        anomaly["explanation"] = f"Value {item[field]} is {z_score:.2f} standard deviations from the mean ({mean:.2f})"
                                    
                                    field_anomalies.append(anomaly)
                except (statistics.StatisticsError, ZeroDivisionError):
                    pass
            
            # Reference data comparison
            if "reference_comparison" in methods and field in reference_values and reference_values[field]:
                ref_mean = statistics.mean(reference_values[field])
                ref_stdev = statistics.stdev(reference_values[field]) if len(reference_values[field]) > 1 else 0
                
                if ref_stdev > 0:
                    z_threshold = 3.0 * (1.0 - sensitivity)
                    
                    for i, item in enumerate(data):
                        if field in item and item[field] is not None and isinstance(item[field], (int, float)):
                            z_score = abs((item[field] - ref_mean) / ref_stdev)
                            
                            if z_score > z_threshold:
                                anomaly = {
                                    "index": i,
                                    "field": field,
                                    "value": item[field],
                                    "method": "reference_comparison",
                                    "score": z_score
                                }
                                
                                if include_explanation:
                                    anomaly["explanation"] = f"Value {item[field]} is {z_score:.2f} standard deviations from the reference mean ({ref_mean:.2f})"
                                
                                field_anomalies.append(anomaly)
            
            # Add field anomalies to the overall list
            anomalies.extend(field_anomalies)
        
        return {
            "anomalies_detected": len(anomalies) > 0,
            "total_items": len(data),
            "anomalies": anomalies,
            "fields_checked": fields_to_check,
            "methods_used": methods,
            "sensitivity": sensitivity,
            "timestamp": datetime.now().isoformat()
        }
    
    async def _execute_check_completeness(self, task: Task) -> Dict[str, Any]:
        """
        Execute a data completeness checking task.
        
        Args:
            task: The task to execute.
            
        Returns:
            A dictionary containing the completeness check results.
        """
        params = task.parameters
        
        # Get required parameters
        data = params.get("data", [])
        required_fields = params.get("required_fields", [])
        
        if not data:
            return {"complete": True, "missing_fields": [], "message": "No data to check"}
        
        if not required_fields:
            return {"complete": True, "missing_fields": [], "message": "No required fields specified"}
        
        # Get optional parameters
        completeness_threshold = params.get("completeness_threshold", 0.8)  # 80% of fields must be present
        field_weights = params.get("field_weights", {field: 1.0 for field in required_fields})
        
        # Check completeness for each item
        item_results = []
        all_complete = True
        
        for i, item in enumerate(data):
            # Count present fields and calculate weighted completeness
            present_fields = [field for field in required_fields if field in item and item[field] is not None]
            missing_fields = [field for field in required_fields if field not in item or item[field] is None]
            
            # Calculate weighted completeness
            total_weight = sum(field_weights.get(field, 1.0) for field in required_fields)
            present_weight = sum(field_weights.get(field, 1.0) for field in present_fields)
            
            completeness_score = present_weight / total_weight if total_weight > 0 else 1.0
            is_complete = completeness_score >= completeness_threshold
            
            if not is_complete:
                all_complete = False
            
            item_results.append({
                "index": i,
                "complete": is_complete,
                "completeness_score": completeness_score,
                "present_fields": present_fields,
                "missing_fields": missing_fields
            })
        
        # Calculate overall completeness
        overall_completeness = sum(r["completeness_score"] for r in item_results) / len(item_results) if item_results else 1.0
        
        return {
            "complete": all_complete,
            "overall_completeness": overall_completeness,
            "total_items": len(data),
            "complete_items": sum(1 for r in item_results if r["complete"]),
            "incomplete_items": sum(1 for r in item_results if not r["complete"]),
            "required_fields": required_fields,
            "completeness_threshold": completeness_threshold,
            "results": item_results,
            "timestamp": datetime.now().isoformat()
        }
    
    async def _execute_verify_consistency(self, task: Task) -> Dict[str, Any]:
        """
        Execute a data consistency verification task.
        
        Args:
            task: The task to execute.
            
        Returns:
            A dictionary containing the consistency verification results.
        """
        params = task.parameters
        
        # Get required parameters
        data = params.get("data", [])
        consistency_rules = params.get("consistency_rules", [])
        
        if not data:
            return {"consistent": True, "violations": [], "message": "No data to verify"}
        
        if not consistency_rules:
            return {"consistent": True, "violations": [], "message": "No consistency rules specified"}
        
        # Get optional parameters
        check_cross_record_consistency = params.get("check_cross_record_consistency", True)
        
        # Verify consistency for each item
        item_violations = []
        all_consistent = True
        
        for i, item in enumerate(data):
            record_violations = []
            
            # Check each rule
            for rule in consistency_rules:
                rule_expr = rule["rule"]
                error_message = rule.get("error_message", f"Violated rule: {rule_expr}")
                
                # Evaluate the rule in the context of the item
                try:
                    # Create a safe evaluation context with the item's fields
                    eval_context = {**item}
                    
                    # Add some safe functions
                    eval_context["len"] = len
                    
                    # Evaluate the rule
                    is_consistent = eval(rule_expr, {"__builtins__": {}}, eval_context)
                    
                    if not is_consistent:
                        record_violations.append({
                            "rule": rule_expr,
                            "message": error_message
                        })
                except Exception as e:
                    record_violations.append({
                        "rule": rule_expr,
                        "message": f"Error evaluating rule: {str(e)}"
                    })
            
            if record_violations:
                all_consistent = False
                item_violations.append({
                    "index": i,
                    "violations": record_violations
                })
        
        # Check cross-record consistency if requested
        cross_record_violations = []
        if check_cross_record_consistency and len(data) > 1:
            # Example: Check for duplicate IDs
            id_counts = {}
            for i, item in enumerate(data):
                if "id" in item:
                    item_id = item["id"]
                    if item_id in id_counts:
                        id_counts[item_id].append(i)
                    else:
                        id_counts[item_id] = [i]
            
            # Report duplicates
            for item_id, indices in id_counts.items():
                if len(indices) > 1:
                    all_consistent = False
                    cross_record_violations.append({
                        "type": "duplicate_id",
                        "id": item_id,
                        "indices": indices,
                        "message": f"Duplicate ID '{item_id}' found in records {', '.join(str(idx) for idx in indices)}"
                    })
        
        return {
            "consistent": all_consistent,
            "total_items": len(data),
            "consistent_items": len(data) - len(item_violations),
            "inconsistent_items": len(item_violations),
            "item_violations": item_violations,
            "cross_record_violations": cross_record_violations,
            "timestamp": datetime.now().isoformat()
        }
    
    async def _execute_score_data_quality(self, task: Task) -> Dict[str, Any]:
        """
        Execute a data quality scoring task.
        
        Args:
            task: The task to execute.
            
        Returns:
            A dictionary containing the quality scoring results.
        """
        params = task.parameters
        
        # Get required parameters
        data = params.get("data", [])
        
        if not data:
            return {"quality_score": 1.0, "dimensions": {}, "message": "No data to score"}
        
        # Get optional parameters
        dimensions = params.get("dimensions", ["completeness", "accuracy", "consistency", "timeliness"])
        weights = params.get("weights", {
            "completeness": 0.3,
            "accuracy": 0.3,
            "consistency": 0.2,
            "timeliness": 0.2
        })
        reference_data = params.get("reference_data")
        generate_report = params.get("generate_report", True)
        
        # Normalize weights
        total_weight = sum(weights.values())
        normalized_weights = {k: v / total_weight for k, v in weights.items()}
        
        # Score each dimension
        dimension_scores = {}
        
        # Completeness
        if "completeness" in dimensions:
            # Create a completeness checking task
            completeness_task = Task(
                type=TaskType.CHECK_COMPLETENESS,
                parameters={
                    "data": data,
                    "required_fields": list(data[0].keys()) if data else []
                }
            )
            
            completeness_result = await self._execute_check_completeness(completeness_task)
            dimension_scores["completeness"] = completeness_result["overall_completeness"]
        
        # Accuracy (using anomaly detection as a proxy)
        if "accuracy" in dimensions:
            # Create an anomaly detection task
            anomaly_task = Task(
                type=TaskType.DETECT_DATA_ANOMALIES,
                parameters={
                    "data": data,
                    "reference_data": reference_data
                }
            )
            
            anomaly_result = await self._execute_detect_data_anomalies(anomaly_task)
            anomaly_rate = len(anomaly_result["anomalies"]) / len(data) if data else 0
            dimension_scores["accuracy"] = 1.0 - anomaly_rate
        
        # Consistency
        if "consistency" in dimensions:
            # Create a consistency verification task
            consistency_task = Task(
                type=TaskType.VERIFY_CONSISTENCY,
                parameters={
                    "data": data,
                    "consistency_rules": []  # No specific rules, just check for cross-record consistency
                }
            )
            
            consistency_result = await self._execute_verify_consistency(consistency_task)
            inconsistent_rate = len(consistency_result["item_violations"]) / len(data) if data else 0
            dimension_scores["consistency"] = 1.0 - inconsistent_rate
        
        # Timeliness (if timestamp field exists)
        if "timeliness" in dimensions:
            timeliness_score = 1.0
            
            # Check if there's a timestamp field
            timestamp_fields = ["timestamp", "created_at", "updated_at", "date", "time"]
            timestamp_field = next((field for field in timestamp_fields if field in data[0]), None) if data else None
            
            if timestamp_field:
                # Calculate timeliness based on age
                now = datetime.now()
                try:
                    timestamps = []
                    for item in data:
                        if timestamp_field in item and item[timestamp_field]:
                            try:
                                ts = datetime.fromisoformat(item[timestamp_field])
                                timestamps.append(ts)
                            except (ValueError, TypeError):
                                pass
                    
                    if timestamps:
                        # Calculate average age in days
                        ages = [(now - ts).total_seconds() / 86400 for ts in timestamps]
                        avg_age = sum(ages) / len(ages)
                        
                        # Score decreases with age (1.0 for fresh data, 0.5 for 7-day old data)
                        timeliness_score = max(0.0, min(1.0, 1.0 - (avg_age / 14.0)))
                except Exception:
                    pass
            
            dimension_scores["timeliness"] = timeliness_score
        
        # Calculate overall quality score
        quality_score = sum(dimension_scores.get(dim, 0.0) * normalized_weights.get(dim, 0.0) for dim in dimensions)
        
        # Generate report if requested
        report = None
        if generate_report:
            report = {
                "quality_score": quality_score,
                "dimension_scores": dimension_scores,
                "weights": normalized_weights,
                "sample_size": len(data),
                "timestamp": datetime.now().isoformat()
            }
            
            # Add recommendations based on scores
            recommendations = []
            
            if "completeness" in dimension_scores and dimension_scores["completeness"] < 0.8:
                recommendations.append({
                    "dimension": "completeness",
                    "message": "Improve data completeness by ensuring all required fields are present",
                    "priority": "high" if dimension_scores["completeness"] < 0.6 else "medium"
                })
            
            if "accuracy" in dimension_scores and dimension_scores["accuracy"] < 0.9:
                recommendations.append({
                    "dimension": "accuracy",
                    "message": "Address data anomalies to improve accuracy",
                    "priority": "high" if dimension_scores["accuracy"] < 0.7 else "medium"
                })
            
            if "consistency" in dimension_scores and dimension_scores["consistency"] < 0.9:
                recommendations.append({
                    "dimension": "consistency",
                    "message": "Resolve data consistency issues",
                    "priority": "medium"
                })
            
            if "timeliness" in dimension_scores and dimension_scores["timeliness"] < 0.7:
                recommendations.append({
                    "dimension": "timeliness",
                    "message": "Update data more frequently to improve timeliness",
                    "priority": "medium"
                })
            
            report["recommendations"] = recommendations
            
            # Save report
            report_file = os.path.join(self.data_dir, f"quality_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
            with open(report_file, "w") as f:
                json.dump(report, f, indent=2)
            
            report["report_file"] = report_file
        
        return {
            "quality_score": quality_score,
            "dimension_scores": dimension_scores,
            "dimensions": dimensions,
            "weights": normalized_weights,
            "report": report,
            "timestamp": datetime.now().isoformat()
        }
    
    async def _handle_data_quality_config_message(self, message: Message) -> None:
        """
        Handle a data quality configuration message.
        
        Args:
            message: The message to handle.
        """
        if not hasattr(message, "config") or not message.config:
            self.logger.warning("Received data quality config message without config")
            return
        
        self.logger.info("Updating data quality configuration")
        
        # Update schemas if provided
        if "schemas" in message.config:
            for schema_name, schema in message.config["schemas"].items():
                self.schemas[schema_name] = schema
        
        # Update reference data if provided
        if "reference_data" in message.config:
            for dataset_name, dataset in message.config["reference_data"].items():
                self.reference_data[dataset_name] = dataset
        
        # Send acknowledgment
        status_message = StatusMessage(
            sender_id=self.agent_id,
            recipient_id=message.sender_id,
            status="data_quality_config_updated",
            details={"schemas": list(self.schemas.keys())}
        )
        self.outbox.put(status_message)
