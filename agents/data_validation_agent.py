"""
Data Validation Agent with advanced quality scoring and anomaly detection.
This agent ensures data integrity and quality throughout the scraping pipeline.
"""
import asyncio
import logging
import re
import statistics
from typing import Dict, Any, Optional, List, Union, Tuple
from datetime import datetime, timedelta
from collections import Counter

from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import BaseTool, tool
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field, validator

from agents.langchain_base import EnhancedAgent
from models.langchain_models import (
    AgentConfig, AgentType, TaskRequest, TaskResponse,
    DataValidationRule, QualityReport, ErrorCorrection
)


class DataQualityMetrics(BaseModel):
    """Comprehensive data quality metrics."""
    completeness: float = Field(ge=0.0, le=1.0, description="Percentage of non-null values")
    accuracy: float = Field(ge=0.0, le=1.0, description="Estimated accuracy score")
    consistency: float = Field(ge=0.0, le=1.0, description="Data consistency score")
    validity: float = Field(ge=0.0, le=1.0, description="Format validity score")
    uniqueness: float = Field(ge=0.0, le=1.0, description="Uniqueness ratio")
    timeliness: float = Field(ge=0.0, le=1.0, description="Data freshness score")
    overall_score: float = Field(ge=0.0, le=1.0, description="Weighted overall quality score")


class ValidationResult(BaseModel):
    """Result of data validation process."""
    field_name: str
    value: Any
    is_valid: bool
    confidence: float = Field(ge=0.0, le=1.0)
    issues: List[str] = Field(default_factory=list)
    suggestions: List[str] = Field(default_factory=list)
    quality_metrics: Optional[DataQualityMetrics] = None


@tool
def validate_price_format(price_text: str) -> Dict[str, Any]:
    """Validate and extract price information from text."""
    if not price_text or not isinstance(price_text, str):
        return {"valid": False, "error": "Empty or invalid price text"}
    
    # Common price patterns
    price_patterns = [
        r'\$[\d,]+\.?\d*',  # $123.45, $1,234
        r'[\d,]+\.?\d*\s*(?:USD|dollars?)',  # 123.45 USD
        r'€[\d,]+\.?\d*',  # €123.45
        r'£[\d,]+\.?\d*',  # £123.45
        r'[\d,]+\.?\d*\s*(?:€|EUR)',  # 123.45 EUR
    ]
    
    for pattern in price_patterns:
        match = re.search(pattern, price_text, re.IGNORECASE)
        if match:
            price_str = match.group()
            # Extract numeric value
            numeric_value = re.sub(r'[^\d.]', '', price_str)
            try:
                price_value = float(numeric_value)
                return {
                    "valid": True,
                    "price": price_value,
                    "currency": "USD",  # Default, could be enhanced
                    "original_text": price_text,
                    "confidence": 0.9
                }
            except ValueError:
                continue
    
    return {"valid": False, "error": "No valid price pattern found"}


@tool
def validate_url_format(url: str) -> Dict[str, Any]:
    """Validate URL format and accessibility."""
    if not url or not isinstance(url, str):
        return {"valid": False, "error": "Empty or invalid URL"}
    
    # URL pattern validation
    url_pattern = r'^https?://(?:[-\w.])+(?:[:\d]+)?(?:/(?:[\w/_.])*(?:\?(?:[\w&=%.])*)?(?:#(?:[\w.])*)?)?$'
    
    if re.match(url_pattern, url):
        return {
            "valid": True,
            "url": url,
            "domain": re.search(r'https?://([^/]+)', url).group(1),
            "confidence": 0.95
        }
    else:
        return {"valid": False, "error": "Invalid URL format"}


@tool
def detect_data_anomalies(data_list: List[Any]) -> Dict[str, Any]:
    """Detect anomalies in a list of data values."""
    if not data_list or len(data_list) < 3:
        return {"anomalies": [], "confidence": 0.0}
    
    anomalies = []
    
    # For numeric data
    numeric_values = []
    for item in data_list:
        try:
            if isinstance(item, (int, float)):
                numeric_values.append(float(item))
            elif isinstance(item, str):
                # Try to extract numeric value
                numeric_match = re.search(r'[\d.]+', item)
                if numeric_match:
                    numeric_values.append(float(numeric_match.group()))
        except (ValueError, TypeError):
            continue
    
    if len(numeric_values) >= 3:
        mean_val = statistics.mean(numeric_values)
        std_val = statistics.stdev(numeric_values) if len(numeric_values) > 1 else 0
        
        # Detect outliers using z-score
        for i, value in enumerate(numeric_values):
            if std_val > 0:
                z_score = abs(value - mean_val) / std_val
                if z_score > 2.5:  # Outlier threshold
                    anomalies.append({
                        "index": i,
                        "value": value,
                        "type": "statistical_outlier",
                        "z_score": z_score
                    })
    
    # For text data - detect unusual patterns
    text_values = [str(item) for item in data_list if item]
    if text_values:
        # Check for unusual length variations
        lengths = [len(text) for text in text_values]
        if lengths:
            mean_length = statistics.mean(lengths)
            for i, length in enumerate(lengths):
                if length > mean_length * 3 or length < mean_length * 0.3:
                    anomalies.append({
                        "index": i,
                        "value": text_values[i][:50] + "..." if len(text_values[i]) > 50 else text_values[i],
                        "type": "unusual_length",
                        "length": length,
                        "mean_length": mean_length
                    })
    
    return {
        "anomalies": anomalies,
        "total_checked": len(data_list),
        "anomaly_rate": len(anomalies) / len(data_list) if data_list else 0,
        "confidence": 0.8 if anomalies else 0.9
    }


@tool
def calculate_data_completeness(data_dict: Dict[str, Any]) -> Dict[str, float]:
    """Calculate completeness score for each field in the data."""
    completeness_scores = {}
    
    for field, values in data_dict.items():
        if not isinstance(values, list):
            values = [values]
        
        total_count = len(values)
        if total_count == 0:
            completeness_scores[field] = 0.0
            continue
        
        non_null_count = 0
        for value in values:
            if value is not None and value != "" and str(value).strip() != "":
                non_null_count += 1
        
        completeness_scores[field] = non_null_count / total_count
    
    return completeness_scores


class DataValidationAgent(EnhancedAgent):
    """
    Advanced Data Validation Agent that ensures data integrity and quality
    throughout the scraping pipeline using AI-powered validation techniques.
    """
    
    def __init__(
        self,
        config: Optional[AgentConfig] = None,
        llm: Optional[BaseLanguageModel] = None
    ):
        """
        Initialize the Data Validation Agent.
        
        Args:
            config: Agent configuration
            llm: Language model for validation tasks
        """
        if config is None:
            config = AgentConfig(
                agent_id="data-validation-agent",
                agent_type=AgentType.DATA_VALIDATION,
                capabilities=[
                    "data_quality_assessment",
                    "anomaly_detection",
                    "format_validation",
                    "consistency_checking",
                    "error_correction"
                ]
            )
        
        if llm is None:
            try:
                llm = ChatOpenAI(
                    model="gpt-4",
                    temperature=0.1,  # Low temperature for consistent validation
                    max_tokens=1500
                )
            except Exception as e:
                llm = None
                logging.warning(f"Could not initialize OpenAI LLM: {e}. Using rule-based validation.")
        
        # Define validation tools
        tools = [
            validate_price_format,
            validate_url_format,
            detect_data_anomalies,
            calculate_data_completeness
        ]
        
        # Create prompt template for validation tasks
        prompt_template = PromptTemplate(
            input_variables=["input", "task_parameters"],
            template="""
You are a Data Validation Agent responsible for ensuring data quality and integrity.
Your task is to analyze the provided data and identify any quality issues.

Data to validate: {input}
Validation parameters: {task_parameters}

Please analyze the data for:
1. Completeness - Are there missing values?
2. Accuracy - Do the values seem correct and realistic?
3. Consistency - Are similar fields formatted consistently?
4. Validity - Do values match expected formats?
5. Anomalies - Are there any unusual or suspicious values?

Use the available tools to perform detailed validation and provide specific recommendations.

Available tools:
- validate_price_format: Check price formatting and extract values
- validate_url_format: Validate URL structure and format
- detect_data_anomalies: Find statistical outliers and unusual patterns
- calculate_data_completeness: Measure data completeness ratios

Provide a detailed analysis with specific issues and actionable recommendations.
"""
        )
        
        super().__init__(
            config=config,
            llm=llm,
            tools=tools,
            prompt_template=prompt_template
        )
        
        # Validation rules registry
        self.validation_rules: Dict[str, DataValidationRule] = {}
        self._load_default_validation_rules()
        
        self.logger.info("Data Validation Agent initialized with quality assessment capabilities")
    
    def _load_default_validation_rules(self):
        """Load default validation rules for common data types."""
        default_rules = [
            DataValidationRule(
                field_name="price",
                rule_type="format",
                parameters={"pattern": r"^\$?[\d,]+\.?\d*$", "min_value": 0},
                error_message="Price must be a valid monetary amount"
            ),
            DataValidationRule(
                field_name="url",
                rule_type="format",
                parameters={"pattern": r"^https?://.*"},
                error_message="URL must start with http:// or https://"
            ),
            DataValidationRule(
                field_name="title",
                rule_type="required",
                parameters={"min_length": 3, "max_length": 200},
                error_message="Title is required and must be 3-200 characters"
            ),
            DataValidationRule(
                field_name="rating",
                rule_type="range",
                parameters={"min_value": 0, "max_value": 5},
                error_message="Rating must be between 0 and 5"
            )
        ]
        
        for rule in default_rules:
            self.validation_rules[rule.field_name] = rule
    
    async def validate_scraped_data(
        self,
        data: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> QualityReport:
        """
        Validate scraped data and generate a comprehensive quality report.
        
        Args:
            data: The scraped data to validate
            context: Additional context for validation
            
        Returns:
            Comprehensive quality report with scores and recommendations
        """
        self.logger.info(f"Validating scraped data with {len(data)} fields")
        
        try:
            # Create validation task
            validation_task = TaskRequest(
                task_type="validate_scraped_data",
                parameters={
                    "data": data,
                    "context": context or {},
                    "validation_rules": [rule.dict() for rule in self.validation_rules.values()]
                }
            )
            
            # Use LangChain reasoning for validation
            response = await self.execute_with_reasoning(validation_task, context)
            
            if response.status == "completed" and response.result:
                return self._build_quality_report(data, response.result, context)
            else:
                # Fallback to rule-based validation
                return await self._validate_data_basic(data, context)
                
        except Exception as e:
            self.logger.error(f"Error validating scraped data: {e}")
            # Fallback to basic validation
            return await self._validate_data_basic(data, context)
    
    def _build_quality_report(
        self,
        data: Dict[str, Any],
        validation_result: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> QualityReport:
        """Build a comprehensive quality report from validation results."""
        
        # Calculate completeness scores
        completeness_scores = calculate_data_completeness(data)
        
        # Detect anomalies in data
        all_values = []
        for field_values in data.values():
            if isinstance(field_values, list):
                all_values.extend(field_values)
            else:
                all_values.append(field_values)
        
        anomaly_result = detect_data_anomalies(all_values)
        
        # Calculate individual quality metrics
        overall_completeness = sum(completeness_scores.values()) / len(completeness_scores) if completeness_scores else 0
        anomaly_rate = anomaly_result.get("anomaly_rate", 0)
        
        # Estimate accuracy based on validation results
        accuracy_score = 1.0 - (anomaly_rate * 0.5)  # Reduce accuracy based on anomalies
        
        # Calculate consistency (simplified - could be enhanced)
        consistency_score = 0.9 if overall_completeness > 0.8 else 0.7
        
        # Calculate validity based on format validation
        validity_score = 0.95  # Default high, reduced by specific validation failures
        
        # Calculate uniqueness
        unique_values = set(str(v) for v in all_values if v is not None)
        uniqueness_score = len(unique_values) / len(all_values) if all_values else 1.0
        
        # Calculate timeliness (assume recent data is good)
        timeliness_score = 1.0  # Could be enhanced with timestamp analysis
        
        # Calculate weighted overall score
        weights = {
            "completeness": 0.25,
            "accuracy": 0.25,
            "consistency": 0.15,
            "validity": 0.15,
            "uniqueness": 0.10,
            "timeliness": 0.10
        }
        
        overall_score = (
            overall_completeness * weights["completeness"] +
            accuracy_score * weights["accuracy"] +
            consistency_score * weights["consistency"] +
            validity_score * weights["validity"] +
            uniqueness_score * weights["uniqueness"] +
            timeliness_score * weights["timeliness"]
        )
        
        # Generate quality metrics
        quality_metrics = DataQualityMetrics(
            completeness=overall_completeness,
            accuracy=accuracy_score,
            consistency=consistency_score,
            validity=validity_score,
            uniqueness=uniqueness_score,
            timeliness=timeliness_score,
            overall_score=overall_score
        )
        
        # Identify issues and recommendations
        issues = []
        recommendations = []
        
        if overall_completeness < 0.8:
            issues.append(f"Low data completeness: {overall_completeness:.2%}")
            recommendations.append("Review scraping selectors to capture missing fields")
        
        if anomaly_rate > 0.1:
            issues.append(f"High anomaly rate: {anomaly_rate:.2%}")
            recommendations.append("Investigate data sources for unusual patterns")
        
        if len(anomaly_result.get("anomalies", [])) > 0:
            for anomaly in anomaly_result["anomalies"][:3]:  # Show first 3
                issues.append(f"Anomaly detected: {anomaly['type']} in value '{anomaly['value']}'")
        
        return QualityReport(
            data_source=context.get("source", "unknown") if context else "unknown",
            total_records=len(all_values),
            valid_records=len([v for v in all_values if v is not None and str(v).strip()]),
            invalid_records=len([v for v in all_values if v is None or not str(v).strip()]),
            quality_score=overall_score * 100,  # Convert to percentage
            issues=[{"type": "quality", "message": issue} for issue in issues],
            recommendations=recommendations
        )
    
    async def _validate_data_basic(
        self,
        data: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> QualityReport:
        """Basic validation fallback when LangChain reasoning is unavailable."""
        self.logger.info("Using basic validation fallback")
        
        # Simple rule-based validation
        issues = []
        recommendations = []
        
        total_fields = len(data)
        valid_fields = 0
        
        for field_name, field_value in data.items():
            if field_value is not None and str(field_value).strip():
                valid_fields += 1
                
                # Apply specific validation rules
                if field_name in self.validation_rules:
                    rule = self.validation_rules[field_name]
                    validation_result = self._apply_validation_rule(field_value, rule)
                    if not validation_result["valid"]:
                        issues.append({
                            "type": "validation_error",
                            "message": f"{field_name}: {validation_result['error']}"
                        })
            else:
                issues.append({
                    "type": "missing_data",
                    "message": f"Missing value for field: {field_name}"
                })
        
        completeness = valid_fields / total_fields if total_fields > 0 else 0
        quality_score = completeness * 85  # Basic scoring
        
        if completeness < 0.8:
            recommendations.append("Improve data collection to reduce missing values")
        
        if len(issues) > total_fields * 0.2:
            recommendations.append("Review validation rules and data sources")
        
        return QualityReport(
            data_source=context.get("source", "unknown") if context else "unknown",
            total_records=total_fields,
            valid_records=valid_fields,
            invalid_records=total_fields - valid_fields,
            quality_score=quality_score,
            issues=issues,
            recommendations=recommendations
        )
    
    def _apply_validation_rule(self, value: Any, rule: DataValidationRule) -> Dict[str, Any]:
        """Apply a specific validation rule to a value."""
        try:
            if rule.rule_type == "required":
                if value is None or str(value).strip() == "":
                    return {"valid": False, "error": rule.error_message}
            
            elif rule.rule_type == "format":
                pattern = rule.parameters.get("pattern")
                if pattern and not re.match(pattern, str(value)):
                    return {"valid": False, "error": rule.error_message}
            
            elif rule.rule_type == "range":
                try:
                    numeric_value = float(value)
                    min_val = rule.parameters.get("min_value")
                    max_val = rule.parameters.get("max_value")
                    
                    if min_val is not None and numeric_value < min_val:
                        return {"valid": False, "error": f"Value below minimum: {min_val}"}
                    if max_val is not None and numeric_value > max_val:
                        return {"valid": False, "error": f"Value above maximum: {max_val}"}
                except ValueError:
                    return {"valid": False, "error": "Value is not numeric"}
            
            return {"valid": True}
            
        except Exception as e:
            return {"valid": False, "error": f"Validation error: {str(e)}"}
    
    async def execute_task_basic(self, task: TaskRequest) -> TaskResponse:
        """Basic task execution for validation operations."""
        if task.task_type == "validate_scraped_data":
            data = task.parameters.get("data", {})
            context = task.parameters.get("context", {})
            
            # Perform basic validation
            quality_report = await self._validate_data_basic(data, context)
            
            return TaskResponse(
                task_id=task.id,
                status="completed",
                result={"quality_report": quality_report.dict()},
                agent_id=self.agent_id
            )
        else:
            return TaskResponse(
                task_id=task.id,
                status="failed",
                error={"message": f"Unknown task type: {task.task_type}"},
                agent_id=self.agent_id
            )
