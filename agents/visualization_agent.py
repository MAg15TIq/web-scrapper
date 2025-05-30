"""
Visualization Agent for automatic chart generation and reporting.
This agent creates intelligent visualizations and generates insights from scraped data.
"""
import asyncio
import logging
import json
import base64
from io import BytesIO
from typing import Dict, Any, Optional, List, Union, Tuple
from datetime import datetime, timedelta
from collections import Counter, defaultdict

from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import BaseTool, tool
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

try:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    import seaborn as sns
    import pandas as pd
    import numpy as np
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False
    logging.warning("Visualization libraries not available. Install matplotlib, seaborn, pandas for full functionality.")

from agents.langchain_base import EnhancedAgent
from models.langchain_models import (
    AgentConfig, AgentType, TaskRequest, TaskResponse
)


class ChartConfiguration(BaseModel):
    """Configuration for chart generation."""
    chart_type: str = Field(..., description="Type of chart (bar, line, pie, scatter, etc.)")
    title: str = Field(..., description="Chart title")
    x_axis_label: Optional[str] = None
    y_axis_label: Optional[str] = None
    color_scheme: str = Field(default="viridis", description="Color scheme for the chart")
    width: int = Field(default=12, description="Chart width in inches")
    height: int = Field(default=8, description="Chart height in inches")
    style: str = Field(default="whitegrid", description="Chart style")
    show_legend: bool = Field(default=True)
    show_grid: bool = Field(default=True)


class VisualizationRequest(BaseModel):
    """Request for visualization generation."""
    data: Dict[str, Any] = Field(..., description="Data to visualize")
    chart_config: ChartConfiguration = Field(..., description="Chart configuration")
    insights_required: bool = Field(default=True, description="Whether to generate insights")
    export_format: str = Field(default="png", description="Export format (png, svg, pdf)")


class InsightReport(BaseModel):
    """Generated insights from data analysis."""
    summary: str = Field(..., description="Executive summary of findings")
    key_insights: List[str] = Field(default_factory=list, description="Key insights discovered")
    trends: List[str] = Field(default_factory=list, description="Identified trends")
    anomalies: List[str] = Field(default_factory=list, description="Detected anomalies")
    recommendations: List[str] = Field(default_factory=list, description="Actionable recommendations")
    confidence_score: float = Field(ge=0.0, le=1.0, description="Confidence in insights")


@tool
def analyze_price_trends(price_data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze price trends from scraped e-commerce data."""
    if not price_data:
        return {"error": "No price data provided"}
    
    trends = {
        "average_price": 0,
        "price_range": {"min": float('inf'), "max": 0},
        "trend_direction": "stable",
        "volatility": 0,
        "insights": []
    }
    
    prices = []
    for item in price_data:
        try:
            if isinstance(item, dict) and 'price' in item:
                price = float(str(item['price']).replace('$', '').replace(',', ''))
                prices.append(price)
            elif isinstance(item, (int, float)):
                prices.append(float(item))
        except (ValueError, TypeError):
            continue
    
    if prices:
        trends["average_price"] = sum(prices) / len(prices)
        trends["price_range"]["min"] = min(prices)
        trends["price_range"]["max"] = max(prices)
        
        # Calculate volatility (coefficient of variation)
        if len(prices) > 1:
            mean_price = trends["average_price"]
            variance = sum((p - mean_price) ** 2 for p in prices) / len(prices)
            std_dev = variance ** 0.5
            trends["volatility"] = std_dev / mean_price if mean_price > 0 else 0
        
        # Determine trend direction (simplified)
        if len(prices) >= 3:
            first_third = prices[:len(prices)//3]
            last_third = prices[-len(prices)//3:]
            
            avg_first = sum(first_third) / len(first_third)
            avg_last = sum(last_third) / len(last_third)
            
            if avg_last > avg_first * 1.05:
                trends["trend_direction"] = "increasing"
            elif avg_last < avg_first * 0.95:
                trends["trend_direction"] = "decreasing"
        
        # Generate insights
        if trends["volatility"] > 0.2:
            trends["insights"].append("High price volatility detected across sources")
        
        price_spread = trends["price_range"]["max"] - trends["price_range"]["min"]
        if price_spread > trends["average_price"] * 0.5:
            trends["insights"].append("Significant price differences between sources")
    
    return trends


@tool
def identify_chart_type(data_structure: Dict[str, Any]) -> str:
    """Automatically identify the best chart type for given data structure."""
    
    # Analyze data characteristics
    numeric_fields = []
    categorical_fields = []
    temporal_fields = []
    
    for field, values in data_structure.items():
        if not isinstance(values, list):
            values = [values]
        
        # Check if field contains numeric data
        numeric_count = 0
        for value in values[:10]:  # Sample first 10 values
            try:
                float(str(value).replace('$', '').replace(',', ''))
                numeric_count += 1
            except (ValueError, TypeError):
                pass
        
        if numeric_count / len(values[:10]) > 0.7:
            numeric_fields.append(field)
        elif any(keyword in field.lower() for keyword in ['date', 'time', 'created', 'updated']):
            temporal_fields.append(field)
        else:
            categorical_fields.append(field)
    
    # Recommend chart type based on data structure
    if len(temporal_fields) > 0 and len(numeric_fields) > 0:
        return "line"  # Time series data
    elif len(categorical_fields) == 1 and len(numeric_fields) == 1:
        return "bar"  # Single categorical vs numeric
    elif len(numeric_fields) >= 2:
        return "scatter"  # Multiple numeric fields
    elif len(categorical_fields) == 1:
        return "pie"  # Single categorical for distribution
    else:
        return "bar"  # Default fallback


@tool
def generate_color_palette(data_size: int, style: str = "viridis") -> List[str]:
    """Generate an appropriate color palette for the data."""
    if not VISUALIZATION_AVAILABLE:
        # Return basic colors if matplotlib not available
        basic_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                       '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
        return basic_colors[:data_size] if data_size <= len(basic_colors) else basic_colors * (data_size // len(basic_colors) + 1)
    
    try:
        import matplotlib.cm as cm
        colormap = cm.get_cmap(style)
        colors = [colormap(i / max(data_size - 1, 1)) for i in range(data_size)]
        return [f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}" for r, g, b, a in colors]
    except Exception:
        # Fallback to basic colors
        return ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'][:data_size]


@tool
def extract_insights_from_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """Extract key insights and patterns from scraped data."""
    insights = {
        "summary": "",
        "key_findings": [],
        "patterns": [],
        "recommendations": []
    }
    
    # Analyze data distribution
    total_items = 0
    field_analysis = {}
    
    for field, values in data.items():
        if not isinstance(values, list):
            values = [values]
        
        total_items = max(total_items, len(values))
        
        # Analyze each field
        non_null_count = sum(1 for v in values if v is not None and str(v).strip())
        completeness = non_null_count / len(values) if values else 0
        
        field_analysis[field] = {
            "completeness": completeness,
            "unique_values": len(set(str(v) for v in values if v is not None)),
            "sample_values": values[:3]
        }
    
    # Generate summary
    insights["summary"] = f"Analyzed {total_items} items across {len(data)} fields"
    
    # Identify patterns
    high_completeness_fields = [f for f, a in field_analysis.items() if a["completeness"] > 0.9]
    low_completeness_fields = [f for f, a in field_analysis.items() if a["completeness"] < 0.5]
    
    if high_completeness_fields:
        insights["key_findings"].append(f"High data quality in: {', '.join(high_completeness_fields)}")
    
    if low_completeness_fields:
        insights["key_findings"].append(f"Data gaps detected in: {', '.join(low_completeness_fields)}")
        insights["recommendations"].append("Improve data collection for incomplete fields")
    
    # Look for price-related insights
    price_fields = [f for f in data.keys() if 'price' in f.lower() or 'cost' in f.lower()]
    if price_fields:
        insights["patterns"].append("Price data available for analysis")
        insights["recommendations"].append("Consider price trend analysis and competitive monitoring")
    
    return insights


class VisualizationAgent(EnhancedAgent):
    """
    Visualization Agent that creates intelligent charts and generates insights
    from scraped data using AI-powered analysis and automatic chart selection.
    """
    
    def __init__(
        self,
        config: Optional[AgentConfig] = None,
        llm: Optional[BaseLanguageModel] = None
    ):
        """
        Initialize the Visualization Agent.
        
        Args:
            config: Agent configuration
            llm: Language model for insight generation
        """
        if config is None:
            config = AgentConfig(
                agent_id="visualization-agent",
                agent_type=AgentType.VISUALIZATION,
                capabilities=[
                    "chart_generation",
                    "insight_analysis",
                    "trend_detection",
                    "report_creation",
                    "data_storytelling"
                ]
            )
        
        if llm is None:
            try:
                llm = ChatOpenAI(
                    model="gpt-4",
                    temperature=0.3,  # Moderate creativity for insights
                    max_tokens=2000
                )
            except Exception as e:
                llm = None
                logging.warning(f"Could not initialize OpenAI LLM: {e}. Using basic analysis.")
        
        # Define visualization tools
        tools = [
            analyze_price_trends,
            identify_chart_type,
            generate_color_palette,
            extract_insights_from_data
        ]
        
        # Create prompt template for visualization tasks
        prompt_template = PromptTemplate(
            input_variables=["input", "task_parameters"],
            template="""
You are a Visualization Agent specialized in creating meaningful charts and extracting insights from data.
Your role is to analyze data patterns and create compelling visualizations that tell a story.

Data to analyze: {input}
Visualization parameters: {task_parameters}

Please analyze the data and:
1. Identify the most appropriate chart type for the data structure
2. Extract key insights and patterns from the data
3. Detect trends, anomalies, and interesting findings
4. Generate actionable recommendations based on the analysis
5. Create a compelling narrative that explains the data story

Use the available tools to perform detailed analysis and provide specific insights.

Available tools:
- analyze_price_trends: Analyze pricing patterns and trends
- identify_chart_type: Determine optimal chart type for data
- generate_color_palette: Create appropriate color schemes
- extract_insights_from_data: Find patterns and generate insights

Focus on creating visualizations that are both informative and actionable for business decisions.
"""
        )
        
        super().__init__(
            config=config,
            llm=llm,
            tools=tools,
            prompt_template=prompt_template
        )
        
        # Set up matplotlib style if available
        if VISUALIZATION_AVAILABLE:
            plt.style.use('seaborn-v0_8')
            sns.set_palette("husl")
        
        self.logger.info("Visualization Agent initialized with chart generation capabilities")
    
    async def create_visualization(
        self,
        data: Dict[str, Any],
        chart_config: Optional[ChartConfiguration] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Create a visualization from scraped data.
        
        Args:
            data: The data to visualize
            chart_config: Chart configuration (auto-generated if None)
            context: Additional context for visualization
            
        Returns:
            Visualization result with chart and insights
        """
        self.logger.info(f"Creating visualization for data with {len(data)} fields")
        
        try:
            # Auto-generate chart config if not provided
            if chart_config is None:
                chart_type = identify_chart_type(data)
                chart_config = ChartConfiguration(
                    chart_type=chart_type,
                    title=context.get("title", "Data Analysis") if context else "Data Analysis"
                )
            
            # Create visualization task
            viz_task = TaskRequest(
                task_type="create_visualization",
                parameters={
                    "data": data,
                    "chart_config": chart_config.dict(),
                    "context": context or {}
                }
            )
            
            # Use LangChain reasoning for analysis
            response = await self.execute_with_reasoning(viz_task, context)
            
            if response.status == "completed" and response.result:
                return await self._build_visualization(data, chart_config, response.result, context)
            else:
                # Fallback to basic visualization
                return await self._create_visualization_basic(data, chart_config, context)
                
        except Exception as e:
            self.logger.error(f"Error creating visualization: {e}")
            # Fallback to basic visualization
            return await self._create_visualization_basic(data, chart_config, context)
    
    async def _build_visualization(
        self,
        data: Dict[str, Any],
        chart_config: ChartConfiguration,
        analysis_result: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Build visualization with AI-generated insights."""
        
        # Extract insights from analysis
        insights = extract_insights_from_data(data)
        
        # Generate chart if visualization libraries are available
        chart_data = None
        if VISUALIZATION_AVAILABLE:
            chart_data = await self._generate_chart(data, chart_config)
        
        # Create insight report
        insight_report = InsightReport(
            summary=insights.get("summary", "Data analysis completed"),
            key_insights=insights.get("key_findings", []),
            trends=insights.get("patterns", []),
            recommendations=insights.get("recommendations", []),
            confidence_score=0.85  # Default confidence
        )
        
        return {
            "chart_config": chart_config.dict(),
            "chart_data": chart_data,
            "insights": insight_report.dict(),
            "data_summary": {
                "total_fields": len(data),
                "total_records": max(len(v) if isinstance(v, list) else 1 for v in data.values()) if data else 0,
                "chart_type": chart_config.chart_type
            },
            "generated_at": datetime.now().isoformat()
        }
    
    async def _create_visualization_basic(
        self,
        data: Dict[str, Any],
        chart_config: Optional[ChartConfiguration] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Basic visualization creation fallback."""
        self.logger.info("Using basic visualization fallback")
        
        if chart_config is None:
            chart_config = ChartConfiguration(
                chart_type="bar",
                title="Data Overview"
            )
        
        # Basic insights extraction
        insights = extract_insights_from_data(data)
        
        # Generate basic chart if possible
        chart_data = None
        if VISUALIZATION_AVAILABLE:
            chart_data = await self._generate_chart(data, chart_config)
        
        return {
            "chart_config": chart_config.dict(),
            "chart_data": chart_data,
            "insights": {
                "summary": insights.get("summary", "Basic analysis completed"),
                "key_insights": insights.get("key_findings", []),
                "recommendations": insights.get("recommendations", [])
            },
            "data_summary": {
                "total_fields": len(data),
                "chart_type": chart_config.chart_type
            }
        }
    
    async def _generate_chart(
        self,
        data: Dict[str, Any],
        chart_config: ChartConfiguration
    ) -> Optional[str]:
        """Generate chart and return as base64 encoded image."""
        if not VISUALIZATION_AVAILABLE:
            return None
        
        try:
            # Create figure
            plt.figure(figsize=(chart_config.width, chart_config.height))
            
            # Prepare data for plotting
            if chart_config.chart_type == "bar":
                # Create bar chart
                fields = list(data.keys())[:5]  # Limit to 5 fields
                values = []
                
                for field in fields:
                    field_data = data[field]
                    if isinstance(field_data, list):
                        # Count non-null values
                        values.append(sum(1 for v in field_data if v is not None))
                    else:
                        values.append(1 if field_data is not None else 0)
                
                plt.bar(fields, values)
                plt.title(chart_config.title)
                plt.xlabel("Fields")
                plt.ylabel("Count")
                plt.xticks(rotation=45)
            
            elif chart_config.chart_type == "pie":
                # Create pie chart from first categorical field
                first_field = list(data.keys())[0]
                field_data = data[first_field]
                
                if isinstance(field_data, list):
                    # Count occurrences
                    counter = Counter(str(v) for v in field_data if v is not None)
                    labels = list(counter.keys())[:8]  # Limit to 8 slices
                    sizes = [counter[label] for label in labels]
                else:
                    labels = [first_field]
                    sizes = [1]
                
                plt.pie(sizes, labels=labels, autopct='%1.1f%%')
                plt.title(chart_config.title)
            
            else:
                # Default to simple data overview
                plt.text(0.5, 0.5, f"Chart Type: {chart_config.chart_type}\nData Fields: {len(data)}", 
                        ha='center', va='center', transform=plt.gca().transAxes)
                plt.title(chart_config.title)
            
            # Save to base64
            buffer = BytesIO()
            plt.savefig(buffer, format='png', bbox_inches='tight', dpi=150)
            buffer.seek(0)
            
            chart_base64 = base64.b64encode(buffer.getvalue()).decode()
            plt.close()
            
            return chart_base64
            
        except Exception as e:
            self.logger.error(f"Error generating chart: {e}")
            plt.close()
            return None
    
    async def execute_task_basic(self, task: TaskRequest) -> TaskResponse:
        """Basic task execution for visualization operations."""
        if task.task_type == "create_visualization":
            data = task.parameters.get("data", {})
            chart_config_dict = task.parameters.get("chart_config", {})
            context = task.parameters.get("context", {})
            
            # Create chart config
            chart_config = ChartConfiguration(**chart_config_dict)
            
            # Create basic visualization
            result = await self._create_visualization_basic(data, chart_config, context)
            
            return TaskResponse(
                task_id=task.id,
                status="completed",
                result={"visualization": result},
                agent_id=self.agent_id
            )
        else:
            return TaskResponse(
                task_id=task.id,
                status="failed",
                error={"message": f"Unknown task type: {task.task_type}"},
                agent_id=self.agent_id
            )
