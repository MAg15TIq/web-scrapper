"""
Enhanced Planning Agent with LangChain reasoning capabilities.
This agent creates sophisticated execution strategies for web scraping operations.
"""
import asyncio
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from urllib.parse import urlparse

from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import BaseTool, tool
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from agents.langchain_base import EnhancedAgent
from models.langchain_models import (
    AgentConfig, AgentType, TaskRequest, TaskResponse,
    ScrapingRequest, ScrapeStrategy, ExecutionPlan, Priority
)


class SiteAnalysisResult(BaseModel):
    """Result of website analysis for strategy planning."""
    site_url: str
    site_type: str = "unknown"  # e-commerce, news, social, etc.
    requires_js: bool = False
    has_rate_limiting: bool = False
    has_anti_bot: bool = False
    estimated_complexity: str = "medium"  # low, medium, high
    recommended_strategy: str = "standard"
    risk_factors: List[str] = []


@tool
def analyze_website_structure(url: str) -> Dict[str, Any]:
    """Analyze website structure to determine scraping strategy."""
    parsed_url = urlparse(url)
    domain = parsed_url.netloc.lower()
    
    # Common website patterns and their characteristics
    site_patterns = {
        'amazon': {
            'type': 'e-commerce',
            'requires_js': True,
            'has_anti_bot': True,
            'complexity': 'high',
            'strategy': 'anti_detection'
        },
        'ebay': {
            'type': 'e-commerce',
            'requires_js': True,
            'has_anti_bot': True,
            'complexity': 'high',
            'strategy': 'anti_detection'
        },
        'shopify': {
            'type': 'e-commerce',
            'requires_js': False,
            'has_anti_bot': False,
            'complexity': 'medium',
            'strategy': 'standard'
        },
        'wordpress': {
            'type': 'blog',
            'requires_js': False,
            'has_anti_bot': False,
            'complexity': 'low',
            'strategy': 'standard'
        }
    }
    
    # Check for known patterns
    for pattern, characteristics in site_patterns.items():
        if pattern in domain:
            return characteristics
    
    # Default analysis for unknown sites
    return {
        'type': 'unknown',
        'requires_js': False,
        'has_anti_bot': False,
        'complexity': 'medium',
        'strategy': 'standard'
    }


@tool
def estimate_resource_requirements(
    sites: List[str],
    data_points: List[str],
    complexity: str = "medium"
) -> Dict[str, Any]:
    """Estimate resource requirements for scraping operation."""
    base_time_per_site = {
        'low': 30,      # seconds
        'medium': 120,  # seconds
        'high': 300     # seconds
    }
    
    time_per_site = base_time_per_site.get(complexity, 120)
    total_estimated_time = len(sites) * time_per_site
    
    # Adjust for data points complexity
    data_point_multiplier = 1 + (len(data_points) * 0.1)
    total_estimated_time *= data_point_multiplier
    
    return {
        'estimated_duration': int(total_estimated_time),
        'memory_requirement': len(sites) * 50,  # MB
        'cpu_requirement': 'medium',
        'network_bandwidth': 'medium',
        'concurrent_agents': min(len(sites), 5)
    }


@tool
def assess_risks_and_challenges(sites: List[str], strategies: List[Dict[str, Any]]) -> List[str]:
    """Assess potential risks and challenges for the scraping operation."""
    risks = []
    
    for i, site in enumerate(sites):
        strategy = strategies[i] if i < len(strategies) else {}
        
        if strategy.get('has_anti_bot', False):
            risks.append(f"Anti-bot detection on {site}")
        
        if strategy.get('requires_js', False):
            risks.append(f"JavaScript rendering required for {site}")
        
        if strategy.get('complexity') == 'high':
            risks.append(f"High complexity site: {site}")
    
    # General risks
    if len(sites) > 10:
        risks.append("Large number of sites may trigger rate limiting")
    
    return risks


@tool
def create_fallback_strategies(primary_strategies: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Create fallback strategies for when primary strategies fail."""
    fallbacks = []
    
    for strategy in primary_strategies:
        fallback = {
            'type': 'fallback',
            'original_strategy': strategy.get('strategy', 'standard'),
            'fallback_actions': []
        }
        
        if strategy.get('requires_js', False):
            fallback['fallback_actions'].append('try_without_js')
        
        if strategy.get('has_anti_bot', False):
            fallback['fallback_actions'].extend([
                'increase_delays',
                'rotate_user_agents',
                'use_proxy_rotation'
            ])
        
        fallbacks.append(fallback)
    
    return fallbacks


class EnhancedPlanningAgent(EnhancedAgent):
    """
    Enhanced Planning Agent that uses LangChain reasoning to create
    sophisticated execution strategies for web scraping operations.
    """
    
    def __init__(
        self,
        config: Optional[AgentConfig] = None,
        llm: Optional[BaseLanguageModel] = None
    ):
        """
        Initialize the Enhanced Planning Agent.
        
        Args:
            config: Agent configuration
            llm: Language model for planning tasks
        """
        if config is None:
            config = AgentConfig(
                agent_id="enhanced-planning-agent",
                agent_type=AgentType.PLANNING_AGENT,
                capabilities=[
                    "strategy_planning",
                    "resource_estimation",
                    "risk_assessment",
                    "workflow_optimization"
                ]
            )
        
        if llm is None:
            llm = ChatOpenAI(
                model="gpt-4",
                temperature=0.2,  # Slightly higher for creative planning
                max_tokens=2000
            )
        
        # Define tools for planning tasks
        tools = [
            analyze_website_structure,
            estimate_resource_requirements,
            assess_risks_and_challenges,
            create_fallback_strategies
        ]
        
        # Create prompt template for planning tasks
        prompt_template = PromptTemplate(
            input_variables=["input", "task_parameters"],
            template="""
You are an Enhanced Planning Agent for a sophisticated web scraping system.
Your role is to analyze scraping requests and create optimal execution strategies.

Scraping Request: {input}
Parameters: {task_parameters}

Please create a comprehensive execution plan that includes:

1. **Site Analysis**: Analyze each target site to understand its characteristics
2. **Strategy Selection**: Choose the best scraping strategy for each site
3. **Resource Planning**: Estimate time, memory, and computational requirements
4. **Risk Assessment**: Identify potential challenges and risks
5. **Fallback Planning**: Create backup strategies for when primary approaches fail
6. **Optimization**: Suggest optimizations for efficiency and success rate

Use the available tools to gather information and make informed decisions.

Available tools:
- analyze_website_structure: Analyze website characteristics
- estimate_resource_requirements: Calculate resource needs
- assess_risks_and_challenges: Identify potential issues
- create_fallback_strategies: Plan backup approaches

Think step by step and create a detailed, actionable plan.
"""
        )
        
        super().__init__(
            config=config,
            llm=llm,
            tools=tools,
            prompt_template=prompt_template
        )
        
        self.logger.info("Enhanced Planning Agent initialized with strategic planning capabilities")
    
    async def create_execution_plan(
        self,
        scraping_request: ScrapingRequest,
        context: Optional[Dict[str, Any]] = None
    ) -> ExecutionPlan:
        """
        Create a comprehensive execution plan for a scraping request.
        
        Args:
            scraping_request: The scraping request to plan for
            context: Additional context information
            
        Returns:
            Detailed execution plan
        """
        self.logger.info(f"Creating execution plan for request: {scraping_request.id}")
        
        try:
            # Create a task request for planning
            planning_task = TaskRequest(
                task_type="create_execution_plan",
                parameters={
                    "scraping_request": scraping_request.dict(),
                    "context": context or {}
                },
                priority=scraping_request.priority
            )
            
            # Use LangChain reasoning to create the plan
            response = await self.execute_with_reasoning(planning_task, context)
            
            if response.status == "completed" and response.result:
                return self._build_execution_plan(scraping_request, response.result, context)
            else:
                # Fallback to basic planning
                return await self._create_plan_basic(scraping_request, context)
                
        except Exception as e:
            self.logger.error(f"Error creating execution plan: {e}")
            # Fallback to basic planning
            return await self._create_plan_basic(scraping_request, context)
    
    def _build_execution_plan(
        self,
        scraping_request: ScrapingRequest,
        planning_result: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> ExecutionPlan:
        """
        Build an ExecutionPlan from planning results.
        
        Args:
            scraping_request: Original scraping request
            planning_result: Results from LangChain planning
            context: Additional context
            
        Returns:
            Comprehensive execution plan
        """
        # Analyze each target site
        strategies = []
        for site in scraping_request.target_sites:
            site_analysis = analyze_website_structure(site)
            
            strategy = ScrapeStrategy(
                site_url=site,
                strategy_type=site_analysis.get('strategy', 'standard'),
                use_javascript=site_analysis.get('requires_js', False),
                use_proxy=site_analysis.get('has_anti_bot', False),
                rate_limit_delay=self._calculate_rate_limit(site_analysis),
                selectors=self._generate_selectors(scraping_request.data_points),
                anti_detection_config=self._create_anti_detection_config(site_analysis)
            )
            strategies.append(strategy)
        
        # Estimate resources
        resource_requirements = estimate_resource_requirements(
            scraping_request.target_sites,
            scraping_request.data_points,
            "medium"  # Default complexity
        )
        
        # Assess risks
        strategy_dicts = [s.dict() for s in strategies]
        risks = assess_risks_and_challenges(scraping_request.target_sites, strategy_dicts)
        
        # Create fallback plans
        fallbacks = create_fallback_strategies(strategy_dicts)
        
        return ExecutionPlan(
            request_id=scraping_request.id,
            strategies=strategies,
            estimated_duration=resource_requirements.get('estimated_duration', 300),
            resource_requirements=resource_requirements,
            risk_assessment={
                'identified_risks': risks,
                'risk_level': self._calculate_risk_level(risks),
                'mitigation_strategies': self._create_mitigation_strategies(risks)
            },
            fallback_plans=fallbacks
        )
    
    async def _create_plan_basic(
        self,
        scraping_request: ScrapingRequest,
        context: Optional[Dict[str, Any]] = None
    ) -> ExecutionPlan:
        """
        Create a basic execution plan as fallback.
        
        Args:
            scraping_request: The scraping request
            context: Additional context
            
        Returns:
            Basic execution plan
        """
        self.logger.info("Using basic planning fallback")
        
        # Create simple strategies for each site
        strategies = []
        for site in scraping_request.target_sites:
            strategy = ScrapeStrategy(
                site_url=site,
                strategy_type="standard",
                rate_limit_delay=1.0,
                selectors=self._generate_selectors(scraping_request.data_points)
            )
            strategies.append(strategy)
        
        return ExecutionPlan(
            request_id=scraping_request.id,
            strategies=strategies,
            estimated_duration=len(scraping_request.target_sites) * 60,  # 1 minute per site
            resource_requirements={
                'memory_requirement': 100,
                'cpu_requirement': 'low',
                'concurrent_agents': 2
            },
            risk_assessment={
                'identified_risks': ['basic_planning_used'],
                'risk_level': 'medium'
            }
        )
    
    def _calculate_rate_limit(self, site_analysis: Dict[str, Any]) -> float:
        """Calculate appropriate rate limiting delay."""
        if site_analysis.get('has_anti_bot', False):
            return 3.0  # 3 seconds for anti-bot sites
        elif site_analysis.get('complexity') == 'high':
            return 2.0  # 2 seconds for complex sites
        else:
            return 1.0  # 1 second for standard sites
    
    def _generate_selectors(self, data_points: List[str]) -> Dict[str, str]:
        """Generate CSS selectors for common data points."""
        selector_mapping = {
            'title': 'h1, .title, .product-title, .name',
            'price': '.price, .cost, .amount, [class*="price"]',
            'description': '.description, .summary, .details',
            'rating': '.rating, .stars, .score',
            'image': 'img, .image, .photo',
            'link': 'a, .link',
            'availability': '.availability, .stock, .in-stock'
        }
        
        return {point: selector_mapping.get(point, f'.{point}') for point in data_points}
    
    def _create_anti_detection_config(self, site_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Create anti-detection configuration."""
        config = {
            'rotate_user_agents': site_analysis.get('has_anti_bot', False),
            'use_random_delays': True,
            'simulate_human_behavior': site_analysis.get('has_anti_bot', False)
        }
        
        if site_analysis.get('complexity') == 'high':
            config.update({
                'use_proxy_rotation': True,
                'randomize_request_order': True,
                'add_random_headers': True
            })
        
        return config
    
    def _calculate_risk_level(self, risks: List[str]) -> str:
        """Calculate overall risk level based on identified risks."""
        if len(risks) >= 5:
            return 'high'
        elif len(risks) >= 3:
            return 'medium'
        elif len(risks) >= 1:
            return 'low'
        else:
            return 'minimal'
    
    def _create_mitigation_strategies(self, risks: List[str]) -> List[str]:
        """Create mitigation strategies for identified risks."""
        mitigations = []
        
        for risk in risks:
            if 'anti-bot' in risk.lower():
                mitigations.append('Use proxy rotation and user agent randomization')
            elif 'javascript' in risk.lower():
                mitigations.append('Enable browser automation with Playwright')
            elif 'rate limiting' in risk.lower():
                mitigations.append('Implement adaptive rate limiting')
            elif 'complexity' in risk.lower():
                mitigations.append('Use specialized extraction strategies')
        
        return mitigations
    
    async def execute_task_basic(self, task: TaskRequest) -> TaskResponse:
        """
        Basic task execution for planning operations.
        
        Args:
            task: The task to execute
            
        Returns:
            Task response
        """
        if task.task_type == "create_execution_plan":
            scraping_request_data = task.parameters.get("scraping_request", {})
            
            # Create a basic plan
            result = {
                "plan_created": True,
                "strategies_count": len(scraping_request_data.get("target_sites", [])),
                "estimated_duration": 300,  # 5 minutes default
                "risk_level": "medium"
            }
            
            return TaskResponse(
                task_id=task.id,
                status="completed",
                result={"data": result},
                agent_id=self.agent_id
            )
        else:
            return TaskResponse(
                task_id=task.id,
                status="failed",
                error={"message": f"Unknown task type: {task.task_type}"},
                agent_id=self.agent_id
            )
