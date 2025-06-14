"""
Enhanced Error Recovery Agent with Advanced AI & Intelligence Features.

This agent provides intelligent error recovery, pattern-based failure prediction,
and adaptive retry strategies that learn from success/failure patterns.
"""
import asyncio
import logging
import time
import json
import hashlib
from typing import Dict, Any, List, Optional, Union, Tuple
from collections import defaultdict, Counter, deque
from enum import Enum
import random
import math

from agents.base import Agent
from models.message import Message, TaskMessage, ResultMessage, ErrorMessage, StatusMessage, Priority
from models.task import Task, TaskStatus, TaskType

# Enhanced AI features
try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import LabelEncoder
    from sklearn.model_selection import train_test_split
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    logging.warning("Machine learning libraries not available. Install scikit-learn for ML features.")


class ErrorType(Enum):
    """Types of errors that can occur during scraping."""
    NETWORK_ERROR = "network_error"
    TIMEOUT_ERROR = "timeout_error"
    PARSING_ERROR = "parsing_error"
    AUTHENTICATION_ERROR = "authentication_error"
    RATE_LIMIT_ERROR = "rate_limit_error"
    CAPTCHA_ERROR = "captcha_error"
    JAVASCRIPT_ERROR = "javascript_error"
    SELECTOR_ERROR = "selector_error"
    DATA_VALIDATION_ERROR = "data_validation_error"
    UNKNOWN_ERROR = "unknown_error"


class RecoveryStrategy(Enum):
    """Recovery strategies for different error types."""
    RETRY_WITH_DELAY = "retry_with_delay"
    RETRY_WITH_DIFFERENT_AGENT = "retry_with_different_agent"
    RETRY_WITH_PROXY = "retry_with_proxy"
    RETRY_WITH_DIFFERENT_SELECTORS = "retry_with_different_selectors"
    FALLBACK_TO_SIMPLE_REQUEST = "fallback_to_simple_request"
    SKIP_AND_CONTINUE = "skip_and_continue"
    ESCALATE_TO_HUMAN = "escalate_to_human"
    ABORT_TASK = "abort_task"


class EnhancedErrorRecoveryAgent(Agent):
    """
    Enhanced Error Recovery Agent with Advanced AI & Intelligence Features.
    
    Features:
    - Pattern-based failure prediction using ML
    - Adaptive retry strategies that learn from success/failure patterns
    - Context-aware error handling with intelligent fallbacks
    - Self-healing capabilities for common issues
    - Proactive error prevention
    """
    
    def __init__(self, agent_id: Optional[str] = None, coordinator_id: str = "coordinator"):
        """
        Initialize a new Enhanced Error Recovery Agent.

        Args:
            agent_id: Unique identifier for the agent. If None, a UUID will be generated.
            coordinator_id: ID of the coordinator agent. Used for message routing.
        """
        super().__init__(agent_id=agent_id, agent_type="enhanced_error_recovery", coordinator_id=coordinator_id)
        
        # Error tracking and pattern recognition
        self.error_history: deque = deque(maxlen=10000)  # Circular buffer for error history
        self.error_patterns: Dict[str, Dict[str, Any]] = defaultdict(dict)
        self.success_patterns: Dict[str, Dict[str, Any]] = defaultdict(dict)
        
        # ML models for failure prediction
        self.ml_models = self._initialize_ml_models()
        
        # Adaptive retry policies
        self.retry_policies = self._initialize_adaptive_retry_policies()
        
        # Recovery strategies mapping
        self.recovery_strategies = self._initialize_recovery_strategies()
        
        # Performance metrics
        self.performance_metrics = {
            "total_errors": 0,
            "successful_recoveries": 0,
            "failed_recoveries": 0,
            "predictions_made": 0,
            "predictions_accurate": 0,
            "self_healing_actions": 0
        }
        
        # Context tracking for intelligent recovery
        self.context_tracker = {
            "current_tasks": {},
            "site_reliability": defaultdict(float),
            "agent_performance": defaultdict(dict),
            "recent_failures": deque(maxlen=100)
        }
        
        # Register enhanced message handlers
        self.register_handler("handle_error", self._handle_error)
        self.register_handler("predict_failure", self._handle_predict_failure)
        self.register_handler("suggest_recovery", self._handle_suggest_recovery)
        self.register_handler("learn_from_outcome", self._handle_learn_from_outcome)
        self.register_handler("analyze_patterns", self._handle_analyze_patterns)
        self.register_handler("self_heal", self._handle_self_heal)
        
        # Start enhanced periodic tasks
        self._start_enhanced_periodic_tasks()
    
    def _initialize_ml_models(self) -> Dict[str, Any]:
        """Initialize machine learning models for failure prediction."""
        models = {}
        
        if ML_AVAILABLE:
            # Random Forest for failure prediction
            models['failure_predictor'] = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )
            
            # Label encoder for error types
            models['error_encoder'] = LabelEncoder()
            
            # Label encoder for recovery strategies
            models['strategy_encoder'] = LabelEncoder()
            
            self.logger.info("ML models for error recovery initialized")
        else:
            self.logger.warning("ML models not available - using rule-based error recovery")
            
        return models
    
    def _initialize_adaptive_retry_policies(self) -> Dict[str, Dict[str, Any]]:
        """Initialize adaptive retry policies for different error types."""
        return {
            ErrorType.NETWORK_ERROR.value: {
                "max_retries": 5,
                "base_delay": 2.0,
                "backoff_factor": 2.0,
                "max_delay": 300.0,
                "jitter": True,
                "success_rate": 0.7
            },
            ErrorType.TIMEOUT_ERROR.value: {
                "max_retries": 3,
                "base_delay": 5.0,
                "backoff_factor": 1.5,
                "max_delay": 120.0,
                "jitter": True,
                "success_rate": 0.6
            },
            ErrorType.RATE_LIMIT_ERROR.value: {
                "max_retries": 10,
                "base_delay": 60.0,
                "backoff_factor": 1.2,
                "max_delay": 3600.0,
                "jitter": True,
                "success_rate": 0.9
            },
            ErrorType.PARSING_ERROR.value: {
                "max_retries": 2,
                "base_delay": 1.0,
                "backoff_factor": 1.0,
                "max_delay": 5.0,
                "jitter": False,
                "success_rate": 0.4
            },
            ErrorType.AUTHENTICATION_ERROR.value: {
                "max_retries": 3,
                "base_delay": 10.0,
                "backoff_factor": 2.0,
                "max_delay": 180.0,
                "jitter": False,
                "success_rate": 0.5
            },
            ErrorType.CAPTCHA_ERROR.value: {
                "max_retries": 1,
                "base_delay": 30.0,
                "backoff_factor": 1.0,
                "max_delay": 30.0,
                "jitter": False,
                "success_rate": 0.2
            },
            ErrorType.JAVASCRIPT_ERROR.value: {
                "max_retries": 3,
                "base_delay": 3.0,
                "backoff_factor": 1.5,
                "max_delay": 30.0,
                "jitter": True,
                "success_rate": 0.6
            },
            ErrorType.SELECTOR_ERROR.value: {
                "max_retries": 2,
                "base_delay": 1.0,
                "backoff_factor": 1.0,
                "max_delay": 2.0,
                "jitter": False,
                "success_rate": 0.3
            }
        }
    
    def _initialize_recovery_strategies(self) -> Dict[ErrorType, List[RecoveryStrategy]]:
        """Initialize recovery strategies for different error types."""
        return {
            ErrorType.NETWORK_ERROR: [
                RecoveryStrategy.RETRY_WITH_DELAY,
                RecoveryStrategy.RETRY_WITH_PROXY,
                RecoveryStrategy.RETRY_WITH_DIFFERENT_AGENT
            ],
            ErrorType.TIMEOUT_ERROR: [
                RecoveryStrategy.RETRY_WITH_DELAY,
                RecoveryStrategy.FALLBACK_TO_SIMPLE_REQUEST,
                RecoveryStrategy.RETRY_WITH_DIFFERENT_AGENT
            ],
            ErrorType.RATE_LIMIT_ERROR: [
                RecoveryStrategy.RETRY_WITH_DELAY,
                RecoveryStrategy.RETRY_WITH_PROXY,
                RecoveryStrategy.SKIP_AND_CONTINUE
            ],
            ErrorType.PARSING_ERROR: [
                RecoveryStrategy.RETRY_WITH_DIFFERENT_SELECTORS,
                RecoveryStrategy.FALLBACK_TO_SIMPLE_REQUEST,
                RecoveryStrategy.SKIP_AND_CONTINUE
            ],
            ErrorType.AUTHENTICATION_ERROR: [
                RecoveryStrategy.RETRY_WITH_DIFFERENT_AGENT,
                RecoveryStrategy.ESCALATE_TO_HUMAN,
                RecoveryStrategy.SKIP_AND_CONTINUE
            ],
            ErrorType.CAPTCHA_ERROR: [
                RecoveryStrategy.RETRY_WITH_PROXY,
                RecoveryStrategy.ESCALATE_TO_HUMAN,
                RecoveryStrategy.SKIP_AND_CONTINUE
            ],
            ErrorType.JAVASCRIPT_ERROR: [
                RecoveryStrategy.RETRY_WITH_DELAY,
                RecoveryStrategy.FALLBACK_TO_SIMPLE_REQUEST,
                RecoveryStrategy.RETRY_WITH_DIFFERENT_AGENT
            ],
            ErrorType.SELECTOR_ERROR: [
                RecoveryStrategy.RETRY_WITH_DIFFERENT_SELECTORS,
                RecoveryStrategy.FALLBACK_TO_SIMPLE_REQUEST,
                RecoveryStrategy.SKIP_AND_CONTINUE
            ]
        }
    
    def _start_enhanced_periodic_tasks(self) -> None:
        """Start enhanced periodic tasks."""
        asyncio.create_task(self._periodic_pattern_analysis())
        asyncio.create_task(self._periodic_model_training())
        asyncio.create_task(self._periodic_self_healing())
        asyncio.create_task(self._periodic_performance_analysis())
    
    async def _periodic_pattern_analysis(self) -> None:
        """Periodically analyze error patterns for insights."""
        while self.running:
            try:
                self.logger.debug("Analyzing error patterns")
                
                # Analyze error patterns
                await self._analyze_error_patterns()
                
                # Update recovery strategies based on patterns
                await self._update_recovery_strategies()
                
                # Clean old patterns
                await self._cleanup_old_patterns()
                
                self.logger.debug("Error pattern analysis completed")
                
            except Exception as e:
                self.logger.error(f"Error in pattern analysis: {e}")
            
            # Sleep for 30 minutes
            await asyncio.sleep(1800)
    
    async def _periodic_model_training(self) -> None:
        """Periodically retrain ML models with new error data."""
        while self.running:
            try:
                if ML_AVAILABLE and len(self.error_history) > 100:
                    self.logger.debug("Retraining error prediction models")
                    
                    # Extract training data
                    training_data = self._extract_training_data()
                    
                    if training_data:
                        # Retrain failure prediction model
                        await self._retrain_failure_predictor(training_data)
                        
                        self.logger.debug("Error prediction models retrained")
                
            except Exception as e:
                self.logger.error(f"Error in model training: {e}")
            
            # Sleep for 2 hours
            await asyncio.sleep(7200)
    
    async def _periodic_self_healing(self) -> None:
        """Periodically perform self-healing actions."""
        while self.running:
            try:
                self.logger.debug("Performing self-healing analysis")
                
                # Identify recurring issues
                recurring_issues = await self._identify_recurring_issues()
                
                # Apply self-healing actions
                for issue in recurring_issues:
                    await self._apply_self_healing(issue)
                
                self.logger.debug("Self-healing analysis completed")
                
            except Exception as e:
                self.logger.error(f"Error in self-healing: {e}")
            
            # Sleep for 1 hour
            await asyncio.sleep(3600)
    
    async def _periodic_performance_analysis(self) -> None:
        """Periodically analyze performance metrics."""
        while self.running:
            try:
                self.logger.debug("Analyzing error recovery performance")
                
                total_errors = self.performance_metrics['total_errors']
                if total_errors > 0:
                    recovery_rate = self.performance_metrics['successful_recoveries'] / total_errors
                    prediction_accuracy = (
                        self.performance_metrics['predictions_accurate'] / 
                        max(self.performance_metrics['predictions_made'], 1)
                    )
                    
                    self.logger.info(
                        f"Error Recovery Performance - Recovery rate: {recovery_rate:.2%}, "
                        f"Prediction accuracy: {prediction_accuracy:.2%}, "
                        f"Self-healing actions: {self.performance_metrics['self_healing_actions']}"
                    )
                
                # Reset metrics for next period
                self.performance_metrics = {
                    "total_errors": 0,
                    "successful_recoveries": 0,
                    "failed_recoveries": 0,
                    "predictions_made": 0,
                    "predictions_accurate": 0,
                    "self_healing_actions": 0
                }
                
            except Exception as e:
                self.logger.error(f"Error in performance analysis: {e}")
            
            # Sleep for 1 hour
            await asyncio.sleep(3600)
