"""
Enhanced Quality Assurance Agent with Advanced AI & Intelligence Features.

This agent provides AI-powered quality assessment, anomaly detection,
and intelligent data validation with confidence metrics.
"""
import asyncio
import logging
import time
import json
import statistics
from typing import Dict, Any, List, Optional, Union, Tuple
from collections import defaultdict, Counter
import numpy as np

from agents.base import Agent
from models.message import Message, TaskMessage, ResultMessage, ErrorMessage, StatusMessage, Priority
from models.task import Task, TaskStatus, TaskType

# Enhanced AI features
try:
    import spacy
    from textblob import TextBlob
    ADVANCED_NLP_AVAILABLE = True
except ImportError:
    ADVANCED_NLP_AVAILABLE = False
    logging.warning("Advanced NLP libraries not available. Install spacy and textblob for enhanced features.")

try:
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import DBSCAN
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    logging.warning("Machine learning libraries not available. Install scikit-learn for ML features.")


class EnhancedQualityAssuranceAgent(Agent):
    """
    Enhanced Quality Assurance Agent with Advanced AI & Intelligence Features.
    
    Features:
    - AI-powered quality assessment with confidence metrics
    - Anomaly detection using machine learning
    - Intelligent data validation and completeness analysis
    - Pattern-based quality scoring
    - Adaptive quality thresholds
    """
    
    def __init__(self, agent_id: Optional[str] = None, coordinator_id: str = "coordinator"):
        """
        Initialize a new Enhanced Quality Assurance Agent.

        Args:
            agent_id: Unique identifier for the agent. If None, a UUID will be generated.
            coordinator_id: ID of the coordinator agent. Used for message routing.
        """
        super().__init__(agent_id=agent_id, agent_type="enhanced_quality_assurance", coordinator_id=coordinator_id)
        
        # Enhanced quality metrics with AI support
        self.quality_metrics = self._initialize_quality_metrics()
        
        # ML models for anomaly detection and quality assessment
        self.ml_models = self._initialize_ml_models()
        
        # Adaptive quality thresholds
        self.adaptive_thresholds = self._initialize_adaptive_thresholds()
        
        # Quality assessment history for learning
        self.assessment_history: List[Dict[str, Any]] = []
        
        # Pattern recognition for quality issues
        self.quality_patterns = defaultdict(list)
        
        # Performance metrics
        self.performance_metrics = {
            "total_assessments": 0,
            "anomalies_detected": 0,
            "quality_improvements": 0,
            "false_positives": 0
        }
        
        # Register enhanced message handlers
        self.register_handler("assess_quality", self._handle_assess_quality)
        self.register_handler("detect_anomalies", self._handle_detect_anomalies)
        self.register_handler("validate_completeness", self._handle_validate_completeness)
        self.register_handler("analyze_consistency", self._handle_analyze_consistency)
        self.register_handler("suggest_improvements", self._handle_suggest_improvements)
        self.register_handler("learn_from_feedback", self._handle_learn_from_feedback)
        
        # Start enhanced periodic tasks
        self._start_enhanced_periodic_tasks()
    
    def _initialize_quality_metrics(self) -> Dict[str, Any]:
        """Initialize enhanced quality metrics system."""
        return {
            'core_metrics': {
                'completeness': {'weight': 0.25, 'threshold': 0.85},
                'accuracy': {'weight': 0.30, 'threshold': 0.90},
                'consistency': {'weight': 0.20, 'threshold': 0.80},
                'timeliness': {'weight': 0.15, 'threshold': 0.75},
                'relevance': {'weight': 0.10, 'threshold': 0.70}
            },
            'advanced_metrics': {
                'data_density': {'weight': 0.15, 'threshold': 0.70},
                'semantic_coherence': {'weight': 0.20, 'threshold': 0.75},
                'structural_integrity': {'weight': 0.25, 'threshold': 0.80},
                'content_freshness': {'weight': 0.20, 'threshold': 0.65},
                'extraction_confidence': {'weight': 0.20, 'threshold': 0.85}
            },
            'confidence_levels': {
                'high': 0.85,
                'medium': 0.65,
                'low': 0.45
            }
        }
    
    def _initialize_ml_models(self) -> Dict[str, Any]:
        """Initialize machine learning models for quality assessment."""
        models = {}
        
        if ML_AVAILABLE:
            # Isolation Forest for anomaly detection
            models['anomaly_detector'] = IsolationForest(
                contamination=0.1,
                random_state=42
            )
            
            # DBSCAN for clustering quality patterns
            models['pattern_clusterer'] = DBSCAN(
                eps=0.5,
                min_samples=5
            )
            
            # Standard scaler for feature normalization
            models['scaler'] = StandardScaler()
            
            self.logger.info("ML models for quality assessment initialized")
        else:
            self.logger.warning("ML models not available - using rule-based quality assessment")
            
        return models
    
    def _initialize_adaptive_thresholds(self) -> Dict[str, float]:
        """Initialize adaptive quality thresholds."""
        return {
            'completeness_threshold': 0.85,
            'accuracy_threshold': 0.90,
            'consistency_threshold': 0.80,
            'anomaly_threshold': 0.15,
            'confidence_threshold': 0.75,
            'adaptation_rate': 0.1
        }
    
    def _start_enhanced_periodic_tasks(self) -> None:
        """Start enhanced periodic tasks."""
        asyncio.create_task(self._periodic_model_training())
        asyncio.create_task(self._periodic_threshold_adaptation())
        asyncio.create_task(self._periodic_pattern_analysis())
        asyncio.create_task(self._periodic_performance_analysis())
    
    async def _periodic_model_training(self) -> None:
        """Periodically retrain ML models with new quality data."""
        while self.running:
            try:
                if ML_AVAILABLE and len(self.assessment_history) > 50:
                    self.logger.debug("Retraining quality assessment models")
                    
                    # Extract features from assessment history
                    features = self._extract_quality_features()
                    
                    if features:
                        # Retrain anomaly detector
                        self.ml_models['anomaly_detector'].fit(features)
                        
                        # Update pattern clusters
                        clusters = self.ml_models['pattern_clusterer'].fit_predict(features)
                        self._update_quality_patterns(clusters, features)
                        
                        self.logger.debug("Quality assessment models retrained")
                
            except Exception as e:
                self.logger.error(f"Error in model training: {e}")
            
            # Sleep for 4 hours
            await asyncio.sleep(14400)
    
    async def _periodic_threshold_adaptation(self) -> None:
        """Periodically adapt quality thresholds based on performance."""
        while self.running:
            try:
                if len(self.assessment_history) > 20:
                    self.logger.debug("Adapting quality thresholds")
                    
                    # Analyze recent assessments
                    recent_assessments = self.assessment_history[-20:]
                    
                    # Calculate average quality scores
                    avg_scores = self._calculate_average_scores(recent_assessments)
                    
                    # Adapt thresholds
                    self._adapt_thresholds(avg_scores)
                    
                    self.logger.debug("Quality thresholds adapted")
                
            except Exception as e:
                self.logger.error(f"Error in threshold adaptation: {e}")
            
            # Sleep for 2 hours
            await asyncio.sleep(7200)
    
    async def _periodic_pattern_analysis(self) -> None:
        """Periodically analyze quality patterns for insights."""
        while self.running:
            try:
                self.logger.debug("Analyzing quality patterns")
                
                # Analyze common quality issues
                await self._analyze_quality_patterns()
                
                # Update pattern recognition
                await self._update_pattern_recognition()
                
                self.logger.debug("Quality pattern analysis completed")
                
            except Exception as e:
                self.logger.error(f"Error in pattern analysis: {e}")
            
            # Sleep for 1 hour
            await asyncio.sleep(3600)
    
    async def _periodic_performance_analysis(self) -> None:
        """Periodically analyze performance metrics."""
        while self.running:
            try:
                self.logger.debug("Analyzing quality assurance performance")
                
                total = self.performance_metrics['total_assessments']
                if total > 0:
                    anomaly_rate = self.performance_metrics['anomalies_detected'] / total
                    improvement_rate = self.performance_metrics['quality_improvements'] / total
                    false_positive_rate = self.performance_metrics['false_positives'] / total
                    
                    self.logger.info(
                        f"QA Performance - Anomaly rate: {anomaly_rate:.2%}, "
                        f"Improvement rate: {improvement_rate:.2%}, "
                        f"False positive rate: {false_positive_rate:.2%}"
                    )
                
                # Reset metrics for next period
                self.performance_metrics = {
                    "total_assessments": 0,
                    "anomalies_detected": 0,
                    "quality_improvements": 0,
                    "false_positives": 0
                }
                
            except Exception as e:
                self.logger.error(f"Error in performance analysis: {e}")
            
            # Sleep for 1 hour
            await asyncio.sleep(3600)

    # ===== ENHANCED QUALITY ASSESSMENT METHODS =====

    async def assess_data_quality(self, data: Union[Dict[str, Any], List[Dict[str, Any]]],
                                 schema: Optional[Dict[str, Any]] = None,
                                 context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Perform comprehensive AI-powered quality assessment.

        Args:
            data: Data to assess (single object or list)
            schema: Optional schema for validation
            context: Optional context information (URL, extraction method, etc.)

        Returns:
            Comprehensive quality assessment with confidence metrics
        """
        self.performance_metrics['total_assessments'] += 1

        try:
            # Normalize data to list format
            data_list = data if isinstance(data, list) else [data]

            # Core quality metrics
            core_assessment = await self._assess_core_quality(data_list, schema)

            # Advanced AI-powered metrics
            advanced_assessment = await self._assess_advanced_quality(data_list, context)

            # Anomaly detection
            anomaly_assessment = await self._detect_quality_anomalies(data_list)

            # Combine assessments
            combined_assessment = self._combine_quality_assessments(
                core_assessment, advanced_assessment, anomaly_assessment
            )

            # Generate confidence score
            confidence_score = self._calculate_confidence_score(combined_assessment)

            # Generate recommendations
            recommendations = await self._generate_quality_recommendations(combined_assessment, data_list)

            # Store assessment for learning
            assessment_record = {
                'timestamp': time.time(),
                'data_size': len(data_list),
                'assessment': combined_assessment,
                'confidence': confidence_score,
                'context': context or {}
            }

            self.assessment_history.append(assessment_record)

            # Limit history size
            if len(self.assessment_history) > 1000:
                self.assessment_history = self.assessment_history[-1000:]

            return {
                'overall_quality': combined_assessment['overall_score'],
                'confidence': confidence_score,
                'core_metrics': core_assessment,
                'advanced_metrics': advanced_assessment,
                'anomalies': anomaly_assessment,
                'recommendations': recommendations,
                'assessment_id': f"qa_{int(time.time())}_{len(self.assessment_history)}"
            }

        except Exception as e:
            self.logger.error(f"Error in quality assessment: {e}")
            return {
                'error': str(e),
                'overall_quality': 0.0,
                'confidence': 0.0
            }

    async def _assess_core_quality(self, data_list: List[Dict[str, Any]],
                                  schema: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Assess core quality metrics."""
        metrics = {}

        # Completeness assessment
        completeness_score = await self._assess_completeness(data_list, schema)
        metrics['completeness'] = completeness_score

        # Accuracy assessment
        accuracy_score = await self._assess_accuracy(data_list, schema)
        metrics['accuracy'] = accuracy_score

        # Consistency assessment
        consistency_score = await self._assess_consistency(data_list)
        metrics['consistency'] = consistency_score

        # Timeliness assessment
        timeliness_score = await self._assess_timeliness(data_list)
        metrics['timeliness'] = timeliness_score

        # Relevance assessment
        relevance_score = await self._assess_relevance(data_list)
        metrics['relevance'] = relevance_score

        return metrics

    async def _assess_completeness(self, data_list: List[Dict[str, Any]],
                                  schema: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Assess data completeness with AI-enhanced analysis."""
        try:
            total_fields = 0
            filled_fields = 0
            missing_patterns = defaultdict(int)

            # If schema provided, use it for completeness check
            if schema and 'required' in schema:
                required_fields = schema['required']

                for item in data_list:
                    total_fields += len(required_fields)
                    for field in required_fields:
                        if field in item and item[field] is not None and str(item[field]).strip():
                            filled_fields += 1
                        else:
                            missing_patterns[field] += 1
            else:
                # Infer expected fields from data
                all_fields = set()
                for item in data_list:
                    all_fields.update(item.keys())

                for item in data_list:
                    total_fields += len(all_fields)
                    for field in all_fields:
                        if field in item and item[field] is not None and str(item[field]).strip():
                            filled_fields += 1
                        else:
                            missing_patterns[field] += 1

            # Calculate completeness score
            completeness_ratio = filled_fields / total_fields if total_fields > 0 else 0.0

            # Adjust score based on critical missing patterns
            critical_missing = sum(1 for count in missing_patterns.values() if count > len(data_list) * 0.5)
            penalty = critical_missing * 0.1
            adjusted_score = max(0.0, completeness_ratio - penalty)

            return {
                'score': adjusted_score,
                'ratio': completeness_ratio,
                'total_fields': total_fields,
                'filled_fields': filled_fields,
                'missing_patterns': dict(missing_patterns),
                'critical_missing': critical_missing
            }

        except Exception as e:
            self.logger.error(f"Error assessing completeness: {e}")
            return {'score': 0.0, 'error': str(e)}

    async def _assess_accuracy(self, data_list: List[Dict[str, Any]],
                              schema: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Assess data accuracy using pattern recognition and validation."""
        try:
            accuracy_issues = []
            total_checks = 0
            passed_checks = 0

            for i, item in enumerate(data_list):
                for field, value in item.items():
                    if value is None:
                        continue

                    total_checks += 1
                    value_str = str(value).strip()

                    # Basic accuracy checks
                    is_accurate = True
                    issue_type = None

                    # Check for common accuracy issues
                    if not value_str:
                        is_accurate = False
                        issue_type = 'empty_value'
                    elif len(value_str) > 10000:  # Suspiciously long values
                        is_accurate = False
                        issue_type = 'excessive_length'
                    elif value_str.count('?') > len(value_str) * 0.3:  # Too many question marks
                        is_accurate = False
                        issue_type = 'placeholder_text'
                    elif 'error' in value_str.lower() or 'not found' in value_str.lower():
                        is_accurate = False
                        issue_type = 'error_text'

                    # Type-specific validation
                    if field.lower() in ['email', 'e-mail']:
                        if '@' not in value_str or '.' not in value_str:
                            is_accurate = False
                            issue_type = 'invalid_email'
                    elif field.lower() in ['url', 'link', 'website']:
                        if not value_str.startswith(('http://', 'https://', 'www.')):
                            is_accurate = False
                            issue_type = 'invalid_url'
                    elif field.lower() in ['price', 'cost', 'amount']:
                        if not any(char.isdigit() for char in value_str):
                            is_accurate = False
                            issue_type = 'invalid_price'

                    if is_accurate:
                        passed_checks += 1
                    else:
                        accuracy_issues.append({
                            'item_index': i,
                            'field': field,
                            'value': value_str[:100],  # Truncate for logging
                            'issue_type': issue_type
                        })

            # Calculate accuracy score
            accuracy_ratio = passed_checks / total_checks if total_checks > 0 else 1.0

            # Group issues by type for analysis
            issue_counts = Counter(issue['issue_type'] for issue in accuracy_issues)

            return {
                'score': accuracy_ratio,
                'total_checks': total_checks,
                'passed_checks': passed_checks,
                'issues_found': len(accuracy_issues),
                'issue_types': dict(issue_counts),
                'sample_issues': accuracy_issues[:5]  # First 5 issues for review
            }

        except Exception as e:
            self.logger.error(f"Error assessing accuracy: {e}")
            return {'score': 0.0, 'error': str(e)}
