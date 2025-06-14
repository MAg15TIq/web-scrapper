"""
Performance Benchmarking Framework for Web Scraper
Measures and analyzes scraping performance metrics.
"""

import asyncio
import time
import statistics
import logging
import psutil
import gc
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import json

from .automated_testing import TestCase, TestResult, TestStatus


@dataclass
class PerformanceMetrics:
    """Performance metrics for a benchmark run."""
    execution_time: float
    memory_usage_mb: float
    cpu_usage_percent: float
    network_requests: int = 0
    data_processed_mb: float = 0
    items_extracted: int = 0
    error_count: int = 0
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class BenchmarkResult:
    """Result of a performance benchmark."""
    benchmark_name: str
    test_case: str
    metrics: List[PerformanceMetrics]
    summary_stats: Dict[str, Any] = field(default_factory=dict)
    baseline_comparison: Optional[Dict[str, Any]] = None
    passed: bool = True
    notes: str = ""


class PerformanceBenchmark:
    """
    Individual performance benchmark definition.
    """
    
    def __init__(
        self,
        name: str,
        test_function: Callable,
        config: Dict[str, Any],
        iterations: int = 5,
        warmup_iterations: int = 1,
        timeout: int = 300,
        memory_threshold_mb: float = 1000,
        cpu_threshold_percent: float = 80,
        execution_time_threshold_seconds: float = 60
    ):
        """
        Initialize performance benchmark.
        
        Args:
            name: Benchmark name
            test_function: Function to benchmark
            config: Test configuration
            iterations: Number of test iterations
            warmup_iterations: Number of warmup iterations
            timeout: Timeout in seconds
            memory_threshold_mb: Memory usage threshold in MB
            cpu_threshold_percent: CPU usage threshold percentage
            execution_time_threshold_seconds: Execution time threshold
        """
        self.name = name
        self.test_function = test_function
        self.config = config
        self.iterations = iterations
        self.warmup_iterations = warmup_iterations
        self.timeout = timeout
        self.memory_threshold_mb = memory_threshold_mb
        self.cpu_threshold_percent = cpu_threshold_percent
        self.execution_time_threshold_seconds = execution_time_threshold_seconds
        self.logger = logging.getLogger(f"benchmark.{name}")


class BenchmarkRunner:
    """
    Performance benchmark runner and analyzer.
    """
    
    def __init__(self):
        """Initialize benchmark runner."""
        self.logger = logging.getLogger("benchmark_runner")
        self.benchmarks: Dict[str, PerformanceBenchmark] = {}
        self.results: Dict[str, List[BenchmarkResult]] = {}
        self.baselines: Dict[str, BenchmarkResult] = {}
    
    def register_benchmark(self, benchmark: PerformanceBenchmark) -> None:
        """Register a performance benchmark."""
        self.benchmarks[benchmark.name] = benchmark
        self.logger.info(f"Registered benchmark: {benchmark.name}")
    
    async def run_benchmark(self, benchmark_name: str, save_as_baseline: bool = False) -> BenchmarkResult:
        """
        Run a specific benchmark.
        
        Args:
            benchmark_name: Name of benchmark to run
            save_as_baseline: Whether to save results as baseline
            
        Returns:
            Benchmark result
        """
        if benchmark_name not in self.benchmarks:
            raise ValueError(f"Benchmark not found: {benchmark_name}")
        
        benchmark = self.benchmarks[benchmark_name]
        self.logger.info(f"Running benchmark: {benchmark_name}")
        
        # Warmup iterations
        if benchmark.warmup_iterations > 0:
            self.logger.info(f"Running {benchmark.warmup_iterations} warmup iterations")
            for i in range(benchmark.warmup_iterations):
                try:
                    await asyncio.wait_for(
                        benchmark.test_function(benchmark.config),
                        timeout=benchmark.timeout
                    )
                except Exception as e:
                    self.logger.warning(f"Warmup iteration {i+1} failed: {e}")
        
        # Collect garbage before benchmarking
        gc.collect()
        
        # Run benchmark iterations
        metrics = []
        for i in range(benchmark.iterations):
            self.logger.info(f"Running iteration {i+1}/{benchmark.iterations}")
            
            try:
                metric = await self._run_single_iteration(benchmark)
                metrics.append(metric)
                
                # Small delay between iterations
                await asyncio.sleep(0.1)
                
            except Exception as e:
                self.logger.error(f"Iteration {i+1} failed: {e}")
                # Create error metric
                error_metric = PerformanceMetrics(
                    execution_time=0,
                    memory_usage_mb=0,
                    cpu_usage_percent=0,
                    error_count=1
                )
                metrics.append(error_metric)
        
        # Calculate summary statistics
        summary_stats = self._calculate_summary_stats(metrics)
        
        # Check against thresholds
        passed = self._check_thresholds(benchmark, summary_stats)
        
        # Create result
        result = BenchmarkResult(
            benchmark_name=benchmark_name,
            test_case=benchmark.config.get("test_case", "default"),
            metrics=metrics,
            summary_stats=summary_stats,
            passed=passed
        )
        
        # Compare with baseline if available
        if benchmark_name in self.baselines:
            result.baseline_comparison = self._compare_with_baseline(result, self.baselines[benchmark_name])
        
        # Save results
        if benchmark_name not in self.results:
            self.results[benchmark_name] = []
        self.results[benchmark_name].append(result)
        
        # Save as baseline if requested
        if save_as_baseline:
            self.baselines[benchmark_name] = result
            self.logger.info(f"Saved benchmark {benchmark_name} as baseline")
        
        self.logger.info(f"Benchmark {benchmark_name} completed: {'PASSED' if passed else 'FAILED'}")
        return result
    
    async def _run_single_iteration(self, benchmark: PerformanceBenchmark) -> PerformanceMetrics:
        """Run a single benchmark iteration and collect metrics."""
        # Get initial system state
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Start monitoring
        start_time = time.time()
        start_cpu_time = process.cpu_percent()
        
        try:
            # Execute test function
            result = await asyncio.wait_for(
                benchmark.test_function(benchmark.config),
                timeout=benchmark.timeout
            )
            
            # Calculate metrics
            end_time = time.time()
            execution_time = end_time - start_time
            
            final_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_usage = final_memory - initial_memory
            
            cpu_usage = process.cpu_percent()
            
            # Extract additional metrics from result if available
            network_requests = 0
            data_processed_mb = 0
            items_extracted = 0
            
            if isinstance(result, dict):
                network_requests = result.get("network_requests", 0)
                data_processed_mb = result.get("data_processed_mb", 0)
                items_extracted = result.get("items_extracted", 0)
            
            return PerformanceMetrics(
                execution_time=execution_time,
                memory_usage_mb=memory_usage,
                cpu_usage_percent=cpu_usage,
                network_requests=network_requests,
                data_processed_mb=data_processed_mb,
                items_extracted=items_extracted,
                error_count=0
            )
            
        except Exception as e:
            end_time = time.time()
            execution_time = end_time - start_time
            
            return PerformanceMetrics(
                execution_time=execution_time,
                memory_usage_mb=0,
                cpu_usage_percent=0,
                error_count=1
            )
    
    def _calculate_summary_stats(self, metrics: List[PerformanceMetrics]) -> Dict[str, Any]:
        """Calculate summary statistics from metrics."""
        if not metrics:
            return {}
        
        # Filter out error metrics for calculations
        valid_metrics = [m for m in metrics if m.error_count == 0]
        
        if not valid_metrics:
            return {
                "total_errors": sum(m.error_count for m in metrics),
                "success_rate": 0.0
            }
        
        execution_times = [m.execution_time for m in valid_metrics]
        memory_usages = [m.memory_usage_mb for m in valid_metrics]
        cpu_usages = [m.cpu_usage_percent for m in valid_metrics]
        
        return {
            "iterations": len(metrics),
            "successful_iterations": len(valid_metrics),
            "total_errors": sum(m.error_count for m in metrics),
            "success_rate": len(valid_metrics) / len(metrics) * 100,
            
            "execution_time": {
                "mean": statistics.mean(execution_times),
                "median": statistics.median(execution_times),
                "min": min(execution_times),
                "max": max(execution_times),
                "std_dev": statistics.stdev(execution_times) if len(execution_times) > 1 else 0
            },
            
            "memory_usage_mb": {
                "mean": statistics.mean(memory_usages),
                "median": statistics.median(memory_usages),
                "min": min(memory_usages),
                "max": max(memory_usages),
                "std_dev": statistics.stdev(memory_usages) if len(memory_usages) > 1 else 0
            },
            
            "cpu_usage_percent": {
                "mean": statistics.mean(cpu_usages),
                "median": statistics.median(cpu_usages),
                "min": min(cpu_usages),
                "max": max(cpu_usages),
                "std_dev": statistics.stdev(cpu_usages) if len(cpu_usages) > 1 else 0
            },
            
            "total_network_requests": sum(m.network_requests for m in valid_metrics),
            "total_data_processed_mb": sum(m.data_processed_mb for m in valid_metrics),
            "total_items_extracted": sum(m.items_extracted for m in valid_metrics)
        }
    
    def _check_thresholds(self, benchmark: PerformanceBenchmark, summary_stats: Dict[str, Any]) -> bool:
        """Check if benchmark results meet performance thresholds."""
        if not summary_stats or summary_stats.get("success_rate", 0) < 80:
            return False
        
        # Check execution time threshold
        mean_execution_time = summary_stats.get("execution_time", {}).get("mean", 0)
        if mean_execution_time > benchmark.execution_time_threshold_seconds:
            return False
        
        # Check memory threshold
        max_memory_usage = summary_stats.get("memory_usage_mb", {}).get("max", 0)
        if max_memory_usage > benchmark.memory_threshold_mb:
            return False
        
        # Check CPU threshold
        max_cpu_usage = summary_stats.get("cpu_usage_percent", {}).get("max", 0)
        if max_cpu_usage > benchmark.cpu_threshold_percent:
            return False
        
        return True
    
    def _compare_with_baseline(self, current: BenchmarkResult, baseline: BenchmarkResult) -> Dict[str, Any]:
        """Compare current results with baseline."""
        comparison = {}
        
        current_stats = current.summary_stats
        baseline_stats = baseline.summary_stats
        
        # Compare execution time
        current_time = current_stats.get("execution_time", {}).get("mean", 0)
        baseline_time = baseline_stats.get("execution_time", {}).get("mean", 0)
        
        if baseline_time > 0:
            time_change_percent = ((current_time - baseline_time) / baseline_time) * 100
            comparison["execution_time_change_percent"] = time_change_percent
            comparison["execution_time_regression"] = time_change_percent > 10  # 10% threshold
        
        # Compare memory usage
        current_memory = current_stats.get("memory_usage_mb", {}).get("mean", 0)
        baseline_memory = baseline_stats.get("memory_usage_mb", {}).get("mean", 0)
        
        if baseline_memory > 0:
            memory_change_percent = ((current_memory - baseline_memory) / baseline_memory) * 100
            comparison["memory_usage_change_percent"] = memory_change_percent
            comparison["memory_usage_regression"] = memory_change_percent > 20  # 20% threshold
        
        # Compare success rate
        current_success_rate = current_stats.get("success_rate", 0)
        baseline_success_rate = baseline_stats.get("success_rate", 0)
        
        comparison["success_rate_change"] = current_success_rate - baseline_success_rate
        comparison["success_rate_regression"] = current_success_rate < baseline_success_rate - 5  # 5% threshold
        
        # Overall regression check
        comparison["has_regression"] = (
            comparison.get("execution_time_regression", False) or
            comparison.get("memory_usage_regression", False) or
            comparison.get("success_rate_regression", False)
        )
        
        return comparison
    
    async def run_all_benchmarks(self) -> Dict[str, BenchmarkResult]:
        """Run all registered benchmarks."""
        results = {}
        
        for benchmark_name in self.benchmarks:
            try:
                result = await self.run_benchmark(benchmark_name)
                results[benchmark_name] = result
            except Exception as e:
                self.logger.error(f"Failed to run benchmark {benchmark_name}: {e}")
        
        return results
    
    def generate_report(self, format: str = "json") -> str:
        """Generate performance benchmark report."""
        if format == "json":
            return self._generate_json_report()
        elif format == "html":
            return self._generate_html_report()
        else:
            raise ValueError(f"Unsupported report format: {format}")
    
    def _generate_json_report(self) -> str:
        """Generate JSON format report."""
        report_data = {
            "timestamp": datetime.now().isoformat(),
            "benchmarks": {},
            "summary": {
                "total_benchmarks": len(self.benchmarks),
                "total_results": sum(len(results) for results in self.results.values())
            }
        }
        
        for benchmark_name, results in self.results.items():
            if results:
                latest_result = results[-1]
                report_data["benchmarks"][benchmark_name] = {
                    "passed": latest_result.passed,
                    "summary_stats": latest_result.summary_stats,
                    "baseline_comparison": latest_result.baseline_comparison,
                    "iterations": len(latest_result.metrics),
                    "timestamp": latest_result.metrics[0].timestamp.isoformat() if latest_result.metrics else None
                }
        
        return json.dumps(report_data, indent=2)
    
    def _generate_html_report(self) -> str:
        """Generate HTML format report."""
        html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Performance Benchmark Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                .benchmark { margin-bottom: 30px; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }
                .passed { border-left: 5px solid #28a745; }
                .failed { border-left: 5px solid #dc3545; }
                .stats { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 10px; }
                .stat { background: #f8f9fa; padding: 10px; border-radius: 3px; }
            </style>
        </head>
        <body>
            <h1>Performance Benchmark Report</h1>
        """
        
        for benchmark_name, results in self.results.items():
            if results:
                latest_result = results[-1]
                status_class = "passed" if latest_result.passed else "failed"
                
                html += f"""
                <div class="benchmark {status_class}">
                    <h3>{benchmark_name}</h3>
                    <p>Status: {'PASSED' if latest_result.passed else 'FAILED'}</p>
                    <div class="stats">
                """
                
                stats = latest_result.summary_stats
                if "execution_time" in stats:
                    html += f"""
                    <div class="stat">
                        <strong>Execution Time</strong><br>
                        Mean: {stats['execution_time']['mean']:.2f}s<br>
                        Min: {stats['execution_time']['min']:.2f}s<br>
                        Max: {stats['execution_time']['max']:.2f}s
                    </div>
                    """
                
                html += """
                    </div>
                </div>
                """
        
        html += """
        </body>
        </html>
        """
        
        return html
