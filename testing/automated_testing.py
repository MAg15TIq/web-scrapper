"""
Automated Testing Framework for Web Scraper Configurations
"""

import asyncio
import logging
import time
import json
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
import pytest
from datetime import datetime

from models.task import Task, TaskType, TaskStatus


class TestStatus(str, Enum):
    """Test execution status."""
    PENDING = "pending"
    RUNNING = "running"
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    ERROR = "error"


class TestSeverity(str, Enum):
    """Test failure severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class TestResult:
    """Result of a test execution."""
    test_id: str
    status: TestStatus
    execution_time: float
    message: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    severity: TestSeverity = TestSeverity.MEDIUM
    timestamp: datetime = field(default_factory=datetime.now)
    artifacts: List[str] = field(default_factory=list)


@dataclass
class TestCase:
    """Individual test case definition."""
    test_id: str
    name: str
    description: str
    test_function: Callable
    config: Dict[str, Any] = field(default_factory=dict)
    expected_results: Dict[str, Any] = field(default_factory=dict)
    timeout: int = 30
    retry_count: int = 0
    tags: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    severity: TestSeverity = TestSeverity.MEDIUM


class TestSuite:
    """Collection of related test cases."""
    
    def __init__(self, name: str, description: str = ""):
        """Initialize test suite."""
        self.name = name
        self.description = description
        self.test_cases: Dict[str, TestCase] = {}
        self.setup_function: Optional[Callable] = None
        self.teardown_function: Optional[Callable] = None
        self.logger = logging.getLogger(f"test_suite.{name}")
    
    def add_test(self, test_case: TestCase) -> None:
        """Add a test case to the suite."""
        self.test_cases[test_case.test_id] = test_case
        self.logger.info(f"Added test case: {test_case.test_id}")
    
    def remove_test(self, test_id: str) -> bool:
        """Remove a test case from the suite."""
        if test_id in self.test_cases:
            del self.test_cases[test_id]
            self.logger.info(f"Removed test case: {test_id}")
            return True
        return False
    
    def get_test(self, test_id: str) -> Optional[TestCase]:
        """Get a specific test case."""
        return self.test_cases.get(test_id)
    
    def list_tests(self) -> List[str]:
        """List all test case IDs."""
        return list(self.test_cases.keys())
    
    def set_setup(self, setup_function: Callable) -> None:
        """Set suite setup function."""
        self.setup_function = setup_function
    
    def set_teardown(self, teardown_function: Callable) -> None:
        """Set suite teardown function."""
        self.teardown_function = teardown_function


class AutomatedTestRunner:
    """
    Automated test runner for web scraper configurations.
    
    Executes test suites and individual test cases with comprehensive
    reporting and artifact collection.
    """
    
    def __init__(self):
        """Initialize the test runner."""
        self.logger = logging.getLogger("automated_test_runner")
        self.test_suites: Dict[str, TestSuite] = {}
        self.test_results: Dict[str, TestResult] = {}
        self.execution_history: List[Dict[str, Any]] = []
        self.parallel_execution = True
        self.max_workers = 4
    
    def register_suite(self, test_suite: TestSuite) -> None:
        """Register a test suite."""
        self.test_suites[test_suite.name] = test_suite
        self.logger.info(f"Registered test suite: {test_suite.name}")
    
    def unregister_suite(self, suite_name: str) -> bool:
        """Unregister a test suite."""
        if suite_name in self.test_suites:
            del self.test_suites[suite_name]
            self.logger.info(f"Unregistered test suite: {suite_name}")
            return True
        return False
    
    async def run_test_case(self, test_case: TestCase, context: Optional[Dict[str, Any]] = None) -> TestResult:
        """
        Execute a single test case.
        
        Args:
            test_case: Test case to execute
            context: Optional execution context
            
        Returns:
            Test execution result
        """
        start_time = time.time()
        test_result = TestResult(
            test_id=test_case.test_id,
            status=TestStatus.RUNNING
        )
        
        try:
            self.logger.info(f"Starting test: {test_case.test_id}")
            
            # Execute test function with timeout
            if asyncio.iscoroutinefunction(test_case.test_function):
                result = await asyncio.wait_for(
                    test_case.test_function(test_case.config, context),
                    timeout=test_case.timeout
                )
            else:
                result = test_case.test_function(test_case.config, context)
            
            # Validate results
            if self._validate_test_result(result, test_case.expected_results):
                test_result.status = TestStatus.PASSED
                test_result.message = "Test passed successfully"
            else:
                test_result.status = TestStatus.FAILED
                test_result.message = "Test validation failed"
                test_result.details = {
                    "expected": test_case.expected_results,
                    "actual": result
                }
            
        except asyncio.TimeoutError:
            test_result.status = TestStatus.FAILED
            test_result.message = f"Test timed out after {test_case.timeout} seconds"
            test_result.severity = TestSeverity.HIGH
            
        except Exception as e:
            test_result.status = TestStatus.ERROR
            test_result.message = f"Test execution error: {str(e)}"
            test_result.severity = TestSeverity.CRITICAL
            self.logger.error(f"Test {test_case.test_id} failed with error: {e}")
        
        finally:
            test_result.execution_time = time.time() - start_time
            self.test_results[test_case.test_id] = test_result
            
            self.logger.info(
                f"Test {test_case.test_id} completed: {test_result.status} "
                f"({test_result.execution_time:.2f}s)"
            )
        
        return test_result
    
    async def run_test_suite(self, suite_name: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, TestResult]:
        """
        Execute all tests in a test suite.
        
        Args:
            suite_name: Name of the test suite to run
            context: Optional execution context
            
        Returns:
            Dictionary of test results
        """
        if suite_name not in self.test_suites:
            raise ValueError(f"Test suite not found: {suite_name}")
        
        suite = self.test_suites[suite_name]
        suite_results = {}
        
        try:
            # Run suite setup
            if suite.setup_function:
                self.logger.info(f"Running setup for suite: {suite_name}")
                if asyncio.iscoroutinefunction(suite.setup_function):
                    await suite.setup_function(context)
                else:
                    suite.setup_function(context)
            
            # Execute test cases
            if self.parallel_execution:
                # Run tests in parallel
                tasks = []
                for test_case in suite.test_cases.values():
                    task = asyncio.create_task(self.run_test_case(test_case, context))
                    tasks.append(task)
                
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                for i, result in enumerate(results):
                    test_case = list(suite.test_cases.values())[i]
                    if isinstance(result, Exception):
                        suite_results[test_case.test_id] = TestResult(
                            test_id=test_case.test_id,
                            status=TestStatus.ERROR,
                            execution_time=0,
                            message=f"Parallel execution error: {str(result)}"
                        )
                    else:
                        suite_results[test_case.test_id] = result
            else:
                # Run tests sequentially
                for test_case in suite.test_cases.values():
                    result = await self.run_test_case(test_case, context)
                    suite_results[test_case.test_id] = result
            
        finally:
            # Run suite teardown
            if suite.teardown_function:
                self.logger.info(f"Running teardown for suite: {suite_name}")
                try:
                    if asyncio.iscoroutinefunction(suite.teardown_function):
                        await suite.teardown_function(context)
                    else:
                        suite.teardown_function(context)
                except Exception as e:
                    self.logger.error(f"Suite teardown failed: {e}")
        
        # Record execution history
        self.execution_history.append({
            "suite_name": suite_name,
            "timestamp": datetime.now().isoformat(),
            "results": {test_id: result.status.value for test_id, result in suite_results.items()},
            "total_tests": len(suite_results),
            "passed": sum(1 for r in suite_results.values() if r.status == TestStatus.PASSED),
            "failed": sum(1 for r in suite_results.values() if r.status == TestStatus.FAILED),
            "errors": sum(1 for r in suite_results.values() if r.status == TestStatus.ERROR)
        })
        
        return suite_results
    
    async def run_all_suites(self, context: Optional[Dict[str, Any]] = None) -> Dict[str, Dict[str, TestResult]]:
        """
        Execute all registered test suites.
        
        Args:
            context: Optional execution context
            
        Returns:
            Dictionary mapping suite names to test results
        """
        all_results = {}
        
        for suite_name in self.test_suites.keys():
            try:
                suite_results = await self.run_test_suite(suite_name, context)
                all_results[suite_name] = suite_results
            except Exception as e:
                self.logger.error(f"Failed to run test suite {suite_name}: {e}")
                all_results[suite_name] = {}
        
        return all_results
    
    def _validate_test_result(self, actual: Any, expected: Dict[str, Any]) -> bool:
        """
        Validate test results against expected values.
        
        Args:
            actual: Actual test result
            expected: Expected result criteria
            
        Returns:
            True if validation passes, False otherwise
        """
        if not expected:
            return True  # No validation criteria
        
        try:
            # Check if result is a dictionary
            if isinstance(actual, dict):
                for key, expected_value in expected.items():
                    if key not in actual:
                        return False
                    
                    if isinstance(expected_value, dict) and "operator" in expected_value:
                        # Handle complex validation rules
                        if not self._evaluate_condition(actual[key], expected_value):
                            return False
                    else:
                        # Simple equality check
                        if actual[key] != expected_value:
                            return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Validation error: {e}")
            return False
    
    def _evaluate_condition(self, value: Any, condition: Dict[str, Any]) -> bool:
        """Evaluate a validation condition."""
        operator = condition.get("operator")
        expected = condition.get("value")
        
        if operator == "equals":
            return value == expected
        elif operator == "greater_than":
            return value > expected
        elif operator == "less_than":
            return value < expected
        elif operator == "contains":
            return expected in value
        elif operator == "not_empty":
            return bool(value)
        elif operator == "length":
            return len(value) == expected
        
        return False
    
    def get_test_results(self, suite_name: Optional[str] = None) -> Dict[str, TestResult]:
        """Get test results for a specific suite or all tests."""
        if suite_name:
            suite = self.test_suites.get(suite_name)
            if suite:
                return {
                    test_id: result for test_id, result in self.test_results.items()
                    if test_id in suite.test_cases
                }
            return {}
        
        return self.test_results.copy()
    
    def generate_report(self, format: str = "json") -> str:
        """
        Generate a test execution report.
        
        Args:
            format: Report format ("json", "html", "text")
            
        Returns:
            Formatted report string
        """
        if format == "json":
            return self._generate_json_report()
        elif format == "html":
            return self._generate_html_report()
        elif format == "text":
            return self._generate_text_report()
        else:
            raise ValueError(f"Unsupported report format: {format}")
    
    def _generate_json_report(self) -> str:
        """Generate JSON format report."""
        report_data = {
            "timestamp": datetime.now().isoformat(),
            "summary": self._get_summary_stats(),
            "test_results": {
                test_id: {
                    "status": result.status.value,
                    "execution_time": result.execution_time,
                    "message": result.message,
                    "severity": result.severity.value,
                    "timestamp": result.timestamp.isoformat()
                }
                for test_id, result in self.test_results.items()
            },
            "execution_history": self.execution_history
        }
        
        return json.dumps(report_data, indent=2)
    
    def _generate_html_report(self) -> str:
        """Generate HTML format report."""
        summary = self._get_summary_stats()
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Web Scraper Test Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .summary {{ background: #f8f9fa; padding: 15px; border-radius: 5px; margin-bottom: 20px; }}
                .passed {{ color: #28a745; }}
                .failed {{ color: #dc3545; }}
                .error {{ color: #fd7e14; }}
                table {{ width: 100%; border-collapse: collapse; }}
                th, td {{ padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <h1>Web Scraper Test Report</h1>
            <div class="summary">
                <h3>Summary</h3>
                <p>Total Tests: {summary['total']}</p>
                <p class="passed">Passed: {summary['passed']}</p>
                <p class="failed">Failed: {summary['failed']}</p>
                <p class="error">Errors: {summary['errors']}</p>
                <p>Success Rate: {summary['success_rate']:.1f}%</p>
            </div>
            
            <h3>Test Results</h3>
            <table>
                <tr>
                    <th>Test ID</th>
                    <th>Status</th>
                    <th>Execution Time</th>
                    <th>Message</th>
                </tr>
        """
        
        for test_id, result in self.test_results.items():
            status_class = result.status.value
            html += f"""
                <tr>
                    <td>{test_id}</td>
                    <td class="{status_class}">{result.status.value.upper()}</td>
                    <td>{result.execution_time:.2f}s</td>
                    <td>{result.message}</td>
                </tr>
            """
        
        html += """
            </table>
        </body>
        </html>
        """
        
        return html
    
    def _generate_text_report(self) -> str:
        """Generate plain text format report."""
        summary = self._get_summary_stats()
        
        report = f"""
Web Scraper Test Report
=======================

Summary:
--------
Total Tests: {summary['total']}
Passed: {summary['passed']}
Failed: {summary['failed']}
Errors: {summary['errors']}
Success Rate: {summary['success_rate']:.1f}%

Test Results:
-------------
"""
        
        for test_id, result in self.test_results.items():
            report += f"""
{test_id}: {result.status.value.upper()} ({result.execution_time:.2f}s)
  Message: {result.message}
"""
        
        return report
    
    def _get_summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics for test results."""
        total = len(self.test_results)
        passed = sum(1 for r in self.test_results.values() if r.status == TestStatus.PASSED)
        failed = sum(1 for r in self.test_results.values() if r.status == TestStatus.FAILED)
        errors = sum(1 for r in self.test_results.values() if r.status == TestStatus.ERROR)
        
        success_rate = (passed / total * 100) if total > 0 else 0
        
        return {
            "total": total,
            "passed": passed,
            "failed": failed,
            "errors": errors,
            "success_rate": success_rate
        }
    
    def clear_results(self) -> None:
        """Clear all test results and history."""
        self.test_results.clear()
        self.execution_history.clear()
        self.logger.info("Test results and history cleared")


# Built-in test functions for common scenarios
async def test_url_accessibility(config: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Test if a URL is accessible."""
    import aiohttp

    url = config.get("url")
    if not url:
        raise ValueError("URL not provided in config")

    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            return {
                "status_code": response.status,
                "accessible": response.status == 200,
                "response_time": response.headers.get("X-Response-Time", "unknown")
            }


async def test_selector_extraction(config: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Test CSS selector extraction."""
    from bs4 import BeautifulSoup
    import aiohttp

    url = config.get("url")
    selectors = config.get("selectors", {})

    if not url or not selectors:
        raise ValueError("URL and selectors required in config")

    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            html = await response.text()

    soup = BeautifulSoup(html, 'html.parser')
    results = {}

    for name, selector in selectors.items():
        elements = soup.select(selector)
        results[name] = {
            "found": len(elements),
            "selector": selector,
            "sample_text": elements[0].get_text().strip() if elements else None
        }

    return results


def test_data_validation(config: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Test data validation rules."""
    data = config.get("data")
    rules = config.get("validation_rules", {})

    if not data or not rules:
        raise ValueError("Data and validation rules required in config")

    results = {
        "valid": True,
        "errors": [],
        "warnings": []
    }

    for field, rule in rules.items():
        if field not in data:
            results["errors"].append(f"Missing required field: {field}")
            results["valid"] = False
            continue

        value = data[field]

        # Check data type
        if "type" in rule:
            expected_type = rule["type"]
            if expected_type == "string" and not isinstance(value, str):
                results["errors"].append(f"Field {field} should be string, got {type(value).__name__}")
                results["valid"] = False
            elif expected_type == "number" and not isinstance(value, (int, float)):
                results["errors"].append(f"Field {field} should be number, got {type(value).__name__}")
                results["valid"] = False

        # Check length constraints
        if "min_length" in rule and len(str(value)) < rule["min_length"]:
            results["errors"].append(f"Field {field} too short (min: {rule['min_length']})")
            results["valid"] = False

        if "max_length" in rule and len(str(value)) > rule["max_length"]:
            results["warnings"].append(f"Field {field} very long (max recommended: {rule['max_length']})")

    return results
