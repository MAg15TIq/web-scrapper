"""
Advanced Testing Framework for Web Scraper
Automated testing, regression testing, and performance benchmarking.
"""

from .automated_testing import AutomatedTestRunner, TestCase, TestSuite
from .regression_testing import RegressionTester, WebsiteChangeDetector
from .performance_benchmarks import PerformanceBenchmark, BenchmarkRunner
from .mock_server import MockServer, MockResponse

__all__ = [
    'AutomatedTestRunner',
    'TestCase',
    'TestSuite',
    'RegressionTester',
    'WebsiteChangeDetector',
    'PerformanceBenchmark',
    'BenchmarkRunner',
    'MockServer',
    'MockResponse'
]

__version__ = "1.0.0"
