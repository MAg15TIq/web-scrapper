"""
Regression Testing Framework for Web Scraper
Detects website changes and validates scraping configurations.
"""

import asyncio
import logging
import hashlib
import json
import time
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
import aiohttp
from bs4 import BeautifulSoup
import difflib

from .automated_testing import TestCase, TestResult, TestStatus, TestSeverity


@dataclass
class WebsiteSnapshot:
    """Snapshot of a website's structure and content."""
    url: str
    timestamp: datetime
    html_hash: str
    structure_hash: str
    content_hashes: Dict[str, str] = field(default_factory=dict)
    selectors: Dict[str, List[str]] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ChangeDetection:
    """Detected change in website structure or content."""
    change_type: str  # "structure", "content", "selector", "metadata"
    severity: str     # "low", "medium", "high", "critical"
    description: str
    old_value: Any = None
    new_value: Any = None
    affected_selectors: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)


class WebsiteChangeDetector:
    """
    Detects changes in website structure and content that might
    affect scraping configurations.
    """
    
    def __init__(self, storage_path: str = "snapshots"):
        """Initialize the change detector."""
        self.logger = logging.getLogger("website_change_detector")
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(exist_ok=True)
        self.snapshots: Dict[str, List[WebsiteSnapshot]] = {}
        self.change_thresholds = {
            "structure_similarity": 0.8,
            "content_similarity": 0.7,
            "selector_tolerance": 0.9
        }
    
    async def create_snapshot(self, url: str, selectors: Optional[Dict[str, str]] = None) -> WebsiteSnapshot:
        """
        Create a snapshot of a website's current state.
        
        Args:
            url: URL to snapshot
            selectors: Optional CSS selectors to track
            
        Returns:
            Website snapshot
        """
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    html_content = await response.text()
                    status_code = response.status
            
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Generate hashes
            html_hash = hashlib.md5(html_content.encode()).hexdigest()
            structure_hash = self._generate_structure_hash(soup)
            
            # Track specific selectors
            content_hashes = {}
            selector_results = {}
            
            if selectors:
                for name, selector in selectors.items():
                    elements = soup.select(selector)
                    if elements:
                        content = ' '.join(el.get_text().strip() for el in elements)
                        content_hashes[name] = hashlib.md5(content.encode()).hexdigest()
                        selector_results[name] = [el.get_text().strip() for el in elements[:5]]  # First 5 results
            
            snapshot = WebsiteSnapshot(
                url=url,
                timestamp=datetime.now(),
                html_hash=html_hash,
                structure_hash=structure_hash,
                content_hashes=content_hashes,
                selectors=selector_results,
                metadata={
                    "status_code": status_code,
                    "content_length": len(html_content),
                    "title": soup.title.string if soup.title else None,
                    "meta_description": self._get_meta_description(soup)
                }
            )
            
            # Store snapshot
            self._store_snapshot(snapshot)
            
            self.logger.info(f"Created snapshot for {url}")
            return snapshot
            
        except Exception as e:
            self.logger.error(f"Failed to create snapshot for {url}: {e}")
            raise
    
    def _generate_structure_hash(self, soup: BeautifulSoup) -> str:
        """Generate a hash representing the page structure."""
        # Extract structural elements (tags, classes, IDs)
        structure_elements = []
        
        for element in soup.find_all(True):
            tag_info = element.name
            if element.get('class'):
                tag_info += f".{'.'.join(element['class'])}"
            if element.get('id'):
                tag_info += f"#{element['id']}"
            structure_elements.append(tag_info)
        
        structure_string = '|'.join(structure_elements)
        return hashlib.md5(structure_string.encode()).hexdigest()
    
    def _get_meta_description(self, soup: BeautifulSoup) -> Optional[str]:
        """Extract meta description from page."""
        meta_desc = soup.find('meta', attrs={'name': 'description'})
        return meta_desc.get('content') if meta_desc else None
    
    def _store_snapshot(self, snapshot: WebsiteSnapshot) -> None:
        """Store snapshot to disk."""
        url_hash = hashlib.md5(snapshot.url.encode()).hexdigest()
        snapshot_file = self.storage_path / f"{url_hash}_{snapshot.timestamp.strftime('%Y%m%d_%H%M%S')}.json"
        
        snapshot_data = {
            "url": snapshot.url,
            "timestamp": snapshot.timestamp.isoformat(),
            "html_hash": snapshot.html_hash,
            "structure_hash": snapshot.structure_hash,
            "content_hashes": snapshot.content_hashes,
            "selectors": snapshot.selectors,
            "metadata": snapshot.metadata
        }
        
        with open(snapshot_file, 'w') as f:
            json.dump(snapshot_data, f, indent=2)
        
        # Update in-memory storage
        if snapshot.url not in self.snapshots:
            self.snapshots[snapshot.url] = []
        self.snapshots[snapshot.url].append(snapshot)
    
    def load_snapshots(self, url: str) -> List[WebsiteSnapshot]:
        """Load all snapshots for a URL."""
        url_hash = hashlib.md5(url.encode()).hexdigest()
        snapshots = []
        
        for snapshot_file in self.storage_path.glob(f"{url_hash}_*.json"):
            try:
                with open(snapshot_file, 'r') as f:
                    data = json.load(f)
                
                snapshot = WebsiteSnapshot(
                    url=data["url"],
                    timestamp=datetime.fromisoformat(data["timestamp"]),
                    html_hash=data["html_hash"],
                    structure_hash=data["structure_hash"],
                    content_hashes=data["content_hashes"],
                    selectors=data["selectors"],
                    metadata=data["metadata"]
                )
                snapshots.append(snapshot)
                
            except Exception as e:
                self.logger.error(f"Failed to load snapshot {snapshot_file}: {e}")
        
        # Sort by timestamp
        snapshots.sort(key=lambda s: s.timestamp)
        self.snapshots[url] = snapshots
        return snapshots
    
    def detect_changes(self, current_snapshot: WebsiteSnapshot, previous_snapshot: WebsiteSnapshot) -> List[ChangeDetection]:
        """
        Detect changes between two snapshots.
        
        Args:
            current_snapshot: Current website snapshot
            previous_snapshot: Previous website snapshot
            
        Returns:
            List of detected changes
        """
        changes = []
        
        # Check HTML changes
        if current_snapshot.html_hash != previous_snapshot.html_hash:
            changes.append(ChangeDetection(
                change_type="content",
                severity="medium",
                description="HTML content has changed",
                old_value=previous_snapshot.html_hash,
                new_value=current_snapshot.html_hash
            ))
        
        # Check structure changes
        if current_snapshot.structure_hash != previous_snapshot.structure_hash:
            changes.append(ChangeDetection(
                change_type="structure",
                severity="high",
                description="Page structure has changed",
                old_value=previous_snapshot.structure_hash,
                new_value=current_snapshot.structure_hash
            ))
        
        # Check selector-specific content changes
        for selector_name in current_snapshot.content_hashes:
            if selector_name in previous_snapshot.content_hashes:
                old_hash = previous_snapshot.content_hashes[selector_name]
                new_hash = current_snapshot.content_hashes[selector_name]
                
                if old_hash != new_hash:
                    changes.append(ChangeDetection(
                        change_type="selector",
                        severity="medium",
                        description=f"Content for selector '{selector_name}' has changed",
                        old_value=previous_snapshot.selectors.get(selector_name, []),
                        new_value=current_snapshot.selectors.get(selector_name, []),
                        affected_selectors=[selector_name]
                    ))
            else:
                changes.append(ChangeDetection(
                    change_type="selector",
                    severity="low",
                    description=f"New selector '{selector_name}' detected",
                    new_value=current_snapshot.selectors.get(selector_name, []),
                    affected_selectors=[selector_name]
                ))
        
        # Check for removed selectors
        for selector_name in previous_snapshot.content_hashes:
            if selector_name not in current_snapshot.content_hashes:
                changes.append(ChangeDetection(
                    change_type="selector",
                    severity="critical",
                    description=f"Selector '{selector_name}' no longer returns results",
                    old_value=previous_snapshot.selectors.get(selector_name, []),
                    affected_selectors=[selector_name]
                ))
        
        # Check metadata changes
        for key in ["title", "meta_description"]:
            old_value = previous_snapshot.metadata.get(key)
            new_value = current_snapshot.metadata.get(key)
            
            if old_value != new_value:
                changes.append(ChangeDetection(
                    change_type="metadata",
                    severity="low",
                    description=f"Page {key} has changed",
                    old_value=old_value,
                    new_value=new_value
                ))
        
        return changes
    
    async def monitor_url(self, url: str, selectors: Dict[str, str], check_interval: int = 3600) -> None:
        """
        Continuously monitor a URL for changes.
        
        Args:
            url: URL to monitor
            selectors: CSS selectors to track
            check_interval: Check interval in seconds
        """
        self.logger.info(f"Starting monitoring for {url} (interval: {check_interval}s)")
        
        while True:
            try:
                # Create new snapshot
                current_snapshot = await self.create_snapshot(url, selectors)
                
                # Load previous snapshots
                snapshots = self.load_snapshots(url)
                
                if len(snapshots) > 1:
                    # Compare with previous snapshot
                    previous_snapshot = snapshots[-2]  # Second to last
                    changes = self.detect_changes(current_snapshot, previous_snapshot)
                    
                    if changes:
                        self.logger.warning(f"Detected {len(changes)} changes for {url}")
                        for change in changes:
                            self.logger.warning(f"  - {change.description} (severity: {change.severity})")
                    else:
                        self.logger.info(f"No changes detected for {url}")
                
                await asyncio.sleep(check_interval)
                
            except Exception as e:
                self.logger.error(f"Error monitoring {url}: {e}")
                await asyncio.sleep(60)  # Wait 1 minute before retrying


class RegressionTester:
    """
    Regression testing framework for web scraper configurations.
    
    Validates that scraping configurations continue to work correctly
    after website changes.
    """
    
    def __init__(self, change_detector: WebsiteChangeDetector):
        """Initialize regression tester."""
        self.logger = logging.getLogger("regression_tester")
        self.change_detector = change_detector
        self.test_configurations: Dict[str, Dict[str, Any]] = {}
        self.baseline_results: Dict[str, Dict[str, Any]] = {}
    
    def register_configuration(self, config_id: str, config: Dict[str, Any]) -> None:
        """
        Register a scraping configuration for regression testing.
        
        Args:
            config_id: Unique identifier for the configuration
            config: Scraping configuration
        """
        self.test_configurations[config_id] = config
        self.logger.info(f"Registered configuration: {config_id}")
    
    async def create_baseline(self, config_id: str) -> Dict[str, Any]:
        """
        Create baseline results for a configuration.
        
        Args:
            config_id: Configuration identifier
            
        Returns:
            Baseline test results
        """
        if config_id not in self.test_configurations:
            raise ValueError(f"Configuration not found: {config_id}")
        
        config = self.test_configurations[config_id]
        
        # Execute scraping with current configuration
        baseline_results = await self._execute_scraping_test(config)
        
        # Store baseline
        self.baseline_results[config_id] = {
            "timestamp": datetime.now().isoformat(),
            "results": baseline_results,
            "config": config.copy()
        }
        
        self.logger.info(f"Created baseline for configuration: {config_id}")
        return baseline_results
    
    async def run_regression_test(self, config_id: str) -> TestResult:
        """
        Run regression test for a configuration.
        
        Args:
            config_id: Configuration identifier
            
        Returns:
            Test result
        """
        if config_id not in self.test_configurations:
            raise ValueError(f"Configuration not found: {config_id}")
        
        if config_id not in self.baseline_results:
            raise ValueError(f"No baseline found for configuration: {config_id}")
        
        start_time = time.time()
        
        try:
            # Execute current test
            current_results = await self._execute_scraping_test(self.test_configurations[config_id])
            
            # Compare with baseline
            baseline = self.baseline_results[config_id]["results"]
            comparison = self._compare_results(baseline, current_results)
            
            # Determine test status
            if comparison["passed"]:
                status = TestStatus.PASSED
                message = "Regression test passed - results match baseline"
            else:
                status = TestStatus.FAILED
                message = f"Regression test failed - {comparison['differences']} differences found"
            
            return TestResult(
                test_id=f"regression_{config_id}",
                status=status,
                execution_time=time.time() - start_time,
                message=message,
                details=comparison
            )
            
        except Exception as e:
            return TestResult(
                test_id=f"regression_{config_id}",
                status=TestStatus.ERROR,
                execution_time=time.time() - start_time,
                message=f"Regression test error: {str(e)}"
            )
    
    async def _execute_scraping_test(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a scraping test and return results."""
        # This is a simplified implementation
        # In a real system, this would execute the actual scraping workflow
        
        url = config.get("url")
        selectors = config.get("selectors", {})
        
        if not url:
            raise ValueError("URL required in configuration")
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                html_content = await response.text()
        
        soup = BeautifulSoup(html_content, 'html.parser')
        results = {}
        
        for name, selector in selectors.items():
            elements = soup.select(selector)
            results[name] = {
                "count": len(elements),
                "sample_texts": [el.get_text().strip() for el in elements[:3]],
                "selector": selector
            }
        
        return results
    
    def _compare_results(self, baseline: Dict[str, Any], current: Dict[str, Any]) -> Dict[str, Any]:
        """Compare current results with baseline."""
        comparison = {
            "passed": True,
            "differences": 0,
            "details": {}
        }
        
        # Check for missing selectors
        for selector_name in baseline:
            if selector_name not in current:
                comparison["passed"] = False
                comparison["differences"] += 1
                comparison["details"][selector_name] = {
                    "type": "missing_selector",
                    "baseline": baseline[selector_name],
                    "current": None
                }
                continue
            
            baseline_data = baseline[selector_name]
            current_data = current[selector_name]
            
            # Compare counts (with tolerance)
            baseline_count = baseline_data.get("count", 0)
            current_count = current_data.get("count", 0)
            
            count_tolerance = 0.1  # 10% tolerance
            if abs(baseline_count - current_count) > baseline_count * count_tolerance:
                comparison["passed"] = False
                comparison["differences"] += 1
                comparison["details"][selector_name] = {
                    "type": "count_mismatch",
                    "baseline_count": baseline_count,
                    "current_count": current_count,
                    "tolerance": count_tolerance
                }
        
        # Check for new selectors
        for selector_name in current:
            if selector_name not in baseline:
                comparison["details"][selector_name] = {
                    "type": "new_selector",
                    "current": current[selector_name]
                }
        
        return comparison
