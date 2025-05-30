"""
Real-time monitoring dashboard for the web scraping system.
"""
import time
import asyncio
from typing import Dict, Any, Optional, List, Union
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, BarColumn, TextColumn, TaskProgressColumn, TimeRemainingColumn
from rich.panel import Panel
from rich.layout import Layout
from rich.live import Live
from rich.text import Text
from rich import box

class ScrapingDashboard:
    """
    Real-time monitoring dashboard for the web scraping system.
    """
    def __init__(self, console: Optional[Console] = None):
        """
        Initialize a new scraping dashboard.
        
        Args:
            console: Rich console to use for output. If None, a new console is created.
        """
        self.console = console or Console()
        self.layout = self._create_layout()
        self.live = Live(self.layout, console=self.console, refresh_per_second=4)
        
        # Initialize statistics
        self.stats = {
            "scraping_progress": 0.0,
            "active_agents": 0,
            "failed_tasks": 0,
            "data_collected": 0,
            "throughput": 0.0,
            "start_time": time.time(),
            "last_update": time.time(),
            "requests_total": 0,
            "requests_success": 0,
            "requests_failed": 0,
            "agent_stats": {},
            "domain_stats": {},
        }
        
        # Request rate tracking
        self.request_times = []
        self.request_window = 60  # Calculate throughput over last 60 seconds
    
    def _create_layout(self) -> Layout:
        """
        Create the dashboard layout.
        
        Returns:
            A Layout object for the dashboard.
        """
        layout = Layout()
        
        # Split into sections
        layout.split(
            Layout(name="header", size=3),
            Layout(name="main"),
            Layout(name="footer", size=3)
        )
        
        # Split main section
        layout["main"].split_row(
            Layout(name="left", ratio=2),
            Layout(name="right", ratio=3)
        )
        
        # Split left section
        layout["left"].split(
            Layout(name="progress", size=5),
            Layout(name="stats")
        )
        
        # Split right section
        layout["right"].split(
            Layout(name="agents", ratio=1),
            Layout(name="domains", ratio=1)
        )
        
        # Set initial content
        layout["header"].update(Panel(
            Text("Web Scraping System - Real-Time Monitoring", style="bold blue", justify="center"),
            box=box.ROUNDED
        ))
        layout["footer"].update(Panel(
            Text("Press Ctrl+C to exit", style="italic", justify="center"),
            box=box.ROUNDED
        ))
        
        return layout
    
    def start(self) -> None:
        """Start the dashboard."""
        self.live.start()
    
    def stop(self) -> None:
        """Stop the dashboard."""
        self.live.stop()
    
    def update(self, stats: Dict[str, Any]) -> None:
        """
        Update the dashboard with new statistics.
        
        Args:
            stats: Dictionary of statistics to update.
        """
        # Update statistics
        self.stats.update(stats)
        
        # Update last update time
        self.stats["last_update"] = time.time()
        
        # Calculate throughput
        current_time = time.time()
        self.request_times.append(current_time)
        
        # Remove requests older than the window
        while self.request_times and self.request_times[0] < current_time - self.request_window:
            self.request_times.pop(0)
        
        # Calculate requests per second
        if len(self.request_times) > 1:
            window_duration = max(1, current_time - self.request_times[0])
            self.stats["throughput"] = len(self.request_times) / window_duration
        
        # Update layout
        self._update_layout()
        
        # Refresh the display
        self.live.refresh()
    
    def _update_layout(self) -> None:
        """Update the layout with current statistics."""
        # Update progress section
        self.layout["progress"].update(self._create_progress_panel())
        
        # Update stats section
        self.layout["stats"].update(self._create_stats_panel())
        
        # Update agents section
        self.layout["agents"].update(self._create_agents_panel())
        
        # Update domains section
        self.layout["domains"].update(self._create_domains_panel())
    
    def _create_progress_panel(self) -> Panel:
        """
        Create a panel showing the scraping progress.
        
        Returns:
            A Panel object containing the progress bar.
        """
        progress = self.stats["scraping_progress"]
        progress_int = int(progress * 100)
        
        # Create progress bar
        bar_width = 40
        filled = int(bar_width * progress)
        bar = f"[{'█' * filled}{' ' * (bar_width - filled)}]"
        
        text = Text()
        text.append("\n")
        text.append("Scraping Progress ", style="bold")
        text.append(f"{bar} ", style="bold blue")
        text.append(f"{progress_int}%\n", style="bold green")
        
        return Panel(text, title="Progress", border_style="green", box=box.ROUNDED)
    
    def _create_stats_panel(self) -> Panel:
        """
        Create a panel showing key statistics.
        
        Returns:
            A Panel object containing the statistics.
        """
        # Create table
        table = Table(show_header=False, box=None)
        table.add_column("Stat")
        table.add_column("Value")
        
        # Format data collected
        data_collected = self.stats["data_collected"]
        if data_collected >= 1_000_000:
            data_str = f"{data_collected / 1_000_000:.1f}M rows"
        elif data_collected >= 1_000:
            data_str = f"{data_collected / 1_000:.1f}K rows"
        else:
            data_str = f"{data_collected} rows"
        
        # Add rows
        table.add_row("Active Agents", f"[bold]{self.stats['active_agents']}[/bold]")
        table.add_row("Failed Tasks", f"[bold red]{self.stats['failed_tasks']}[/bold red]")
        table.add_row("Data Collected", f"[bold green]{data_str}[/bold green]")
        table.add_row("Throughput", f"[bold blue]{self.stats['throughput']:.1f}[/bold blue] reqs/sec")
        
        # Calculate elapsed time
        elapsed = time.time() - self.stats["start_time"]
        hours, remainder = divmod(elapsed, 3600)
        minutes, seconds = divmod(remainder, 60)
        table.add_row("Elapsed Time", f"{int(hours):02}:{int(minutes):02}:{int(seconds):02}")
        
        # Add success rate
        total_requests = self.stats["requests_success"] + self.stats["requests_failed"]
        if total_requests > 0:
            success_rate = self.stats["requests_success"] / total_requests * 100
            table.add_row("Success Rate", f"[bold green]{success_rate:.1f}%[/bold green]")
        
        return Panel(table, title="Statistics", border_style="blue", box=box.ROUNDED)
    
    def _create_agents_panel(self) -> Panel:
        """
        Create a panel showing agent information.
        
        Returns:
            A Panel object containing agent information.
        """
        # Create table
        table = Table(show_header=True, box=box.SIMPLE)
        table.add_column("Agent ID", style="cyan")
        table.add_column("Type", style="green")
        table.add_column("Status", style="yellow")
        table.add_column("Tasks", justify="right")
        
        # Add rows for each agent
        for agent_id, agent_info in self.stats.get("agent_stats", {}).items():
            status_style = "green" if agent_info.get("status") == "idle" else "yellow"
            table.add_row(
                agent_id,
                agent_info.get("type", "unknown"),
                f"[{status_style}]{agent_info.get('status', 'unknown')}[/{status_style}]",
                str(agent_info.get("tasks", 0))
            )
        
        return Panel(table, title="Active Agents", border_style="cyan", box=box.ROUNDED)
    
    def _create_domains_panel(self) -> Panel:
        """
        Create a panel showing domain statistics.
        
        Returns:
            A Panel object containing domain statistics.
        """
        # Create table
        table = Table(show_header=True, box=box.SIMPLE)
        table.add_column("Domain", style="cyan")
        table.add_column("Requests", justify="right")
        table.add_column("Rate", justify="right")
        table.add_column("Success", justify="right", style="green")
        
        # Add rows for each domain
        for domain, domain_info in self.stats.get("domain_stats", {}).items():
            success_rate = domain_info.get("success_rate", 1.0) * 100
            rate_style = "green" if success_rate >= 90 else "yellow" if success_rate >= 70 else "red"
            
            table.add_row(
                domain,
                str(domain_info.get("request_count", 0)),
                f"{domain_info.get('current_rate', 0.0):.2f}/s",
                f"[{rate_style}]{success_rate:.1f}%[/{rate_style}]"
            )
        
        return Panel(table, title="Domain Statistics", border_style="cyan", box=box.ROUNDED)

class DashboardManager:
    """
    Manager for the scraping dashboard.
    """
    def __init__(self):
        """Initialize a new dashboard manager."""
        self.dashboard = ScrapingDashboard()
        self.running = False
        self.update_interval = 0.5  # seconds
    
    async def start(self) -> None:
        """Start the dashboard."""
        self.running = True
        self.dashboard.start()
        
        # Start update loop
        asyncio.create_task(self._update_loop())
    
    async def stop(self) -> None:
        """Stop the dashboard."""
        self.running = False
        self.dashboard.stop()
    
    async def _update_loop(self) -> None:
        """Update loop for the dashboard."""
        while self.running:
            # Collect statistics from various sources
            stats = await self._collect_stats()
            
            # Update the dashboard
            self.dashboard.update(stats)
            
            # Wait for next update
            await asyncio.sleep(self.update_interval)
    
    async def _collect_stats(self) -> Dict[str, Any]:
        """
        Collect statistics from various sources.
        
        Returns:
            A dictionary of statistics.
        """
        # This is a placeholder for actual statistics collection
        # In a real implementation, you would collect statistics from agents, rate limiters, etc.
        return {}
    
    def update_stats(self, stats: Dict[str, Any]) -> None:
        """
        Update statistics manually.
        
        Args:
            stats: Dictionary of statistics to update.
        """
        self.dashboard.update(stats)

# Singleton instance
dashboard_manager = DashboardManager()
