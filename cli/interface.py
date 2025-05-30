"""
Command Line Interface for the web scraping system.
"""
import os
import sys
import asyncio
import logging
from typing import Dict, Any, Optional, List, Union
import typer
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn
from rich.panel import Panel
from rich.syntax import Syntax
from rich.tree import Tree
from rich.prompt import Prompt, Confirm
import yaml

from agents.coordinator import CoordinatorAgent
from agents.scraper import ScraperAgent
from agents.parser import ParserAgent
from agents.storage import StorageAgent
from agents.javascript import JavaScriptAgent
from agents.authentication import AuthenticationAgent
from agents.anti_detection import AntiDetectionAgent
from agents.data_transformation import DataTransformationAgent
from agents.error_recovery import ErrorRecoveryAgent
from agents.data_extractor import DataExtractorAgent
from models.task import Task, TaskType, TaskStatus


app = typer.Typer(help="Multi-Agent Web Scraping System")
console = Console()


@app.command()
def scrape(
    url: str = typer.Option(..., "--url", "-u", help="URL to scrape"),
    selectors: str = typer.Option(None, "--selectors", "-s", help="CSS selectors for data extraction (format: field1:selector1,field2:selector2)"),
    output: str = typer.Option("output.json", "--output", "-o", help="Output file path"),
    format: str = typer.Option("json", "--format", "-f", help="Output format (json, csv, excel, sqlite)"),
    max_pages: int = typer.Option(1, "--max-pages", "-p", help="Maximum number of pages to scrape"),
    render_js: bool = typer.Option(False, "--render-js", "-j", help="Render JavaScript before scraping"),
    anti_detection: bool = typer.Option(False, "--anti-detection", "-a", help="Use anti-detection measures"),
    clean_data: bool = typer.Option(False, "--clean-data", "-d", help="Clean and normalize data before storing"),
    config: Optional[str] = typer.Option(None, "--config", "-c", help="Path to configuration file")
):
    """
    Scrape data from a website.
    """
    # Load configuration if provided
    config_data = {}
    if config:
        try:
            with open(config, "r") as f:
                config_data = yaml.safe_load(f)
            console.print(f"Loaded configuration from [bold]{config}[/bold]")
        except Exception as e:
            console.print(f"[bold red]Error loading configuration:[/bold red] {str(e)}")
            return

    # Parse selectors
    selector_dict = {}
    if selectors:
        try:
            for item in selectors.split(","):
                field, selector = item.split(":", 1)
                selector_dict[field.strip()] = selector.strip()
        except Exception as e:
            console.print(f"[bold red]Error parsing selectors:[/bold red] {str(e)}")
            console.print("Format should be: field1:selector1,field2:selector2")
            return

    # Merge config with command line options
    url = url or config_data.get("url")
    selector_dict = selector_dict or config_data.get("selectors", {})
    output = output or config_data.get("output", "output.json")
    format = format or config_data.get("format", "json")
    max_pages = max_pages or config_data.get("max_pages", 1)
    render_js = render_js or config_data.get("render_js", False)
    anti_detection = anti_detection or config_data.get("anti_detection", False)
    clean_data = clean_data or config_data.get("clean_data", False)

    if not url:
        console.print("[bold red]Error:[/bold red] URL is required")
        return

    # Run the scraping process
    asyncio.run(_run_scrape(url, selector_dict, output, format, max_pages, render_js, anti_detection, clean_data))


@app.command()
def interactive():
    """
    Start an interactive scraping session.
    """
    console.print(Panel.fit(
        "[bold]Multi-Agent Web Scraping System - Interactive Mode[/bold]\n\n"
        "Type [bold]help[/bold] to see available commands.",
        title="Interactive Mode",
        border_style="blue"
    ))

    # Run the interactive session
    asyncio.run(_run_interactive())


async def _run_scrape(
    url: str,
    selectors: Dict[str, str],
    output: str,
    format: str,
    max_pages: int,
    render_js: bool,
    use_anti_detection: bool = False,
    clean_data: bool = False
):
    """
    Run the scraping process.

    Args:
        url: URL to scrape.
        selectors: Dictionary of field names to CSS selectors.
        output: Output file path.
        format: Output format.
        max_pages: Maximum number of pages to scrape.
        render_js: Whether to render JavaScript before scraping.
        use_anti_detection: Whether to use anti-detection measures.
        clean_data: Whether to clean the data before storing.
    """
    # Create progress display
    with Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeRemainingColumn(),
        console=console
    ) as progress:
        # Create progress tasks
        setup_task = progress.add_task("[cyan]Setting up agents...", total=1)
        scrape_task = progress.add_task("[green]Scraping content...", total=max_pages, visible=False)
        parse_task = progress.add_task("[yellow]Parsing content...", total=1, visible=False)
        store_task = progress.add_task("[magenta]Storing data...", total=1, visible=False)

        try:
            # Create agents
            coordinator = CoordinatorAgent()
            scraper = ScraperAgent(coordinator_id=coordinator.agent_id)
            parser = ParserAgent(coordinator_id=coordinator.agent_id)
            storage = StorageAgent(coordinator_id=coordinator.agent_id)
            javascript = JavaScriptAgent(coordinator_id=coordinator.agent_id)
            authentication = AuthenticationAgent(coordinator_id=coordinator.agent_id)
            anti_detection = AntiDetectionAgent(coordinator_id=coordinator.agent_id)
            data_transformation = DataTransformationAgent(coordinator_id=coordinator.agent_id)
            error_recovery = ErrorRecoveryAgent(coordinator_id=coordinator.agent_id)
            data_extractor = DataExtractorAgent(coordinator_id=coordinator.agent_id)

            # Register agents with coordinator
            coordinator.register_agent(scraper)
            coordinator.register_agent(parser)
            coordinator.register_agent(storage)
            coordinator.register_agent(javascript)
            coordinator.register_agent(authentication)
            coordinator.register_agent(anti_detection)
            coordinator.register_agent(data_transformation)
            coordinator.register_agent(error_recovery)
            coordinator.register_agent(data_extractor)

            # Complete setup task
            progress.update(setup_task, completed=1)

            # Make scrape task visible
            progress.update(scrape_task, visible=True)

            # Create scraping tasks
            results = []

            # Add anti-detection progress task if needed
            anti_detection_task = None
            if use_anti_detection:
                anti_detection_task = progress.add_task("[blue]Setting up anti-detection...", total=1, visible=True)

                # Generate fingerprint for the domain
                from urllib.parse import urlparse
                domain = urlparse(url).netloc
                fingerprint_task = Task(
                    type=TaskType.GENERATE_FINGERPRINT,
                    parameters={
                        "domain": domain,
                        "consistent": True
                    }
                )

                fingerprint_task_id = await coordinator.submit_task(fingerprint_task)

                # Wait for task to complete
                while True:
                    task_status = coordinator.get_task_status(fingerprint_task_id)
                    if task_status and task_status["status"] in [TaskStatus.COMPLETED, TaskStatus.FAILED]:
                        break
                    await asyncio.sleep(0.1)

                # Check if task succeeded
                if task_status["status"] == TaskStatus.FAILED:
                    console.print(f"[bold red]Error generating fingerprint:[/bold red] {task_status['error']['message']}")
                else:
                    progress.update(anti_detection_task, completed=1)

                # Check if site is blocking
                check_blocking_task = Task(
                    type=TaskType.CHECK_BLOCKING,
                    parameters={
                        "url": url,
                        "check_methods": ["status_code", "content_analysis", "redirect"]
                    }
                )

                blocking_task_id = await coordinator.submit_task(check_blocking_task)

                # Wait for task to complete
                while True:
                    task_status = coordinator.get_task_status(blocking_task_id)
                    if task_status and task_status["status"] in [TaskStatus.COMPLETED, TaskStatus.FAILED]:
                        break
                    await asyncio.sleep(0.1)

                # Check if task succeeded
                if task_status["status"] == TaskStatus.FAILED:
                    console.print(f"[bold red]Error checking blocking:[/bold red] {task_status['error']['message']}")
                elif task_status["result"]["is_blocked"]:
                    console.print(f"[bold yellow]Warning: Site may be blocking scraping:[/bold yellow] {task_status['result']['reason']}")

            for page in range(1, max_pages + 1):
                # Update progress
                progress.update(scrape_task, description=f"[green]Scraping page {page}/{max_pages}...")

                # Create task for scraping
                current_url = url
                if page > 1:
                    # Simple pagination - add page parameter
                    if "?" in url:
                        current_url = f"{url}&page={page}"
                    else:
                        current_url = f"{url}?page={page}"

                # Determine task type based on whether to render JavaScript
                task_type = TaskType.RENDER_JS if render_js else TaskType.FETCH_URL

                # Create and submit task
                scrape_task_obj = Task(
                    type=task_type,
                    parameters={"url": current_url}
                )

                task_id = await coordinator.submit_task(scrape_task_obj)

                # Wait for task to complete
                while True:
                    task_status = coordinator.get_task_status(task_id)
                    if task_status and task_status["status"] in [TaskStatus.COMPLETED, TaskStatus.FAILED]:
                        break
                    await asyncio.sleep(0.1)

                # Check if task succeeded
                if task_status["status"] == TaskStatus.FAILED:
                    console.print(f"[bold red]Error scraping page {page}:[/bold red] {task_status['error']['message']}")
                    continue

                # Store result
                results.append(task_status["result"])

                # Update progress
                progress.update(scrape_task, advance=1)

                # Optimize request pattern if using anti-detection
                if use_anti_detection and page < max_pages:
                    domain = urlparse(url).netloc
                    optimize_task = Task(
                        type=TaskType.OPTIMIZE_REQUEST_PATTERN,
                        parameters={
                            "domain": domain,
                            "aggressive": False
                        }
                    )

                    optimize_task_id = await coordinator.submit_task(optimize_task)

                    # Wait for task to complete
                    while True:
                        task_status = coordinator.get_task_status(optimize_task_id)
                        if task_status and task_status["status"] in [TaskStatus.COMPLETED, TaskStatus.FAILED]:
                            break
                        await asyncio.sleep(0.1)

                    # Check if task succeeded
                    if task_status["status"] == TaskStatus.COMPLETED:
                        # Add delay between requests
                        delay = task_status["result"]["delay"]
                        await asyncio.sleep(delay)

            # Make parse task visible
            progress.update(parse_task, visible=True)

            # Parse each page
            parsed_data = []

            for i, result in enumerate(results):
                # Update progress
                progress.update(parse_task, description=f"[yellow]Parsing page {i+1}/{len(results)}...")

                # Create task for parsing
                parse_task_obj = Task(
                    type=TaskType.PARSE_CONTENT,
                    parameters={
                        "content": result["content"],
                        "selectors": selectors,
                        "normalize": True
                    }
                )

                task_id = await coordinator.submit_task(parse_task_obj)

                # Wait for task to complete
                while True:
                    task_status = coordinator.get_task_status(task_id)
                    if task_status and task_status["status"] in [TaskStatus.COMPLETED, TaskStatus.FAILED]:
                        break
                    await asyncio.sleep(0.1)

                # Check if task succeeded
                if task_status["status"] == TaskStatus.FAILED:
                    console.print(f"[bold red]Error parsing page {i+1}:[/bold red] {task_status['error']['message']}")
                    continue

                # Add parsed data
                if "extracted_data" in task_status["result"]:
                    parsed_data.append(task_status["result"]["extracted_data"])

                # Update progress
                progress.update(parse_task, completed=1)

            # Make store task visible
            progress.update(store_task, visible=True)

            # Clean data if requested
            if clean_data and parsed_data:
                clean_task = progress.add_task("[cyan]Cleaning data...", total=1, visible=True)

                # Create task for cleaning data
                clean_data_task = Task(
                    type=TaskType.CLEAN_DATA,
                    parameters={
                        "data": parsed_data,
                        "operations": [
                            {"field": "*", "operation": "strip_whitespace"},
                            {"field": "*", "operation": "remove_empty"}
                        ],
                        "add_metadata": True
                    }
                )

                clean_task_id = await coordinator.submit_task(clean_data_task)

                # Wait for task to complete
                while True:
                    task_status = coordinator.get_task_status(clean_task_id)
                    if task_status and task_status["status"] in [TaskStatus.COMPLETED, TaskStatus.FAILED]:
                        break
                    await asyncio.sleep(0.1)

                # Check if task succeeded
                if task_status["status"] == TaskStatus.FAILED:
                    console.print(f"[bold red]Error cleaning data:[/bold red] {task_status['error']['message']}")
                else:
                    # Update parsed data with cleaned data
                    parsed_data = task_status["result"]["data"]
                    console.print(f"[bold green]Data cleaned:[/bold green] {task_status['result']['original_count']} items -> {task_status['result']['cleaned_count']} items")

                    # Update progress
                    progress.update(clean_task, completed=1)

            # Store the data
            store_task_obj = Task(
                type=TaskType.STORE_DATA,
                parameters={
                    "data": parsed_data,
                    "format": format,
                    "path": output
                }
            )

            task_id = await coordinator.submit_task(store_task_obj)

            # Wait for task to complete
            while True:
                task_status = coordinator.get_task_status(task_id)
                if task_status and task_status["status"] in [TaskStatus.COMPLETED, TaskStatus.FAILED]:
                    break
                await asyncio.sleep(0.1)

            # Check if task succeeded
            if task_status["status"] == TaskStatus.FAILED:
                console.print(f"[bold red]Error storing data:[/bold red] {task_status['error']['message']}")
            else:
                # Update progress
                progress.update(store_task, completed=1)

                # Show success message
                console.print(f"\n[bold green]Successfully scraped and stored data to {output}[/bold green]")
                console.print(f"Scraped {len(results)} pages and extracted {len(parsed_data)} items")

        except Exception as e:
            console.print(f"[bold red]Error:[/bold red] {str(e)}")


async def _run_interactive():
    """Run the interactive scraping session."""
    # Create agents
    coordinator = CoordinatorAgent()
    scraper = ScraperAgent(coordinator_id=coordinator.agent_id)
    parser = ParserAgent(coordinator_id=coordinator.agent_id)
    storage = StorageAgent(coordinator_id=coordinator.agent_id)
    javascript = JavaScriptAgent(coordinator_id=coordinator.agent_id)
    authentication = AuthenticationAgent(coordinator_id=coordinator.agent_id)
    anti_detection = AntiDetectionAgent(coordinator_id=coordinator.agent_id)
    data_transformation = DataTransformationAgent(coordinator_id=coordinator.agent_id)
    error_recovery = ErrorRecoveryAgent(coordinator_id=coordinator.agent_id)
    data_extractor = DataExtractorAgent(coordinator_id=coordinator.agent_id)

    # Register agents with coordinator
    coordinator.register_agent(scraper)
    coordinator.register_agent(parser)
    coordinator.register_agent(storage)
    coordinator.register_agent(javascript)
    coordinator.register_agent(authentication)
    coordinator.register_agent(anti_detection)
    coordinator.register_agent(data_transformation)
    coordinator.register_agent(error_recovery)
    coordinator.register_agent(data_extractor)

    # Command handlers
    async def handle_scrape(args):
        """Handle the scrape command."""
        if not args:
            console.print("[bold red]Error:[/bold red] URL is required")
            return

        url = args[0]

        # Get selectors
        selectors = {}
        if len(args) > 1:
            try:
                for item in args[1].split(","):
                    field, selector = item.split(":", 1)
                    selectors[field.strip()] = selector.strip()
            except Exception:
                console.print("[bold red]Error:[/bold red] Invalid selector format")
                console.print("Format should be: field1:selector1,field2:selector2")
                return
        else:
            # Prompt for selectors
            selector_input = Prompt.ask("Enter selectors (field1:selector1,field2:selector2)")
            if selector_input:
                try:
                    for item in selector_input.split(","):
                        field, selector = item.split(":", 1)
                        selectors[field.strip()] = selector.strip()
                except Exception:
                    console.print("[bold red]Error:[/bold red] Invalid selector format")
                    return

        # Get output file
        output = Prompt.ask("Enter output file path", default="output.json")

        # Get output format
        format = Prompt.ask("Enter output format", choices=["json", "csv", "excel", "sqlite"], default="json")

        # Get max pages
        max_pages = int(Prompt.ask("Enter maximum number of pages to scrape", default="1"))

        # Get render JS option
        render_js = Confirm.ask("Render JavaScript before scraping?", default=False)

        # Get anti-detection option
        anti_detection = Confirm.ask("Use anti-detection measures?", default=False)

        # Get clean data option
        clean_data = Confirm.ask("Clean and normalize data before storing?", default=False)

        # Run the scraping process
        await _run_scrape(url, selectors, output, format, max_pages, render_js, anti_detection, clean_data)

    async def handle_help(args):
        """Handle the help command."""
        help_table = Table(title="Available Commands")
        help_table.add_column("Command", style="cyan")
        help_table.add_column("Description", style="green")
        help_table.add_column("Usage", style="yellow")

        # Basic commands
        help_table.add_row(
            "scrape",
            "Scrape data from a website",
            "scrape <url> [selectors]"
        )
        help_table.add_row(
            "preview",
            "Preview data from a file",
            "preview <file_path>"
        )
        help_table.add_row(
            "agents",
            "Show agent status",
            "agents"
        )
        help_table.add_row(
            "help",
            "Show this help message",
            "help"
        )
        help_table.add_row(
            "exit",
            "Exit the interactive session",
            "exit"
        )

        # Anti-detection commands
        help_table.add_row(
            "fingerprint",
            "Generate a browser fingerprint",
            "fingerprint [domain]"
        )
        help_table.add_row(
            "check-blocking",
            "Check if a site is blocking scraping",
            "check-blocking <url>"
        )

        # Data transformation commands
        help_table.add_row(
            "analyze-text",
            "Analyze text content",
            "analyze-text <text>"
        )
        help_table.add_row(
            "clean-data",
            "Clean and normalize data",
            "clean-data <file_path>"
        )

        # Error recovery commands
        help_table.add_row(
            "check-system",
            "Check system health",
            "check-system"
        )
        help_table.add_row(
            "recover-system",
            "Recover from system issues",
            "recover-system [type]"
        )
        help_table.add_row(
            "show-errors",
            "Show recent error history",
            "show-errors"
        )

        # Data extraction commands
        help_table.add_row(
            "extract-data",
            "Extract data from various sources",
            "extract-data <url> [type]"
        )
        help_table.add_row(
            "validate-data",
            "Validate extracted data",
            "validate-data <file_path>"
        )
        help_table.add_row(
            "cache-stats",
            "Show cache statistics",
            "cache-stats"
        )

        console.print(help_table)

    async def handle_preview(args):
        """Handle the preview command."""
        if not args:
            console.print("[bold red]Error:[/bold red] File path is required")
            return

        file_path = args[0]

        if not os.path.exists(file_path):
            console.print(f"[bold red]Error:[/bold red] File {file_path} does not exist")
            return

        # Determine file type from extension
        _, ext = os.path.splitext(file_path)
        ext = ext.lower()

        try:
            if ext == ".json":
                import json
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                if isinstance(data, list):
                    # Show first 5 items
                    preview_data = data[:5]
                    console.print(f"[bold]Showing first {len(preview_data)} of {len(data)} items:[/bold]")

                    # Create table
                    if preview_data:
                        table = Table(title=f"Preview of {file_path}")

                        # Add columns
                        columns = set()
                        for item in preview_data:
                            columns.update(item.keys())

                        for column in columns:
                            table.add_column(column)

                        # Add rows
                        for item in preview_data:
                            row = [str(item.get(column, "")) for column in columns]
                            table.add_row(*row)

                        console.print(table)
                else:
                    # Show the data as JSON
                    syntax = Syntax(json.dumps(data, indent=2), "json")
                    console.print(syntax)

            elif ext == ".csv":
                import csv
                with open(file_path, "r", encoding="utf-8") as f:
                    reader = csv.reader(f)
                    headers = next(reader)
                    rows = []
                    for i, row in enumerate(reader):
                        if i < 5:  # Show first 5 rows
                            rows.append(row)
                        else:
                            break

                # Create table
                table = Table(title=f"Preview of {file_path}")

                # Add columns
                for header in headers:
                    table.add_column(header)

                # Add rows
                for row in rows:
                    table.add_row(*row)

                console.print(table)
                console.print(f"[bold]Showing first {len(rows)} rows[/bold]")

            elif ext == ".xlsx" or ext == ".xls":
                import pandas as pd
                df = pd.read_excel(file_path)

                # Show first 5 rows
                preview_df = df.head(5)

                # Create table
                table = Table(title=f"Preview of {file_path}")

                # Add columns
                for column in preview_df.columns:
                    table.add_column(str(column))

                # Add rows
                for _, row in preview_df.iterrows():
                    table.add_row(*[str(val) for val in row])

                console.print(table)
                console.print(f"[bold]Showing first {len(preview_df)} of {len(df)} rows[/bold]")

            else:
                console.print(f"[bold yellow]Warning:[/bold yellow] Unsupported file format: {ext}")
                console.print("Supported formats: .json, .csv, .xlsx, .xls")

        except Exception as e:
            console.print(f"[bold red]Error previewing file:[/bold red] {str(e)}")

    async def handle_agents(args):
        """Handle the agents command."""
        # Create table
        table = Table(title="Agent Status")
        table.add_column("ID", style="cyan")
        table.add_column("Type", style="green")
        table.add_column("Status", style="yellow")
        table.add_column("Tasks", style="magenta")

        # Add rows for each agent
        for agent_id, agent_info in coordinator.agents.items():
            table.add_row(
                agent_id,
                agent_info["type"],
                agent_info["status"],
                str(agent_info["tasks"])
            )

        console.print(table)

    async def handle_fingerprint(args):
        """Handle the fingerprint command."""
        domain = None
        if args:
            domain = args[0]
        else:
            domain = Prompt.ask("Enter domain to generate fingerprint for", default="example.com")

        # Create task for generating fingerprint
        task = Task(
            type=TaskType.GENERATE_FINGERPRINT,
            parameters={
                "domain": domain,
                "consistent": True
            }
        )

        console.print(f"[bold]Generating fingerprint for {domain}...[/bold]")

        # Submit task
        task_id = await coordinator.submit_task(task)

        # Wait for task to complete
        while True:
            task_status = coordinator.get_task_status(task_id)
            if task_status and task_status["status"] in [TaskStatus.COMPLETED, TaskStatus.FAILED]:
                break
            await asyncio.sleep(0.1)

        # Check if task succeeded
        if task_status["status"] == TaskStatus.FAILED:
            console.print(f"[bold red]Error generating fingerprint:[/bold red] {task_status['error']['message']}")
            return

        # Show fingerprint
        fingerprint = task_status["result"]["fingerprint"]
        headers = task_status["result"]["headers"]

        # Create fingerprint table
        table = Table(title=f"Browser Fingerprint for {domain}")
        table.add_column("Property", style="cyan")
        table.add_column("Value", style="green")

        for key, value in fingerprint.items():
            table.add_row(key, str(value))

        console.print(table)

        # Create headers table
        headers_table = Table(title="HTTP Headers")
        headers_table.add_column("Header", style="cyan")
        headers_table.add_column("Value", style="green")

        for key, value in headers.items():
            headers_table.add_row(key, str(value))

        console.print(headers_table)

    async def handle_check_blocking(args):
        """Handle the check-blocking command."""
        if not args:
            console.print("[bold red]Error:[/bold red] URL is required")
            return

        url = args[0]

        # Create task for checking blocking
        task = Task(
            type=TaskType.CHECK_BLOCKING,
            parameters={
                "url": url,
                "check_methods": ["status_code", "content_analysis", "redirect"]
            }
        )

        console.print(f"[bold]Checking if {url} is blocking scraping...[/bold]")

        # Submit task
        task_id = await coordinator.submit_task(task)

        # Wait for task to complete
        while True:
            task_status = coordinator.get_task_status(task_id)
            if task_status and task_status["status"] in [TaskStatus.COMPLETED, TaskStatus.FAILED]:
                break
            await asyncio.sleep(0.1)

        # Check if task succeeded
        if task_status["status"] == TaskStatus.FAILED:
            console.print(f"[bold red]Error checking blocking:[/bold red] {task_status['error']['message']}")
            return

        # Show results
        result = task_status["result"]

        if result["is_blocked"]:
            console.print(f"[bold red]Site is blocking scraping:[/bold red] {result['reason']}")
        else:
            console.print(f"[bold green]Site is not blocking scraping[/bold green]")

        # Show blocking stats
        stats = result["blocking_stats"]

        stats_table = Table(title="Blocking Statistics")
        stats_table.add_column("Metric", style="cyan")
        stats_table.add_column("Value", style="green")

        stats_table.add_row("Total Requests", str(stats["total_requests"]))
        stats_table.add_row("Blocked Count", str(stats["blocked_count"]))
        stats_table.add_row("Last Blocked Time", str(stats["last_blocked_time"]))

        console.print(stats_table)

        # Show recommendations
        if result["should_backoff"]:
            console.print(f"[bold yellow]Recommendation:[/bold yellow] Back off for {result['backoff_time']} seconds")

    async def handle_analyze_text(args):
        """Handle the analyze-text command."""
        text = None
        if args:
            text = " ".join(args)
        else:
            text = Prompt.ask("Enter text to analyze")

        if not text:
            console.print("[bold red]Error:[/bold red] Text is required")
            return

        # Get analyses to perform
        analyses = ["sentiment", "entities", "keywords", "language"]

        # Create task for analyzing text
        task = Task(
            type=TaskType.ANALYZE_TEXT,
            parameters={
                "text": text,
                "analyses": analyses,
                "language": "en"
            }
        )

        console.print(f"[bold]Analyzing text...[/bold]")

        # Submit task
        task_id = await coordinator.submit_task(task)

        # Wait for task to complete
        while True:
            task_status = coordinator.get_task_status(task_id)
            if task_status and task_status["status"] in [TaskStatus.COMPLETED, TaskStatus.FAILED]:
                break
            await asyncio.sleep(0.1)

        # Check if task succeeded
        if task_status["status"] == TaskStatus.FAILED:
            console.print(f"[bold red]Error analyzing text:[/bold red] {task_status['error']['message']}")
            return

        # Show results
        results = task_status["result"]["results"]

        # Show statistics
        if "statistics" in results:
            stats = results["statistics"]

            stats_table = Table(title="Text Statistics")
            stats_table.add_column("Metric", style="cyan")
            stats_table.add_column("Value", style="green")

            for key, value in stats.items():
                stats_table.add_row(key.replace("_", " ").title(), str(value))

            console.print(stats_table)

        # Show sentiment
        if "sentiment" in results:
            sentiment = results["sentiment"]

            if "error" not in sentiment:
                sentiment_table = Table(title="Sentiment Analysis")
                sentiment_table.add_column("Metric", style="cyan")
                sentiment_table.add_column("Value", style="green")

                sentiment_color = "green" if sentiment["sentiment"] == "positive" else "red" if sentiment["sentiment"] == "negative" else "yellow"

                sentiment_table.add_row("Sentiment", f"[bold {sentiment_color}]{sentiment['sentiment']}[/bold {sentiment_color}]")
                sentiment_table.add_row("Compound Score", f"{sentiment['compound_score']:.2f}")
                sentiment_table.add_row("Positive Score", f"{sentiment['positive_score']:.2f}")
                sentiment_table.add_row("Negative Score", f"{sentiment['negative_score']:.2f}")
                sentiment_table.add_row("Neutral Score", f"{sentiment['neutral_score']:.2f}")

                console.print(sentiment_table)
            else:
                console.print(f"[bold yellow]Sentiment analysis not available:[/bold yellow] {sentiment['error']}")

        # Show keywords
        if "keywords" in results:
            keywords = results["keywords"]

            if isinstance(keywords, list) and keywords and "error" not in keywords[0]:
                keywords_table = Table(title="Keywords")
                keywords_table.add_column("Keyword", style="cyan")
                keywords_table.add_column("Frequency", style="green")

                for keyword in keywords[:10]:  # Show top 10
                    keywords_table.add_row(keyword["word"], str(keyword["frequency"]))

                console.print(keywords_table)
            else:
                console.print("[bold yellow]Keyword extraction not available[/bold yellow]")

        # Show language
        if "language" in results:
            language = results["language"]

            language_table = Table(title="Language Detection")
            language_table.add_column("Language", style="cyan")
            language_table.add_column("Confidence", style="green")

            language_table.add_row(language["language_code"], f"{language['confidence']:.2f}")

            console.print(language_table)

    async def handle_clean_data(args):
        """Handle the clean-data command."""
        if not args:
            console.print("[bold red]Error:[/bold red] File path is required")
            return

        file_path = args[0]

        if not os.path.exists(file_path):
            console.print(f"[bold red]Error:[/bold red] File {file_path} does not exist")
            return

        # Determine file type from extension
        _, ext = os.path.splitext(file_path)
        ext = ext.lower()

        # Load data
        data = None
        try:
            if ext == ".json":
                import json
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            elif ext == ".csv":
                import pandas as pd
                df = pd.read_csv(file_path)
                data = df.to_dict(orient="records")
            elif ext == ".xlsx" or ext == ".xls":
                import pandas as pd
                df = pd.read_excel(file_path)
                data = df.to_dict(orient="records")
            else:
                console.print(f"[bold red]Error:[/bold red] Unsupported file format: {ext}")
                console.print("Supported formats: .json, .csv, .xlsx, .xls")
                return
        except Exception as e:
            console.print(f"[bold red]Error loading file:[/bold red] {str(e)}")
            return

        if not data:
            console.print("[bold red]Error:[/bold red] No data found in file")
            return

        # Get operations
        operations = []

        # Ask for operations
        console.print("[bold]Available cleaning operations:[/bold]")
        console.print("1. Strip whitespace")
        console.print("2. Convert to lowercase")
        console.print("3. Convert to uppercase")
        console.print("4. Remove HTML tags")
        console.print("5. Extract numbers (e.g., from prices)")
        console.print("6. Remove empty values")

        while True:
            op_choice = Prompt.ask("Select operation (1-6) or 'done' to finish", default="done")

            if op_choice.lower() == "done":
                break

            try:
                op_num = int(op_choice)
                if op_num < 1 or op_num > 6:
                    console.print("[bold red]Invalid choice[/bold red]")
                    continue

                # Get field to apply operation to
                field = Prompt.ask("Enter field name to apply operation to (or '*' for all fields)")

                # Add operation
                if op_num == 1:
                    operations.append({"field": field, "operation": "strip_whitespace"})
                elif op_num == 2:
                    operations.append({"field": field, "operation": "lowercase"})
                elif op_num == 3:
                    operations.append({"field": field, "operation": "uppercase"})
                elif op_num == 4:
                    operations.append({"field": field, "operation": "remove_html"})
                elif op_num == 5:
                    operations.append({"field": field, "operation": "extract_number"})
                elif op_num == 6:
                    operations.append({"field": field, "operation": "remove_empty"})
            except ValueError:
                console.print("[bold red]Invalid choice[/bold red]")

        if not operations:
            console.print("[bold yellow]No operations selected[/bold yellow]")
            return

        # Get output file
        output_file = Prompt.ask("Enter output file path", default=f"cleaned_{os.path.basename(file_path)}")

        # Create task for cleaning data
        task = Task(
            type=TaskType.CLEAN_DATA,
            parameters={
                "data": data,
                "operations": operations,
                "add_metadata": True
            }
        )

        console.print(f"[bold]Cleaning data with {len(operations)} operations...[/bold]")

        # Submit task
        task_id = await coordinator.submit_task(task)

        # Wait for task to complete
        while True:
            task_status = coordinator.get_task_status(task_id)
            if task_status and task_status["status"] in [TaskStatus.COMPLETED, TaskStatus.FAILED]:
                break
            await asyncio.sleep(0.1)

        # Check if task succeeded
        if task_status["status"] == TaskStatus.FAILED:
            console.print(f"[bold red]Error cleaning data:[/bold red] {task_status['error']['message']}")
            return

        # Get cleaned data
        cleaned_data = task_status["result"]["data"]

        # Save cleaned data
        try:
            _, out_ext = os.path.splitext(output_file)
            out_ext = out_ext.lower()

            if out_ext == ".json":
                import json
                with open(output_file, "w", encoding="utf-8") as f:
                    json.dump(cleaned_data, f, indent=2)
            elif out_ext == ".csv":
                import pandas as pd
                pd.DataFrame(cleaned_data).to_csv(output_file, index=False)
            elif out_ext == ".xlsx":
                import pandas as pd
                pd.DataFrame(cleaned_data).to_excel(output_file, index=False)
            else:
                # Default to JSON
                import json
                with open(output_file, "w", encoding="utf-8") as f:
                    json.dump(cleaned_data, f, indent=2)

            console.print(f"[bold green]Cleaned data saved to {output_file}[/bold green]")
            console.print(f"Original items: {task_status['result']['original_count']}, Cleaned items: {task_status['result']['cleaned_count']}")
        except Exception as e:
            console.print(f"[bold red]Error saving cleaned data:[/bold red] {str(e)}")

    async def handle_check_system(args):
        """Handle the check-system command."""
        # Create task for checking system health
        task = Task(
            type=TaskType.CHECK_SYSTEM,
            parameters={
                "check_metrics": True,
                "check_resources": True,
                "check_services": True
            }
        )

        console.print("[bold]Checking system health...[/bold]")

        # Submit task
        task_id = await coordinator.submit_task(task)

        # Wait for task to complete
        while True:
            task_status = coordinator.get_task_status(task_id)
            if task_status and task_status["status"] in [TaskStatus.COMPLETED, TaskStatus.FAILED]:
                break
            await asyncio.sleep(0.1)

        # Check if task succeeded
        if task_status["status"] == TaskStatus.FAILED:
            console.print(f"[bold red]Error checking system:[/bold red] {task_status['error']['message']}")
            return

        # Show results
        result = task_status["result"]

        # Show metrics
        metrics_table = Table(title="System Metrics")
        metrics_table.add_column("Metric", style="cyan")
        metrics_table.add_column("Current", style="green")
        metrics_table.add_column("Threshold", style="yellow")
        metrics_table.add_column("Status", style="magenta")

        for metric, data in result["metrics"].items():
            status = "OK" if data["current"] <= data["threshold"] else "WARNING"
            status_color = "green" if status == "OK" else "red"
            metrics_table.add_row(
                metric,
                f"{data['current']:.1f}%",
                f"{data['threshold']}%",
                f"[{status_color}]{status}[/{status_color}]"
            )

        console.print(metrics_table)

        # Show issues if any
        if result["issues"]:
            issues_table = Table(title="System Issues")
            issues_table.add_column("Type", style="cyan")
            issues_table.add_column("Description", style="red")

            for issue in result["issues"]:
                issues_table.add_row(issue["type"], issue["description"])

            console.print(issues_table)

    async def handle_recover_system(args):
        """Handle the recover-system command."""
        recovery_type = None
        if args:
            recovery_type = args[0]
        else:
            recovery_type = Prompt.ask(
                "Enter recovery type",
                choices=["system", "process", "service"],
                default="system"
            )

        # Create task for system recovery
        task = Task(
            type=TaskType.RECOVER_SYSTEM,
            parameters={
                "recovery_type": recovery_type,
                "options": {
                    "force": False,
                    "backup": True
                }
            }
        )

        console.print(f"[bold]Recovering {recovery_type}...[/bold]")

        # Submit task
        task_id = await coordinator.submit_task(task)

        # Wait for task to complete
        while True:
            task_status = coordinator.get_task_status(task_id)
            if task_status and task_status["status"] in [TaskStatus.COMPLETED, TaskStatus.FAILED]:
                break
            await asyncio.sleep(0.1)

        # Check if task succeeded
        if task_status["status"] == TaskStatus.FAILED:
            console.print(f"[bold red]Error recovering system:[/bold red] {task_status['error']['message']}")
            return

        # Show results
        result = task_status["result"]

        console.print(f"[bold green]Recovery completed successfully[/bold green]")
        console.print(f"Strategy used: {result['strategy']}")
        console.print(f"Attempt: {result['attempt']}")

    async def handle_show_errors(args):
        """Handle the show-errors command."""
        # Create task for getting error history
        task = Task(
            type=TaskType.GET_ERROR_HISTORY,
            parameters={
                "limit": 10,
                "include_patterns": True
            }
        )

        console.print("[bold]Fetching error history...[/bold]")

        # Submit task
        task_id = await coordinator.submit_task(task)

        # Wait for task to complete
        while True:
            task_status = coordinator.get_task_status(task_id)
            if task_status and task_status["status"] in [TaskStatus.COMPLETED, TaskStatus.FAILED]:
                break
            await asyncio.sleep(0.1)

        # Check if task succeeded
        if task_status["status"] == TaskStatus.FAILED:
            console.print(f"[bold red]Error fetching error history:[/bold red] {task_status['error']['message']}")
            return

        # Show results
        result = task_status["result"]

        # Show recent errors
        errors_table = Table(title="Recent Errors")
        errors_table.add_column("Time", style="cyan")
        errors_table.add_column("Type", style="green")
        errors_table.add_column("Message", style="yellow")
        errors_table.add_column("Impact", style="magenta")

        for error in result["errors"]:
            errors_table.add_row(
                error["timestamp"],
                error["type"],
                error["message"],
                str(error["impact"])
            )

        console.print(errors_table)

        # Show error patterns if available
        if result["patterns"]:
            patterns_table = Table(title="Error Patterns")
            patterns_table.add_column("Type", style="cyan")
            patterns_table.add_column("Frequency", style="green")
            patterns_table.add_column("Impact", style="yellow")
            patterns_table.add_column("Strategy", style="magenta")

            for error_type, pattern in result["patterns"].items():
                patterns_table.add_row(
                    error_type,
                    f"{pattern['frequency']:.2f}",
                    f"{pattern['impact']:.2f}",
                    pattern["recovery_strategy"]
                )

            console.print(patterns_table)

    async def handle_extract_data(args):
        """Handle the extract-data command."""
        if not args:
            console.print("[bold red]Error:[/bold red] URL is required")
            return

        url = args[0]
        extraction_type = args[1] if len(args) > 1 else Prompt.ask(
            "Enter extraction type",
            choices=["html", "json", "xml", "pdf", "image", "table", "dynamic"],
            default="html"
        )

        # Get extraction options
        options = {}
        if extraction_type == "html":
            selectors = Prompt.ask("Enter selectors (field1:selector1,field2:selector2)")
            if selectors:
                try:
                    for item in selectors.split(","):
                        field, selector = item.split(":", 1)
                        options["selectors"] = options.get("selectors", {})
                        options["selectors"][field.strip()] = selector.strip()
                except Exception:
                    console.print("[bold red]Error:[/bold red] Invalid selector format")
                    return

        # Create task for data extraction
        task = Task(
            type=TaskType.EXTRACT_DATA,
            parameters={
                "url": url,
                "extraction_type": extraction_type,
                "options": options
            }
        )

        console.print(f"[bold]Extracting {extraction_type} data from {url}...[/bold]")

        # Submit task
        task_id = await coordinator.submit_task(task)

        # Wait for task to complete
        while True:
            task_status = coordinator.get_task_status(task_id)
            if task_status and task_status["status"] in [TaskStatus.COMPLETED, TaskStatus.FAILED]:
                break
            await asyncio.sleep(0.1)

        # Check if task succeeded
        if task_status["status"] == TaskStatus.FAILED:
            console.print(f"[bold red]Error extracting data:[/bold red] {task_status['error']['message']}")
            return

        # Show results
        result = task_status["result"]

        # Create result table
        result_table = Table(title=f"Extracted Data from {url}")
        result_table.add_column("Field", style="cyan")
        result_table.add_column("Value", style="green")

        for key, value in result.items():
            if isinstance(value, (list, dict)):
                value = str(value)
            result_table.add_row(key, value)

        console.print(result_table)

    async def handle_validate_data(args):
        """Handle the validate-data command."""
        if not args:
            console.print("[bold red]Error:[/bold red] File path is required")
            return

        file_path = args[0]

        if not os.path.exists(file_path):
            console.print(f"[bold red]Error:[/bold red] File {file_path} does not exist")
            return

        # Load data
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            console.print(f"[bold red]Error loading file:[/bold red] {str(e)}")
            return

        # Get validation rules
        rules = {}
        console.print("[bold]Enter validation rules:[/bold]")

        # Required fields
        required = Prompt.ask("Enter required fields (comma-separated)")
        if required:
            rules["required_fields"] = [f.strip() for f in required.split(",")]

        # Field types
        types = Prompt.ask("Enter field types (field:type,field:type)")
        if types:
            try:
                rules["field_types"] = {}
                for item in types.split(","):
                    field, type_name = item.split(":")
                    rules["field_types"][field.strip()] = type_name.strip()
            except Exception:
                console.print("[bold red]Error:[/bold red] Invalid type format")

        # Create task for data validation
        task = Task(
            type=TaskType.VALIDATE_DATA,
            parameters={
                "data": data,
                "rules": rules
            }
        )

        console.print("[bold]Validating data...[/bold]")

        # Submit task
        task_id = await coordinator.submit_task(task)

        # Wait for task to complete
        while True:
            task_status = coordinator.get_task_status(task_id)
            if task_status and task_status["status"] in [TaskStatus.COMPLETED, TaskStatus.FAILED]:
                break
            await asyncio.sleep(0.1)

        # Check if task succeeded
        if task_status["status"] == TaskStatus.FAILED:
            console.print(f"[bold red]Error validating data:[/bold red] {task_status['error']['message']}")
            return

        # Show results
        result = task_status["result"]

        if result["valid"]:
            console.print("[bold green]Data validation successful[/bold green]")
        else:
            console.print("[bold red]Data validation failed[/bold red]")

            # Show errors
            errors_table = Table(title="Validation Errors")
            errors_table.add_column("Error", style="red")

            for error in result["errors"]:
                errors_table.add_row(error)

            console.print(errors_table)

    async def handle_cache_stats(args):
        """Handle the cache-stats command."""
        # Create task for getting cache statistics
        task = Task(
            type=TaskType.GET_CACHE_STATS,
            parameters={}
        )

        console.print("[bold]Fetching cache statistics...[/bold]")

        # Submit task
        task_id = await coordinator.submit_task(task)

        # Wait for task to complete
        while True:
            task_status = coordinator.get_task_status(task_id)
            if task_status and task_status["status"] in [TaskStatus.COMPLETED, TaskStatus.FAILED]:
                break
            await asyncio.sleep(0.1)

        # Check if task succeeded
        if task_status["status"] == TaskStatus.FAILED:
            console.print(f"[bold red]Error fetching cache statistics:[/bold red] {task_status['error']['message']}")
            return

        # Show results
        result = task_status["result"]

        # Show statistics
        stats_table = Table(title="Cache Statistics")
        stats_table.add_column("Metric", style="cyan")
        stats_table.add_column("Value", style="green")

        for key, value in result["stats"].items():
            stats_table.add_row(key.replace("_", " ").title(), str(value))

        console.print(stats_table)

        # Show cache configuration
        config_table = Table(title="Cache Configuration")
        config_table.add_column("Setting", style="cyan")
        config_table.add_column("Value", style="green")

        config_table.add_row("Cache Size", str(result["cache_size"]))
        config_table.add_row("Cache TTL", f"{result['cache_ttl']} seconds")
        config_table.add_row("Max Cache Size", str(result["max_cache_size"]))
        config_table.add_row("Memory Limit", f"{result['cache_memory_limit'] / (1024*1024):.1f} MB")

        console.print(config_table)

    # Command mapping
    commands = {
        "scrape": handle_scrape,
        "help": handle_help,
        "preview": handle_preview,
        "agents": handle_agents,
        "fingerprint": handle_fingerprint,
        "check-blocking": handle_check_blocking,
        "analyze-text": handle_analyze_text,
        "clean-data": handle_clean_data,
        "check-system": handle_check_system,
        "recover-system": handle_recover_system,
        "show-errors": handle_show_errors,
        "extract-data": handle_extract_data,
        "validate-data": handle_validate_data,
        "cache-stats": handle_cache_stats
    }

    # Main interactive loop
    running = True
    while running:
        try:
            # Get command
            command_line = Prompt.ask("\n[bold blue]scraper[/bold blue]")

            # Parse command
            parts = command_line.strip().split()
            if not parts:
                continue

            command = parts[0].lower()
            args = parts[1:]

            # Handle exit command
            if command == "exit":
                running = False
                console.print("[bold]Exiting interactive mode...[/bold]")
                continue

            # Handle other commands
            if command in commands:
                await commands[command](args)
            else:
                console.print(f"[bold red]Unknown command:[/bold red] {command}")
                console.print("Type [bold]help[/bold] to see available commands.")

        except KeyboardInterrupt:
            running = False
            console.print("\n[bold]Exiting interactive mode...[/bold]")
        except Exception as e:
            console.print(f"[bold red]Error:[/bold red] {str(e)}")


if __name__ == "__main__":
    app()
