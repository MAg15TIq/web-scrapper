"""
Visualizer for the CLI interface.
"""
import time
from typing import Dict, Any, Optional, List, Union
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn
from rich.panel import Panel
from rich.tree import Tree
from rich.syntax import Syntax
from rich.layout import Layout
from rich.live import Live
import pandas as pd


class Visualizer:
    """
    Visualizer for displaying information in the CLI.
    """
    def __init__(self, console: Optional[Console] = None):
        """
        Initialize a new visualizer.
        
        Args:
            console: Rich console to use for output. If None, a new console is created.
        """
        self.console = console or Console()
    
    def display_table(self, title: str, data: List[Dict[str, Any]], columns: Optional[List[str]] = None) -> None:
        """
        Display data in a table.
        
        Args:
            title: Title of the table.
            data: List of dictionaries containing the data.
            columns: List of column names to display. If None, all columns are displayed.
        """
        if not data:
            self.console.print(f"[bold yellow]No data to display for: {title}[/bold yellow]")
            return
        
        # Create table
        table = Table(title=title)
        
        # Determine columns
        if not columns:
            columns = list(data[0].keys())
        
        # Add columns
        for column in columns:
            table.add_column(column)
        
        # Add rows
        for item in data:
            row = [str(item.get(column, "")) for column in columns]
            table.add_row(*row)
        
        # Print table
        self.console.print(table)
    
    def display_progress(self, tasks: List[Dict[str, Any]]) -> Progress:
        """
        Create and return a progress display for multiple tasks.
        
        Args:
            tasks: List of task dictionaries with 'description', 'total', and optional 'completed' keys.
            
        Returns:
            A Progress object that can be updated.
        """
        progress = Progress(
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeRemainingColumn(),
            console=self.console
        )
        
        # Add tasks
        for task in tasks:
            description = task.get("description", "Task")
            total = task.get("total", 100)
            completed = task.get("completed", 0)
            task_id = progress.add_task(description, total=total, completed=completed)
            task["id"] = task_id
        
        return progress
    
    def display_data_preview(self, data: Union[List[Dict[str, Any]], pd.DataFrame], title: str, max_rows: int = 5) -> None:
        """
        Display a preview of data.
        
        Args:
            data: Data to preview, either as a list of dictionaries or a pandas DataFrame.
            title: Title for the preview.
            max_rows: Maximum number of rows to display.
        """
        # Convert to DataFrame if necessary
        if not isinstance(data, pd.DataFrame):
            df = pd.DataFrame(data)
        else:
            df = data
        
        # Limit rows
        preview_df = df.head(max_rows)
        
        # Create table
        table = Table(title=title)
        
        # Add columns
        for column in preview_df.columns:
            table.add_column(str(column))
        
        # Add rows
        for _, row in preview_df.iterrows():
            table.add_row(*[str(val) for val in row])
        
        # Print table
        self.console.print(table)
        self.console.print(f"[bold]Showing {len(preview_df)} of {len(df)} rows[/bold]")
    
    def display_code(self, code: str, language: str = "python") -> None:
        """
        Display syntax-highlighted code.
        
        Args:
            code: Code to display.
            language: Programming language for syntax highlighting.
        """
        syntax = Syntax(code, language, theme="monokai", line_numbers=True)
        self.console.print(syntax)
    
    def display_tree(self, title: str, data: Dict[str, Any]) -> None:
        """
        Display hierarchical data as a tree.
        
        Args:
            title: Title for the tree.
            data: Hierarchical data to display.
        """
        tree = Tree(title)
        
        def add_branch(branch, data):
            if isinstance(data, dict):
                for key, value in data.items():
                    if isinstance(value, (dict, list)):
                        sub_branch = branch.add(key)
                        add_branch(sub_branch, value)
                    else:
                        branch.add(f"{key}: {value}")
            elif isinstance(data, list):
                for i, item in enumerate(data):
                    if isinstance(item, (dict, list)):
                        sub_branch = branch.add(f"Item {i+1}")
                        add_branch(sub_branch, item)
                    else:
                        branch.add(str(item))
        
        add_branch(tree, data)
        self.console.print(tree)
    
    def create_dashboard(self) -> Layout:
        """
        Create a dashboard layout.
        
        Returns:
            A Layout object that can be updated.
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
            Layout(name="left"),
            Layout(name="right")
        )
        
        # Set initial content
        layout["header"].update(Panel("Multi-Agent Web Scraping System", style="bold blue"))
        layout["footer"].update(Panel("Press Ctrl+C to exit", style="bold"))
        
        return layout
    
    def update_dashboard(self, layout: Layout, sections: Dict[str, Any]) -> None:
        """
        Update sections of a dashboard layout.
        
        Args:
            layout: The layout to update.
            sections: Dictionary mapping section names to content.
        """
        for section, content in sections.items():
            if section in layout:
                layout[section].update(content)
    
    def display_stats(self, stats: Dict[str, Any]) -> Panel:
        """
        Create a panel displaying statistics.
        
        Args:
            stats: Dictionary of statistics to display.
            
        Returns:
            A Panel object containing the statistics.
        """
        # Create table
        table = Table(show_header=False, box=None)
        table.add_column("Stat")
        table.add_column("Value")
        
        # Add rows
        for key, value in stats.items():
            # Format the key
            formatted_key = key.replace("_", " ").title()
            
            # Format the value
            if isinstance(value, (int, float)):
                if key.endswith("_time") or key.endswith("_seconds"):
                    # Format as time
                    formatted_value = f"{value:.2f}s"
                elif key.endswith("_rate") or key.endswith("_ratio"):
                    # Format as percentage
                    formatted_value = f"{value * 100:.1f}%"
                else:
                    formatted_value = str(value)
            else:
                formatted_value = str(value)
            
            table.add_row(formatted_key, formatted_value)
        
        return Panel(table, title="Statistics", border_style="green")
    
    def create_live_display(self) -> Live:
        """
        Create a live display that can be updated.
        
        Returns:
            A Live object that can be updated.
        """
        return Live(console=self.console, auto_refresh=False)
    
    def update_live_display(self, live: Live, content: Any) -> None:
        """
        Update a live display with new content.
        
        Args:
            live: The live display to update.
            content: The new content to display.
        """
        live.update(content)
        live.refresh()
