"""
Interactive tool testing script.

Run this to manually test tools with custom queries.
Usage: python tests/interactive_test.py
"""

import sys
import os
import json

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tools.implementations import ToolExecutor
from tools.definitions import get_tool_definitions
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt, Confirm
from rich.table import Table
from rich.markdown import Markdown
from rich import print as rprint


class MockRAG:
    """Mock RAG for testing."""
    def __init__(self):
        self.auto_index = False
    
    def index_search_results(self, *args, **kwargs):
        return 0
    
    def index_single_page(self, *args, **kwargs):
        return 0


def show_tool_list(console: Console):
    """Display available tools."""
    tools = get_tool_definitions()
    
    table = Table(title="Available Tools", show_header=True, header_style="bold cyan")
    table.add_column("#", style="dim", width=3)
    table.add_column("Tool Name", style="cyan")
    table.add_column("Description", style="white")
    
    for idx, tool in enumerate(tools, 1):
        name = tool['function']['name']
        desc = tool['function']['description'][:80] + "..." if len(tool['function']['description']) > 80 else tool['function']['description']
        table.add_row(str(idx), name, desc)
    
    console.print(table)


def show_tool_details(tool_name: str, console: Console):
    """Show detailed information about a tool."""
    tools = get_tool_definitions()
    tool = next((t for t in tools if t['function']['name'] == tool_name), None)
    
    if not tool:
        console.print(f"[red]Tool '{tool_name}' not found[/red]")
        return
    
    func = tool['function']
    
    console.print(Panel(f"[bold cyan]{func['name']}[/bold cyan]", expand=False))
    console.print(f"\n[bold]Description:[/bold]")
    console.print(func['description'])
    
    if 'parameters' in func and 'properties' in func['parameters']:
        console.print(f"\n[bold]Parameters:[/bold]")
        props = func['parameters']['properties']
        required = func['parameters'].get('required', [])
        
        param_table = Table(show_header=True, header_style="bold magenta")
        param_table.add_column("Parameter", style="cyan")
        param_table.add_column("Type", style="yellow")
        param_table.add_column("Required", style="red")
        param_table.add_column("Description", style="white")
        
        for param_name, param_info in props.items():
            is_required = "Yes" if param_name in required else "No"
            param_type = param_info.get('type', 'any')
            param_desc = param_info.get('description', 'No description')[:60]
            if 'default' in param_info:
                param_desc += f" (default: {param_info['default']})"
            
            param_table.add_row(param_name, param_type, is_required, param_desc)
        
        console.print(param_table)


def get_tool_examples(tool_name: str) -> list:
    """Get example queries for each tool."""
    examples = {
        'web_search': [
            {'query': 'Python programming language', 'iterations': 1},
            {'query': 'latest AI news 2024', 'iterations': 1},
            {'query': 'how to use Docker containers', 'iterations': 1}
        ],
        'news_search': [
            {'keywords': 'technology', 'max_results': 5},
            {'keywords': 'artificial intelligence', 'timelimit': 'd', 'max_results': 3},
            {'keywords': 'climate change', 'timelimit': 'w', 'max_results': 5}
        ],
        'fetch_url_content': [
            {'url': 'https://en.wikipedia.org/wiki/Python_(programming_language)', 'max_length': 2000},
            {'url': 'https://en.wikipedia.org/wiki/Artificial_intelligence', 'max_length': 1500}
        ],
        'calculate': [
            {'expression': '2 + 2'},
            {'expression': 'sqrt(144)'},
            {'expression': '(10 + 5) * 2 - 8'},
            {'expression': 'sin(pi/2)'}
        ],
        'get_current_time': [{}],
        'search_wikipedia': [
            {'query': 'Quantum computing', 'top_k': 1},
            {'query': 'Machine learning', 'top_k': 2},
            {'query': 'Python programming language', 'top_k': 1}
        ],
        'search_arxiv': [
            {'query': 'transformer neural networks', 'max_results': 2},
            {'query': 'quantum computing algorithms', 'max_results': 3},
            {'query': 'graph neural networks', 'max_results': 2}
        ],
        'search_and_fetch': [
            {'query': 'FastAPI Python framework', 'max_fetch_pages': 3},
            {'query': 'Docker containers tutorial', 'max_fetch_pages': 2}
        ]
    }
    return examples.get(tool_name, [])


def test_tool_with_example(executor: ToolExecutor, tool_name: str, args: dict, console: Console):
    """Test a tool with given arguments."""
    console.print(Panel(f"[bold cyan]Testing: {tool_name}[/bold cyan]"))
    console.print(f"[dim]Arguments: {json.dumps(args, indent=2)}[/dim]\n")
    
    try:
        result = executor.execute(tool_name, args)
        
        if result.startswith("Error"):
            console.print(Panel(result, title="[red]Error[/red]", border_style="red"))
        else:
            # Truncate very long results for display
            display_result = result if len(result) <= 2000 else result[:2000] + f"\n\n... [truncated {len(result) - 2000} characters]"
            console.print(Panel(display_result, title="[green]Result[/green]", border_style="green"))
        
        return result
        
    except Exception as e:
        console.print(Panel(f"Exception: {str(e)}", title="[red]Error[/red]", border_style="red"))
        return None


def interactive_test_menu(console: Console):
    """Interactive testing menu."""
    config = {
        'web_search_enabled': True,
        'max_search_results': 20,
        'news_search_auto_index': False,
        'show_chunk_previews': False
    }
    
    mock_rag = MockRAG()
    executor = ToolExecutor(config=config, console=console, web_search_rag=mock_rag)
    
    console.print(Panel.fit(
        "[bold cyan]Interactive Tool Testing[/bold cyan]\n"
        "Test tools with pre-defined examples or custom queries",
        border_style="cyan"
    ))
    
    while True:
        console.print("\n[bold]Options:[/bold]")
        console.print("1. List all tools")
        console.print("2. Test tool with example")
        console.print("3. Test tool with custom arguments")
        console.print("4. View tool details")
        console.print("5. Exit")
        
        choice = Prompt.ask("\nSelect option", choices=["1", "2", "3", "4", "5"], default="1")
        
        if choice == "1":
            show_tool_list(console)
        
        elif choice == "2":
            tool_name = Prompt.ask("\nEnter tool name")
            examples = get_tool_examples(tool_name)
            
            if not examples:
                console.print(f"[yellow]No examples available for '{tool_name}'[/yellow]")
                continue
            
            console.print(f"\n[bold]Available examples for {tool_name}:[/bold]")
            for idx, example in enumerate(examples, 1):
                console.print(f"{idx}. {json.dumps(example)}")
            
            example_idx = Prompt.ask(
                "Select example number",
                choices=[str(i) for i in range(1, len(examples) + 1)],
                default="1"
            )
            
            selected_example = examples[int(example_idx) - 1]
            test_tool_with_example(executor, tool_name, selected_example, console)
        
        elif choice == "3":
            tool_name = Prompt.ask("\nEnter tool name")
            console.print("[dim]Enter arguments as JSON (e.g., {\"query\": \"test\"})[/dim]")
            args_json = Prompt.ask("Arguments")
            
            try:
                args = json.loads(args_json)
                test_tool_with_example(executor, tool_name, args, console)
            except json.JSONDecodeError as e:
                console.print(f"[red]Invalid JSON: {e}[/red]")
        
        elif choice == "4":
            tool_name = Prompt.ask("\nEnter tool name")
            show_tool_details(tool_name, console)
        
        elif choice == "5":
            console.print("[cyan]Goodbye![/cyan]")
            break


def quick_test_all_tools(console: Console):
    """Run a quick test on all tools with one example each."""
    config = {
        'web_search_enabled': True,
        'max_search_results': 20,
        'news_search_auto_index': False,
        'show_chunk_previews': False
    }
    
    mock_rag = MockRAG()
    executor = ToolExecutor(config=config, console=console, web_search_rag=mock_rag)
    
    console.print(Panel.fit(
        "[bold cyan]Quick Test: All Tools[/bold cyan]\n"
        "Running one example for each tool",
        border_style="cyan"
    ))
    
    tools_to_test = [
        ('calculate', {'expression': '2 + 2'}),
        ('get_current_time', {}),
        ('web_search', {'query': 'Python programming', 'iterations': 1}),
        ('news_search', {'keywords': 'technology', 'max_results': 3}),
        ('search_wikipedia', {'query': 'Python programming language', 'top_k': 1, 'auto_index': False}),
        ('search_arxiv', {'query': 'neural networks', 'max_results': 2, 'auto_index': False}),
    ]
    
    for tool_name, args in tools_to_test:
        console.print(f"\n{'='*60}")
        result = test_tool_with_example(executor, tool_name, args, console)
        
        if Confirm.ask("\nContinue to next tool?", default=True):
            continue
        else:
            break
    
    console.print(f"\n{'='*60}")
    console.print("[green]Quick test complete![/green]")


if __name__ == "__main__":
    console = Console()
    
    # Check if user wants quick test or interactive mode
    if len(sys.argv) > 1 and sys.argv[1] == "--quick":
        quick_test_all_tools(console)
    else:
        interactive_test_menu(console)
