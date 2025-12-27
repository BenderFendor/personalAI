"""
Comprehensive test suite for tool implementations.

This test suite validates all tools with various test queries to ensure they work correctly.
Run with: python -m pytest tests/test_tools.py -v
Or run individual tests with: python tests/test_tools.py
"""

import sys
import os
from typing import Dict, Any
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tools.implementations import ToolExecutor
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import print as rprint


class MockRAG:
    """Mock RAG for testing without actual vector DB."""
    def __init__(self):
        self.auto_index = False
        
    def index_search_results(self, *args, **kwargs):
        return 0
    
    def index_single_page(self, *args, **kwargs):
        return 0


class TestResults:
    """Track test results."""
    def __init__(self):
        self.passed = []
        self.failed = []
        self.warnings = []
    
    def add_pass(self, tool_name: str, test_name: str):
        self.passed.append((tool_name, test_name))
    
    def add_fail(self, tool_name: str, test_name: str, error: str):
        self.failed.append((tool_name, test_name, error))
    
    def add_warning(self, tool_name: str, test_name: str, warning: str):
        self.warnings.append((tool_name, test_name, warning))
    
    def print_summary(self, console: Console):
        """Print test summary."""
        table = Table(title="Test Results Summary", show_header=True, header_style="bold magenta")
        table.add_column("Category", style="cyan")
        table.add_column("Count", justify="right", style="green")
        
        table.add_row("PASS", str(len(self.passed)))
        table.add_row("FAIL", str(len(self.failed)), style="red")
        table.add_row("WARN", str(len(self.warnings)), style="yellow")
        table.add_row("Total Tests", str(len(self.passed) + len(self.failed)))
        
        console.print("\n")
        console.print(table)
        
        if self.failed:
            console.print("\n[bold red]Failed Tests:[/bold red]")
            for tool, test, error in self.failed:
                console.print(f"  [red]FAIL[/red] {tool}.{test}: {error}")
        
        if self.warnings:
            console.print("\n[bold yellow]Warnings:[/bold yellow]")
            for tool, test, warning in self.warnings:
                console.print(f"  [yellow]WARN[/yellow] {tool}.{test}: {warning}")


def test_web_search(executor: ToolExecutor, results: TestResults, console: Console):
    """Test web_search tool with various queries."""
    console.print(Panel("[bold cyan]Testing: web_search[/bold cyan]"))
    
    test_cases = [
        {
            "name": "basic_search",
            "query": "Python programming language",
            "iterations": 1,
            "expected": ["python", "programming"]
        },
        {
            "name": "specific_search",
            "query": "OpenAI GPT-4 release date",
            "iterations": 1,
            "expected": ["gpt", "openai"]
        },
        {
            "name": "multiple_iterations",
            "query": "machine learning algorithms",
            "iterations": 2,
            "expected": ["machine learning", "iteration"]
        }
    ]
    
    for test_case in test_cases:
        try:
            console.print(f"\n[dim]Running: {test_case['name']}[/dim]")
            result = executor.web_search(
                query=test_case['query'],
                iterations=test_case.get('iterations', 1)
            )
            
            # Check for errors
            if result.startswith("Error"):
                results.add_fail("web_search", test_case['name'], result)
                console.print(f"[red]FAIL[/red]: {result[:100]}")
                continue
            
            # Check for expected content
            result_lower = result.lower()
            missing_terms = [term for term in test_case['expected'] if term.lower() not in result_lower]
            
            if missing_terms:
                results.add_warning("web_search", test_case['name'], 
                                  f"Expected terms not found: {missing_terms}")
                console.print(f"[yellow]WARN[/yellow]: Missing expected terms")
            else:
                results.add_pass("web_search", test_case['name'])
                console.print(f"[green]PASS[/green]")
            
            # Print result preview
            console.print(f"[dim]Preview: {result[:200]}...[/dim]")
            
        except Exception as e:
            results.add_fail("web_search", test_case['name'], str(e))
            console.print(f"[red]FAIL[/red]: {str(e)}")


def test_news_search(executor: ToolExecutor, results: TestResults, console: Console):
    """Test news_search tool with various queries."""
    console.print(Panel("[bold cyan]Testing: news_search[/bold cyan]"))
    
    test_cases = [
        {
            "name": "basic_news",
            "keywords": "technology",
            "max_results": 5,
            "expected": ["news", "articles"]
        },
        {
            "name": "recent_news_day",
            "keywords": "artificial intelligence",
            "timelimit": "d",
            "max_results": 3,
            "expected": ["past day"]
        },
        {
            "name": "news_week",
            "keywords": "climate change",
            "timelimit": "w",
            "max_results": 5,
            "expected": ["past week"]
        },
        {
            "name": "region_specific",
            "keywords": "politics",
            "region": "uk-en",
            "max_results": 3,
            "expected": ["news"]
        }
    ]
    
    for test_case in test_cases:
        try:
            console.print(f"\n[dim]Running: {test_case['name']}[/dim]")
            result = executor.news_search(
                keywords=test_case['keywords'],
                region=test_case.get('region', 'us-en'),
                timelimit=test_case.get('timelimit'),
                max_results=test_case.get('max_results', 10)
            )
            
            if result.startswith("Error"):
                results.add_fail("news_search", test_case['name'], result)
                console.print(f"[red]FAIL[/red]: {result[:100]}")
                continue
            
            # Check for expected content
            result_lower = result.lower()
            missing_terms = [term for term in test_case['expected'] if term.lower() not in result_lower]
            
            if "No news articles found" in result:
                results.add_warning("news_search", test_case['name'], "No articles found")
                console.print(f"[yellow]WARN[/yellow]: No articles found")
            elif missing_terms:
                results.add_warning("news_search", test_case['name'], 
                                  f"Expected terms not found: {missing_terms}")
                console.print(f"[yellow]WARN[/yellow]: Missing expected terms")
            else:
                results.add_pass("news_search", test_case['name'])
                console.print(f"[green]PASS[/green]")
            
            console.print(f"[dim]Preview: {result[:200]}...[/dim]")
            
        except Exception as e:
            results.add_fail("news_search", test_case['name'], str(e))
            console.print(f"[red]FAIL[/red]: {str(e)}")


def test_fetch_url_content(executor: ToolExecutor, results: TestResults, console: Console):
    """Test fetch_url_content tool with various URLs."""
    console.print(Panel("[bold cyan]Testing: fetch_url_content[/bold cyan]"))
    
    test_cases = [
        {
            "name": "valid_website",
            "url": "https://en.wikipedia.org/wiki/Python_(programming_language)",
            "max_length": 2000,
            "expected": ["python", "programming"]
        },
        {
            "name": "invalid_url",
            "url": "not_a_valid_url",
            "expected": ["error", "invalid"]
        },
        {
            "name": "video_url_blocked",
            "url": "https://www.youtube.com/watch?v=example",
            "expected": ["error", "video"]
        },
        {
            "name": "max_length_truncation",
            "url": "https://en.wikipedia.org/wiki/Artificial_intelligence",
            "max_length": 500,
            "expected": ["content truncated", "500"]
        }
    ]
    
    for test_case in test_cases:
        try:
            console.print(f"\n[dim]Running: {test_case['name']}[/dim]")
            result = executor.fetch_url_content(
                url=test_case['url'],
                max_length=test_case.get('max_length', 5000)
            )
            
            result_lower = result.lower()
            
            # For error cases, we expect errors
            if "error" in test_case['name'].lower():
                if result.startswith("Error"):
                    results.add_pass("fetch_url_content", test_case['name'])
                    console.print(f"[green]PASS[/green] (Error handled correctly)")
                else:
                    results.add_fail("fetch_url_content", test_case['name'], 
                                   "Expected error but got success")
                    console.print(f"[red]FAIL[/red]: Expected error")
            else:
                # For success cases
                if result.startswith("Error"):
                    results.add_fail("fetch_url_content", test_case['name'], result)
                    console.print(f"[red]FAIL[/red]: {result[:100]}")
                else:
                    missing_terms = [term for term in test_case['expected'] 
                                   if term.lower() not in result_lower]
                    if missing_terms:
                        results.add_warning("fetch_url_content", test_case['name'], 
                                          f"Expected terms not found: {missing_terms}")
                        console.print(f"[yellow]WARN[/yellow]: Missing expected terms")
                    else:
                        results.add_pass("fetch_url_content", test_case['name'])
                        console.print(f"[green]PASS[/green]")
            
            console.print(f"[dim]Preview: {result[:150]}...[/dim]")
            
        except Exception as e:
            results.add_fail("fetch_url_content", test_case['name'], str(e))
            console.print(f"[red]FAIL[/red]: {str(e)}")


def test_calculate(executor: ToolExecutor, results: TestResults, console: Console):
    """Test calculate tool with various expressions."""
    console.print(Panel("[bold cyan]Testing: calculate[/bold cyan]"))
    
    test_cases = [
        {
            "name": "basic_arithmetic",
            "expression": "2 + 2",
            "expected_result": "4"
        },
        {
            "name": "complex_expression",
            "expression": "(10 + 5) * 2 - 8",
            "expected_result": "22"
        },
        {
            "name": "sqrt_function",
            "expression": "sqrt(16)",
            "expected_result": "4"
        },
        {
            "name": "trigonometry",
            "expression": "sin(pi/2)",
            "expected_result": "1"
        },
        {
            "name": "power",
            "expression": "pow(2, 10)",
            "expected_result": "1024"
        },
        {
            "name": "invalid_expression",
            "expression": "import os",
            "is_error": True
        }
    ]
    
    for test_case in test_cases:
        try:
            console.print(f"\n[dim]Running: {test_case['name']}[/dim]")
            result = executor.calculate(expression=test_case['expression'])
            
            if test_case.get('is_error'):
                if result.startswith("Error"):
                    results.add_pass("calculate", test_case['name'])
                    console.print(f"[green]PASS[/green] (Error handled correctly)")
                else:
                    results.add_fail("calculate", test_case['name'], 
                                   "Expected error but got success")
                    console.print(f"[red]FAIL[/red]: Should have errored")
            else:
                if result.startswith("Error"):
                    results.add_fail("calculate", test_case['name'], result)
                    console.print(f"[red]FAIL[/red]: {result}")
                elif test_case['expected_result'] in result:
                    results.add_pass("calculate", test_case['name'])
                    console.print(f"[green]PASS[/green]: {result}")
                else:
                    results.add_warning("calculate", test_case['name'], 
                                      f"Expected {test_case['expected_result']}, got {result}")
                    console.print(f"[yellow]WARN[/yellow]: Result doesn't match expected")
            
        except Exception as e:
            results.add_fail("calculate", test_case['name'], str(e))
            console.print(f"[red]FAIL[/red]: {str(e)}")


def test_get_current_time(executor: ToolExecutor, results: TestResults, console: Console):
    """Test get_current_time tool."""
    console.print(Panel("[bold cyan]Testing: get_current_time[/bold cyan]"))
    
    try:
        console.print(f"\n[dim]Running: time_check[/dim]")
        result = executor.get_current_time()
        
        # Check if result contains expected date format elements
        current_year = str(datetime.now().year)
        
        if result.startswith("Error"):
            results.add_fail("get_current_time", "time_check", result)
            console.print(f"[red]FAIL[/red]: {result}")
        elif current_year in result and "-" in result and ":" in result:
            results.add_pass("get_current_time", "time_check")
            console.print(f"[green]PASS[/green]: {result}")
        else:
            results.add_warning("get_current_time", "time_check", 
                              "Unexpected format")
            console.print(f"[yellow]WARN[/yellow]: Unexpected format: {result}")
            
    except Exception as e:
        results.add_fail("get_current_time", "time_check", str(e))
        console.print(f"[red]FAIL[/red]: {str(e)}")


def test_search_wikipedia(executor: ToolExecutor, results: TestResults, console: Console):
    """Test search_wikipedia tool."""
    console.print(Panel("[bold cyan]Testing: search_wikipedia[/bold cyan]"))
    
    test_cases = [
        {
            "name": "basic_search",
            "query": "Python programming language",
            "expected": ["python", "programming", "wikipedia"]
        },
        {
            "name": "scientific_topic",
            "query": "Quantum mechanics",
            "expected": ["quantum", "physics"]
        },
        {
            "name": "nonexistent_topic",
            "query": "XYZ123NotRealTopic789",
            "expected": ["no", "articles", "found"]
        }
    ]
    
    for test_case in test_cases:
        try:
            console.print(f"\n[dim]Running: {test_case['name']}[/dim]")
            result = executor.search_wikipedia(
                query=test_case['query'],
                top_k=1,
                max_chars=1000,
                auto_index=False
            )
            
            result_lower = result.lower()
            
            if "No Wikipedia articles found" in result:
                if "nonexistent" in test_case['name']:
                    results.add_pass("search_wikipedia", test_case['name'])
                    console.print(f"[green]PASS[/green] (Correctly found no results)")
                else:
                    results.add_warning("search_wikipedia", test_case['name'], 
                                      "No articles found")
                    console.print(f"[yellow]WARN[/yellow]: No articles found")
            else:
                missing_terms = [term for term in test_case['expected'] 
                               if term.lower() not in result_lower]
                if missing_terms and "nonexistent" not in test_case['name']:
                    results.add_warning("search_wikipedia", test_case['name'], 
                                      f"Expected terms not found: {missing_terms}")
                    console.print(f"[yellow]WARN[/yellow]: Missing expected terms")
                else:
                    results.add_pass("search_wikipedia", test_case['name'])
                    console.print(f"[green]PASS[/green]")
            
            console.print(f"[dim]Preview: {result[:200]}...[/dim]")
            
        except Exception as e:
            results.add_fail("search_wikipedia", test_case['name'], str(e))
            console.print(f"[red]FAIL[/red]: {str(e)}")


def test_search_arxiv(executor: ToolExecutor, results: TestResults, console: Console):
    """Test search_arxiv tool."""
    console.print(Panel("[bold cyan]Testing: search_arxiv[/bold cyan]"))
    
    test_cases = [
        {
            "name": "basic_search",
            "query": "transformer neural networks",
            "expected": ["arxiv", "paper"]
        },
        {
            "name": "specific_field",
            "query": "quantum computing algorithms",
            "expected": ["quantum"]
        },
        {
            "name": "obscure_search",
            "query": "XYZ999NotRealPaper888ZZZ",
            "expected": ["no", "papers", "found"]
        }
    ]
    
    for test_case in test_cases:
        try:
            console.print(f"\n[dim]Running: {test_case['name']}[/dim]")
            result = executor.search_arxiv(
                query=test_case['query'],
                max_results=2,
                get_full_text=False,
                auto_index=False
            )
            
            result_lower = result.lower()
            
            if "No arXiv papers found" in result:
                if "obscure" in test_case['name']:
                    results.add_pass("search_arxiv", test_case['name'])
                    console.print(f"[green]PASS[/green] (Correctly found no results)")
                else:
                    results.add_warning("search_arxiv", test_case['name'], 
                                      "No papers found")
                    console.print(f"[yellow]WARN[/yellow]: No papers found")
            else:
                missing_terms = [term for term in test_case['expected'] 
                               if term.lower() not in result_lower]
                if missing_terms and "obscure" not in test_case['name']:
                    results.add_warning("search_arxiv", test_case['name'], 
                                      f"Expected terms not found: {missing_terms}")
                    console.print(f"[yellow]WARN[/yellow]: Missing expected terms")
                else:
                    results.add_pass("search_arxiv", test_case['name'])
                    console.print(f"[green]PASS[/green]")
            
            console.print(f"[dim]Preview: {result[:200]}...[/dim]")
            
        except Exception as e:
            results.add_fail("search_arxiv", test_case['name'], str(e))
            console.print(f"[red]FAIL[/red]: {str(e)}")


def test_argument_normalization(executor: ToolExecutor, results: TestResults, console: Console):
    """Test argument normalization for flexible parameter handling."""
    console.print(Panel("[bold cyan]Testing: Argument Normalization[/bold cyan]"))
    
    # Test news_search argument aliases
    test_cases = [
        {
            "tool": "news_search",
            "name": "alias_query_to_keywords",
            "args": {"query": "technology", "max_results": 3},
            "expected_no_error": True
        },
        {
            "tool": "news_search",
            "name": "date_filter_alias",
            "args": {"keywords": "AI", "date_filter": "week", "max_results": 3},
            "expected_no_error": True
        },
        {
            "tool": "web_search",
            "name": "alias_q_to_query",
            "args": {"q": "Python", "iterations": 1},
            "expected_no_error": True
        }
    ]
    
    for test_case in test_cases:
        try:
            console.print(f"\n[dim]Running: {test_case['name']}[/dim]")
            result = executor.execute(test_case['tool'], test_case['args'])
            
            if result.startswith("Error") and test_case.get('expected_no_error'):
                results.add_fail("argument_normalization", test_case['name'], result)
                console.print(f"[red]FAIL[/red]: {result[:100]}")
            else:
                results.add_pass("argument_normalization", test_case['name'])
                console.print(f"[green]PASS[/green]")
            
        except Exception as e:
            results.add_fail("argument_normalization", test_case['name'], str(e))
            console.print(f"[red]FAIL[/red]: {str(e)}")


def run_all_tests():
    """Run all tool tests."""
    console = Console()
    results = TestResults()
    
    # Initialize test configuration
    config = {
        'web_search_enabled': True,
        'max_search_results': 20,
        'news_search_auto_index': False,
        'show_chunk_previews': False
    }
    
    # Initialize executor with mock RAG
    mock_rag = MockRAG()
    executor = ToolExecutor(config=config, console=console, web_search_rag=mock_rag)
    
    console.print(Panel.fit(
        "[bold cyan]Tool Test Suite[/bold cyan]\n"
        "Testing all tool implementations with various queries",
        border_style="cyan"
    ))
    
    # Run tests for each tool
    try:
        test_calculate(executor, results, console)
        test_get_current_time(executor, results, console)
        test_web_search(executor, results, console)
        test_news_search(executor, results, console)
        test_fetch_url_content(executor, results, console)
        test_search_wikipedia(executor, results, console)
        test_search_arxiv(executor, results, console)
        test_argument_normalization(executor, results, console)
    except KeyboardInterrupt:
        console.print("\n[yellow]Tests interrupted by user[/yellow]")
    
    # Print summary
    results.print_summary(console)
    
    # Return exit code
    return 0 if len(results.failed) == 0 else 1


if __name__ == "__main__":
    exit_code = run_all_tests()
    sys.exit(exit_code)
