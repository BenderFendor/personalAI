# Tool Testing Suite

This directory contains comprehensive testing utilities for the PersonalAI chatbot tools.

## Test Files

### 1. `test_tools.py` - Automated Test Suite
Comprehensive automated tests for all tools with predefined test cases.

**Features:**
- Tests all 8 tools with multiple scenarios each
- Validates success/failure cases
- Checks for expected content in results
- Provides detailed test summary with pass/fail/warning counts
- Handles edge cases and error conditions

**Usage:**
```bash
# Run all tests
python tests/test_tools.py

# Or with pytest (if installed)
pytest tests/test_tools.py -v
```

**What it tests:**
- ✅ `web_search`: Basic search, specific queries, multiple iterations
- ✅ `news_search`: Recent news, time filters, region-specific searches
- ✅ `fetch_url_content`: Valid URLs, invalid URLs, video blocking, truncation
- ✅ `calculate`: Arithmetic, functions (sqrt, sin, cos), security (import blocking)
- ✅ `get_current_time`: Date/time format validation
- ✅ `search_wikipedia`: Encyclopedia searches, nonexistent topics
- ✅ `search_arxiv`: Academic paper searches, field-specific queries
- ✅ Argument normalization: Parameter aliases and flexible inputs

### 2. `interactive_test.py` - Interactive Testing Tool
Manual testing interface for exploring tools with custom queries.

**Features:**
- Interactive menu-driven interface
- Pre-defined example queries for each tool
- Custom argument testing with JSON input
- Detailed tool documentation viewer
- Quick test mode for all tools

**Usage:**
```bash
# Interactive mode (default)
python tests/interactive_test.py

# Quick test mode (runs one example per tool)
python tests/interactive_test.py --quick
```

**Interactive Menu Options:**
1. **List all tools** - View all available tools and descriptions
2. **Test tool with example** - Choose from pre-defined test queries
3. **Test tool with custom arguments** - Enter JSON arguments manually
4. **View tool details** - See parameters, types, and descriptions
5. **Exit** - Quit the testing interface

## Quick Start

### Run Automated Tests
```bash
# Make sure you're in the project root with venv activated
source .venv/bin/activate

# Run the test suite
python tests/test_tools.py
```

Expected output:
```
╭─────────────────────────────────────╮
│ Tool Test Suite                     │
│ Testing all tool implementations    │
╰─────────────────────────────────────�╯

Testing: web_search
Running: basic_search
✓ PASSED
...

Test Results Summary
Category    Count
─────────────────
✓ Passed      25
✗ Failed       0
⚠ Warnings     3
Total Tests   25
```

### Run Interactive Tests
```bash
# Interactive mode
python tests/interactive_test.py

# Quick test all tools
python tests/interactive_test.py --quick
```

## Example Test Cases

### Web Search Examples
```python
# Basic search
web_search(query="Python programming language", iterations=1)

# Specific technical query
web_search(query="OpenAI GPT-4 release date", iterations=1)

# Multi-iteration research
web_search(query="machine learning algorithms", iterations=2)
```

### News Search Examples
```python
# Recent technology news
news_search(keywords="technology", max_results=5)

# Past day filter
news_search(keywords="artificial intelligence", timelimit="d", max_results=3)

# Regional news
news_search(keywords="politics", region="uk-en", max_results=3)
```

### Calculate Examples
```python
# Basic math
calculate(expression="2 + 2")  # Result: 4

# Functions
calculate(expression="sqrt(16)")  # Result: 4.0
calculate(expression="sin(pi/2)")  # Result: 1.0

# Complex expressions
calculate(expression="(10 + 5) * 2 - 8")  # Result: 22
```

### Wikipedia Search Examples
```python
# Encyclopedia search
search_wikipedia(query="Quantum computing", top_k=1)

# Multiple articles
search_wikipedia(query="Machine learning", top_k=2, max_chars=1000)
```

### arXiv Search Examples
```python
# Academic papers
search_arxiv(query="transformer neural networks", max_results=2)

# With full PDF text (slower)
search_arxiv(query="quantum algorithms", max_results=1, get_full_text=True)
```

## Test Configuration

Both test scripts use a mock configuration:

```python
config = {
    'web_search_enabled': True,
    'max_search_results': 20,
    'news_search_auto_index': False,  # Disable RAG indexing in tests
    'show_chunk_previews': False
}
```

This ensures tests run quickly without requiring a full ChromaDB setup.

## Testing Tips

### For Development
1. **Use interactive mode** when implementing new features
2. **Test edge cases** with custom JSON arguments
3. **Verify error handling** with invalid inputs

### For Validation
1. **Run automated suite** before committing changes
2. **Check all tests pass** (or have acceptable warnings)
3. **Review warnings** - they may indicate API changes or content issues

### Common Issues

**"No results found" warnings:**
- Often harmless - search APIs may have no results for some queries
- Check if the query is too specific or if API rate limits apply

**Network errors:**
- Tests require internet connection
- DuckDuckGo, Wikipedia, and arXiv APIs must be accessible
- Consider retry logic for flaky network conditions

**Timeout issues:**
- Some tools (especially `fetch_url_content`) may timeout on slow connections
- Adjust timeout values in `tools/implementations.py` if needed

## Adding New Tests

### To add a test to automated suite:

1. Add a new test function in `test_tools.py`:
```python
def test_new_tool(executor: ToolExecutor, results: TestResults, console: Console):
    """Test new_tool with various queries."""
    test_cases = [
        {
            "name": "test_case_1",
            "arg1": "value1",
            "expected": ["expected", "terms"]
        }
    ]
    # Test logic here...
```

2. Call it in `run_all_tests()`:
```python
test_new_tool(executor, results, console)
```

### To add examples to interactive test:

1. Add examples to `get_tool_examples()` in `interactive_test.py`:
```python
'new_tool': [
    {'arg1': 'value1', 'arg2': 'value2'},
    {'arg1': 'different_value'}
]
```

## Test Results Interpretation

### ✓ PASSED
- Tool executed successfully
- All expected content found in results
- No errors or exceptions

### ✗ FAILED
- Tool raised an exception
- Unexpected error returned
- Critical validation failed

### ⚠ WARNING
- Tool succeeded but with unexpected results
- Missing some expected terms (might be API variance)
- Edge case behavior (e.g., no search results)

## Dependencies

Tests use the same dependencies as the main application:
- `rich` - Console output and formatting
- `duckduckgo_search` - Web/news search
- `trafilatura` - URL content extraction
- `wikipedia-api` - Wikipedia search
- `arxiv` - arXiv paper search
- `PyMuPDF` - PDF parsing (for arXiv full text)

All dependencies should already be installed via `requirements.txt`.

## CI/CD Integration

To integrate with CI/CD pipelines:

```yaml
# Example GitHub Actions workflow
- name: Run tool tests
  run: |
    source .venv/bin/activate
    python tests/test_tools.py
```

Exit codes:
- `0` - All tests passed
- `1` - One or more tests failed

## Contributing

When adding new tools:
1. Add tool tests to `test_tools.py`
2. Add examples to `interactive_test.py`
3. Update this README with usage examples
4. Ensure all tests pass before submitting PR
