"""Tool definitions for Ollama function calling."""

from typing import List, Dict, Any


def get_tool_definitions() -> List[Dict[str, Any]]:
    """Get tool definitions for Ollama.
    
    Returns:
        List of tool definition dictionaries
    """
    return [
        {
            'type': 'function',
            'function': {
                'name': 'web_search',
                'description': 'Search the web for current information using DuckDuckGo. Can perform multiple searches with reflection between each search. Use iterations parameter to control how many times to search and refine the query.',
                'parameters': {
                    'type': 'object',
                    'required': ['query'],
                    'properties': {
                        'query': {
                            'type': 'string',
                            'description': 'The search query'
                        },
                        'iterations': {
                            'type': 'integer',
                            'description': 'Number of search iterations to perform (1-5). After each search, results are analyzed before the next search. Default is 1.',
                            'default': 1,
                            'minimum': 1,
                            'maximum': 5
                        }
                    }
                }
            }
        },
        {
            'type': 'function',
            'function': {
                'name': 'news_search',
                'description': 'Search for recent news articles using DuckDuckGo News. Best for current events, breaking news, and time-sensitive information. Returns articles with publication dates, sources, and images.',
                'parameters': {
                    'type': 'object',
                    'required': ['keywords'],
                    'properties': {
                        'keywords': {
                            'type': 'string',
                            'description': 'Keywords to search for in news articles'
                        },
                        'region': {
                            'type': 'string',
                            'description': 'Region code (e.g., "us-en", "uk-en", "wt-wt" for worldwide). Default is "us-en".',
                            'default': 'us-en'
                        },
                        'safesearch': {
                            'type': 'string',
                            'description': 'Safe search level: "on", "moderate", or "off". Default is "moderate".',
                            'default': 'moderate',
                            'enum': ['on', 'moderate', 'off']
                        },
                        'timelimit': {
                            'type': 'string',
                            'description': 'Time filter: "d" (day), "w" (week), "m" (month), or null for all time. Default is null.',
                            'enum': ['d', 'w', 'm']
                        },
                        'max_results': {
                            'type': 'integer',
                            'description': 'Maximum number of news articles to return. Default is 10.',
                            'default': 10,
                            'minimum': 1,
                            'maximum': 50
                        }
                    }
                }
            }
        },
        {
            'type': 'function',
            'function': {
                'name': 'fetch_url_content',
                'description': 'Fetch and extract clean text content from a webpage URL. Removes ads, navigation, and other boilerplate to return only the main article content. Use this to read full articles from search results.',
                'parameters': {
                    'type': 'object',
                    'required': ['url'],
                    'properties': {
                        'url': {
                            'type': 'string',
                            'description': 'The URL to fetch content from'
                        },
                        'max_length': {
                            'type': 'integer',
                            'description': 'Maximum character length of extracted content (to avoid context overflow). Default is 5000.',
                            'default': 5000,
                            'minimum': 500,
                            'maximum': 20000
                        }
                    }
                }
            }
        },
        {
            'type': 'function',
            'function': {
                'name': 'calculate',
                'description': 'Perform mathematical calculations',
                'parameters': {
                    'type': 'object',
                    'required': ['expression'],
                    'properties': {
                        'expression': {
                            'type': 'string',
                            'description': 'Mathematical expression to evaluate (e.g., "2 + 2", "sqrt(16)")'
                        }
                    }
                }
            }
        },
        {
            'type': 'function',
            'function': {
                'name': 'get_current_time',
                'description': 'Get the current date and time',
                'parameters': {
                    'type': 'object',
                    'properties': {}
                }
            }
        }
    ]
