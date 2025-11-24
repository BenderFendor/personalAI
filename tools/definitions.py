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
                'description': (
                    'Search the web for current information using DuckDuckGo. '
                    'Use this when you lack required facts, need to verify claims, the topic is time-sensitive, '
                    'or the query references specific entities, releases, or documentation. '
                    'Keep queries concise and constraint-rich (entities, year/version/OS). '
                    'Helpful operators: quotes for exact phrases, site:example.com, filetype:pdf, OR for alternatives. '
                    'Can perform multiple searches with reflection between each search via the iterations parameter.'
                ),
                'parameters': {
                    'type': 'object',
                    'required': ['query'],
                    'properties': {
                        'query': {
                            'type': 'string',
                            'description': 'The search query (concise; include key entities and constraints like year/version/OS/region)'
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
                'name': 'search_and_fetch',
                'description': (
                    'Perform a ranked web search then automatically fetch, chunk, and index the most relevant pages into the RAG vector store. '
                    'Use this when you need synthesized, source-cited answers from fresh web content. Always yields both a search summary and '
                    'the indexed chunk previews for transparency. Respects content-type safety filters (skips video/PDF/binary). '
                    'Chunk previews can be toggled via configuration.'
                ),
                'parameters': {
                    'type': 'object',
                    'required': ['query'],
                    'properties': {
                        'query': {
                            'type': 'string',
                            'description': 'The search query (concise; include key entities, constraints, version/date qualifiers)'
                        },
                        'max_search_results': {
                            'type': 'integer',
                            'description': 'Maximum raw search results to pull before ranking (default 15).',
                            'default': 15,
                            'minimum': 3,
                            'maximum': 50
                        },
                        'max_fetch_pages': {
                            'type': 'integer',
                            'description': 'Maximum number of pages to fetch & index after ranking (default 5).',
                            'default': 5,
                            'minimum': 1,
                            'maximum': 10
                        },
                        'similarity_threshold': {
                            'type': 'number',
                            'description': 'Minimum semantic similarity (0-1) for a URL to be fetched (default 0.55).',
                            'default': 0.55,
                            'minimum': 0.0,
                            'maximum': 1.0
                        },
                        'diversity_lambda': {
                            'type': 'number',
                            'description': 'MMR diversity weighting (0=similarity only, typical 0.3–0.7). Default 0.4.',
                            'default': 0.4,
                            'minimum': 0.0,
                            'maximum': 1.0
                        },
                        'fetch_concurrency': {
                            'type': 'integer',
                            'description': 'Parallel fetch concurrency limit (default 3).',
                            'default': 3,
                            'minimum': 1,
                            'maximum': 8
                        },
                        'include_chunks': {
                            'type': 'boolean',
                            'description': 'Force include chunk previews even if global flag disabled (override).',
                            'default': False
                        }
                    }
                }
            }
        },
        {
            'type': 'function',
            'function': {
                'name': 'news_search',
                'description': (
                    'Search for recent news articles using DuckDuckGo News. Best for current events, breaking news, and time-sensitive information. '
                    'Returns articles with publication dates, sources, and images. When auto_fetch is enabled, automatically fetches, chunks, and indexes '
                    'the most relevant articles into the RAG vector store using semantic ranking.'
                ),
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
                        },
                        'auto_fetch': {
                            'type': 'boolean',
                            'description': 'Enable automatic fetching, chunking, and RAG indexing of top-ranked articles. Default is false.',
                            'default': False
                        },
                        'max_fetch_pages': {
                            'type': 'integer',
                            'description': 'Maximum number of articles to fetch & index after ranking (only used when auto_fetch=true). Default is 5.',
                            'default': 5,
                            'minimum': 1,
                            'maximum': 10
                        },
                        'similarity_threshold': {
                            'type': 'number',
                            'description': 'Minimum semantic similarity (0-1) for an article to be fetched (only used when auto_fetch=true). Default is 0.35.',
                            'default': 0.35,
                            'minimum': 0.0,
                            'maximum': 1.0
                        },
                        'include_chunks': {
                            'type': 'boolean',
                            'description': 'Force include chunk previews even if global flag disabled (only used when auto_fetch=true). Default is false.',
                            'default': False
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
        },
        {
            'type': 'function',
            'function': {
                'name': 'search_vector_db',
                'description': 'Search the internal vector database (RAG) for relevant indexed documents, files, web pages, or news articles. Use this to find information from previously indexed content, including fetched news articles with dates and sources. Results include chunk text and metadata (source, title, date, publisher for news).',
                'parameters': {
                    'type': 'object',
                    'required': ['query'],
                    'properties': {
                        'query': {
                            'type': 'string',
                            'description': 'The search query to run against the vector database.'
                        },
                        'top_k': {
                            'type': 'integer',
                            'description': 'The number of top results to return. Default is 3.',
                            'default': 3,
                            'minimum': 1,
                            'maximum': 10
                        }
                    }
                }
            }
        },
        {
            'type': 'function',
            'function': {
                'name': 'search_wikipedia',
                'description': (
                    'Search Wikipedia for encyclopedic, factual information. Returns article summaries with inline URL citations. '
                    'Best for: definitions, historical facts, biographical information, scientific concepts, geographic data. '
                    'Handles disambiguation automatically by ranking results semantically. Content is automatically chunked and '
                    'indexed into RAG vector store for later retrieval. Use this for authoritative reference material, not current events.'
                ),
                'parameters': {
                    'type': 'object',
                    'required': ['query'],
                    'properties': {
                        'query': {
                            'type': 'string',
                            'description': 'Search query (e.g., "quantum computing", "Marie Curie", "Python programming language")'
                        },
                        'top_k': {
                            'type': 'integer',
                            'description': 'Number of Wikipedia articles to retrieve (1-5). Default is 2.',
                            'default': 2,
                            'minimum': 1,
                            'maximum': 5
                        },
                        'max_chars': {
                            'type': 'integer',
                            'description': 'Maximum characters per article summary for display. Default is 3000.',
                            'default': 3000,
                            'minimum': 500,
                            'maximum': 10000
                        },
                        'auto_index': {
                            'type': 'boolean',
                            'description': 'Automatically chunk and index full article text into RAG vector store. Default is true.',
                            'default': True
                        }
                    }
                }
            }
        },
        {
            'type': 'function',
            'function': {
                'name': 'search_arxiv',
                'description': (
                    'Search arXiv.org for academic papers and research articles in physics, mathematics, computer science, '
                    'and related fields. Returns paper metadata (title, authors, abstract, publication date) with inline URL citations. '
                    'Optionally fetches and parses full PDF content. All content is automatically chunked and indexed into RAG vector store. '
                    'Best for: academic research, technical papers, algorithm documentation, scientific studies, latest research developments.'
                ),
                'parameters': {
                    'type': 'object',
                    'required': ['query'],
                    'properties': {
                        'query': {
                            'type': 'string',
                            'description': 'Search query (e.g., "transformer neural networks", "quantum machine learning", "graph algorithms")'
                        },
                        'max_results': {
                            'type': 'integer',
                            'description': 'Maximum number of papers to retrieve (1-10). Default is 3.',
                            'default': 3,
                            'minimum': 1,
                            'maximum': 10
                        },
                        'get_full_text': {
                            'type': 'boolean',
                            'description': 'Download and parse full PDF content (slower, provides complete paper text). Default is false (abstract only).',
                            'default': False
                        },
                        'sort_by': {
                            'type': 'string',
                            'description': 'Sort order: "relevance", "lastUpdatedDate", or "submittedDate". Default is "relevance".',
                            'default': 'relevance',
                            'enum': ['relevance', 'lastUpdatedDate', 'submittedDate']
                        },
                        'auto_index': {
                            'type': 'boolean',
                            'description': 'Automatically chunk and index paper content into RAG vector store. Default is true.',
                            'default': True
                        }
                    }
                }
            }
        },
        {
            'type': 'function',
            'function': {
                'name': 'deep_research',
                'description': (
                    'Conducts comprehensive multi-step research by recursively breaking down complex topics, '
                    'searching multiple aspects across web and academic sources, and synthesizing findings into '
                    'a detailed report with citations. Use for: broad inquiries ("state of AI in healthcare"), '
                    'complex questions requiring multi-source integration, topics needing both breadth and depth, '
                    'literature reviews, comparative analyses, and research summaries.'
                ),
                'parameters': {
                    'type': 'object',
                    'required': ['topic'],
                    'properties': {
                        'topic': {
                            'type': 'string',
                            'description': 'The main research subject or question'
                        },
                        'depth': {
                            'type': 'integer',
                            'description': 'Number of recursive research iterations (1-5). Higher depth = more thorough but slower. Default: 3',
                            'default': 3,
                            'minimum': 1,
                            'maximum': 5
                        },
                        'breadth': {
                            'type': 'integer',
                            'description': 'Number of sub-topics to investigate per iteration (2-6). Higher breadth = broader coverage. Default: 4',
                            'default': 4,
                            'minimum': 2,
                            'maximum': 6
                        },
                        'quality_threshold': {
                            'type': 'number',
                            'description': 'Quality score (0.0-1.0) to stop early if reached. Default: 0.75',
                            'default': 0.75,
                            'minimum': 0.0,
                            'maximum': 1.0
                        },
                        'include_academic': {
                            'type': 'boolean',
                            'description': 'Include academic sources (Semantic Scholar, arXiv). Default: true',
                            'default': True
                        }
                    }
                }
            }
        },
        {
            'type': 'function',
            'function': {
                'name': 'search_academic',
                'description': (
                    'Searches Semantic Scholar for peer-reviewed academic papers, research studies, and scholarly articles. '
                    'Ideal for: empirical data, citations, academic credibility, research methodologies, theoretical foundations. '
                    'Returns papers with abstracts, citation counts, and PDFs when available. '
                    'Content is automatically indexed into RAG vector store.'
                ),
                'parameters': {
                    'type': 'object',
                    'required': ['query'],
                    'properties': {
                        'query': {
                            'type': 'string',
                            'description': 'Academic search query (e.g., "transformer architectures attention mechanisms")'
                        },
                        'limit': {
                            'type': 'integer',
                            'description': 'Number of papers to retrieve (1-100). Default: 10',
                            'default': 10,
                            'minimum': 1,
                            'maximum': 100
                        },
                        'year_filter': {
                            'type': 'string',
                            'description': 'Year range (e.g., "2020-2024" for range, "2023-" for 2023 onwards). Omit for all years.'
                        },
                        'fields_of_study': {
                            'type': 'array',
                            'items': {'type': 'string'},
                            'description': 'Filter by fields (e.g., ["Computer Science", "Medicine"]). Omit for all fields.'
                        }
                    }
                }
            }
        },
        {
            'type': 'function',
            'function': {
                'name': 'search_pubmed',
                'description': (
                    'Searches PubMed for biomedical and life sciences literature. '
                    'Best for: medical research, clinical studies, drug information, disease mechanisms, healthcare topics. '
                    'Returns peer-reviewed articles with abstracts and PubMed IDs. '
                    'Content is automatically indexed into RAG vector store.'
                ),
                'parameters': {
                    'type': 'object',
                    'required': ['query'],
                    'properties': {
                        'query': {
                            'type': 'string',
                            'description': 'PubMed search query (supports MeSH terms and Boolean operators)'
                        },
                        'limit': {
                            'type': 'integer',
                            'description': 'Number of results (1-50). Default: 10',
                            'default': 10,
                            'minimum': 1,
                            'maximum': 50
                        },
                        'sort': {
                            'type': 'string',
                            'enum': ['relevance', 'date'],
                            'description': 'Sort order. Default: relevance',
                            'default': 'relevance'
                        }
                    }
                }
            }
        }
    ]
