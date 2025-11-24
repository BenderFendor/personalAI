# Deep Research Agent Implementation Plan

This document outlines a comprehensive plan to transform the PersonalAI chatbot from a single-turn search assistant into a **deep research agent** capable of recursive inquiry, multi-source integration, and synthesized reporting.

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Phase 1: Recursive Research Loop](#phase-1-recursive-research-loop)
3. [Phase 2: Multi-Source Academic Integration](#phase-2-multi-source-academic-integration)
4. [Phase 3: SearXNG Integration](#phase-3-searxng-integration)
5. [Phase 4: Advanced Features](#phase-4-advanced-features)
6. [Configuration & Testing](#configuration--testing)
7. [Performance Optimization](#performance-optimization)

---

## Architecture Overview

### Current State
- **Single-turn web search** via DuckDuckGo
- **RAG indexing** with ChromaDB for web content
- **Tool-based execution** pattern with Ollama function calling
- **Agentic loop** (max 5 iterations) for tool chaining

### Target State: Deep Research Agent
```
User Query
    ↓
Planning Agent (Decomposition)
    ↓
[Parallel Execution Layer]
    ├─→ Web Search (SearXNG)
    ├─→ Academic Search (Semantic Scholar, PubMed)
    ├─→ Vector DB Retrieval (ChromaDB)
    └─→ Content Fetcher (Trafilatura)
    ↓
[Evaluation & Reflection]
    ├─→ Quality Assessment
    ├─→ Gap Analysis
    └─→ Follow-up Query Generation
    ↓
[Recursive Loop or Synthesis]
    ├─→ Iterate if depth > 0
    └─→ Generate Final Report
```

### Key Design Principles (from Research)

**From LangGraph/CrewAI Patterns:**
- **State-based workflows**: Use graph structures for complex multi-step reasoning
- **Supervisor-worker pattern**: Global planner + local executors
- **Escalation mechanisms**: Stop recursive loops when quality threshold is met

**From Academic Research (Tree/Graph of Thought):**
- **Multi-path reasoning**: Explore parallel research directions (breadth)
- **Recursive depth control**: Iterative refinement with depth limits
- **Graph-based thought modeling**: Dependencies between research sub-topics

**From Production RAG Systems:**
- **Semantic ranking**: Auto-fetch only URLs with >0.6 similarity
- **Chunking strategy**: 500 tokens/chunk, 100 token overlap (already implemented)
- **Citation requirements**: 2-4 sources minimum for web-backed claims

---

## Phase 1: Recursive Research Loop

### 1.1 Implementation: Planner-Executor Architecture

#### **File: `tools/implementations.py`**

Add the following method to `ToolExecutor`:

```python
import json
from typing import List, Dict, Any

def deep_research(
    self,
    topic: str,
    depth: int = 3,
    breadth: int = 4,
    quality_threshold: float = 0.75
) -> str:
    """
    Performs recursive research with planning, execution, evaluation, and synthesis.
    
    Args:
        topic: Main research subject
        depth: How many recursive iterations (1-5)
        breadth: Number of sub-queries per iteration (2-6)
        quality_threshold: Minimum quality score to stop early (0.0-1.0)
    
    Returns:
        Comprehensive research report with citations
    """
    import ollama
    from rich.progress import Progress, SpinnerColumn, TextColumn
    
    self.console.print(f"[bold magenta]🕵️ Deep Research Initiated: {topic}[/bold magenta]")
    
    # State tracking
    research_state = {
        "topic": topic,
        "current_depth": 0,
        "max_depth": depth,
        "findings": [],
        "indexed_sources": set(),
        "quality_scores": []
    }
    
    # Recursive research loop
    while research_state["current_depth"] < research_state["max_depth"]:
        current_depth = research_state["current_depth"]
        self.console.print(f"\n[cyan]📊 Research Depth: {current_depth + 1}/{depth}[/cyan]")
        
        # STEP 1: PLANNING - Generate sub-queries
        sub_queries = self._generate_sub_queries(
            topic=topic,
            previous_findings=research_state["findings"],
            breadth=breadth,
            depth_level=current_depth
        )
        
        self.console.print(f"[dim]Sub-queries: {sub_queries}[/dim]")
        
        # STEP 2: EXECUTION - Multi-source parallel search
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console
        ) as progress:
            search_task = progress.add_task(
                f"[cyan]Investigating {len(sub_queries)} sub-topics...",
                total=len(sub_queries)
            )
            
            iteration_findings = []
            for i, query in enumerate(sub_queries):
                # Multi-source search (will expand in Phase 2)
                findings = self._execute_multi_source_search(query)
                iteration_findings.append({
                    "query": query,
                    "content": findings,
                    "depth": current_depth
                })
                research_state["findings"].extend(iteration_findings)
                progress.update(search_task, advance=1)
        
        # STEP 3: EVALUATION - Assess research quality
        quality_score = self._evaluate_research_quality(
            topic=topic,
            findings=research_state["findings"]
        )
        research_state["quality_scores"].append(quality_score)
        
        self.console.print(
            f"[yellow]Quality Score: {quality_score:.2f}/1.0[/yellow]"
        )
        
        # STEP 4: ESCALATION CHECK - Stop if quality threshold met
        if quality_score >= quality_threshold:
            self.console.print(
                f"[green]✓ Quality threshold reached. Stopping early.[/green]"
            )
            break
        
        research_state["current_depth"] += 1
    
    # STEP 5: SYNTHESIS - Generate final report
    report = self._synthesize_report(
        topic=topic,
        findings=research_state["findings"],
        depth_reached=research_state["current_depth"]
    )
    
    return report


def _generate_sub_queries(
    self,
    topic: str,
    previous_findings: List[Dict],
    breadth: int,
    depth_level: int
) -> List[str]:
    """
    Uses LLM to decompose topic into searchable sub-questions.
    Adapts based on previous findings (recursive refinement).
    """
    import ollama
    
    # Context from previous iteration (if any)
    context = ""
    if previous_findings:
        recent = previous_findings[-breadth:]
        context = "\n".join([f"- {f['query']}: {f['content'][:200]}..." for f in recent])
    
    planning_prompt = f"""You are a research planning assistant. Break down the following research topic into {breadth} distinct, specific sub-questions for investigation.

Research Topic: {topic}
Current Depth: {depth_level + 1}

{f"Previous Findings (use to identify gaps):\n{context}" if context else ""}

Requirements:
- Each sub-question must be searchable and specific
- Avoid redundancy with previous queries
- Focus on different aspects: overview, recent developments, controversies, applications, limitations
- Return ONLY a JSON array of strings

Example: ["What are the core principles of X?", "What are recent breakthroughs in X?"]"""
    
    try:
        response = ollama.chat(
            model=self.config.get('model', 'qwen3'),
            messages=[{'role': 'user', 'content': planning_prompt}],
            format='json',
            options={'temperature': 0.7}  # Slightly creative
        )
        
        parsed = json.loads(response['message']['content'])
        
        # Handle different JSON formats
        if isinstance(parsed, dict) and 'queries' in parsed:
            return parsed['queries'][:breadth]
        elif isinstance(parsed, list):
            return parsed[:breadth]
        else:
            raise ValueError("Unexpected JSON format")
            
    except Exception as e:
        self.console.print(f"[yellow]Planning fallback: {e}[/yellow]")
        # Fallback sub-queries
        return [
            f"{topic} - overview and key concepts",
            f"{topic} - recent developments and research",
            f"{topic} - practical applications",
            f"{topic} - challenges and limitations"
        ][:breadth]


def _execute_multi_source_search(self, query: str) -> str:
    """
    Executes search across multiple sources (web + academic).
    Phase 1: Web only. Phase 2: Add academic sources.
    """
    # Use existing search_and_fetch with optimized params
    return self.search_and_fetch(
        query=query,
        max_search_results=5,
        max_fetch_pages=2,
        similarity_threshold=0.5,
        auto_fetch=True
    )


def _evaluate_research_quality(
    self,
    topic: str,
    findings: List[Dict]
) -> float:
    """
    Uses LLM to assess if current research sufficiently addresses the topic.
    Returns quality score (0.0-1.0).
    """
    import ollama
    
    # Prepare findings summary
    findings_text = "\n\n".join([
        f"Sub-topic: {f['query']}\nFindings: {f['content'][:500]}..."
        for f in findings[-6:]  # Last 6 findings
    ])
    
    evaluation_prompt = f"""You are a research quality evaluator. Assess if the following findings adequately address the research topic.

Research Topic: {topic}

Findings:
{findings_text}

Evaluate on these criteria:
1. Coverage: Are multiple aspects of the topic addressed?
2. Depth: Is there sufficient detail in each aspect?
3. Credibility: Are findings from reliable sources?
4. Coherence: Do findings fit together logically?

Return ONLY a JSON object with:
{{
  "score": <float 0.0-1.0>,
  "reasoning": "<brief explanation>",
  "gaps": ["<missing aspect 1>", "<missing aspect 2>"]
}}"""
    
    try:
        response = ollama.chat(
            model=self.config.get('model', 'qwen3'),
            messages=[{'role': 'user', 'content': evaluation_prompt}],
            format='json',
            options={'temperature': 0.3}  # More deterministic
        )
        
        result = json.loads(response['message']['content'])
        score = float(result.get('score', 0.5))
        
        self.console.print(f"[dim]Evaluation: {result.get('reasoning', 'N/A')}[/dim]")
        if result.get('gaps'):
            self.console.print(f"[dim]Gaps identified: {', '.join(result['gaps'])}[/dim]")
        
        return max(0.0, min(1.0, score))  # Clamp to [0, 1]
        
    except Exception as e:
        self.console.print(f"[yellow]Evaluation error: {e}. Using default score.[/yellow]")
        return 0.5  # Neutral score on error


def _synthesize_report(
    self,
    topic: str,
    findings: List[Dict],
    depth_reached: int
) -> str:
    """
    Generates a comprehensive, citation-rich research report.
    Queries RAG for indexed sources to include in synthesis.
    """
    import ollama
    
    # Retrieve relevant context from RAG
    if self.web_search_rag:
        try:
            rag_context = self.web_search_rag.search(topic, top_k=10)
            rag_summary = "\n".join([
                f"Source: {chunk.metadata.get('source', 'N/A')}\n{chunk.content[:300]}..."
                for chunk in rag_context
            ])
        except:
            rag_summary = ""
    else:
        rag_summary = ""
    
    # Organize findings by depth
    findings_by_depth = {}
    for f in findings:
        d = f.get('depth', 0)
        if d not in findings_by_depth:
            findings_by_depth[d] = []
        findings_by_depth[d].append(f)
    
    findings_text = ""
    for depth, items in sorted(findings_by_depth.items()):
        findings_text += f"\n## Research Phase {depth + 1}\n"
        for item in items:
            findings_text += f"### {item['query']}\n{item['content'][:600]}...\n\n"
    
    synthesis_prompt = f"""You are an expert research synthesizer. Create a comprehensive report on the following topic using the provided research findings.

Topic: {topic}
Depth Reached: {depth_reached + 1} iterations

Research Findings:
{findings_text}

Additional Context from Knowledge Base:
{rag_summary if rag_summary else "No additional indexed sources available."}

Instructions:
1. Write a well-structured report with:
   - Executive Summary
   - Key Findings (organized by theme, not chronologically)
   - Detailed Analysis
   - Limitations & Future Research
2. **Cite sources** inline using [Source: URL] format
3. Synthesize conflicting information objectively
4. Highlight areas with insufficient data
5. Use academic tone but remain accessible
6. Minimum 2-4 unique sources cited

Format: Markdown"""
    
    response = ollama.chat(
        model=self.config.get('model', 'qwen3'),
        messages=[{'role': 'user', 'content': synthesis_prompt}],
        options={'temperature': 0.5, 'num_ctx': 8192}  # Larger context for synthesis
    )
    
    report = response['message']['content']
    
    # Add metadata footer
    report += f"\n\n---\n**Research Metadata**\n"
    report += f"- Topic: {topic}\n"
    report += f"- Research Depth: {depth_reached + 1} iterations\n"
    report += f"- Total Sub-Queries: {len(findings)}\n"
    report += f"- Indexed Sources: {len(self.url_cache) if hasattr(self, 'url_cache') else 'N/A'}\n"
    
    return report
```

#### **File: `tools/definitions.py`**

Add the tool definition:

```python
{
    'type': 'function',
    'function': {
        'name': 'deep_research',
        'description': 'Conducts comprehensive multi-step research by recursively breaking down complex topics, searching multiple aspects, and synthesizing findings into a detailed report with citations. Use for: broad inquiries ("state of AI in healthcare"), complex questions requiring multi-source integration, topics needing both breadth and depth.',
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
                }
            }
        }
    }
}
```

### 1.2 Context Window Optimization

**Current Issue:** Deep research generates large amounts of text that can exceed context limits.

**Solution:** Implement context-aware chunking in `_synthesize_report`:

```python
# In _synthesize_report method, before calling Ollama:

from utils.context import ContextCalculator

context_calc = ContextCalculator(self.config.get('model', 'qwen3'))
available_tokens = context_calc.get_context_size() - 2000  # Reserve for response

# Truncate findings if needed
findings_text_truncated = findings_text[:available_tokens * 4]  # ~4 chars per token
```

---

## Phase 2: Multi-Source Academic Integration

### 2.1 Semantic Scholar API Integration

#### **File: `tools/implementations.py`**

Add academic search tool:

```python
def search_academic(
    self,
    query: str,
    limit: int = 10,
    year_filter: str = None,
    fields_of_study: List[str] = None
) -> str:
    """
    Searches Semantic Scholar for academic papers and auto-indexes results.
    
    Args:
        query: Search query
        limit: Number of papers to retrieve (1-100)
        year_filter: Year range (e.g., "2020-2024" or "2023-")
        fields_of_study: List of fields (e.g., ["Computer Science", "Medicine"])
    
    Returns:
        Formatted academic results with citations
    """
    import requests
    
    url = "https://api.semanticscholar.org/graph/v1/paper/search"
    
    # Build query params
    params = {
        "query": query,
        "limit": min(limit, 100),
        "fields": "title,authors,year,abstract,url,citationCount,openAccessPdf,venue,publicationDate,fieldsOfStudy"
    }
    
    if year_filter:
        params["year"] = year_filter
    
    if fields_of_study:
        params["fieldsOfStudy"] = ",".join(fields_of_study)
    
    try:
        self.console.print(f"[dim]🎓 Querying Semantic Scholar: {query}[/dim]")
        
        headers = {}
        # Add API key if available (optional but recommended for higher rate limits)
        if self.config.get('semantic_scholar_api_key'):
            headers['x-api-key'] = self.config['semantic_scholar_api_key']
        
        response = requests.get(url, params=params, headers=headers, timeout=15)
        
        if response.status_code == 429:
            return "Rate limit exceeded for Semantic Scholar. Please try again later or use an API key."
        
        if not response.ok:
            return f"Academic search failed: HTTP {response.status_code}"
        
        data = response.json()
        
        if not data.get("data"):
            return f"No academic papers found for: {query}"
        
        # Format and index results
        formatted_results = []
        for paper in data['data'][:limit]:
            # Extract key info
            title = paper.get('title', 'Unknown Title')
            year = paper.get('year', 'N/A')
            authors = ", ".join([
                a.get('name', 'Unknown')
                for a in paper.get('authors', [])[:3]
            ])
            if len(paper.get('authors', [])) > 3:
                authors += " et al."
            
            citation_count = paper.get('citationCount', 0)
            abstract = paper.get('abstract', 'No abstract available.')
            url = paper.get('url', '')
            pdf_url = paper.get('openAccessPdf', {}).get('url', 'No PDF')
            venue = paper.get('venue', 'Unknown Venue')
            
            # Format for display
            formatted = f"""
**{title}** ({year})
*Authors:* {authors}
*Venue:* {venue} | *Citations:* {citation_count}
*Abstract:* {abstract[:400]}{"..." if len(abstract) > 400 else ""}
*URL:* {url}
*PDF:* {pdf_url}
"""
            formatted_results.append(formatted)
            
            # AUTO-INDEX into RAG (critical for synthesis)
            if self.web_search_rag:
                full_text = f"""
Title: {title}
Year: {year}
Authors: {authors}
Venue: {venue}
Citations: {citation_count}
Abstract: {abstract}
URL: {url}
PDF: {pdf_url}
"""
                try:
                    self.web_search_rag.index_single_page(
                        url=url or f"s2paper_{paper.get('paperId', 'unknown')}",
                        content=full_text,
                        title=title,
                        metadata={
                            "type": "academic",
                            "year": str(year),
                            "venue": venue,
                            "citation_count": str(citation_count)
                        }
                    )
                except Exception as e:
                    self.console.print(f"[yellow]Failed to index paper: {e}[/yellow]")
        
        result_text = "\n---\n".join(formatted_results)
        return f"Found {len(formatted_results)} academic papers:\n\n{result_text}"
    
    except Exception as e:
        return f"Error in academic search: {str(e)}"


def search_pubmed(
    self,
    query: str,
    limit: int = 10,
    sort: str = "relevance"
) -> str:
    """
    Searches PubMed for biomedical literature.
    
    Args:
        query: Search query (supports PubMed query syntax)
        limit: Number of results (1-50)
        sort: Sort order ("relevance", "date", or "citations")
    
    Returns:
        Formatted PubMed results
    """
    from Bio import Entrez
    import requests
    
    # Set email (required by NCBI)
    Entrez.email = self.config.get('pubmed_email', 'user@example.com')
    
    try:
        self.console.print(f"[dim]🧬 Querying PubMed: {query}[/dim]")
        
        # Search PubMed
        search_handle = Entrez.esearch(
            db="pubmed",
            term=query,
            retmax=limit,
            sort=sort
        )
        search_results = Entrez.read(search_handle)
        search_handle.close()
        
        id_list = search_results["IdList"]
        
        if not id_list:
            return f"No PubMed articles found for: {query}"
        
        # Fetch details
        fetch_handle = Entrez.efetch(
            db="pubmed",
            id=id_list,
            rettype="abstract",
            retmode="xml"
        )
        articles = Entrez.read(fetch_handle)
        fetch_handle.close()
        
        formatted_results = []
        for article in articles['PubmedArticle']:
            medline = article['MedlineCitation']
            article_data = medline['Article']
            
            title = article_data.get('ArticleTitle', 'Unknown Title')
            abstract = article_data.get('Abstract', {}).get('AbstractText', ['No abstract'])
            abstract_text = " ".join(abstract) if isinstance(abstract, list) else str(abstract)
            
            pmid = str(medline['PMID'])
            journal = article_data.get('Journal', {}).get('Title', 'Unknown Journal')
            pub_date = article_data.get('ArticleDate', [{}])[0] if article_data.get('ArticleDate') else {}
            year = pub_date.get('Year', 'N/A')
            
            authors = []
            for author in article_data.get('AuthorList', [])[:3]:
                last_name = author.get('LastName', '')
                initials = author.get('Initials', '')
                if last_name:
                    authors.append(f"{last_name} {initials}")
            authors_str = ", ".join(authors)
            if len(article_data.get('AuthorList', [])) > 3:
                authors_str += " et al."
            
            url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
            
            formatted = f"""
**{title}** ({year})
*Authors:* {authors_str}
*Journal:* {journal} | *PMID:* {pmid}
*Abstract:* {abstract_text[:400]}{"..." if len(abstract_text) > 400 else ""}
*URL:* {url}
"""
            formatted_results.append(formatted)
            
            # Auto-index
            if self.web_search_rag:
                full_text = f"""
Title: {title}
Year: {year}
Authors: {authors_str}
Journal: {journal}
PMID: {pmid}
Abstract: {abstract_text}
URL: {url}
"""
                try:
                    self.web_search_rag.index_single_page(
                        url=url,
                        content=full_text,
                        title=title,
                        metadata={
                            "type": "pubmed",
                            "year": year,
                            "journal": journal,
                            "pmid": pmid
                        }
                    )
                except Exception as e:
                    self.console.print(f"[yellow]Failed to index PubMed article: {e}[/yellow]")
        
        result_text = "\n---\n".join(formatted_results)
        return f"Found {len(formatted_results)} PubMed articles:\n\n{result_text}"
    
    except Exception as e:
        return f"Error in PubMed search: {str(e)}"
```

#### **File: `tools/definitions.py`**

Add academic tool definitions:

```python
{
    'type': 'function',
    'function': {
        'name': 'search_academic',
        'description': 'Searches Semantic Scholar for peer-reviewed academic papers, research studies, and scholarly articles. Ideal for: empirical data, citations, academic credibility, research methodologies, theoretical foundations. Returns papers with abstracts, citation counts, and PDFs when available.',
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
                    'description': 'Year range (e.g., "2020-2024" for range, "2023-" for 2023 onwards). Omit for all years.',
                    'default': None
                },
                'fields_of_study': {
                    'type': 'array',
                    'items': {'type': 'string'},
                    'description': 'Filter by fields (e.g., ["Computer Science", "Medicine"]). Omit for all fields.',
                    'default': None
                }
            }
        }
    }
},
{
    'type': 'function',
    'function': {
        'name': 'search_pubmed',
        'description': 'Searches PubMed for biomedical and life sciences literature. Best for: medical research, clinical studies, drug information, disease mechanisms, healthcare topics. Returns peer-reviewed articles with abstracts and PubMed IDs.',
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
                    'enum': ['relevance', 'date', 'citations'],
                    'description': 'Sort order. Default: relevance',
                    'default': 'relevance'
                }
            }
        }
    }
}
```

### 2.2 Update Multi-Source Search Logic

Modify `_execute_multi_source_search` in Phase 1:

```python
def _execute_multi_source_search(self, query: str) -> str:
    """
    Executes parallel search across web and academic sources.
    Intelligently routes queries to appropriate sources.
    """
    results = []
    
    # Always search web
    web_result = self.search_and_fetch(
        query=query,
        max_search_results=5,
        max_fetch_pages=2,
        similarity_threshold=0.5
    )
    results.append(f"## Web Sources\n{web_result}")
    
    # Heuristic: If query contains academic keywords, search academic sources
    academic_keywords = [
        'research', 'study', 'paper', 'journal', 'empirical',
        'theory', 'model', 'methodology', 'analysis', 'evidence'
    ]
    
    is_academic = any(kw in query.lower() for kw in academic_keywords)
    
    if is_academic:
        # Search Semantic Scholar
        try:
            academic_result = self.search_academic(query, limit=5)
            results.append(f"## Academic Sources\n{academic_result}")
        except Exception as e:
            self.console.print(f"[yellow]Academic search skipped: {e}[/yellow]")
    
    # For medical/biomedical queries, search PubMed
    medical_keywords = [
        'disease', 'treatment', 'drug', 'medical', 'clinical',
        'patient', 'diagnosis', 'therapy', 'health', 'medicine'
    ]
    
    is_medical = any(kw in query.lower() for kw in medical_keywords)
    
    if is_medical:
        try:
            pubmed_result = self.search_pubmed(query, limit=5)
            results.append(f"## Biomedical Sources\n{pubmed_result}")
        except Exception as e:
            self.console.print(f"[yellow]PubMed search skipped: {e}[/yellow]")
    
    return "\n\n".join(results)
```

### 2.3 Dependencies

Add to `requirements.txt`:

```
biopython>=1.81  # For PubMed search
```

Install:
```bash
uv pip install biopython
```

---

## Phase 3: SearXNG Integration

### 3.1 SearXNG Deployment

**Option 1: Docker (Recommended for Development)**


I have searxng installed used localhost:8080 for it

### 3.3 Update Web Search Tool

#### **File: `tools/implementations.py`**

Replace DuckDuckGo with SearXNG:

```python
def search_and_fetch(
    self,
    query: str,
    max_search_results: int = 20,
    max_fetch_pages: int = 3,
    similarity_threshold: float = 0.6,
    auto_fetch: bool = True,
    iterations: int = 1,
    searxng_engines: List[str] = None
) -> str:
    """
    Enhanced search using SearXNG meta-search engine.
    Falls back to DuckDuckGo if SearXNG unavailable.
    
    Args:
        searxng_engines: List of engines to use (e.g., ['google', 'bing', 'duckduckgo'])
    """
    searxng_url = self.config.get('searxng_url', 'http://localhost:8080')
    use_searxng = self.config.get('use_searxng', False)
    
    if use_searxng:
        try:
            results = self._searxng_search(
                query=query,
                max_results=max_search_results,
                engines=searxng_engines
            )
        except Exception as e:
            self.console.print(f"[yellow]SearXNG unavailable, falling back to DuckDuckGo: {e}[/yellow]")
            results = self._ddg_search(query, max_search_results)
    else:
        results = self._ddg_search(query, max_search_results)
    
    # Rest of existing auto-fetch logic remains the same...
    # (semantic ranking, URL fetching, RAG indexing)
    
    return formatted_results


def _searxng_search(
    self,
    query: str,
    max_results: int,
    engines: List[str] = None
) -> List[Dict]:
    """
    Queries SearXNG meta-search API.
    Returns list of search results with URL, title, content.
    """
    import requests
    
    searxng_url = self.config.get('searxng_url', 'http://localhost:8080')
    
    params = {
        'q': query,
        'format': 'json',
        'pageno': 1
    }
    
    if engines:
        params['engines'] = ','.join(engines)
    
    try:
        response = requests.get(
            f"{searxng_url}/search",
            params=params,
            timeout=10
        )
        
        if not response.ok:
            raise Exception(f"HTTP {response.status_code}")
        
        data = response.json()
        results = []
        
        for item in data.get('results', [])[:max_results]:
            results.append({
                'href': item.get('url', ''),
                'title': item.get('title', ''),
                'body': item.get('content', ''),
                'engines': item.get('engines', [])
            })
        
        return results
    
    except Exception as e:
        self.console.print(f"[red]SearXNG search failed: {e}[/red]")
        return []


def _ddg_search(self, query: str, max_results: int) -> List[Dict]:
    """
    Fallback to DuckDuckGo (existing implementation).
    """
    from duckduckgo_search import DDGS
    
    try:
        ddgs = DDGS()
        results = list(ddgs.text(query, max_results=max_results))
        return results
    except Exception as e:
        self.console.print(f"[red]DuckDuckGo search failed: {e}[/red]")
        return []
```

### 3.4 Configuration

Add to `config.json`:

```json
{
  "use_searxng": true,
  "searxng_url": "http://localhost:8080",
  "searxng_default_engines": ["google", "bing", "duckduckgo"],
  "semantic_scholar_api_key": null,
  "pubmed_email": "your_email@example.com"
}
```

---

## Phase 4: Advanced Features

### 4.1 Progressive Reporting

Stream intermediate findings to user during deep research:

```python
# In deep_research method, after each iteration:

yield {
    "type": "progress",
    "depth": current_depth,
    "query": sub_queries,
    "findings": iteration_findings
}
```

Modify `chat.py` to handle streaming updates.

### 4.2 Research Persistence

Save research sessions to disk:

```python
# In _synthesize_report:

import json
from datetime import datetime

report_path = f"research_logs/research_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

with open(report_path, 'w') as f:
    json.dump({
        "topic": topic,
        "timestamp": datetime.now().isoformat(),
        "findings": findings,
        "report": report
    }, f, indent=2)

self.console.print(f"[dim]Research saved to: {report_path}[/dim]")
```

### 4.3 Citation Graph Visualization

Use NetworkX to visualize citation relationships between sources:

```python
import networkx as nx
import matplotlib.pyplot as plt

def visualize_research_graph(findings: List[Dict]):
    """
    Creates a graph showing relationships between research sub-topics.
    """
    G = nx.DiGraph()
    
    for f in findings:
        G.add_node(f['query'], depth=f['depth'])
    
    # Add edges based on depth (parent-child relationships)
    # ... implementation details
    
    plt.figure(figsize=(12, 8))
    nx.draw(G, with_labels=True, node_color='lightblue', node_size=2000)
    plt.savefig('research_graph.png')
    return 'research_graph.png'
```

---

## Configuration & Testing

### Configuration Schema

Complete `config.json` example:

```json
{
  "model": "qwen3",
  "temperature": 0.7,
  "thinking_enabled": true,
  "tools_enabled": true,
  
  "deep_research": {
    "default_depth": 3,
    "default_breadth": 4,
    "quality_threshold": 0.75,
    "max_tokens_per_iteration": 4096
  },
  
  "search": {
    "use_searxng": true,
    "searxng_url": "http://localhost:8080",
    "searxng_engines": ["google", "bing", "duckduckgo", "google scholar"],
    "web_search_rag_enabled": true,
    "max_search_results": 20,
    "auto_fetch_threshold": 0.6
  },
  
  "academic": {
    "semantic_scholar_api_key": null,
    "pubmed_email": "user@example.com",
    "default_year_filter": "2020-"
  },
  
  "rag": {
    "chunk_size": 500,
    "chunk_overlap": 100,
    "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
    "chroma_db_path": "./chroma_db",
    "collection": "rag_documents"
  }
}
```

### Testing Plan

#### Test 1: Shallow Research (Depth=1)
```python
# Expected: 4 sub-queries, ~5 sources, report in <2 min
python main.py
> Use deep_research to investigate "quantum computing applications" with depth=1
```

#### Test 2: Medium Research (Depth=3, Default)
```python
# Expected: 12 sub-queries, ~20 sources, report in 3-5 min
> Research "impact of large language models on software development" in depth
```

#### Test 3: Academic-Heavy Research
```python
# Expected: Mix of web + academic sources, citations visible
> Deep research "transformer attention mechanisms in NLP" focusing on academic sources
```

#### Test 4: Quality Threshold Early Stop
```python
# Expected: Stops at depth=1-2 if high-quality sources found
> Research "Python programming basics" with quality_threshold=0.8
```

### Performance Benchmarks

| Metric | Target | Measurement |
|--------|--------|-------------|
| **Depth=1 Research Time** | <2 minutes | Time from query to report |
| **Depth=3 Research Time** | 3-5 minutes | Full cycle with 12 sub-queries |
| **RAG Indexing Rate** | >10 docs/sec | ChromaDB ingestion speed |
| **Academic API Latency** | <3 seconds | Semantic Scholar response time |
| **Context Overflow Rate** | <5% | Queries exceeding context limit |

---

## Performance Optimization

### 1. Parallel Execution

Current: Sequential sub-query execution  
**Improvement:** Parallel execution with `asyncio`

```python
import asyncio

async def _execute_parallel_searches(self, queries: List[str]) -> List[str]:
    """
    Executes multiple searches concurrently.
    """
    tasks = [
        asyncio.to_thread(self._execute_multi_source_search, q)
        for q in queries
    ]
    return await asyncio.gather(*tasks)
```

### 2. Caching Layer

Add Redis for API result caching:

```python
import redis

class SearchCache:
    def __init__(self):
        self.redis = redis.Redis(host='localhost', port=6379, decode_responses=True)
    
    def get(self, key: str) -> Optional[str]:
        return self.redis.get(f"search:{key}")
    
    def set(self, key: str, value: str, ttl: int = 3600):
        self.redis.setex(f"search:{key}", ttl, value)
```

### 3. Streaming Responses

Use Ollama's streaming API for real-time progress:

```python
for chunk in ollama.chat(
    model='qwen3',
    messages=messages,
    stream=True
):
    print(chunk['message']['content'], end='', flush=True)
```

### 4. Resource Limits

Prevent runaway searches:

```python
# In config.json
{
  "deep_research": {
    "max_total_queries": 30,
    "max_indexed_sources": 100,
    "timeout_per_iteration": 120  # seconds
  }
}
```

---

## Migration Path

### Phase 1 (Week 1): Core Infrastructure
- [ ] Implement `deep_research` tool with planning/execution/evaluation
- [ ] Add quality assessment logic
- [ ] Test with web-only sources
- [ ] Update system prompt with deep research guidance

### Phase 2 (Week 2): Academic Integration
- [ ] Add `search_academic` and `search_pubmed` tools
- [ ] Update `_execute_multi_source_search` for multi-source routing
- [ ] Install Biopython dependency
- [ ] Test with academic queries

### Phase 3 (Week 3): SearXNG Deployment
- [ ] Deploy SearXNG via Docker
- [ ] Configure custom `settings.yml`
- [ ] Update `search_and_fetch` with SearXNG support
- [ ] Add fallback to DuckDuckGo
- [ ] Performance testing

### Phase 4 (Week 4): Polish & Optimization
- [ ] Implement parallel search execution
- [ ] Add Redis caching layer
- [ ] Progressive reporting (streaming)
- [ ] Research persistence & visualization
- [ ] Comprehensive testing & benchmarking

---

## Expected Outcomes

### Before (Current State)
```
User: "What are the latest developments in quantum computing?"
Bot: [Searches DuckDuckGo once, fetches 2 pages, responds in 30 seconds]
Output: 300-word summary from 2 web sources
```

### After (Deep Research Agent)
```
User: "What are the latest developments in quantum computing?"
Bot: [Initiates deep research]
  - Phase 1: Generates 4 sub-queries (hardware, algorithms, applications, challenges)
  - Phase 2: Searches web + Semantic Scholar (20 sources)
  - Phase 3: Evaluates quality (score: 0.68), continues to depth=2
  - Phase 4: Refines with 4 follow-up queries (error correction, commercial status)
  - Phase 5: Fetches 15 URLs, indexes to RAG
  - Phase 6: Synthesizes report with citations

Output: 2000-word comprehensive report with:
  - Executive summary
  - 4 thematic sections
  - 12 academic citations
  - 8 web sources
  - Generated in 4 minutes
```

---

## Conclusion

This plan transforms the PersonalAI chatbot into a **production-grade deep research agent** by implementing:

1. **Recursive reasoning architecture** (Plan-Execute-Evaluate-Synthesize)
2. **Multi-source integration** (Web, Semantic Scholar, PubMed)
3. **Unlimited search capacity** (SearXNG meta-search)
4. **Quality-driven iteration** (Early stopping via LLM evaluation)
5. **Citation-rich reporting** (Academic rigor)

The architecture follows patterns from leading research (LangGraph, ReAct, Tree-of-Thought) and production systems (RAG best practices, context management).

**Key Advantages:**
- **Free infrastructure** (SearXNG, Ollama, ChromaDB)
- **Modular design** (phases can be implemented incrementally)
- **Fallback mechanisms** (DuckDuckGo if SearXNG unavailable)
- **Explainable reasoning** (step-by-step progress tracking)

**Next Steps:**
1. Review this plan
2. Begin Phase 1 implementation
3. Iterate based on testing results
4. Deploy SearXNG in production environment
