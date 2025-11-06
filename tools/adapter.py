"""Adapter to convert existing JSON-based tool definitions to LangChain tools."""
from __future__ import annotations
from typing import List, Dict, Any, Callable

try:
    from langchain.tools import Tool
except Exception:
    Tool = None  # type: ignore

class LangChainToolAdapter:
    def __init__(self, executor):
        self.executor = executor  # ToolExecutor instance

    def build_tools(self, tool_definitions: List[Dict[str, Any]]) -> List[Any]:
        lc_tools: List[Any] = []
        for td in tool_definitions:
            fn = td.get("function", {})
            name = fn.get("name")
            description = fn.get("description", "")
            parameters = fn.get("parameters", {})

            # Build a callable that executes via ToolExecutor
            def _make_call(n: str) -> Callable[..., str]:
                def _call(**kwargs) -> str:
                    return self.executor.execute(n, kwargs)
                return _call

            call_fn = _make_call(name)

            # StructuredTool requires args schema; we'll pass pydantic-like validation via JSON schema using kwargs
            if Tool is not None:
                lc_tools.append(Tool(name=name, description=description, func=call_fn))
        return lc_tools
