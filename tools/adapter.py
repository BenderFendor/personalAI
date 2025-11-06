"""Adapter to convert existing JSON-based tool definitions to LangChain tools.

This prefers structured tools (with argument schemas) so Gemini can reliably
invoke functions with named parameters, falling back to simple Tool objects
when StructuredTool or Pydantic aren't available in the environment.
"""
from __future__ import annotations
from typing import List, Dict, Any, Callable

try:
    from langchain.tools import Tool, StructuredTool  # type: ignore
except Exception:
    Tool = None  # type: ignore
    StructuredTool = None  # type: ignore

try:
    # Pydantic is used to create args schemas dynamically for StructuredTool
    from pydantic import BaseModel, create_model  # type: ignore
except Exception:
    BaseModel = None  # type: ignore
    create_model = None  # type: ignore

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

            # Prefer StructuredTool with an args schema so models know parameter names/types
            args_schema = None
            if StructuredTool is not None and BaseModel is not None and create_model is not None and isinstance(parameters, dict):
                try:
                    props: Dict[str, Any] = parameters.get("properties", {}) or {}
                    required = set(parameters.get("required", []) or [])

                    field_defs: Dict[str, tuple] = {}
                    # Map JSON schema types to Python types
                    type_map = {
                        "string": str,
                        "integer": int,
                        "number": float,
                        "boolean": bool,
                        "object": dict,
                        "array": list,
                    }

                    for pname, pschema in props.items():
                        jtype = (pschema or {}).get("type", "string")
                        py_type = type_map.get(jtype, str)
                        default = pschema.get("default") if isinstance(pschema, dict) else None
                        if pname in required and default is None:
                            field_defs[pname] = (py_type, ...)
                        else:
                            # Optional field; if no default provided, default to None
                            field_defs[pname] = (py_type, default if default is not None else None)

                    # Dynamically create a Pydantic model for tool args
                    args_schema = create_model(f"{name.title().replace('_','')}Args", **field_defs)  # type: ignore[arg-type]
                except Exception:
                    args_schema = None

            try:
                if StructuredTool is not None and args_schema is not None:
                    lc_tools.append(StructuredTool(name=name, description=description, func=call_fn, args_schema=args_schema))
                elif Tool is not None:
                    # Fallback: unstructured tool; model may pass a single 'tool_input'
                    lc_tools.append(Tool(name=name, description=description, func=call_fn))
            except Exception:
                # As a last resort, skip tool to avoid crashing the run
                continue
        return lc_tools
