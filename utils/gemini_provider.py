"""Gemini provider integration using LangChain ChatGoogleGenerativeAI.

Provides streaming responses and tool calling compatible with existing ChatBot loop.
"""
from __future__ import annotations
from typing import List, Dict, Any, Iterator

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import (
    SystemMessage,
    HumanMessage,
    AIMessage,
    ToolMessage,
    BaseMessage,
)

try:
    from google.generativeai.types import HarmCategory, HarmBlockThreshold  # type: ignore
except Exception:  # pragma: no cover - fallback if types not available
    HarmCategory = None  # type: ignore
    HarmBlockThreshold = None  # type: ignore

class GeminiProvider:
    """Wrapper for Gemini via LangChain providing a unified streaming interface."""

    def __init__(self, model: str, api_key: str, temperature: float, minimal_safety: bool = True):
        if not api_key:
            raise ValueError("GOOGLE_API_KEY missing. Set in .env or environment.")

        safety_settings = None
        if minimal_safety and HarmCategory and HarmBlockThreshold:
            # BLOCK_NONE to reduce hallucinated blocking while retaining explicit disallowed content filtering server-side
            safety_settings = {
                cat: HarmBlockThreshold.BLOCK_NONE  # type: ignore
                for cat in [
                    HarmCategory.HARM_CATEGORY_HATE_SPEECH,  # type: ignore
                    HarmCategory.HARM_CATEGORY_HARASSMENT,  # type: ignore
                    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,  # type: ignore
                    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,  # type: ignore
                ]
            }

        # Normalize model name: strip "models/" prefix if present
        normalized_model = model.replace("models/", "") if model.startswith("models/") else model
        
        # Some environments might not accept "-latest" variants; prefer a fallback list
        # Priority: requested model, gemini-2.5-pro, gemini-2.5-flash, gemini-2.0-flash-exp, gemini-1.5-pro
        preferred_models = [normalized_model, "gemini-2.5-pro", "gemini-2.5-flash", "gemini-2.0-flash-exp", "gemini-1.5-pro"]
        last_err = None
        self.llm = None
        for mid in preferred_models:
            try:
                self.llm = ChatGoogleGenerativeAI(
                    model=mid,
                    temperature=temperature,
                    api_key=api_key,
                    safety_settings=safety_settings,
                    streaming=True,
                )
                break
            except Exception as e:  # lazy fallback on model id errors
                last_err = e
                continue
        if self.llm is None:
            raise RuntimeError(f"Failed to initialize Gemini model. Last error: {last_err}")
        self._bound_tools = None

    def bind_tools(self, tools: List[Any]):  # tools are LangChain Tool objects
        base = self.llm
        if tools and hasattr(base, 'bind_tools'):
            try:
                self._bound_tools = base.bind_tools(tools)  # type: ignore[attr-defined]
            except Exception:
                # Fallback: ignore tools if binding fails
                self._bound_tools = base
        else:
            self._bound_tools = base

    @staticmethod
    def _convert_messages(messages: List[Dict[str, Any]]) -> List[BaseMessage]:
        lc_messages: List[BaseMessage] = []
        for m in messages:
            role = m.get("role")
            content = m.get("content", "")
            if role == "system":
                lc_messages.append(SystemMessage(content=content))
            elif role == "user":
                lc_messages.append(HumanMessage(content=content))
            elif role == "assistant":
                # We ignore tool_calls here; they will be handled by appended ToolMessages
                lc_messages.append(AIMessage(content=content))
            elif role == "tool":
                # tool result; use provided tool_call_id when available for correct linking
                tool_call_id = m.get("tool_call_id", "tool-result")
                lc_messages.append(ToolMessage(content=content, tool_call_id=tool_call_id))
        return lc_messages

    def stream(self, messages: List[Dict[str, Any]]) -> Iterator[Dict[str, Any]]:
        lc_messages = self._convert_messages(messages)
        llm = self._bound_tools if self._bound_tools else self.llm
        if not hasattr(llm, 'stream'):
            raise RuntimeError("Gemini LLM object does not support streaming in this environment.")
        for chunk in llm.stream(lc_messages):  # type: ignore[attr-defined]
            tool_calls = []
            if getattr(chunk, "tool_calls", None):
                for tc in chunk.tool_calls:
                    tool_calls.append({
                        "id": tc.get("id"),
                        "function": {
                            "name": tc.get("name"),
                            "arguments": tc.get("args", {})
                        }
                    })
            yield {
                "content": chunk.content,
                "tool_calls": tool_calls,
            }

    def supports_tools(self) -> bool:
        return True
