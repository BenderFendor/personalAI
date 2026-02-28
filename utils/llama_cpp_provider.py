"""Llama.cpp server provider using OpenAI-compatible API.

Provides streaming chat completions and embeddings via HTTP requests to the llama.cpp server.
"""

from __future__ import annotations
import json
import requests
from typing import Dict, Any, Iterator, List, Optional


class LlamaCppProvider:
    """Wrapper for llama.cpp server via OpenAI-compatible API."""

    def __init__(
        self,
        model: str,
        base_url: str = "http://localhost:8080",
        temperature: float = 0.7,
        api_key: Optional[str] = None,
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature
        self.api_key = api_key or "not-needed"
        self._headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

    def _build_url(self, endpoint: str) -> str:
        return f"{self.base_url}/v1/{endpoint.lstrip('/')}"

    def chat(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        temperature: Optional[float] = None,
        stream: bool = False,
    ) -> Dict[str, Any] | Iterator[Dict[str, Any]]:
        """Send chat completion request to llama.cpp server.

        Args:
            messages: List of message dicts with role and content
            tools: Optional list of tool definitions
            temperature: Optional temperature override
            stream: Whether to stream the response

        Returns:
            Response dict (if not streaming) or iterator of chunks
        """
        payload: Dict[str, Any] = {
            "messages": messages,
            "stream": stream,
        }

        if tools:
            payload["tools"] = tools

        temp = temperature if temperature is not None else self.temperature
        if temp is not None:
            payload["temperature"] = temp

        url = self._build_url("/chat/completions")

        if stream:
            return self._stream_chat(url, payload)
        else:
            response = requests.post(
                url, json=payload, headers=self._headers, timeout=300
            )
            response.raise_for_status()
            return response.json()

    def _stream_chat(
        self, url: str, payload: Dict[str, Any]
    ) -> Iterator[Dict[str, Any]]:
        """Handle streaming chat responses from llama.cpp server.

        Llama.cpp uses SSE (Server-Sent Events) format:
        data: {"choices": [...]}

        Yields:
            Parsed response chunks
        """
        response = requests.post(
            url, json=payload, headers=self._headers, stream=True, timeout=300
        )
        response.raise_for_status()

        buffer = ""
        for chunk in response.iter_content(chunk_size=None, decode_unicode=True):
            if chunk:
                buffer += chunk

                while "\n" in buffer:
                    line, buffer = buffer.split("\n", 1)
                    line = line.strip()

                    if not line:
                        continue
                    if line.startswith("data: "):
                        data = line[6:]  # Remove "data: " prefix
                        if data == "[DONE]":
                            return
                        try:
                            yield json.loads(data)
                        except json.JSONDecodeError:
                            continue

    def embeddings(self, prompt: str) -> List[float]:
        """Generate embeddings for a prompt.

        Args:
            prompt: Text to embed

        Returns:
            List of embedding floats
        """
        payload = {
            "model": self.model,
            "input": prompt,
        }

        url = self._build_url("/embeddings")
        response = requests.post(url, json=payload, headers=self._headers, timeout=60)
        response.raise_for_status()
        result = response.json()

        # OpenAI-compatible response format: {"data": [{"embedding": [...]}]}
        if isinstance(result, dict) and "data" in result:
            return result["data"][0].get("embedding", [])
        raise RuntimeError(f"Unexpected embeddings response format: {result}")

    def get_model_info(self) -> Dict[str, Any]:
        """Get model information from llama.cpp server.

        Returns:
            Dict with model details including context length
        """
        url = self._build_url(f"/models/{self.model}")
        response = requests.get(url, headers=self._headers, timeout=30)

        if response.status_code == 404:
            # Try listing all models
            url = self._build_url("/models")
            response = requests.get(url, headers=self._headers, timeout=30)
            response.raise_for_status()
            result = response.json()
            # Find our model in the list
            if isinstance(result, dict) and "data" in result:
                for model in result["data"]:
                    if model.get("id") == self.model:
                        return model
            return {"id": self.model, "context_length": 8192}  # fallback

        response.raise_for_status()
        return response.json()

    def list_models(self) -> List[str]:
        """List available models on the server.

        Returns:
            List of model IDs
        """
        url = self._build_url("/models")
        response = requests.get(url, headers=self._headers, timeout=30)
        response.raise_for_status()
        result = response.json()

        models = []
        if isinstance(result, dict) and "data" in result:
            for model in result["data"]:
                models.append(model.get("id", ""))
        return [m for m in models if m]

    def health_check(self) -> bool:
        """Check if the llama.cpp server is running.

        Returns:
            True if server is healthy
        """
        try:
            url = self._build_url("/health")
            response = requests.get(url, timeout=5)
            return response.status_code == 200
        except Exception:
            return False
