"""Maintain a manifest of chat sessions for sidebar navigation."""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any


@dataclass
class SessionEntry:
    id: str
    file: str
    started_at: str
    title: str
    tokens_in: int = 0
    tokens_out: int = 0


class SessionIndex:
    """Simple JSON-backed index of chat sessions."""

    def __init__(self, index_path: str | Path = "chat_logs/index.json") -> None:
        self.index_path = Path(index_path)
        self.index_path.parent.mkdir(parents=True, exist_ok=True)

    def _load(self) -> Dict[str, Any]:
        if not self.index_path.exists():
            return {"version": 1, "sessions": []}
        try:
            return json.loads(self.index_path.read_text(encoding="utf-8"))
        except Exception:
            # Corrupted index; start fresh but keep a backup
            try:
                backup = self.index_path.with_suffix(".bak.json")
                self.index_path.rename(backup)
            except Exception:
                pass
            return {"version": 1, "sessions": []}

    def _atomic_write(self, data: Dict[str, Any]) -> None:
        tmp = self.index_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(self.index_path)

    def add_session(
        self,
        session_id: str,
        file_path: str,
        title: str,
        tokens_in: int = 0,
        tokens_out: int = 0,
        started_at: datetime | None = None,
    ) -> None:
        data = self._load()
        entry = SessionEntry(
            id=session_id,
            file=file_path,
            started_at=(started_at or datetime.now()).isoformat(timespec="seconds"),
            title=title,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
        )
        # Prepend newest session
        sessions: List[Dict[str, Any]] = data.get("sessions", [])
        sessions.insert(0, asdict(entry))
        data["sessions"] = sessions[:500]  # cap to avoid unbounded growth
        self._atomic_write(data)
