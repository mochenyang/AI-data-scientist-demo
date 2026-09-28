"""Chat sessions: workspace, kernel, Claude conversation history and UI event log."""

from __future__ import annotations

import asyncio
import itertools
import keyword
import re
import shutil
import uuid
from pathlib import Path
from typing import Any

from .kernel import SessionKernel

SUPPORTED_EXTENSIONS = {".csv", ".tsv", ".txt", ".xlsx", ".xls", ".xlsm", ".json", ".jsonl", ".parquet"}
# Consecutive deltas of these types are merged in the stored log, so replaying
# a session (page reload, reconnect) sends a compact history.
MERGEABLE = {"text_delta", "thinking_delta"}
# Live-only events: useful while streaming, redundant once the final event lands.
EPHEMERAL = {"code_delta"}


def safe_var_name(stem: str, taken: set[str]) -> str:
    name = re.sub(r"\W+", "_", stem).strip("_").lower() or "data"
    if name[0].isdigit():
        name = f"df_{name}"
    if keyword.iskeyword(name) or name in {"df", "datasets", "pd", "np", "plt", "sns"}:
        name = f"{name}_df"
    base, n = name, 2
    while name in taken:
        name, n = f"{base}_{n}", n + 1
    return name


class Session:
    def __init__(self, sid: str, root: Path):
        self.id = sid
        self.workdir = root / sid
        (self.workdir / "data").mkdir(parents=True, exist_ok=True)
        (self.workdir / "figures").mkdir(exist_ok=True)
        self.kernel = SessionKernel(self.workdir)
        self.messages: list[dict[str, Any]] = []  # Claude conversation history (append-only)
        self.pending_notes: list[str] = []  # dataset context to prepend to the next user turn
        self.datasets: list[dict[str, Any]] = []
        self.log: list[dict[str, Any]] = []
        self.subscribers: set[asyncio.Queue] = set()
        self.task: asyncio.Task | None = None
        self.cancel = asyncio.Event()
        self._seq = itertools.count(1)
        self._fig = itertools.count(1)

    @property
    def busy(self) -> bool:
        return self.task is not None and not self.task.done()

    def next_figure_path(self) -> Path:
        return self.workdir / "figures" / f"figure_{next(self._fig)}.png"

    # --- event bus -------------------------------------------------------
    def emit(self, event: dict[str, Any]) -> None:
        event = {**event, "seq": next(self._seq)}
        for q in self.subscribers:
            q.put_nowait(event)
        if event["type"] in EPHEMERAL:
            return
        last = self.log[-1] if self.log else None
        if last and event["type"] in MERGEABLE and last["type"] == event["type"]:
            last["text"] += event["text"]
            last["seq"] = event["seq"]
        else:
            self.log.append(dict(event))

    def subscribe(self) -> tuple[asyncio.Queue, list[dict[str, Any]]]:
        """Return a live queue plus a snapshot of the log, atomically."""
        q: asyncio.Queue = asyncio.Queue()
        self.subscribers.add(q)
        return q, [dict(e) for e in self.log]

    def unsubscribe(self, q: asyncio.Queue) -> None:
        self.subscribers.discard(q)

    def state(self) -> dict[str, Any]:
        return {"id": self.id, "busy": self.busy, "datasets": self.datasets}

    # --- datasets --------------------------------------------------------
    async def add_dataset(self, filename: str, path: Path) -> dict[str, Any]:
        var = safe_var_name(Path(filename).stem, {d["var"] for d in self.datasets})
        profile = await self.kernel.run_json(f"_register_dataset({var!r}, {str(path)!r})")
        info = {"name": filename, "var": var, "path": f"data/{path.name}",
                "rows": profile["rows"], "cols": profile["cols"]}
        self.datasets.append(info)
        self.pending_notes.append(dataset_note(info, profile))
        self.emit({"type": "dataset", **info, "profile": profile})
        return info


def dataset_note(info: dict[str, Any], profile: dict[str, Any]) -> str:
    lines = [
        f"[System note: the user uploaded `{info['name']}` (saved at `{info['path']}`). "
        f"It is loaded as DataFrame `{info['var']}` (also `df`, which always points to the most recent upload, "
        f"and `datasets[{info['var']!r}]`). Shape: {profile['rows']:,} rows x {profile['cols']} columns; "
        f"{profile['duplicate_rows']:,} duplicate rows.",
        "Columns (name | dtype | missing | unique | summary):",
    ]
    for c in profile["columns"]:
        if "mean" in c:
            summary = f"min={c['min']:.4g}, median={c['50%']:.4g}, mean={c['mean']:.4g}, max={c['max']:.4g}" \
                if c.get("mean") is not None else "all missing"
        else:
            summary = "top: " + ", ".join(f"{k[:25]} ({v})" for k, v in list(c.get("top", {}).items())[:3])
        lines.append(f"- {c['name']} | {c['dtype']} | {c['missing']} | {c['unique']} | {summary}")
    if profile["truncated_columns"]:
        lines.append(f"- ... and {profile['truncated_columns']} more columns")
    lines.append(f"First rows:\n{profile['head_text']}]")
    return "\n".join(lines)


class SessionManager:
    def __init__(self, root: Path):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.sessions: dict[str, Session] = {}

    async def create(self) -> Session:
        sid = uuid.uuid4().hex[:12]
        session = Session(sid, self.root)
        session.kernel.start()
        self.sessions[sid] = session
        return session

    def get(self, sid: str) -> Session | None:
        return self.sessions.get(sid)

    async def delete(self, sid: str) -> None:
        session = self.sessions.pop(sid, None)
        if session is None:
            return
        if session.busy:
            session.cancel.set()
            session.task.cancel()
        await session.kernel.shutdown()
        shutil.rmtree(session.workdir, ignore_errors=True)

    async def shutdown(self) -> None:
        """Stop all kernels; workspaces (uploads, figures, outputs) stay on disk."""
        for session in self.sessions.values():
            if session.busy:
                session.task.cancel()
            await session.kernel.shutdown()
        self.sessions.clear()
