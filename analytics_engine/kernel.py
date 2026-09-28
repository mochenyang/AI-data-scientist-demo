"""Persistent per-session Python kernels (IPython via jupyter_client).

Each chat session gets its own kernel whose working directory is the session's
workspace, so variables, fitted models and loaded DataFrames persist across
questions exactly like a notebook.
"""

from __future__ import annotations

import asyncio
import json
import re
import shutil
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

from jupyter_client.manager import AsyncKernelManager

ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")

# Runs once when a kernel starts. Sets a consistent, readable chart theme
# (validated categorical palette, thin recessive axes) and defines helpers the
# agent and the upload pipeline rely on.
KERNEL_BOOTSTRAP = r"""
import warnings, json as _json, os as _os
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from cycler import cycler
%matplotlib inline
%config InlineBackend.figure_formats = ['png']
%config InlineBackend.rc = {}

PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
SEQUENTIAL = "Blues"
DIVERGING = "RdBu_r"
sns.set_theme(style="whitegrid", palette=PALETTE)
plt.rcParams.update({
    "figure.figsize": (9, 5), "figure.dpi": 110, "savefig.dpi": 110,
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "savefig.facecolor": "#fcfcfb",
    "axes.prop_cycle": cycler(color=PALETTE),
    "axes.edgecolor": "#c9c8c2", "axes.linewidth": 0.8, "axes.titleweight": "bold",
    "axes.titlesize": 13, "axes.titlelocation": "left", "axes.labelsize": 10.5,
    "axes.labelcolor": "#52514e", "axes.spines.top": False, "axes.spines.right": False,
    "grid.color": "#e6e5e0", "grid.linewidth": 0.7,
    "xtick.color": "#52514e", "ytick.color": "#52514e", "text.color": "#0b0b0b",
    "lines.linewidth": 2, "lines.markersize": 6, "legend.frameon": False,
    "font.family": "DejaVu Sans",
})
pd.set_option("display.max_columns", 50)
pd.set_option("display.width", 160)
pd.set_option("display.max_colwidth", 60)
pd.set_option("display.max_rows", 60)

datasets = {}

def _load_dataset(path):
    ext = _os.path.splitext(path)[1].lower()
    if ext in (".csv", ".txt"):
        return pd.read_csv(path, sep=None, engine="python")
    if ext == ".tsv":
        return pd.read_csv(path, sep="\t")
    if ext in (".xlsx", ".xls", ".xlsm"):
        return pd.read_excel(path)
    if ext == ".json":
        try:
            return pd.read_json(path)
        except ValueError:
            return pd.read_json(path, lines=True)
    if ext == ".jsonl":
        return pd.read_json(path, lines=True)
    if ext == ".parquet":
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported file type: {ext}")

def _profile(df, max_cols=60):
    cols = []
    for c in list(df.columns)[:max_cols]:
        s = df[c]
        info = {"name": str(c), "dtype": str(s.dtype), "missing": int(s.isna().sum()),
                "unique": int(s.nunique(dropna=True))}
        if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
            d = s.describe()
            info.update({k: (None if pd.isna(d.get(k)) else float(d.get(k)))
                         for k in ("mean", "std", "min", "50%", "max")})
        else:
            top = s.astype(str).value_counts().head(5)
            info["top"] = {str(k): int(v) for k, v in top.items()}
        cols.append(info)
    return {"rows": int(df.shape[0]), "cols": int(df.shape[1]), "columns": cols,
            "truncated_columns": max(0, df.shape[1] - max_cols),
            "memory_mb": round(float(df.memory_usage(deep=True).sum()) / 1e6, 2),
            "duplicate_rows": int(df.duplicated().sum()),
            "head_html": df.head(8).to_html(classes="df", border=0, max_cols=20),
            "head_text": df.head(5).to_string(max_cols=20, max_colwidth=30)}

def _register_dataset(var, path):
    df = _load_dataset(path)
    df.columns = [str(c).strip() for c in df.columns]
    globals()[var] = df
    datasets[var] = df
    globals()["df"] = df
    print(_json.dumps(_profile(df)))
"""


@dataclass
class ExecutionResult:
    stdout: str = ""
    stderr: str = ""
    error: str | None = None  # formatted traceback
    images: list[str] = field(default_factory=list)  # base64 PNGs
    html: list[str] = field(default_factory=list)  # rich HTML outputs (tables)
    text_results: list[str] = field(default_factory=list)  # text/plain reprs
    timed_out: bool = False
    interrupted: bool = False
    duration: float = 0.0

    def to_model_text(self, limit: int = 12000) -> str:
        """Plain-text summary of the execution, for the model's tool_result."""
        parts: list[str] = []
        if self.stdout:
            parts.append(f"[stdout]\n{self.stdout}")
        if self.text_results:
            parts.append("[result]\n" + "\n".join(self.text_results))
        if self.stderr:
            parts.append(f"[stderr]\n{self.stderr}")
        if self.images:
            parts.append(f"[{len(self.images)} figure(s) rendered and attached below]")
        if self.error:
            parts.append(f"[error]\n{self.error}")
        if self.timed_out:
            parts.append("[execution timed out and was interrupted]")
        if self.interrupted:
            parts.append("[execution was interrupted by the user]")
        text = "\n\n".join(parts) or "[no output]"
        if len(text) > limit:
            half = limit // 2
            text = text[:half] + f"\n\n... [{len(text) - limit} characters truncated] ...\n\n" + text[-half:]
        return text


class SessionKernel:
    def __init__(self, workdir: Path):
        self.workdir = workdir
        self.km: AsyncKernelManager | None = None
        self.kc = None
        self._lock = asyncio.Lock()
        self._start_task: asyncio.Task | None = None
        self._runtime_dir: str | None = None

    def start(self) -> None:
        """Begin starting the kernel in the background; execute() waits for it."""
        if self._start_task is None:
            self._start_task = asyncio.create_task(self._start())

    async def ready(self) -> None:
        self.start()
        await asyncio.shield(self._start_task)

    async def _start(self) -> None:
        # IPC sockets (local files) rather than TCP ports: nothing listens on the network.
        # Socket paths are limited to ~107 chars, so they live in a short temp dir.
        self._runtime_dir = tempfile.mkdtemp(prefix="aek-")
        self.km = AsyncKernelManager(kernel_name="python3", transport="ipc",
                                     connection_file=f"{self._runtime_dir}/kernel.json")
        # Launch the kernel with the same interpreter as the server (the uv venv),
        # regardless of which kernelspecs happen to be installed system-wide.
        self.km.kernel_spec.argv = [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"]
        await self.km.start_kernel(cwd=str(self.workdir))
        self.kc = self.km.client()
        self.kc.start_channels()
        await self.kc.wait_for_ready(timeout=60)
        async with self._lock:
            res = await self._execute(KERNEL_BOOTSTRAP, timeout=120)
        if res.error:
            raise RuntimeError(f"Kernel bootstrap failed:\n{res.error}")

    async def execute(self, code: str, timeout: float = 300.0) -> ExecutionResult:
        await self.ready()
        async with self._lock:
            return await self._execute(code, timeout)

    async def _execute(self, code: str, timeout: float) -> ExecutionResult:
        assert self.kc is not None
        result = ExecutionResult()
        started = time.monotonic()
        msg_id = self.kc.execute(code, store_history=True, allow_stdin=False)
        deadline = started + timeout
        interrupted_for_timeout = False
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0 and not interrupted_for_timeout:
                await self.interrupt()
                result.timed_out = True
                interrupted_for_timeout = True
                deadline = time.monotonic() + 15
                continue
            try:
                msg = await self.kc.get_iopub_msg(timeout=max(0.1, min(remaining, 1.0)))
            except asyncio.CancelledError:
                raise
            except Exception:  # queue.Empty on timeout
                if interrupted_for_timeout and time.monotonic() > deadline:
                    break
                continue
            if msg.get("parent_header", {}).get("msg_id") != msg_id:
                continue
            mtype, content = msg["msg_type"], msg["content"]
            if mtype == "stream":
                if content["name"] == "stdout":
                    result.stdout += content["text"]
                else:
                    result.stderr += content["text"]
            elif mtype in ("execute_result", "display_data"):
                data = content.get("data", {})
                if "image/png" in data:
                    result.images.append(data["image/png"].replace("\n", ""))
                elif "text/html" in data:
                    result.html.append(data["text/html"])
                    if "text/plain" in data:
                        result.text_results.append(data["text/plain"])
                elif "text/plain" in data:
                    result.text_results.append(data["text/plain"])
            elif mtype == "error":
                tb = "\n".join(content.get("traceback", [])) or f"{content.get('ename')}: {content.get('evalue')}"
                tb = ANSI_RE.sub("", tb)
                if content.get("ename") == "KeyboardInterrupt":
                    result.interrupted = not result.timed_out
                result.error = tb[-6000:]
            elif mtype == "status" and content.get("execution_state") == "idle":
                break
        result.stdout = ANSI_RE.sub("", result.stdout)
        result.stderr = ANSI_RE.sub("", result.stderr)
        result.duration = round(time.monotonic() - started, 2)
        return result

    async def run_json(self, code: str, timeout: float = 120.0) -> dict:
        """Run code whose last stdout line is JSON; return it parsed."""
        res = await self.execute(code, timeout=timeout)
        if res.error:
            raise RuntimeError(res.error.splitlines()[-1] if res.error else "error")
        return json.loads(res.stdout.strip().splitlines()[-1])

    async def interrupt(self) -> None:
        if self.km is not None:
            await self.km.interrupt_kernel()

    async def shutdown(self) -> None:
        if self._start_task is not None and not self._start_task.done():
            self._start_task.cancel()
            try:
                await self._start_task
            except (asyncio.CancelledError, Exception):
                pass
        if self.kc is not None:
            self.kc.stop_channels()
        if self.km is not None:
            await self.km.shutdown_kernel(now=True)
        if self._runtime_dir:
            shutil.rmtree(self._runtime_dir, ignore_errors=True)
