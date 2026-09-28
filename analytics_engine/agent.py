"""The analyst agent: Claude + a `run_python` tool backed by the session kernel."""

from __future__ import annotations

import base64
import json
import logging
import time
from pathlib import Path
from typing import Any

import anthropic

from .config import settings
from .session import Session

log = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are an expert data scientist working inside an analytics application. Users upload datasets and ask questions in a chat. You answer by writing and running Python in a persistent Jupyter kernel with the `run_python` tool, then explaining what you found.

## Environment
- Kernel state persists across tool calls and across the whole conversation, like a notebook. The working directory is the session workspace; uploaded files live in `data/`.
- Uploaded datasets are pre-loaded as pandas DataFrames: each under its own variable name (announced in a system note when uploaded), all of them in the dict `datasets`, and `df` points to the most recent upload.
- Available libraries: pandas, numpy, scipy, statsmodels, scikit-learn, matplotlib, seaborn. There is no internet access and you cannot install packages.
- A chart theme is preset: `PALETTE` (ordered categorical colors, already the default color cycle), `SEQUENTIAL` ("Blues") and `DIVERGING` ("RdBu_r") colormap names. Display figures with `plt.show()`. Each figure is shown to the user and returned to you as an image, so look at it and fix anything unreadable.
- Files you write into the working directory (cleaned data, predictions, saved models, reports) are offered to the user as downloads.

## How to work
- Look before concluding: check shapes, types, missing values and the distributions relevant to the question before modeling or making claims. Every number you report must come from code you ran; never invent or estimate figures you didn't compute.
- Keep each code cell to one logical step with a short, specific title. Print compact summaries rather than whole tables: `.head()`, `.value_counts().head(10)`, rounded numbers.
- If code fails, read the traceback, fix the cause and rerun; don't resubmit identical code.
- Visualization: choose the form that fits the question (line for change over time, sorted bars for comparing categories, histogram/box/violin for distributions, scatter for relationships, heatmap for correlation matrices). Give every chart a title stating the takeaway, axis labels with units, and readable tick labels. Never use dual y-axes; avoid pie charts beyond 3-4 slices; use small multiples (subplots) instead of overcrowding one plot.
- Exploratory analysis: data quality (missingness, duplicates, outliers, impossible values), key distributions, relationships with the target or key metrics, and segment comparisons. When a claim rests on a difference or association, back it with an appropriate statistical test and report the effect size, not just the p-value.
- Predictive modeling: identify the target and task type; prevent leakage (drop identifiers and columns only known after the outcome; fit preprocessing inside a scikit-learn Pipeline/ColumnTransformer); evaluate on a held-out test set or with cross-validation against a naive baseline; compare a simple interpretable model with a stronger one (e.g. gradient boosting or random forest) when worthwhile; report metrics that fit the task (classification: accuracy, precision/recall/F1, ROC AUC, confusion matrix, minding class imbalance; regression: MAE, RMSE, R²) and the most influential features (permutation importance or coefficients). Save predictions or the fitted model to a file when the user is likely to want them.
- Match effort to the question: a simple lookup gets a quick answer; an open-ended "analyze this" gets a structured investigation of several steps.

## Final answer
After the analysis, reply in Markdown for a business audience (plain language; briefly explain any technical term you need):
- Lead with the direct answer or the key findings, with specific numbers.
- Follow with supporting detail and the caveats that matter (data quality, sample size, correlation vs. causation, model limitations).
- Where useful, end with 2-3 concrete next steps or follow-up questions the user could ask.
- Charts and tables are already displayed above your answer: refer to them by what they show, and don't embed images or repeat the code.

If a question needs data and no dataset has been uploaded, ask the user to upload a file (CSV, Excel, JSON or Parquet)."""

RUN_PYTHON_TOOL = {
    "name": "run_python",
    "description": (
        "Execute Python code in the session's persistent Jupyter kernel. Returns stdout, the value of the "
        "last expression, errors with tracebacks, and any figures rendered with plt.show() (as images). "
        "Variables, imports and fitted models persist between calls."
    ),
    "eager_input_streaming": True,
    "input_schema": {
        "type": "object",
        "properties": {
            "title": {
                "type": "string",
                "description": "Short description of what this cell does, shown to the user (3-8 words), "
                               "e.g. 'Plot monthly revenue trend'.",
            },
            "code": {"type": "string", "description": "Python code to execute."},
        },
        "required": ["title", "code"],
    },
}

MAX_IMAGES_TO_MODEL = 4
UI_TEXT_LIMIT = 20000


class TurnAborted(Exception):
    """The turn cannot continue; the message is shown to the user."""


def _client() -> anthropic.AsyncAnthropic:
    return anthropic.AsyncAnthropic(max_retries=3)


def _snapshot_files(workdir: Path) -> dict[str, float]:
    files = {}
    for p in workdir.rglob("*"):
        rel = p.relative_to(workdir)
        if p.is_file() and rel.parts[0] != "figures" and not any(s.startswith(".") for s in rel.parts):
            files[str(rel)] = p.stat().st_mtime
    return files


def _valid_input(args: Any) -> bool:
    return (isinstance(args, dict) and isinstance(args.get("code"), str) and args["code"].strip() != ""
            and isinstance(args.get("title", ""), str))


async def _stream_response(session: Session, client: anthropic.AsyncAnthropic):
    """Stream one model response, forwarding deltas to the UI. Returns None if cancelled."""
    params: dict[str, Any] = dict(
        model=settings.model,
        max_tokens=64000,
        system=[{"type": "text", "text": SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}}],
        tools=[RUN_PYTHON_TOOL],
        messages=session.messages,
        thinking={"type": "adaptive", "display": "summarized"},
        output_config={"effort": settings.effort},
        cache_control={"type": "ephemeral"},
    )
    if settings.fallbacks:
        params.update(betas=["server-side-fallback-2026-07-01"], fallbacks="default")

    for attempt in range(3):
        try:
            async with client.beta.messages.stream(**params) as stream:
                tool_id, last_push = None, 0.0
                async for event in stream:
                    if session.cancel.is_set():
                        return None
                    if event.type == "thinking" and event.thinking:
                        session.emit({"type": "thinking_delta", "text": event.thinking})
                    elif event.type == "text" and event.text:
                        session.emit({"type": "text_delta", "text": event.text})
                    elif event.type == "content_block_start" and event.content_block.type == "tool_use":
                        tool_id = event.content_block.id
                        session.emit({"type": "code_delta", "id": tool_id, "title": "", "code": ""})
                    elif event.type == "input_json" and tool_id and time.monotonic() - last_push > 0.12:
                        snap = event.snapshot if isinstance(event.snapshot, dict) else {}
                        session.emit({"type": "code_delta", "id": tool_id,
                                      "title": str(snap.get("title", "")), "code": str(snap.get("code", ""))})
                        last_push = time.monotonic()
                return await stream.get_final_message()
        except ValueError:
            # Tool-input JSON the SDK could not parse at all (eager input streaming);
            # there is no tool_use_id to answer, so re-issue the request.
            log.warning("Unparseable tool input JSON; retrying (attempt %d)", attempt + 1)
            session.emit({"type": "notice", "text": "Retrying a malformed tool call..."})
    raise TurnAborted("The model repeatedly produced malformed tool input. Please try rephrasing the question.")


async def _run_tool(session: Session, block) -> dict[str, Any]:
    args = block.input
    if not _valid_input(args):
        session.emit({"type": "result", "id": block.id, "error": "Invalid tool input received from the model; retrying."})
        return {"type": "tool_result", "tool_use_id": block.id, "is_error": True,
                "content": json.dumps({"INVALID_JSON": json.dumps(args, default=str)[:4000]})}

    title = args.get("title") or "Run Python"
    session.emit({"type": "code", "id": block.id, "title": title, "code": args["code"]})

    before = _snapshot_files(session.workdir)
    res = await session.kernel.execute(args["code"], timeout=settings.exec_timeout)
    after = _snapshot_files(session.workdir)

    image_urls = []
    for b64 in res.images:
        path = session.next_figure_path()
        path.write_bytes(base64.b64decode(b64))
        image_urls.append(f"/api/sessions/{session.id}/files/figures/{path.name}")
    files = [
        {"name": rel, "url": f"/api/sessions/{session.id}/files/{rel}",
         "size": (session.workdir / rel).stat().st_size}
        for rel, mtime in sorted(after.items()) if before.get(rel) != mtime
    ]
    session.emit({
        "type": "result", "id": block.id,
        "stdout": res.stdout[-UI_TEXT_LIMIT:], "stderr": res.stderr[-5000:], "error": res.error,
        "images": image_urls, "html": [h[:300000] for h in res.html],
        "text": [] if res.html else [t[:UI_TEXT_LIMIT] for t in res.text_results],
        "timed_out": res.timed_out, "interrupted": res.interrupted,
        "duration": res.duration, "files": files,
    })

    text = res.to_model_text()
    if files:
        text += "\n\n[files written/updated: " + ", ".join(f["name"] for f in files) + "]"
    if len(res.images) > MAX_IMAGES_TO_MODEL:
        text += f"\n\n[showing the first {MAX_IMAGES_TO_MODEL} of {len(res.images)} figures]"
    content: list[dict[str, Any]] = [{"type": "text", "text": text}]
    for b64 in res.images[:MAX_IMAGES_TO_MODEL]:
        content.append({"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": b64}})
    return {"type": "tool_result", "tool_use_id": block.id, "content": content, "is_error": bool(res.error)}


async def run_turn(session: Session, user_text: str) -> None:
    """Handle one user message end to end, emitting UI events along the way."""
    session.cancel.clear()
    session.emit({"type": "user", "text": user_text})
    session.emit({"type": "turn_start"})

    content = [{"type": "text", "text": note} for note in session.pending_notes]
    content.append({"type": "text", "text": user_text})
    session.pending_notes.clear()
    session.messages.append({"role": "user", "content": content})

    status = "ok"
    client = _client()
    try:
        for _ in range(settings.max_steps):
            if session.cancel.is_set():
                status = "stopped"
                break
            response = await _stream_response(session, client)
            if response is None:
                status = "stopped"
                break
            if response.stop_reason == "refusal":
                raise TurnAborted("The model declined to continue with this request.")
            tool_uses = [b for b in response.content if b.type == "tool_use"]
            if response.stop_reason == "max_tokens" and tool_uses:
                raise TurnAborted("The response hit the output limit in the middle of a code cell. "
                                  "Try asking for a smaller step.")

            session.messages.append({"role": "assistant", "content": response.content})
            if not tool_uses:
                break

            # Every tool_use must get a tool_result, even if we stop or fail partway,
            # or the conversation history becomes invalid for the next request.
            results: list[dict[str, Any]] = []
            try:
                for block in tool_uses:
                    if session.cancel.is_set():
                        results.append({"type": "tool_result", "tool_use_id": block.id, "is_error": True,
                                        "content": "Skipped: the user stopped the analysis."})
                    else:
                        results.append(await _run_tool(session, block))
            finally:
                done = {r["tool_use_id"] for r in results}
                results += [{"type": "tool_result", "tool_use_id": b.id, "is_error": True,
                             "content": "Execution failed unexpectedly."} for b in tool_uses if b.id not in done]
                session.messages.append({"role": "user", "content": results})
        else:
            session.emit({"type": "notice", "text": f"Stopped after {settings.max_steps} analysis steps. "
                                                    "Ask me to continue if you'd like me to keep going."})
    except TurnAborted as e:
        status = "error"
        session.emit({"type": "error", "message": str(e)})
    except anthropic.AuthenticationError:
        status = "error"
        session.emit({"type": "error", "message": "Claude API authentication failed. Set a valid "
                                                  "ANTHROPIC_API_KEY in the .env file and restart the server."})
    except anthropic.PermissionDeniedError as e:
        status = "error"
        session.emit({"type": "error", "message": f"The API key lacks permission for this request: {e.message}"})
    except anthropic.NotFoundError:
        status = "error"
        session.emit({"type": "error", "message": f"Model '{settings.model}' was not found. Check ANALYST_MODEL."})
    except anthropic.RateLimitError:
        status = "error"
        session.emit({"type": "error", "message": "Rate limited by the Claude API. Wait a moment and try again."})
    except anthropic.BadRequestError as e:
        status = "error"
        session.emit({"type": "error", "message": f"The Claude API rejected the request: {e.message}"})
    except anthropic.APIStatusError as e:
        status = "error"
        session.emit({"type": "error", "message": f"Claude API error ({e.status_code}). Please try again."})
    except anthropic.APIConnectionError:
        status = "error"
        session.emit({"type": "error", "message": "Could not reach the Claude API. Check your network connection."})
    except TypeError as e:
        status = "error"
        if "authentication method" in str(e):
            session.emit({"type": "error", "message": "No Claude API key configured. Add ANTHROPIC_API_KEY=... "
                                                      "to the .env file in the project folder and restart the server."})
        else:
            log.exception("Turn failed")
            session.emit({"type": "error", "message": f"Unexpected error: {e}"})
    except Exception as e:  # noqa: BLE001 - surface anything unexpected to the user, keep the server alive
        log.exception("Turn failed")
        status = "error"
        session.emit({"type": "error", "message": f"Unexpected error: {e}"})
    finally:
        await client.close()
        session.emit({"type": "turn_end", "status": status})
