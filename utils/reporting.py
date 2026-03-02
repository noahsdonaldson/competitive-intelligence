"""
Shared LangGraph-based token & cost reporting utilities.
Used by both the DeepAgents and Prompt Chaining notebooks to produce
a side-by-side comparable report.
"""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult

# ---------------------------------------------------------------------------
# Cost table (USD per 1K tokens) — update as pricing changes
# ---------------------------------------------------------------------------
COST_PER_1K_TOKENS: Dict[str, Dict[str, float]] = {
    "gpt-5-mini": {"input": 0.00040, "output": 0.00160},  # verify at platform.openai.com/docs/pricing
    "gpt-4o": {"input": 0.0025, "output": 0.010},
    "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
    "gpt-4-turbo": {"input": 0.01, "output": 0.03},
    "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015},
    "claude-3-5-sonnet": {"input": 0.003, "output": 0.015},
    "claude-3-haiku": {"input": 0.00025, "output": 0.00125},
}

DEFAULT_MODEL = "gpt-5-mini"


# ---------------------------------------------------------------------------
# Callback handler — plugs into any LangChain LLM call
# ---------------------------------------------------------------------------
class TokenCostTracker(BaseCallbackHandler):
    """
    Tracks token usage and estimated cost across all LLM calls.
    Attach to any chain, agent, or LLM via the `callbacks` parameter.
    """

    def __init__(self, model_name: str = DEFAULT_MODEL):
        super().__init__()
        self.model_name = model_name
        self.prompt_tokens: int = 0
        self.completion_tokens: int = 0
        self.total_tokens: int = 0
        self.llm_calls: int = 0
        self.tool_calls: int = 0
        self.tavily_searches: int = 0
        self.steps: List[Dict[str, Any]] = []
        self._start_time: float = time.time()
        self.elapsed_seconds: float = 0.0
        self._call_start_times: Dict[UUID, float] = {}  # run_id → wall-clock start

    # -- LLM events ----------------------------------------------------------

    def on_llm_start(
        self,
        serialized: Dict[str, Any],
        prompts: List[str],
        *,
        run_id: UUID,
        **kwargs,
    ):
        self.llm_calls += 1
        self._call_start_times[run_id] = time.time()

    def on_llm_end(self, response: LLMResult, *, run_id: UUID, **kwargs):
        elapsed = round(time.time() - self._call_start_times.pop(run_id, time.time()), 2)
        usage = {}
        if response.llm_output:
            usage = response.llm_output.get("token_usage", {})
        pt = usage.get("prompt_tokens", 0)
        ct = usage.get("completion_tokens", 0)
        self.prompt_tokens += pt
        self.completion_tokens += ct
        self.total_tokens += pt + ct
        self.steps.append(
            {
                "type": "llm",
                "prompt_tokens": pt,
                "completion_tokens": ct,
                "total": pt + ct,
                "elapsed_s": elapsed,
            }
        )

    # -- Chat model events (mirrors llm_end for chat models) ----------------

    def on_chat_model_start(
        self,
        serialized: Dict[str, Any],
        messages: Any,
        *,
        run_id: UUID,
        **kwargs,
    ):
        self.llm_calls += 1
        self._call_start_times[run_id] = time.time()

    # -- Tool events ---------------------------------------------------------

    def on_tool_start(
        self,
        serialized: Dict[str, Any],
        input_str: str,
        *,
        run_id: UUID,
        **kwargs,
    ):
        self._call_start_times[run_id] = time.time()
        self.tool_calls += 1
        tool_name = serialized.get("name", "unknown")
        if "tavily" in tool_name.lower() or "search" in tool_name.lower():
            self.tavily_searches += 1
        self.steps.append({"type": "tool", "name": tool_name, "input": input_str[:120], "elapsed_s": None})

    def on_tool_end(self, output: Any, *, run_id: UUID, **kwargs):
        elapsed = round(time.time() - self._call_start_times.pop(run_id, time.time()), 2)
        # Patch elapsed into the most recent matching tool step
        for step in reversed(self.steps):
            if step.get("type") == "tool" and step.get("elapsed_s") is None:
                step["elapsed_s"] = elapsed
                break

    # -- Helpers -------------------------------------------------------------

    def stop(self):
        """Call when execution is complete to capture wall-clock time."""
        self.elapsed_seconds = time.time() - self._start_time

    def estimated_cost_usd(self) -> float:
        pricing = COST_PER_1K_TOKENS.get(
            self.model_name, COST_PER_1K_TOKENS[DEFAULT_MODEL]
        )
        input_cost = (self.prompt_tokens / 1000) * pricing["input"]
        output_cost = (self.completion_tokens / 1000) * pricing["output"]
        return round(input_cost + output_cost, 5)

    def summary(self) -> Dict[str, Any]:
        return {
            "model": self.model_name,
            "llm_calls": self.llm_calls,
            "tool_calls": self.tool_calls,
            "tavily_searches": self.tavily_searches,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "estimated_cost_usd": self.estimated_cost_usd(),
            "elapsed_seconds": round(self.elapsed_seconds, 2),
        }


# ---------------------------------------------------------------------------
# LangGraph reporter — a simple compiled graph that formats & displays results
# ---------------------------------------------------------------------------
from typing import TypedDict
from langgraph.graph import StateGraph, END


class ReportState(TypedDict):
    tracker: Any          # TokenCostTracker instance
    approach: str         # "DeepAgents" | "Prompt Chaining"
    result_text: str      # The final intelligence brief
    report: Optional[str] # Rendered report string


def _format_node(state: ReportState) -> ReportState:
    """Node: build a structured text report from tracker data."""
    t = state["tracker"]
    s = t.summary()
    sep = "─" * 60
    lines = [
        "",
        sep,
        f"  COMPETITIVE INTELLIGENCE REPORT — {state['approach'].upper()}",
        sep,
        f"  Model              : {s['model']}",
        f"  LLM Calls          : {s['llm_calls']}",
        f"  Tool Calls         : {s['tool_calls']}",
        f"    └─ Tavily Searches: {s['tavily_searches']}",
        f"  Prompt Tokens      : {s['prompt_tokens']:,}",
        f"  Completion Tokens  : {s['completion_tokens']:,}",
        f"  Total Tokens       : {s['total_tokens']:,}",
        f"  Estimated Cost     : ${s['estimated_cost_usd']:.5f} USD",
        f"  Wall-Clock Time    : {s['elapsed_seconds']}s",
        sep,
    ]

    # -- Per-step timing breakdown -----------------------------------------
    llm_steps  = [step for step in t.steps if step["type"] == "llm"]
    tool_steps = [step for step in t.steps if step["type"] == "tool"]

    if llm_steps:
        lines.append("")
        lines.append("  LLM Call Breakdown:")
        lines.append(f"  {'#':<4} {'Prompt':>8} {'Compl':>8} {'Total':>8} {'Time':>8}")
        lines.append(f"  {'─'*4} {'─'*8} {'─'*8} {'─'*8} {'─'*8}")
        for i, step in enumerate(llm_steps, 1):
            t_str = f"{step.get('elapsed_s', '?')}s" if step.get('elapsed_s') is not None else "  —"
            lines.append(
                f"  {i:<4} {step['prompt_tokens']:>8,} {step['completion_tokens']:>8,} "
                f"{step['total']:>8,} {t_str:>8}"
            )

    if tool_steps:
        lines.append("")
        lines.append("  Tool Call Breakdown:")
        lines.append(f"  {'#':<4} {'Tool':<28} {'Time':>8}")
        lines.append(f"  {'─'*4} {'─'*28} {'─'*8}")
        for i, step in enumerate(tool_steps, 1):
            t_str = f"{step.get('elapsed_s', '?')}s" if step.get('elapsed_s') is not None else "  —"
            lines.append(f"  {i:<4} {step['name']:<28} {t_str:>8}")

    lines += [sep, ""]
    state["report"] = "\n".join(lines)
    return state


def _print_node(state: ReportState) -> ReportState:
    """Node: print report to stdout."""
    print(state["report"])
    return state


def build_reporter() -> Any:
    """
    Returns a compiled LangGraph graph.
    Invoke with: reporter.invoke({"tracker": tracker, "approach": "...", "result_text": "..."})
    """
    g = StateGraph(ReportState)
    g.add_node("format", _format_node)
    g.add_node("print", _print_node)
    g.set_entry_point("format")
    g.add_edge("format", "print")
    g.add_edge("print", END)
    return g.compile()


# Legacy alias used in notebooks
def build_report(tracker: TokenCostTracker, approach: str, result_text: str = "") -> str:
    """Convenience wrapper — runs the reporter graph and returns the report string."""
    reporter = build_reporter()
    out = reporter.invoke(
        {"tracker": tracker, "approach": approach, "result_text": result_text}
    )
    return out["report"]


# ---------------------------------------------------------------------------
# Comparison display — call after running both notebooks
# ---------------------------------------------------------------------------

def compare_runs(da_summary: Dict[str, Any], pc_summary: Dict[str, Any]) -> None:
    """Pretty-print a side-by-side comparison table."""
    import pandas as pd

    keys = [
        "llm_calls",
        "tool_calls",
        "tavily_searches",
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "estimated_cost_usd",
        "elapsed_seconds",
    ]
    labels = [
        "LLM Calls",
        "Tool Calls",
        "Tavily Searches",
        "Prompt Tokens",
        "Completion Tokens",
        "Total Tokens",
        "Est. Cost (USD)",
        "Elapsed (s)",
    ]
    da_vals = [da_summary.get(k, "—") for k in keys]
    pc_vals = [pc_summary.get(k, "—") for k in keys]

    df = pd.DataFrame(
        {"Metric": labels, "DeepAgents": da_vals, "Prompt Chaining": pc_vals}
    )

    # Highlight winner (lower is better for all metrics)
    def highlight(row):
        if row["DeepAgents"] == "—" or row["Prompt Chaining"] == "—":
            return [""] * 3
        try:
            winner = (
                "DeepAgents"
                if float(row["DeepAgents"]) <= float(row["Prompt Chaining"])
                else "Prompt Chaining"
            )
            styles = ["", "", ""]
            idx = 1 if winner == "DeepAgents" else 2
            styles[idx] = "background-color: #d4edda; font-weight: bold"
            return styles
        except (TypeError, ValueError):
            return [""] * 3

    print("\n" + "═" * 60)
    print("  SIDE-BY-SIDE COMPARISON")
    print("═" * 60)
    try:
        from IPython.display import display
        display(df.style.apply(highlight, axis=1).hide(axis="index"))
    except Exception:
        print(df.to_string(index=False))
