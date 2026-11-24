"""The agent loop.

How tool use works with the Claude API:
  1. We send messages + a list of tools (name, description, JSON schema).
  2. Claude answers with text and/or "tool_use" blocks: {id, name, input}.
  3. We run each tool and send back "tool_result" blocks with the same id.
  4. Repeat until Claude calls submit_findings (our "final answer" tool).

Claude decides which tool to call next - that is the "agentic" part. Our code decides what the
tools can do, how many calls are allowed, and whether the final answer is acceptable.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

from src.agent import tools
from src.agent.cost import Usage
from src.agent.prompts import SYSTEM_PROMPT, build_briefing
from src.agent.validation import (NumberBook, briefing_numbers, check_findings, normalize_findings,
                                  structure_problems)
from src.pipeline import AnalysisResult


@dataclass
class TraceStep:
    kind: str                 # "tool" | "validation" | "note"
    reasoning: str = ""       # Claude's text right before the call (why it called the tool)
    tool: str = ""
    inputs: dict = field(default_factory=dict)
    output: dict = field(default_factory=dict)


@dataclass
class AgentOutput:
    findings: dict | None
    trace: list[TraceStep]
    warnings: list[str]
    model: str
    tool_calls: int
    usage: Usage
    cost_usd: float
    uncached_cost_usd: float

    # kept for the console, report and scorecard
    @property
    def input_tokens(self) -> int:
        return self.usage.total_input

    @property
    def output_tokens(self) -> int:
        return self.usage.output


def _client():
    import anthropic

    if not os.getenv("ANTHROPIC_API_KEY"):
        raise RuntimeError("ANTHROPIC_API_KEY is not set in .env")
    return anthropic.Anthropic()


def _block_to_dict(block) -> dict:
    """SDK content blocks -> plain dicts we can send back in the next request."""
    if block.type == "text":
        return {"type": "text", "text": block.text}
    if block.type == "tool_use":
        return {"type": "tool_use", "id": block.id, "name": block.name, "input": block.input}
    return block.model_dump() if hasattr(block, "model_dump") else dict(block)


def run_agent(r: AnalysisResult, client=None) -> AgentOutput:
    cfg = r.config["agent"]
    model, max_calls = cfg["model"], cfg["max_tool_calls"]
    max_retries, max_turns = cfg.get("max_validation_retries", 1), cfg.get("max_turns", 15)
    client = client or _client()

    briefing = build_briefing(r)
    book = NumberBook()
    book.add(briefing_numbers(briefing))

    driver_names = r.drivers["driver"].tolist()
    tool_schemas = [schema for schema, _ in tools.TOOLS.values()] + [tools.submit_findings_schema(driver_names)]
    messages: list[dict] = [{"role": "user", "content": briefing}]
    trace: list[TraceStep] = []
    calls = retries = 0
    usage = Usage()
    pricing = cfg["pricing_usd_per_mtok"]
    # Automatic caching: one top-level cache_control, and the API caches the longest repeated
    # prefix (tools + system + earlier turns), moving forward as the conversation grows.
    # Sent via extra_body so it works with any SDK version.
    extra = {"extra_body": {"cache_control": {"type": "ephemeral"}}} if cfg.get("prompt_caching") else {}

    def done(findings, warnings):
        return AgentOutput(findings, trace, warnings, model, calls, usage,
                           usage.cost_usd(pricing), usage.uncached_cost_usd(pricing))
    last_findings, last_problems = None, []

    for _ in range(max_turns):
        resp = client.messages.create(model=model, max_tokens=cfg.get("max_tokens", 4000),
                                      system=SYSTEM_PROMPT, tools=tool_schemas, messages=messages, **extra)
        usage.add(resp.usage)
        messages.append({"role": "assistant", "content": [_block_to_dict(b) for b in resp.content]})

        reasoning = " ".join(b.text for b in resp.content if b.type == "text").strip()
        tool_uses = [b for b in resp.content if b.type == "tool_use"]
        if not tool_uses:  # Claude answered in plain text instead of submitting: nudge it
            trace.append(TraceStep("note", reasoning=reasoning))
            messages.append({"role": "user", "content": "Please submit your analysis with the submit_findings tool."})
            continue

        results = []
        for tu in tool_uses:
            if tu.name == "submit_findings":
                findings = normalize_findings(tu.input)
                shape = structure_problems(findings, driver_names)
                if shape:
                    # A malformed answer can't be used at all: always ask for a fix (bounded by max_turns)
                    trace.append(TraceStep("validation", reasoning=reasoning, tool=tu.name,
                                           output={"structure_problems": shape}))
                    results.append({"type": "tool_result", "tool_use_id": tu.id, "is_error": True,
                                    "content": "submit_findings input is malformed: " + "; ".join(shape)
                                               + ". Send arrays as JSON arrays, not strings, and call it again."})
                    continue
                problems = check_findings(findings, book)
                last_findings, last_problems = findings, problems
                trace.append(TraceStep("validation", reasoning=reasoning, tool=tu.name,
                                       output={"unverified_numbers": problems}))
                if not problems or retries >= max_retries:
                    return done(findings, problems)
                retries += 1
                results.append({"type": "tool_result", "tool_use_id": tu.id, "is_error": True, "content":
                                "These numbers do not appear in the briefing or any tool result: "
                                + "; ".join(problems)
                                + ". Use only numbers you were given (same units), or remove them, "
                                  "then call submit_findings again."})
                continue

            if tu.name not in tools.TOOLS:      # e.g. a malformed call: answer, but don't charge the budget
                output = tools.execute(r, tu.name, tu.input)
            elif calls >= max_calls:
                output = {"error": "Tool budget used up. Call submit_findings now with what you know."}
            else:
                calls += 1
                output = tools.execute(r, tu.name, tu.input)
                book.add(output)
            trace.append(TraceStep("tool", reasoning=reasoning, tool=tu.name, inputs=tu.input, output=output))
            reasoning = ""  # the reasoning belongs to the first call of the turn
            results.append({"type": "tool_result", "tool_use_id": tu.id,
                            "content": json.dumps(output), "is_error": "error" in output})
        messages.append({"role": "user", "content": results})

    # Ran out of turns: keep the last submitted findings if there were any
    return done(last_findings, last_problems + ["The agent did not finish within the turn limit."])
