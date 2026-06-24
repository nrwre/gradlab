"""Pluggable AI tutor that explains a training run in plain language.

Provider is chosen via the ``AI_PROVIDER`` environment variable:
    "anthropic" -> calls the Claude API (needs ANTHROPIC_API_KEY)
    "openai"    -> calls the OpenAI API (needs OPENAI_API_KEY)
    "stub"      -> deterministic, rule-based explanation (no key, the default)

The stub means the whole project runs end-to-end with zero credentials; swap in
a real provider by setting two environment variables.
"""
from __future__ import annotations
import os
import json
import urllib.request


def _build_prompt(run: dict) -> str:
    return (
        "You are a concise ML tutor. In 3-4 sentences, explain to a student what "
        "this optimisation/classification run shows. Be specific about the numbers "
        "and the intuition. Avoid headers and bullet points.\n\n"
        f"Run details (JSON):\n{json.dumps(run, indent=2)}"
    )


def _stub(run: dict) -> str:
    kind = run.get("kind", "run")
    if kind == "optimize":
        opt = run.get("optimizer", "the optimizer")
        kappa = run.get("kappa")
        start = run.get("start_loss")
        end = run.get("final_loss")
        msg = (f"Using {opt}, the loss fell from {start:.3g} to {end:.3g} over "
               f"{run.get('steps', '?')} steps.")
        if kappa is not None:
            if kappa >= 30:
                msg += (f" With a high condition number (κ={kappa:g}) the surface is "
                        "stretched and ill-conditioned, so plain gradient descent zig-zags; "
                        "momentum and Adam damp those oscillations and converge faster.")
            else:
                msg += (f" The condition number κ={kappa:g} is mild, so most methods "
                        "converge quickly and the differences between them are small.")
        return msg
    if kind == "classify":
        acc = run.get("accuracy", 0)
        model = run.get("model", "the model")
        ds = run.get("dataset", "the data")
        note = ("A linear boundary is enough here." if run.get("kernel") == "linear"
                else "A non-linear boundary (RBF kernel / tree splits) is needed to capture the curvature.")
        return (f"{model} reached {acc:.1%} accuracy on the {ds} dataset. {note} "
                "Watch how the decision region bends to separate the two classes.")
    return "Run complete."


def explain(run: dict) -> dict:
    provider = os.environ.get("AI_PROVIDER", "stub").lower()
    try:
        if provider == "anthropic" and os.environ.get("ANTHROPIC_API_KEY"):
            return {"text": _anthropic(_build_prompt(run)), "provider": "anthropic"}
        if provider == "openai" and os.environ.get("OPENAI_API_KEY"):
            return {"text": _openai(_build_prompt(run)), "provider": "openai"}
    except Exception as e:  # never fail the request because of the tutor
        return {"text": _stub(run), "provider": f"stub (fallback: {e.__class__.__name__})"}
    return {"text": _stub(run), "provider": "stub"}


def _anthropic(prompt: str) -> str:
    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=json.dumps({
            "model": os.environ.get("ANTHROPIC_MODEL", "claude-3-5-haiku-latest"),
            "max_tokens": 300,
            "messages": [{"role": "user", "content": prompt}],
        }).encode(),
        headers={
            "x-api-key": os.environ["ANTHROPIC_API_KEY"],
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
    )
    with urllib.request.urlopen(req, timeout=20) as r:
        data = json.load(r)
    return data["content"][0]["text"].strip()


def _openai(prompt: str) -> str:
    req = urllib.request.Request(
        "https://api.openai.com/v1/chat/completions",
        data=json.dumps({
            "model": os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
            "max_tokens": 300,
            "messages": [{"role": "user", "content": prompt}],
        }).encode(),
        headers={
            "Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}",
            "Content-Type": "application/json",
        },
    )
    with urllib.request.urlopen(req, timeout=20) as r:
        data = json.load(r)
    return data["choices"][0]["message"]["content"].strip()
