# simulations/traffic/prompts.py
from __future__ import annotations

import random
from typing import Optional


def _rng(seed: Optional[int]) -> random.Random:
    r = random.Random()
    r.seed(0 if seed is None else int(seed))
    return r


def generate_prompt(idx: int, *, seed: Optional[int] = 0) -> str:
    """
    Deterministic prompt generator for /v1/generate.
    Keep prompts short so local dev stays snappy.
    """
    r = _rng(seed)
    # stable but varied
    topics = [
        "a short summary of the benefits of unit tests",
        "a haiku about latency",
        "a 3-bullet explanation of caching",
        "a one-paragraph description of what an SLO is",
        "a concise explanation of p95 vs p99",
    ]
    topic = topics[idx % len(topics)]
    spice = r.choice(["Make it direct.", "Keep it technical.", "No fluff.", "Be precise."])
    return f"Write {topic}. {spice}"


def extract_text(idx: int, *, seed: Optional[int] = 0) -> str:
    """
    Deterministic text generator for /v1/extract.
    These should look like “ticket/invoice-ish” semi-structured blobs.
    """
    r = _rng(seed)
    names = ["A. Rivera", "Chris A.", "M. Chen", "J. Patel"]
    issues = ["Login fails", "Payment declined", "App crash", "Refund request"]
    priorities = ["low", "medium", "high", "urgent"]

    who = names[idx % len(names)]
    issue = issues[idx % len(issues)]
    pr = priorities[(idx + 1) % len(priorities)]
    ticket = 10000 + idx

    # include a couple fields that many schemas want
    return (
        f"TICKET: {ticket}\n"
        f"REQUESTER: {who}\n"
        f"PRIORITY: {pr}\n"
        f"SUBJECT: {issue}\n"
        f"DETAILS: User reports '{issue.lower()}'. Please investigate and respond.\n"
        f"CHANNEL: email\n"
    )