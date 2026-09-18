"""Orchestration: the brain of the platform.

Everything here is expressed against ports, so the whole workflow runs in tests
with a fake LLM, a fake queue and a fake registry — no GPU, no Docker, no
network.
"""
