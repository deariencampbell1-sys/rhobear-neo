"""Advisor service — read-only frontier agent on the build box.

Bound to the tailnet address only.  Receives escalation requests from Neo,
runs a read-only frontier agent (Claude Opus or Codex) in a temp directory,
and returns a structured work order.  Never pushes, comments, or merges.
"""
