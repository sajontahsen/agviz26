"""Staged generation pipeline for the Vis Arena agent.

Public entrypoint: ``orchestrate(workdir)`` (see pipeline.stages), wired into
``example_agent.generate``. The pipeline runs a sequence of short, file-passing
LLM stages: requirements -> profile -> analyst -> planner -> coder.
"""
