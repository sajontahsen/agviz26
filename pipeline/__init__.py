"""Staged pipelines for the Vis Arena agent.

Generation: ``orchestrate(workdir)`` (pipeline.stages) — profile -> planner ->
analyst -> storyboard -> coder.

Evaluation: ``run_evaluation(workdir, artifact_url)`` (pipeline.evaluator) —
structured browser tools with deterministic score aggregation.
"""
