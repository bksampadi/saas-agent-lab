"""Deterministic agent execution on top of the application's services.

No model lives here. The application decides identity (resolver), what may be
touched (goal scope), whether a change happened (business transaction) and
whether the goal holds (verifier).
"""
