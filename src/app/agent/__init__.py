"""Agent execution on top of the application's services.

A model has one job here so far: turning an instruction into an unresolved
intent (``planner``). The application decides everything else: identity
(resolver), what may be touched (goal scope), whether a change happened
(business transaction) and whether the goal holds (verifier).
"""
