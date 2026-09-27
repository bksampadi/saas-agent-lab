"""Agent execution on top of the application's services.

A model has two jobs here: turning an instruction into an unresolved intent
(extraction), and, after resolution, choosing which goal-bound tools to call
and when to conclude (decision). The application decides everything else:
identity (resolver), what may be touched (goal scope), what a model is
shown (observations), how much it may do (decision limits), whether a
change happened (business transaction) and whether the goal holds
(verifier).
"""
