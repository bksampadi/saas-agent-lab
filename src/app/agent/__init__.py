"""Agent execution on top of the application's services.

A model has two jobs here: turning an instruction into an unresolved intent
(extraction), and, after resolution, choosing which goal-bound tools to call
and when to conclude (decision). The application decides everything else:
identity (resolver), what a tool acts on (the persisted goal), whether a
change is allowed (policy), what a model is shown (tools.observe), how much
it may do (decision limits), whether a change happened (the business
transaction) and whether the goal holds (verifier). AgentExecutor.run is
the whole lifecycle.
"""
