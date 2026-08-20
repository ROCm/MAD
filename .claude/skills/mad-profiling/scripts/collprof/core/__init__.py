"""Engine-agnostic half of the package: parsing, aggregation and report composition.

Nothing here may name an engine or state something true of only one workload. Engine facts arrive as
an :class:`~collprof.core.spec.EngineSpec`; report sentences that belong to one engine arrive as
:class:`~collprof.core.spec.ReportNotes`.
"""
