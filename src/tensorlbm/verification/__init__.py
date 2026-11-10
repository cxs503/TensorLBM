"""Verification suites whose only job is to *prove* a capability.

Each subpackage owns one physics face, its acceptance criteria and the artifacts
that record them (``artifacts/<face>/``).  Verification code is deliberately
kept out of the production solver modules so that the acceptance numbers cannot
be silently absorbed by a refactor of the solver.
"""
