"""Immutable value objects of the domain.

Every type here is frozen and compared by value. They carry the invariants that
must hold *everywhere*, so no layer can construct a nonsensical state.
"""
