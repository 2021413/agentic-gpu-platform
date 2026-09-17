"""Explicit run state machine (spec section 10).

Orchestration is a governed transition table, not an uncontrolled recursive
agent loop. Every transition is validated here and nowhere else, so an illegal
move is impossible regardless of which use case is driving the run.
"""

from __future__ import annotations

from types import MappingProxyType

from domain.enums import RunStatus
from domain.exceptions import InvalidStateTransitionError

__all__ = ["RunStateMachine"]

_S = RunStatus

# A cancellation request is legal from any non-terminal state and is therefore
# added programmatically below rather than repeated in every row.
_TRANSITIONS: dict[RunStatus, frozenset[RunStatus]] = {
    _S.CREATED: frozenset({_S.PLANNING, _S.CODING, _S.FAILED}),
    _S.PLANNING: frozenset({_S.PLAN_READY, _S.FAILED}),
    _S.PLAN_READY: frozenset({_S.CODING, _S.PLANNING, _S.FAILED}),
    _S.CODING: frozenset({_S.VALIDATING, _S.REVIEWING, _S.FAILED}),
    _S.VALIDATING: frozenset({_S.REVIEWING, _S.REPAIRING, _S.FAILED}),
    _S.REVIEWING: frozenset({_S.COMPLETED, _S.REPAIRING, _S.FAILED}),
    _S.REPAIRING: frozenset({_S.CODING, _S.PLANNING, _S.FAILED}),
    _S.CANCELLING: frozenset({_S.CANCELLED, _S.FAILED}),
    _S.COMPLETED: frozenset(),
    _S.FAILED: frozenset(),
    _S.CANCELLED: frozenset(),
}

_ALLOWED: MappingProxyType[RunStatus, frozenset[RunStatus]] = MappingProxyType(
    {
        state: (targets | {_S.CANCELLING})
        if not state.is_terminal and state is not _S.CANCELLING
        else targets
        for state, targets in _TRANSITIONS.items()
    }
)


class RunStateMachine:
    """Pure transition authority for :class:`domain.entities.run.Run`."""

    __slots__ = ()

    @staticmethod
    def allowed_from(state: RunStatus) -> frozenset[RunStatus]:
        return _ALLOWED[state]

    @staticmethod
    def can(current: RunStatus, requested: RunStatus) -> bool:
        return requested in _ALLOWED[current]

    @classmethod
    def ensure(cls, current: RunStatus, requested: RunStatus) -> None:
        """Raise unless the transition is legal.

        A self-transition is rejected: re-entering a state is a bug, and
        idempotency is handled by the caller checking ``current`` first.
        """
        if not cls.can(current, requested):
            raise InvalidStateTransitionError("Run", current, requested)
