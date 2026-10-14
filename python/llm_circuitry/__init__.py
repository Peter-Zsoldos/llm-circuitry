"""llm-circuitry: the missing hardware of LLM systems.

If an LLM is a CPU, its context window is RAM, retrieval is the disk and the
weights are ROM, then most of the *other* hardware a computer depends on is
still missing from LLM systems. This package provides it:

=========================  ===================================  ==============================================
Class                      Hardware analogy                     Problem it solves
=========================  ===================================  ==============================================
ProtectedContext           MMU, privilege rings, NX bit         Prompt injection, tainted tool calls
TransactionManager         Journaling filesystem                Undoing half-finished real-world actions
CoherentStore              Cache coherence (MESI)               Agents acting on stale shared state
InterruptController        Interrupt controller                 Interrupting an agent mid-task
Watchdog                   Watchdog timer                       Loops, stalls, runaway budgets
Clock                      Real-time clock                      Models having no sense of time
ExecStack                  Program counter + call stack         Agents losing track of goals
Verifier                   ECC memory                           Hallucinated quotes and numbers
BlobStore                  DMA                                  Copying big data through the context
AddressableDoc             Physical addressing                  Stale line numbers after edits
Pager                      Virtual memory / CLOCK paging        Context overflow
Signer, AuditLog           TPM, secure boot, ACLs               Forged instructions, over-broad permissions
ReplayCache                JTAG / deterministic replay          Unreproducible agent runs
=========================  ===================================  ==============================================
"""

from __future__ import annotations

import enum
import functools
from typing import Any, Callable, Dict, List, Optional, TypeVar

from ._core import (
    AddressableDoc,
    AuditLog,
    BlobStore,
    CapacityError,
    CircuitError,
    Clock,
    CoherentStore,
    ConflictError,
    DeniedError,
    ExecStack,
    IntegrityError,
    InterruptController,
    InvalidStateError,
    NotFoundError,
    Pager,
    ProtectedContext,
    ReplayCache,
    Signer,
    StaleError,
    TransactionManager,
    ValidationError,
    Verifier,
    Watchdog,
    __version__,
    estimate_tokens,
    glob_match,
    scan_injection,
)

__all__ = [
    "AddressableDoc", "AuditLog", "BlobStore", "CapacityError", "CircuitError", "Clock",
    "CoherentStore", "ConflictError", "DeniedError", "ExecStack", "IntegrityError",
    "InterruptController", "InvalidStateError", "NotFoundError", "Pager", "ProtectedContext",
    "ReplayCache", "Ring", "Signer", "StaleError", "Transaction", "TransactionManager",
    "ValidationError", "Verifier", "Watchdog", "__version__", "estimate_tokens", "glob_match",
    "replayable", "scan_injection",
]


class Ring(enum.IntEnum):
    """Privilege level of a piece of context. Lower is more trusted."""

    KERNEL = 0  # system prompt, developer instructions
    USER = 1  # the human the agent works for
    TOOL = 2  # output of tools you control
    UNTRUSTED = 3  # web pages, emails, files, third-party APIs


def _jsonable(value: Any) -> Any:
    """Best effort conversion of a step result into something JSON-serialisable."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    return repr(value)


class Transaction:
    """Run a group of real-world actions as one unit, with undo on failure.

    Every step is written to the journal *before* it runs, so a crash leaves a
    record of what may have happened (see ``TransactionManager.recover()``).

    >>> tm = TransactionManager("journal.jsonl")
    >>> with Transaction(tm, "provision customer") as tx:
    ...     acct = tx.run("create_account", create, undo=delete_account, params={"name": "acme"})
    ...     tx.run("send_welcome", send_email, undo=None)
    ...     # an exception anywhere in the block undoes completed steps, newest first

    With ``dry_run=True`` nothing executes: ``tx.plan`` lists what would run.
    """

    def __init__(self, manager: TransactionManager, label: str, dry_run: bool = False):
        self.manager = manager
        self.label = label
        self.dry_run = dry_run
        self.id: Optional[int] = None
        self.plan: List[Dict[str, Any]] = []
        self.status: Optional[str] = None
        self.compensation_errors: List[str] = []
        self._undo: Dict[int, Optional[Callable[[Any], Any]]] = {}
        self._results: Dict[int, Any] = {}

    def __enter__(self) -> "Transaction":
        if not self.dry_run:
            self.id = self.manager.begin(self.label)
        return self

    def run(
        self,
        name: str,
        do: Callable[[], Any],
        undo: Optional[Callable[[Any], Any]] = None,
        params: Any = None,
    ) -> Any:
        """Journal, then execute ``do()``. ``undo(result)`` reverses it on rollback."""
        if self.dry_run:
            self.plan.append({"name": name, "params": params, "reversible": undo is not None})
            return None
        assert self.id is not None, "use Transaction as a context manager"
        comp = getattr(undo, "__name__", "undo") if undo is not None else None
        step = self.manager.step(self.id, name, _jsonable(params), comp)
        try:
            result = do()
        except BaseException as exc:
            self.manager.fail(self.id, step, f"{type(exc).__name__}: {exc}")
            raise
        self.manager.done(self.id, step, _jsonable(result))
        self._undo[step] = undo
        self._results[step] = result
        return result

    def rollback(self, reason: str = "rolled back") -> str:
        assert self.id is not None
        for step in self.manager.begin_rollback(self.id, reason):
            undo = self._undo.get(step["step"])
            if undo is None:
                if step["status"] == "done":
                    self.manager.compensation_failed(self.id, step["step"], "no undo function (irreversible step)")
                    self.compensation_errors.append(f"{step['name']}: irreversible")
                continue
            try:
                undo(self._results.get(step["step"]))
            except Exception as exc:  # keep undoing the rest
                self.manager.compensation_failed(self.id, step["step"], f"{type(exc).__name__}: {exc}")
                self.compensation_errors.append(f"{step['name']}: {exc}")
            else:
                self.manager.compensated(self.id, step["step"])
        self.status = self.manager.finish_rollback(self.id)
        return self.status

    def __exit__(self, exc_type, exc, tb) -> bool:
        if self.dry_run:
            self.status = "dry_run"
            return False
        if exc_type is None:
            self.manager.commit(self.id)
            self.status = "committed"
        else:
            self.rollback(f"{exc_type.__name__}: {exc}")
        return False


F = TypeVar("F", bound=Callable[..., Any])


def replayable(cache: ReplayCache) -> Callable[[F], F]:
    """Decorator: record/replay a model-calling function keyed by its keyword arguments.

    >>> cache = ReplayCache("auto", "calls.jsonl")
    >>> @replayable(cache)
    ... def complete(**request):
    ...     return client.messages.create(**request).content[0].text
    """

    def deco(fn: F) -> F:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            request = _jsonable({"fn": fn.__qualname__, "args": list(args), "kwargs": kwargs})
            hit = cache.lookup(request)
            if hit is not None:
                return hit
            response = fn(*args, **kwargs)
            cache.store(request, _jsonable(response))
            return response

        return wrapper  # type: ignore[return-value]

    return deco
