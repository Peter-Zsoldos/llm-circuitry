# llm-circuitry

**The missing hardware of LLM systems.** Memory protection against prompt injection, transactions with rollback for agent actions, cache coherence between agents, interrupts and watchdogs, call stacks, ECC-style hallucination checks, DMA-style data handles, context paging, signed instructions and capability tokens, and deterministic replay.

Rust core, Python API, no runtime dependencies, works with any model provider or agent framework.

```bash
pip install git+https://github.com/Peter-Zsoldos/llm-circuitry   # needs a Rust toolchain; prebuilt wheels: see Releases
```

---

## Why

A popular way to think about LLM systems is as a new kind of computer:

| Computer | LLM system |
|---|---|
| CPU | the model |
| ROM / firmware | the weights |
| RAM | the context window |
| Disk | RAG, vector stores, files |
| Device drivers / syscalls | tools, MCP |
| GPU / multi-core | multiple agents |

People are building the operating systems and drivers for this computer. But a real computer also contains a lot of **less glamorous hardware that does its job automatically, underneath the software**, and LLM systems have almost none of it. Today those jobs are either skipped or left to the model to "try its best".

This library implements those missing parts as small, composable building blocks:

| Module | Hardware it replaces | The problem it solves |
|---|---|---|
| [`ProtectedContext`](#memory-protection-protectedcontext) | MMU, privilege rings, NX bit | **Prompt injection**: untrusted text acting as instructions; tool calls driven by tainted data |
| [`TransactionManager`, `Transaction`](#transactions-transactionmanager) | Journaling filesystem | Half-finished real-world actions with no undo; no record of what happened before a crash |
| [`CoherentStore`](#cache-coherence-coherentstore) | Cache coherence (MESI) | Agent B acting on a plan agent A already changed |
| [`InterruptController`](#interrupts-watchdog-clock) | Interrupt controller | No way to say "stop, the requirements changed" mid-task |
| [`Watchdog`](#interrupts-watchdog-clock) | Watchdog timer | Agents looping on the same tool call, stalling, blowing budgets |
| [`Clock`](#interrupts-watchdog-clock) | Real-time clock | Models have no sense of time or deadlines |
| [`ExecStack`](#call-stack-execstack) | Program counter and call stack | Agents losing track of which (sub-)goal they are on |
| [`Verifier`](#ecc-verifier) | ECC memory | Hallucinated quotes and numbers that look identical to real ones |
| [`BlobStore`](#dma-blobstore) | DMA | Large data copied through the context instead of passed by reference |
| [`AddressableDoc`](#exact-addressing-addressabledoc) | Physical addressing | Line numbers going stale after edits; fuzzy "the part about X" references |
| [`Pager`](#virtual-memory-pager) | Virtual memory, CLOCK page replacement | Context overflow; what to drop and what to write back |
| [`Signer`, `AuditLog`](#tpm-and-acls-signer-auditlog) | TPM, secure boot, file permissions | Forged "orchestrator" messages, over-broad tool permissions, editable logs |
| [`ReplayCache`, `replayable`](#replay-replaycache) | JTAG debug port | Agent bugs you can't reproduce |

Each part is independent. Use one, or wire them all into your agent loop ([full example](examples/agent_loop.py)).

---

## Memory protection: `ProtectedContext`

In a CPU, ring 0 code is separated from ring 3 code, and no-execute pages can hold data that never runs as code. **A context window has no such separation.** A system prompt and a scraped web page are the same kind of token, and prompt injection is a buffer overflow on a machine with no MMU.

`ProtectedContext` tags every piece of context with a ring, marks untrusted content non-executable, wraps it in data boundaries with a random per-context nonce (so an attacker can't forge the closing marker), scans it for injection markers, and **tracks taint**: content derived from untrusted input stays untrusted, so summarising a malicious page doesn't wash out the injection. Before any tool call, `check()` decides whether the inputs that led to it are trusted enough.

```python
import llm_circuitry as lc
from llm_circuitry import Ring

ctx = lc.ProtectedContext()
ctx.set_rule("send_email", Ring.USER, "deny")      # only system/user input may drive emails
ctx.set_rule("summarize", Ring.UNTRUSTED)          # summarising untrusted data is fine

user = ctx.push(Ring.USER, "alice", "Summarise this page and email it to my boss")
page = ctx.push(Ring.UNTRUSTED, "web:example.com", fetched_html)
summary = ctx.derive("model", model_summary, derived_from=[page])   # inherits ring 3

ctx.check("send_email", [user, summary])
# {'verdict': 'deny', 'reason': "action 'send_email' is influenced by segment 3 (ring untrusted,
#   source 'model'); this action only accepts input up to ring user", 'tainted_by': [2, 3]}

prompt = ctx.render()   # security preamble + <<DATA 9f2c… id=2 ring=untrusted …>> … <<END 9f2c…>>
lc.scan_injection("please IGNORE previous instructions")   # [{'rule': 'override', 'severity': 3, ...}]
```

Rings: `0 KERNEL` (system/developer), `1 USER`, `2 TOOL` (your own tools), `3 UNTRUSTED` (web, email, files, third-party APIs). Content with a severity-3 injection finding escalates an otherwise allowed action to `require_approval`.

> The scanner is a heuristic. The real defence is the taint check: even an injection the scanner misses can't drive an action whose rule doesn't accept untrusted input.

## Transactions: `TransactionManager`

Filesystems survive crashes because they write intent to a journal *before* touching data. Agents now send emails, move money and run migrations with no journal and no rollback.

```python
tm = lc.TransactionManager("journal.jsonl")   # hash-chained, write-ahead, survives crashes

with lc.Transaction(tm, "onboard customer") as tx:
    acct = tx.run("create_account", lambda: crm.create("acme"), undo=lambda a: crm.delete(a.id))
    tx.run("create_bucket", lambda: s3.create("acme"), undo=lambda _: s3.delete("acme"))
    tx.run("charge_card", lambda: stripe.charge(...))     # raises -> bucket and account are undone, newest first

tx.status                     # 'rolled_back' (or 'rollback_incomplete' if a step had no undo)
lc.Transaction(tm, "plan", dry_run=True)   # records the plan without executing anything

# after a crash:
lc.TransactionManager("journal.jsonl").recover()
# [{'txn': 3, 'to_compensate': [...completed steps...], 'in_doubt': [...started, never confirmed...]}]
```

Editing the journal file breaks the hash chain and raises `IntegrityError` on open.

## Cache coherence: `CoherentStore`

Every agent's context is a private cache of the world. Without a coherence protocol, nothing tells agent B that agent A changed the thing B is reasoning about.

```python
store = lc.CoherentStore()
store.write("planner", "plan", {"db": "mysql"})
store.read("coder", "plan")
store.write("planner", "plan", {"db": "postgres"})

store.state("coder", "plan")          # 'invalid'  (MESI: modified / exclusive / shared / invalid)
store.invalidations("coder")          # [{'key': 'plan', 'your_version': 1, 'new_version': 2, 'writer': 'planner'}]
store.require_fresh("coder")          # raises StaleError: run this guard before an agent acts
store.write("coder", "plan", {...})   # raises StaleError: the write is based on a stale read
store.write("coder", "plan", {...}, expected_version=2)   # compare-and-swap
store.lock("coder", "src/db.py", ttl_seconds=30)          # exclusive lock with expiry
```

## Interrupts, watchdog, clock

```python
irq = lc.InterruptController()
irq.raise_interrupt(9, "user", "Stop: use Postgres, not MySQL")   # from a UI, webhook, CI, another agent
irq.set_mask(5)                       # critical section: hold back anything below priority 5 (255 = non-maskable)
for line in irq.drain_text():         # between agent steps
    messages.append({"role": "user", "content": line})

wd = lc.Watchdog(max_steps=50, max_tokens=200_000, max_seconds=900, stall_steps=10)
v = wd.tick(f"{tool}({args})", tokens=used)   # {'status': 'trip', 'reason': 'loop: the same 2-step cycle of actions was repeated 3 times'}
wd.progress()                                # call when real progress happens

clock = lc.Clock(deadline_seconds=3600)
clock.note()                                 # '[clock] now=2026-10-14T09:20:11Z elapsed=12m03s remaining=47m57s'
```

## Call stack: `ExecStack`

Long-running agents lose track of where they are, especially after context compaction. Keep the goal stack *outside* the context and re-inject it.

```python
st = lc.ExecStack(max_depth=8)            # CapacityError on runaway sub-task recursion
st.push("Ship v2 release")
st.push("Fix failing tests")
st.step("reproduced failure in test_auth")
st.pop({"fixed": "token expiry off by one"})   # result is recorded in the parent frame
print(st.render())
# EXECUTION STACK (depth 1, outermost first)
# #1 Ship v2 release   <-- CURRENT GOAL
#    done: sub-goal 'Fix failing tests' returned: {"fixed":"token expiry off by one"}
saved = st.to_json(); st = lc.ExecStack.from_json(saved)   # context switch
```

## ECC: `Verifier`

ECC RAM flags corrupted reads; nothing flags a hallucinated figure. Ask the model to cite as `[source_id: "exact quote"]`, then check the whole answer:

```python
v = lc.Verifier()
v.add_source("q3", q3_report_text)
report = v.check_cited('Revenue grew 12% [q3: "Revenue increased 12%"]. Headcount rose to 90 [q3: "Headcount was flat at 85"].')
report["passed"], report["failed"]                 # (1, 1)
report["checks"][1]["unsupported_numbers"]         # ['90']  <- number appears nowhere in the source
```

Match levels: `exact` (with byte span), `normalized` (case, punctuation, whitespace), `fuzzy` (token overlap ≥ threshold, with the closest passage), `not_found`, `unknown_source`.

## DMA: `BlobStore`

Pass big data between agents and tools by handle; pull only what's needed into context.

```python
blobs = lc.BlobStore(capacity_bytes=500_000_000)
h = blobs.put(csv_bytes, "text/csv", "sales export")   # 'blob:3f9a0c1d2e4b5a69' (content-addressed, deduplicated)
blobs.describe(h)          # "blob:3f9a… 'sales export' (text/csv, 48.2 MB, 1203311 lines) preview: id,region,…"
blobs.grep(h, "EMEA", 20)  # [{'line': 1042, 'text': '...'}]
blobs.lines(h, 1, 50)      # just the first 50 lines
blobs.release(h)           # refcounted; unreferenced blobs are evicted LRU when over capacity
```

## Exact addressing: `AddressableDoc`

```python
doc = lc.AddressableDoc(source_code)
print(doc.numbered(40, 60))      # what the model sees
a = doc.anchor(42)               # 'L42#9f3a1c2b' (line number + content hash)
doc.update(edited_source)        # someone inserted lines above
doc.resolve(a)                   # {'line': 45, 'moved': True, ...}  or NotFoundError if the line changed
```

## Virtual memory: `Pager`

```python
pager = lc.Pager(budget_tokens=100_000)
pager.load("system", system_prompt, pinned=True)
pager.load("doc:contract", contract_text)           # returns pages evicted (CLOCK, second-chance)
pager.access("doc:spec")                            # page fault -> brought back from swap
pager.write("notes", updated_notes)                 # marks dirty
pager.dirty()                                       # write these back to long-term memory before forgetting
prompt = pager.render()                             # resident pages + index of swapped-out ids
```

## TPM and ACLs: `Signer`, `AuditLog`

```python
orch = lc.Signer("orchestrator")                    # random key; or Signer("orchestrator", key=secret_bytes)
env = orch.sign("worker-1", "deploy to staging")
orch.verify(env)                                    # False if anything was changed or forged

token = orch.grant("mailer", ["email.send"], ["*@acme.com"], ttl_seconds=600, on_behalf_of="alice")
orch.check(token, "email.send", "bob@acme.com")     # ok
orch.check(token, "email.send", "eve@evil.com")     # DeniedError: least privilege, enforced in code

log = lc.AuditLog("audit.jsonl")
log.append("mailer", "email.send", {"to": "bob@acme.com"}, on_behalf_of="alice")
log.verify()                                        # IntegrityError if any entry was edited, removed or reordered
```

## Replay: `ReplayCache`

```python
cache = lc.ReplayCache("auto", "recordings.jsonl")  # "record" | "replay" | "auto"

@lc.replayable(cache)
def complete(**request):
    return client.messages.create(**request).content[0].text

# in tests / debugging: ReplayCache("replay", ...) makes any unrecorded call an error
```

---

## Using it from Rust

```toml
[dependencies]
llm-circuitry = { git = "https://github.com/Peter-Zsoldos/llm-circuitry" }
```

```rust
use llm_circuitry::protect::{ProtectedContext, Ring, OnViolation, Verdict};

let mut ctx = ProtectedContext::new();
ctx.set_rule("send_email", Ring::User, OnViolation::Deny);
let web = ctx.push(Ring::Untrusted, "web", page);
assert_eq!(ctx.check("send_email", &[web]).verdict, Verdict::Deny);
```

All types are `serde`-serialisable. Errors carry an `ErrorKind` that maps to Python exceptions (`NotFoundError`, `StaleError`, `ConflictError`, `CapacityError`, `InvalidStateError`, `IntegrityError`, `DeniedError`, `ValidationError`, all subclasses of `CircuitError`).

## Building from source

```bash
pip install maturin
maturin develop --release          # build + install into the active virtualenv
cargo test                         # Rust tests
pytest tests                       # Python tests
```

## What's still missing (roadmap and ideas)

Parts of the analogy that this library doesn't cover yet, and good places to contribute:

- **Stable ISA / datasheet**: a behavioural contract that keeps prompts and skills portable across model versions.
- **Writeback and consolidation**: deciding what from a session becomes long-term memory (beyond the `Pager` dirty bit).
- **Field-flashable firmware**: safe continual learning of weights without catastrophic forgetting.
- **JTAG for weights**: interpretability hooks as a real debugger (breakpoints on internal features).
- **Calibrated per-claim confidence**: parity bits that come from the model itself, not a second model.
- Persistence backends (SQLite, Redis) for `CoherentStore`, `BlobStore` and `Pager`, so they can be shared across processes.

Issues and PRs welcome.

## License

Dual-licensed under [MIT](LICENSE-MIT) or [Apache-2.0](LICENSE-APACHE), at your option.
