# F.A.B.L.E. Research Log
## Adversarial Multi-LLM Network — Implementation Notes

**Purpose:** Running technical log for research paper. Captures design decisions, engineering challenges, and solutions as they occur.

---

## Architecture Decision: Why Adversarial?

The cooperative pipeline (Analyst → Critic → Synthesizer) produces consensus-driven outputs. Adversarial networks exploit *productive disagreement*: a Critic incentivized to find flaws produces sharper feedback than one incentivized to agree. This is structurally analogous to GAN training (generator vs discriminator) applied to language model reasoning chains.

The six-role design maps to a formal adversarial proof system:
- **Planner** — sets the axioms and success criteria (problem decomposition)
- **Actor** — proposes the theorem/solution (generator)
- **Critic** — constructs a counterexample or falsification (adversary)
- **Validator** — checks logical validity and grounding (proof checker)
- **Refiner** — guides revision (proof assistant)
- **Judge** — decides acceptance (verifier/arbiter)

---

## LLM Role Assignment Rationale

| Role | LLM | Justification |
|------|-----|---------------|
| Planner | Claude Sonnet | Long-horizon structured decomposition; sets the frame for all subsequent agents |
| Actor | GPT-4o | Strong cross-domain generation; produces the primary artifact under adversarial pressure |
| Critic | Groq Llama 3 70B | Low-latency adversarial probing; runs in tight loop — cost efficiency critical |
| Validator | Gemini 1.5 Pro | Large context window (1M tokens) allows reviewing all prior agent outputs simultaneously |
| Refiner | Groq Llama 3 70B | Shares Groq with Critic; fast, directive output; role is structural not generative |
| Judge | Claude Sonnet | Holistic arbitration; convergence detection; produces final user-facing answer |

**Key insight:** Claude appears twice (Planner + Judge) because both roles require the highest-caliber systemic reasoning. Groq appears twice (Critic + Refiner) because both roles are high-frequency and low-semantic-density — precision matters less than speed and cost.

---

## Implementation Log

### Phase 1: Configuration Layer

**File:** `backend/core/config.py`
**Change:** Added 8 new Settings fields — 6 per-role model strings + `adversarial_max_rounds` + `adversarial_judge_threshold`
**Decision:** Default `max_rounds = 2` (not 3) to minimize API credit burn. Judge can terminate after round 1 if score ≥ 0.80.
**Problem/Solution:** None at this stage.

---

### Phase 2: Router Extension

**File:** `backend/router/model_router.py`
**Change:** Added `ROLE_MODEL_MAP` dict and `complete_for_role(role, system, user)` method
**Design note:** Used the `adv:` prefix in role keys to prevent namespace collision with the existing `"critic"` role on the AgentBus. This allows both pipelines to coexist without re-registration conflicts.
**Problem:** If both `register_all()` and `register_adversarial()` use the same role string `"critic"`, the second call overwrites the first — corrupting the standard pipeline. Solution: prefix adversarial roles as `"adv:critic"` etc.

---

### Phase 3: Adversarial Agent Classes

**File:** `backend/agents/adversarial.py` (new)
**6 agents created:** PlannerAgent, ActorAgent, AdversarialCriticAgent, ValidatorAgent, RefinerAgent, JudgeAgent
**Key engineering decision:** `BaseAdversarialAgent` overrides `__call__` to route through `complete_for_role()` instead of `complete()`. This is the only behavioral difference from `BaseAgent` — all other bus/history mechanics are inherited unchanged.
**History access pattern:** `_last_by_role(ctx, role)` searches `reversed(ctx.history)` — O(n) but n is small (max ~12 messages across 2 rounds). Returns the most recent message from that role, enabling round-aware behavior without explicit round tracking.
**Judge JSON output:** The Judge is instructed to return raw JSON. A `_parse_judge_output()` helper strips markdown fences and falls back to `re.search` for embedded JSON — robust against LLM formatting drift.

---

### Phase 4: Registration

**File:** `backend/agents/adversarial_register.py` (new)
**Pattern:** Mirrors `register.py` but with `adv:` prefixed keys. Adversarial agents are registered as a separate namespace on the same `AgentBus` singleton — no bus modification needed.

---

### Phase 5: Adversarial Lifecycle

**File:** `backend/core/adversarial_lifecycle.py` (new)
**Loop structure:**
```
Planner (once)
  └─ for round in range(max_rounds):
       Actor → Critic → Validator → Refiner → Judge
       if Judge.verdict == ACCEPT: break
```
**Termination guarantee:** Judge system prompt forces ACCEPT on final round, preventing infinite loops even if quality is low.
**Fallback:** If Judge JSON parsing fails entirely, lifecycle falls back to the last Actor output as the final answer.
**Credit efficiency:** Groq Llama 3 handles 2 of the 5 per-round roles. If Critic outputs `VERDICT: NO_FLAWS` and Validator outputs `VERDICT: ALL_VALID`, the Refiner produces minimal output and Judge ACCEPTs — round 1 termination in best-case.

---

### Phase 6: API Layer

**Files:** `schemas.py`, `routes/run.py`, `main.py`
**Change:** `RunRequest` gains `mode: Literal["standard", "adversarial"]` and `max_rounds: int | None`. `RunResponse` gains `adversarial_meta: AdversarialMeta | None` (null in standard mode — backward compatible).
**Backward compatibility:** `mode` defaults to `"standard"` — all existing API consumers unaffected.
**New domain:** Added `"general"` to the `domain` Literal so the adversarial pipeline accepts any prompt without requiring a domain label.

---

## Open Questions for Paper

1. **Convergence rate:** At what round does the Judge most frequently accept? Is round 1 acceptance the norm or exception across domains?
2. **Diversity benefit:** Does using 4 different LLMs produce higher-quality outputs than using a single LLM for all roles?
3. **Adversarial vs cooperative:** How do rubric scores compare between `mode=standard` and `mode=adversarial` on the same inputs?
4. **Role ablation:** Which single agent contributes most to final output quality improvement?
5. **Validator grounding:** Does the Validator's large-context review catch errors that the Critic misses?

---

## Problems Encountered

### Problem 1: Bus Role Namespace Collision
**Symptom:** Both the standard pipeline and adversarial pipeline want to register a `"critic"` agent on the same `AgentBus` singleton. The second `register()` call overwrites the first.
**Solution:** Prefix all adversarial role strings with `adv:` (e.g., `"adv:critic"`). The `AgentBus._agents` dict is keyed by arbitrary strings — the prefix acts as a namespace. Standard pipeline uses `"critic"`, adversarial uses `"adv:critic"`. No bus modification needed; they coexist cleanly.
**Research note:** This reveals a design tension in shared-singleton agent registries: global mutability creates implicit coupling between pipeline configurations. A more robust solution would be separate bus instances per pipeline mode, but the prefix approach minimizes complexity for a research prototype.

### Problem 2: Judge Output Parsing Brittleness
**Symptom:** LLMs frequently wrap JSON in markdown code fences (````json ... ````), which breaks `json.loads()`.
**Solution:** Two-stage parser in `_parse_judge_output()`:
1. Strip markdown fences with regex: `re.sub(r"^```(?:json)?\s*|\s*```$", "", content, flags=re.MULTILINE)`
2. Fallback: `re.search(r"\{[\s\S]*\}", content)` finds any embedded JSON object
3. Final fallback: return a `REJECT` verdict to trigger another round (fail-safe, not fail-hard)
**Research note:** Structured output reliability is an open problem in multi-agent LLM systems. The Judge's JSON format requirement is the most fragile contract in the pipeline. Future work: use OpenAI/Anthropic structured output APIs (`response_format: json_object`) to guarantee parseable output.

### Problem 3: Token Budget per Role
**Symptom:** Using a uniform `max_tokens=2048` for all roles wastes credits on low-output roles (Refiner needs ~200 tokens, not 2048).
**Solution:** `_TOKEN_BUDGETS` dict in `adversarial.py` maps each role to a calibrated ceiling:
- Actor: 2048 (needs to produce complete solutions)
- Critic/Validator: 1024 (structured list outputs)
- Planner: 600 (4-section plan)
- Refiner: 512 (surgical specification only)
- Judge: 1024 (JSON + final answer)
**Research note:** Token budgets function as implicit constraints on agent verbosity. Refiner's 512-token cap prevents it from rewriting the Actor's answer (which would blur role boundaries). This is an architectural choice: tight token budgets enforce role discipline.

### Problem 4: Environment Dependencies
**Symptom:** `pydantic_settings`, `structlog`, `fastapi`, etc. not installed in bare Python interpreter during development validation. `pyproject.toml` only lists `anthropic` and `openai` as dependencies.
**Root cause:** The `pyproject.toml` is incomplete — the existing codebase already used `pydantic-settings` and `structlog` before this implementation. This is a pre-existing project setup gap, not introduced by the adversarial implementation.
**Solution (not implemented — out of scope):** Update `pyproject.toml` to list all runtime dependencies. This would be flagged for a separate task.
**Validation used instead:** Python `ast.parse()` on all 8 modified/new files (all passed). Logic verified through code review.

---

## Key References

- GAN paper (Goodfellow et al. 2014) — structural analogy for adversarial generator/discriminator
- Constitutional AI (Bäuerle et al.) — adversarial self-critique in LLM alignment
- Debate as alignment (Irving et al. 2018) — multi-agent debate for AI safety
- ReAct (Yao et al. 2022) — tool-augmented reasoning chains (related to Validator grounding)

---
---

# Phase 2 — Multi-User Platform Foundation (Auth + Providers + Memory)

**Goal:** Transform the single-user, file-based prototype into a multi-tenant,
privacy-isolated platform: per-user provider connections (OAuth + BYOK), encrypted
credentials, and persistent cross-session semantic memory. Backed by Supabase
(Postgres 17 + pgvector + Auth + RLS). Guardrails and UI deferred to later phases.

## Architecture Decisions

### Why Supabase / pgvector
The prototype stored everything in global JSON/JSONL files with a single shared
OpenRouter key — no users, no isolation. Supabase provides four needs in one stack:
(1) Auth (JWT), (2) Postgres for relational data, (3) pgvector for the semantic
memory that already existed in spirit (`knowledge_engine.get_relevant_context` did
cosine search over a NumPy matrix), and (4) Row-Level Security for per-user privacy.
The existing 384-d MiniLM embeddings map directly to `vector(384)` columns, so no
re-embedding was needed.

### Memory vs. Graph split
Rather than rip out the existing knowledge engine, we split responsibilities:
- **Per-user recall → Supabase `memory_chunks`** (RLS-scoped pgvector cosine search).
- **Global 3D graph viz → file-based `knowledge_engine`** (unchanged) so the
  "Knowledge Universe" keeps working.
Both lifecycles now branch on a `multiuser` flag: Supabase memory when authenticated,
file engine otherwise. This preserves a fully working legacy path behind `USE_SUPABASE`.

### Index choice: HNSW over IVFFlat
Chose HNSW (`vector_cosine_ops`) — Supabase's recommended default; better recall/latency
and, crucially, it needs no pre-population (IVFFlat requires training data to build
centroids, awkward for a brand-new per-user table). Embeddings are L2-normalized, so
cosine and inner-product rank identically; cosine chosen for interpretable [0,1] scores.

### Encryption: app-level AES-256-GCM (not pgsodium/Vault)
Provider API keys are encrypted in the FastAPI layer before they touch Postgres, with
the 32-byte key living only in `APP_ENCRYPTION_KEY` (env). Rationale: ciphertext stays
opaque to the database, so a service-role leak or RLS slip exposes only ciphertext.
pgsodium's client-side TCE is being deprecated; Vault suits app-wide secrets, not
per-row user secrets we must decrypt in Python on every call. Layout: base64(nonce(12)
|| ciphertext || GCM tag(16)). Verified: round-trip correct, ciphertext opaque, and
GCM auth-tag rejects single-bit tampering.

### Per-user router via TaskContext (the key refactor)
The hardest design problem: agents received a global `ModelRouter` singleton **at
registration time** (`self.router`), but each request must use the *caller's* credential.
Three solutions considered:
1. ContextVar — implicit, leak-prone across concurrent async tasks, hard to test.
2. Rewrite every agent's constructor / DI container — invasive.
3. **Carry the per-user router on `TaskContext.metadata["router"]`** ✓ — already plumbed
   through every agent and both lifecycles.
Chose (3). The entire agent change is one line in two base classes:
`router = ctx.metadata.get("router") or self.router`. Zero changes to the six agent
classes. `rubric.py` (which built its own client) now takes `router=` too, so scoring
spends the same user's credentials.

### Provider auth reality
Real "log in to OpenAI/Anthropic and authorize inference" OAuth does not exist — those
are API-key-only. The authentic login flow that *does* work is **OpenRouter OAuth PKCE**,
which the project already routes through. Implemented PKCE (S256) start/callback storing
a user-scoped key. BYOK paste-keys cover direct Anthropic/OpenAI/Google access. All keys
encrypted. Documented limitation: per-role multi-vendor routing is OpenRouter-only;
direct BYOK keys use a single provider-default model (OpenRouter slugs like
`anthropic/claude-...` aren't valid on a native OpenAI/Anthropic endpoint).

## Schema (8 tables, all RLS owner-only)
profiles · provider_connections · oauth_states · chat_sessions · chat_messages
· adversarial_runs · adversarial_messages · memory_chunks. Semantic search via SQL
function `match_memory_chunks(p_user_id, query_embedding, match_count)` which filters
by user **before** the ANN order-by so RLS + the index cooperate.

## Problems Encountered (Phase 2)

### P2-1: bytea over PostgREST is painful
Storing AES ciphertext as `bytea` forces hex-encoding gymnastics over PostgREST/supabase-py
(input `\xDEADBEEF`, ambiguous output format). **Fix:** changed `secret_enc` to `text`
holding base64 (migration 07). Security is identical — it's ciphertext either way — and
JSON transport is trivial. Lesson: pick column types for your access path, not just the
data's nature.

### P2-2: pgvector lives in the `extensions` schema
With `create extension vector with schema extensions`, the type is `extensions.vector`
and the `<=>` operator isn't on the default search_path inside a hardened
`security definer`/`set search_path=''` function. **Fix:** fully-qualified the type
(`extensions.vector(384)`) and used `OPERATOR(extensions.<=>)` in the match function. The
HNSW opclass is also qualified (`extensions.vector_cosine_ops`).

### P2-3: passing a vector through PostgREST RPC/insert
JSON arrays don't reliably cast to `vector`. **Fix:** a `vector_literal()` helper formats
embeddings as the pgvector text form `"[0.1,0.2,...]"`, which PostgREST casts cleanly on
both insert and RPC. Verified live: `match_memory_chunks` with a 384-d zero vector executes
and returns 0 rows (no cast error).

### P2-4: SECURITY DEFINER trigger flagged by the linter
`handle_new_user()` (the auto-profile trigger) is `security definer` and lived in `public`,
so Supabase's advisor flagged it as callable via `/rest/v1/rpc/handle_new_user` by anon &
authenticated roles. **Fix (migration 06):** revoked EXECUTE from public/anon/authenticated;
the trigger still fires (it runs as table owner). Security advisors then returned **zero**
findings.

### P2-5: conditional auth without two route trees
`/run` must stay open in legacy mode but require auth in multi-user mode. **Fix:** a
`get_optional_user` dependency that returns None when `USE_SUPABASE=false` and enforces
`get_current_user` otherwise — one route, both modes.

### P2-6: incomplete pyproject dependencies (carried from Phase 1, P4)
`pyproject.toml` only declared `anthropic, openai` though the code used fastapi, pydantic,
structlog, sentence-transformers, faiss, numpy. Added those plus the new platform deps
(`supabase, httpx, pyjwt[crypto], cryptography`). Verification used lightweight installs
(`cryptography`, `pydantic-settings`) to test the security-critical paths without pulling
the heavy ML stack — deliberate, to conserve resources.

## Verification performed
- 8 tables created, RLS enabled on all; security advisors: **0 findings** after hardening.
- `match_memory_chunks` executes live; 384-d vector cast confirmed.
- AES-256-GCM: round-trip correct, ciphertext opaque, tamper rejected by auth tag.
- Config: new settings load; JWKS URL auto-derives; `USE_SUPABASE` defaults to False
  (legacy path intact). All 18 new/modified files pass `ast.parse`.

## Open Questions for Paper (Phase 2)
1. Privacy framing: semantic search requires server-readable plaintext to embed — so this
   is "encryption at rest + RLS isolation," not E2E. How to communicate that honestly?
2. Does cross-session memory measurably improve answer quality/consistency vs. stateless?
3. Multi-provider routing: does letting users mix providers per role (OpenRouter) change
   adversarial dynamics vs. a single provider?
4. Service-role-bypasses-RLS: defense-in-depth analysis — app-layer filter + RLS backstop.
