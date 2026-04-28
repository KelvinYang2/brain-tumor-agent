# Architectural & Process Decisions

A running log. Append, never edit. Every entry: date, what was decided, why.

---

## 2026-04-28 — Use Qwen3-14B as policy LLM (not MedGemma-27B / not Claude API)

**Decision**: All Phase 1 reasoning runs on `unsloth/Qwen3-14B-unsloth-bnb-4bit`
served locally via vLLM.

**Rationale**: Phase 0 finding 6 — Qwen3-14B + KAIST features achieves 96%
diagnosis accuracy on BraTS 2021, statistically equivalent to Claude Opus
(94%, n=50). MedGemma 4B drops to 26% under permutation control. The 14B
threshold isolates the architectural question: at this scale local inference
is enough, frontier API is not required, and per-case cost is ~1000× lower.

**Consequence**: vLLM integration becomes critical-path (Step 1.3). Windows
users go via WSL2.

---

## 2026-04-28 — Tool count reduced from 7 to 5 for Phase 1

**Decision**: Phase 1 implements only `query_segment`, `query_features`,
`query_molecular`, `query_demographics`, `query_history` (+ the terminal
`commit_diagnosis`). Deferred: `query_imaging_quality`, `query_similar_cases`.

**Rationale**: Phase 0 finding 3 — structured features alone match
features+image performance (+2pp from images is within noise). Imaging
quality and case retrieval are nice-to-have, not blocking.

**Consequence**: Saves ~1 week. Reintroduce in Phase 2 if Pareto frontier
plateaus.

---

## 2026-04-28 — Primary evaluation set: UPenn-GBM (not BraTS)

**Decision**: BraTS 2021 demoted to sanity-check (it was Phase 0 dev set);
UPenn-GBM is the Phase 1 primary; UCSF-PDGM reserved for OOD evaluation.

**Rationale**: BraTS labels are GBM-only and partially seen. UPenn-GBM has
mixed grades + IDH/MGMT subtypes + clinical metadata, which the agent
actually needs to reason over. UCSF-PDGM is a different institution → real
OOD test.

**Consequence**: UPenn-GBM download is on the critical path for Week 1.

---

<!-- Next entries go below this line -->
