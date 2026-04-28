# Phase 1 — Week 1 Task Card

**Stage**: Phase 1, Month 1, Week 1
**Goal**: Repo bootstrapped, data pipelines unblocked, outreach in motion.
**End-of-week DoD**: All five P0 items below are checked.

---

## P0 (must-finish-this-week)

### ☐ 1. Bootstrap the repo (Step 1.1) — half day

Drop this `brain-tumor-agent/` skeleton at `D:/brain-tumor-agent/` (or anywhere outside `D:/M3D/`).

```powershell
cd D:\brain-tumor-agent
git init
git add . && git commit -m "chore: bootstrap project skeleton"

uv venv --python 3.11
.\.venv\Scripts\activate
uv pip install -e ".[dev]"

python -c "import src; print(src.__version__)"   # → 0.1.0
pytest tests/ -q                                   # → 3 passed
```

**DoD**:
- [ ] Repo lives under git
- [ ] `uv pip install -e ".[dev]"` succeeds
- [ ] `pytest tests/ -q` shows 3 passed
- [ ] `.env` created from `.env.example` with at least `BRATS_DIR` and `KAIST_PRED_DIR` filled

### ☐ 2. Kick off UPenn-GBM download — start today, runs overnight

UPenn-GBM is public on TCIA (CC-BY). No application needed.

```powershell
# Install nbia-data-retriever (Windows/Mac/Linux installer at:
# https://wiki.cancerimagingarchive.net/display/NBIA/Downloading+TCIA+Images)

# Download the manifest from:
# https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=119705830
# Click "Download" → save the .tcia manifest

nbia-data-retriever <path-to-upenn-gbm.tcia>
# → ~80 GB, leave running overnight
```

Same drill for UCSF-PDGM (~50 GB).

**DoD**:
- [ ] Manifests downloaded and queued
- [ ] At least 200 GB free on the destination drive
- [ ] Download visibly progressing (nbia shows MB/s and ETA)

### ☐ 3. Send two cold emails — 30 min each

Templates are in `playbook_updates.md` (the originals you wrote earlier). Phase 0 numbers to drop in:

- KAIST nnU-Net achieves Dice 0.956 (WT) / 0.931 (TC) / 0.811 (ET) on BraTS 2021 (n=500)
- Qwen3-14B + KAIST features → 96% diagnosis accuracy on BraTS 2021, statistically equivalent to Claude Opus (94%)
- 4B → 14B capability threshold confirmed via permutation control

**Targets**:
- Zongwei Zhou (JHU, R-Super) — collaboration probe
- Silas Ruhrberg Estévez (Cambridge, ACTMED) — methodology probe

**DoD**:
- [ ] Both emails sent
- [ ] Saved a copy in `docs/correspondence/` (gitignored)

---

## P1 (start this week, finish Week 2)

### ☐ 4. Decide your training compute path

You have two practical options:

| Option | Pros | Cons |
|---|---|---|
| Local 24GB GPU (RTX 4090/3090 or similar) | No rental cost, fastest iteration | vLLM is Linux-first → use WSL2 |
| Cloud (Lambda/RunPod, A100 40GB) | No setup pain, predictable | $1-2/hr × hundreds of hours |

**Action**: pick one, write the choice into `docs/decisions.md` with the date.

**DoD**:
- [ ] Decision recorded
- [ ] If WSL2: install Ubuntu 22.04 + CUDA toolkit
- [ ] If cloud: account created, $20 test credit added

### ☐ 5. Smoke-test Qwen3-14B + transformers (no vLLM yet)

You already did this in Phase 0 (46 s/case). Re-run it from inside the new repo to confirm the new env works:

```powershell
# In activated .venv
python scripts/smoke_qwen3.py --case BraTS2021_00000
# → expect single case to print a diagnosis JSON, ~50 s
```

Add `scripts/smoke_qwen3.py` (port your Phase 0 script). This becomes the
target for vLLM optimization in Week 2-3.

**DoD**:
- [ ] One case runs end-to-end through Qwen3-14B 4-bit
- [ ] Latency logged in `experiments/logs/smoke_qwen3.log`

---

## P2 (read-but-don't-do)

- Skim the vLLM docs at `https://docs.vllm.ai/`. You'll need it next week (Step 1.3).
- Read the M3D-LaMed paper once more — useful background, but the model is not on the critical path anymore.

---

## Time budget

Realistic budget for the week:
- Bootstrap + tests: 4 h
- Data downloads (mostly waiting): 1 h interactive setup
- Two cold emails: 1 h
- Compute decision + WSL2 install (if going local): 4-6 h
- Smoke test: 2-3 h
- Buffer + reading: 4 h

Total: ~15-20 h of focused time. Leave at least one full day off.

---

## What "done" looks like at end of week

A 90-second status update should sound like:
> "Repo bootstrapped, tests passing. UPenn-GBM downloading overnight, 60% done.
> Two emails out. Compute path is local WSL2; smoke test of Qwen3-14B reproduces
> the 96% from Phase 0 inside the new env. Next week: vLLM and the first three
> tool wrappers."

If you can say that on Friday, Week 1 is a success. If two of those five items
are missing, **don't push to Week 2** — fix them first.
