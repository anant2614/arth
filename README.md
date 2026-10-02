# Arth

Indian transaction categorization API. Arth reads messy Indian transaction text and turns it into typed answers with calibrated probabilities. It handles bank SMS alerts, statement narrations, UPI strings and Account Aggregator descriptions. Answers use any taxonomy the caller sends. Each taxonomy gets a **certified auto-label threshold**: answers above it carry a 98% precision, 95% confidence guarantee. Everything else goes to a review queue with the top 3 guesses.

This repository implements the V1 scope of the PRD ("Indian Transaction Categorization API", Oct 2026). It includes the runtime cascade, the training pipeline, the evaluation harness with every launch gate, and a test suite that runs evals as tests.

```
request ─► extract (regex) ─► scrub PII ─► render + history tokens ─► embed
                                                                        │
     correction memory ─► rules (gazetteer) ─► fast path ─► general ─► ensemble ─► teacher LLM
     (reviewer labels)    Tier 0              Tier 1a       Tier 1b     (1a+1b)     Tier 2: OOD, 2% audit,
                                                                                    unsure lines, alerts
     first tier whose calibrated confidence clears the taxonomy's certified threshold answers (auto = true)
```

## Quick start

```bash
pip install -e ".[dev,serve]"          # add ".[onnx]" for the bge-small embedder

python -m arth.train --pool-size 20000  # synthetic pool -> students -> calibration -> models/v1
uvicorn arth.api:app                    # ARTH_MODELS_DIR=models ARTH_DB=arth.sqlite by default

pytest -q                               # ~4 min: unit, API, behaviour evals, quality floors
python -m arth.evals.run --seeds 0 1 2  # full launch-gate report -> runs/eval/report.md
python -m arth.evals.behaviour          # named behaviour cases against the current model
```

Set `HF_TOKEN` to switch on the teacher LLM (Tier 2): fallback, the 2% audit slice, teacher labelling (`python -m arth.train --teacher hf`), the zero-shot LLM baseline and the per-slice gate (`python -m arth.evals.run --teacher hf`). The default teacher is `Qwen/Qwen3-235B-A22B-Instruct-2507` via Hugging Face Inference Providers. Override it with `ARTH_TEACHER_MODEL`.

## API

`POST /v1/categorize` takes single or batch requests (up to 1,000 lines) and uses the PRD's request and response shape:

```bash
curl -s localhost:8000/v1/categorize -H 'content-type: application/json' -d '{
  "transactions": [{"id": "t1", "text": "IMPS/P2A/402166/RAMESH K/rent oct", "amount": 18000,
                    "direction": "debit", "history": {"same_counterparty_count": 6, "amount_cv": 0.02, "median_gap_days": 30}}],
  "questions": [{"name": "category", "type": "choice",
                 "options": {"rent": "Monthly house rent", "p2p_transfer": "One-off transfer to a person", "emi": "Fixed loan instalment"}},
                {"name": "is_salary", "type": "bool"}]}'
```

```json
{"results": [{"id": "t1",
  "extracted": {"channel": "IMPS", "direction": "debit", "counterparty_type": "person", "merchant": null, "amount": null, ...},
  "answers": {"category": {"value": "rent", "probs": {"rent": 0.9989, "p2p_transfer": 0.001, "emi": 0.0001}, "auto": false, "route": "general"},
              "is_salary": {"value": false, "prob": 0.0005, "auto": true, "route": "general"}},
  "route": "student", "ood": false, "model_version": "student:v1",
  "review": {"category": "rev_c7a6aea549a74fc5"}}]}
```

`extracted` is what the regex layer read from the text itself (FR-2). Fields the caller sent in the request are not copied into it. `auto` is true only when the answer clears that question's certified threshold. Non-auto answers are queued for review, and `review` gives their queue ids. An unsaved, ad-hoc taxonomy uses the *generic* certificate. That certificate is measured zero-shot on held-out taxonomies, and today it certifies nothing for multi-class questions. To get auto-labels, save the taxonomy with a labelled sample:

| Endpoint | Purpose | PRD |
| --- | --- | --- |
| `POST /v1/categorize` | Single or batch (≤1,000) against `questions`, a `taxonomy_id`, or both | FR-1, 3, 4, 6 |
| `POST /v1/taxonomies` | Save a taxonomy and certify its thresholds on your labelled sample | FR-5 |
| `GET /v1/taxonomies[/{id}]` | Saved and built-in (`builtin:personal_finance`, …) taxonomies with certificates | FR-5 |
| `POST /v1/taxonomies/{id}/recertify` | Retrain the fast path with reviewer corrections and re-certify | FR-8 |
| `GET /v1/review` | Review queue: scrubbed line, top 3 options, route | FR-7 |
| `POST /v1/corrections` | Label a queued item (or a transaction). It applies at once to the same counterparty | FR-8 |
| `GET /v1/corrections/export` | Scrubbed corrections for the next training cycle | FR-8 |
| `GET /v1/models`, `POST /v1/models/rollback`, `POST /v1/models/promote` | `student:vN` registry with one-step rollback | FR-11 |
| `GET /v1/health` | Model version, teacher, audit agreement and alert state | FR-10 |

Validation errors return 422 with a message naming the field. Limits: 255 options per question, 20 questions per request, 1,000 lines per batch. Set `ARTH_API_KEYS=k1,k2` to require `X-API-Key`.

## What is implemented

| PRD item | Where | Status |
| --- | --- | --- |
| FR-1 single + batch ≤1,000 | `api.py` | done |
| FR-2 regex extraction of amount, balance, date, account suffix, channel; never the model | `extract.py` | done (UPI, IMPS, NEFT, RTGS, NACH/ACH/ECS, ATM, card, BBPS, cheque, cash, interest, charges) |
| FR-3 choice (≤255 described options) and yes/no questions | `questions.py`, `students/general.py` | done |
| FR-4 probability per option, top value, `auto` | `engine.py` | done |
| FR-5 saved taxonomy + certified threshold on the caller's sample | `engine.fit_saved_taxonomy`, `certify.py` | done |
| FR-6 rules → gazetteer → student → teacher, route reported | `engine.py` | done |
| FR-7 review queue with top 3 | `store.py`, `api.py` | done |
| FR-8 corrections feed relabelling and training | correction memory, fast-path retrain, export | done |
| FR-9 kNN out-of-distribution check → fallback | `ood.py` | done |
| FR-10 2% audit slice, agreement alert, automatic fallback | `engine.AuditMonitor` | done (needs `HF_TOKEN` to have a teacher to audit against) |
| FR-11 `student:vN` + rollback | `registry.py` | done |
| FR-13 history features from an account's lines | `features.compute_history` | library function (P2; callers send history in V1) |
| FR-12 score questions, FR-14 ONNX/Android, FR-15 lender pack | — | not started (P2) |
| Teacher LLM, constrained JSON, logprobs or 3–5 votes, calibration head | `teacher.py` | done, tested against a mocked router; not run live (no `HF_TOKEN` here) |
| PII scrubber before storage, logs and teacher | `scrub.py` | done |
| No raw text in logs, hashed ids, 30-day retention | `api.py`, `store.purge_expired` | done |

### Models

- **Embedder.** `hash` is char and word n-gram hashing: no download, deterministic, and the default for CI. `bge-small` runs as quantised ONNX on CPU (`pip install -e ".[onnx]"`, `--embedder hash+bge-small`).
- **Tier 1a fast path.** Softmax regression on frozen embeddings, trained against soft labels. There is one per taxonomy. Saved taxonomies get one distilled from the general student's soft labels on the pool.
- **Tier 1b general student.** A bilinear dual encoder: score(line, option) = s·⟨x,o⟩ + ⟨xA, oB⟩ + ⟨o,w⟩. Options enter only through their name and description, so any taxonomy works. It is trained on the built-in taxonomies plus 60 random regroupings with paraphrased descriptions, so it learns meaning rather than one label list. The PRD's production plan for this tier is GLiClass-modern-base. This model has the same contract and serves as the bar GLiClass must clear.
- **Ensemble.** The mean of the two calibrated students. It answers whatever no cheaper tier certified.
- Every tier is temperature-scaled on the calibration split. **One shared threshold per question** is certified for the whole cascade with Clopper–Pearson, strictest-first (fixed-sequence testing). See `certify.certify_cascade`.

### Data

- `arth/synth.py` holds the synthetic pool, used for training only. It covers bank SMS phrasing for 15 banks, statement and UPI/IMPS/NEFT/NACH/card/ATM/BBPS layouts, AA-style descriptions, about 250 gazetteer merchants plus generated local businesses, people from an Indian name lexicon, Hinglish remarks, history features, and deliberately ambiguous rent vs P2P lines.
- `data/gold/proxy_gold.jsonl` holds **420 hand-written, hand-labelled lines. This is a stand-in, not real data.** A separate author wrote them without seeing the generator, and they use more varied formats and more lesser-known merchants. The set is split **by bank** into calibration (178) and eval (242) sets. Gold labels never enter training. Replace this file with the real, scrubbed, consented gold set (PRD: about 1,000 lines) once it exists.

## Evaluation

`python -m arth.evals.run` trains one bundle per seed and scores it on the gold eval split. It checks every PRD gate. A gate that can't run here is reported as `skipped` with the reason, never passed silently. The test suite (`tests/test_evals.py`) also asserts quality floors and 25 named behaviour cases (`arth/evals/behaviour.py`) on every run, so regressions fail CI.

RESULTS_PLACEHOLDER

## Layout

```
arth/
  extract.py      FR-2 regex extraction, counterparty typing
  scrub.py        PII scrubber, keyed hashing
  gazetteer.py    merchant/biller/VPA gazetteer (data/gazetteer.json)
  features.py     student input rendering, history tokens, FR-13
  synth.py        synthetic data pool
  concepts.py     32 concepts; built-in taxonomies as views over them
  embed.py        hashing and bge-small (ONNX) embedders
  students/       fast path (Tier 1a), general student (Tier 1b)
  teacher.py      HF Inference Providers teacher, oracle teacher for tests
  calibration.py  temperature scaling, ECE, reliability
  certify.py      Clopper–Pearson certification, cascade routing
  ood.py          kNN out-of-distribution detector
  engine.py       runtime cascade, audit monitor, saved-taxonomy fitting
  train.py        training pipeline CLI
  registry.py     student:vN registry with rollback
  store.py        SQLite: taxonomies, review queue, corrections, retention
  api.py          FastAPI service
  evals/          gate harness, baselines, metrics, behaviour cases
data/gold/        proxy gold set
tests/            unit, API, privacy, and eval tests
```
