# Arth evaluation report

Gold set: `/home/user/arth/data/gold/proxy_gold.jsonl`, 420 lines (178 calibration / 242 eval, split by bank). Embedder: `hash`. Teacher: `none:generator-labels+smoothing`.

## Gates

| Gate | Pass rule | seed 0 | seed 1 | seed 2 |
| --- | --- | --- | --- | --- |
| accuracy | Macro-F1 beats every baseline on the gold eval split | pass | pass | pass |
| per_slice | Every slice with 30+ items within 2 points of the teacher | skipped | skipped | skipped |
| certified_coverage | 70%+ auto-labelled at 98% precision, 95% confidence | fail | fail | fail |
| calibration | Expected calibration error <= 0.05 | fail | pass | pass |
| generalization | Held-out taxonomy accuracy within 5 points of in-taxonomy accuracy | fail | fail | fail |
| synthetic_gap | Synthetic test accuracy minus gold accuracy under 10 points | pass | fail | pass |
| latency | p95 < 50 ms per line on CPU; batch of 1,000 < 10 s | pass | pass | pass |
| device | ONNX INT8 <= 40 MB and < 30 ms on a budget Android phone | skipped | skipped | skipped |
| stability | No gate flips between seeds | fail | fail | fail |

## Headline numbers (primary taxonomy: personal_finance)

| Seed | Macro-F1 | Accuracy | ECE | Certified coverage (eval) | Auto precision (eval) | Threshold |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 0.870 | 0.855 | 0.050 | 0.0% | - | None |
| 1 | 0.862 | 0.847 | 0.030 | 0.0% | - | None |
| 2 | 0.890 | 0.880 | 0.048 | 0.0% | - | None |

## Baselines vs student (seed 0, macro-F1 on gold eval)

| Model | Macro-F1 | Accuracy |
| --- | --- | --- |
| regex_rules | 0.744 | 0.740 |
| embedding_zero_shot | 0.528 | 0.496 |
| llm_zero_shot | not run |  |
| banking_sms_json_parser_v8 | not run |  |
| student: fast path only | 0.857 | 0.851 |
| student: general only | 0.854 | 0.847 |
| **student: cascade** | **0.870** | **0.855** |

## All built-in questions (seed 0)

| Question | Type | Cascade acc | Macro-F1 | ECE | Cert. threshold | Coverage (eval) | Auto precision |
| --- | --- | --- | --- | --- | --- | --- | --- |
| personal_finance | choice | 0.855 | 0.870 | 0.050 | None | 0.0% | - |
| lender_flags | choice | 0.880 | 0.855 | 0.045 | None | 0.0% | - |
| coarse | choice | 0.926 | 0.888 | 0.051 | 0.99 | 87.6% | 0.972 |
| is_salary | bool | 0.975 | 0.744 | 0.019 | 0.975 | 93.4% | 1.000 |
| is_emi | bool | 0.983 | 0.885 | 0.010 | 0.78 | 100.0% | 0.983 |
| is_p2p | bool | 0.950 | 0.836 | 0.026 | 0.925 | 90.9% | 0.982 |
| is_refund | bool | 0.996 | 0.953 | 0.006 | 0.96 | 96.7% | 0.996 |
| is_bounce | bool | 0.988 | 0.830 | 0.007 | 0.5 | 100.0% | 0.988 |
| is_investment | bool | 0.988 | 0.918 | 0.007 | 0.5 | 100.0% | 0.988 |
| is_cash | bool | 0.992 | 0.956 | 0.003 | 0.5 | 100.0% | 0.992 |
| is_tax | bool | 0.992 | 0.873 | 0.003 | 0.5 | 100.0% | 0.992 |
| gst_ledger | choice | 0.574 | 0.670 | 0.060 | None | 0.0% | - |
| is_rent | bool | 0.992 | 0.942 | 0.019 | 0.98 | 87.6% | 0.995 |
| is_business_income | bool | 0.934 | 0.760 | 0.022 | 0.915 | 86.4% | 0.986 |

## Gate details (seed 0)

- **accuracy**: pass — {"student_macro_f1": 0.8696368342132131, "baselines": {"regex_rules": 0.7436833303385243, "embedding_zero_shot": 0.5281178813589804, "llm_zero_shot": null, "banking_sms_json_parser_v8": null}, "not_run": ["llm_zero_shot", "banking_sms_json_parser_v8"]}
- **per_slice**: skipped — {"reason": "no teacher configured (set HF_TOKEN)", "worst_gated_slice": {"dimension": "merchant_known", "slice": "unseen", "n": 138, "macro_f1": 0.7129642419144137, "accuracy": 0.782608695652174, "gated": true}}
- **certified_coverage**: fail — {"coverage_eval": 0.0, "precision_eval": null, "certificate": {"threshold": null, "target_precision": 0.98, "alpha": 0.05, "n_calibration": 178, "n_above": 0, "precision_above": null, "lower_bound": null, "coverage": 0.0}}
- **calibration**: fail — {"ece": 0.05041448525653392}
- **generalization**: fail — {"in_taxonomy_accuracy": 0.8471074380165289, "heldout_taxonomy": "gst_ledger", "heldout_accuracy": 0.5743801652892562, "gap": 0.2727272727272727}
- **synthetic_gap**: pass — {"synthetic_accuracy": 0.945, "gold_accuracy": 0.8553719008264463, "gap": 0.08962809917355363}
- **latency**: pass — {"p50_ms": 22.4918725000407, "p95_ms": 28.971173349896155, "batch_1000_s": 1.328605540999888, "embedder": "hash"}
- **device**: skipped — {"reason": "ONNX INT8 export and Android SDK are V1.5 (FR-14)"}

## Slices (seed 0, primary taxonomy)

**bank**: AXIS 0.88 (n=36), BOB 0.80 (n=24), CANARA 0.80 (n=25), FEDERAL 0.85 (n=18), IDFC 0.92 (n=23), KOTAK 0.77 (n=36), PAYTMPB 0.59 (n=17), PNB 0.58 (n=21), SBI 0.91 (n=42)

**channel**: ATM 0.63 (n=10), BILLPAY 0.70 (n=5), CARD 0.96 (n=8), CASH 1.00 (n=6), CHARGES 1.00 (n=3), CHEQUE 1.00 (n=3), IMPS 0.78 (n=8), INTEREST 1.00 (n=3), NACH 0.96 (n=19), NEFT 0.93 (n=12), NETBANKING 0.63 (n=7), OTHER 1.00 (n=20), RTGS 1.00 (n=2), UPI 0.77 (n=136)

**merchant_known**: known 0.92 (n=104), unseen 0.71 (n=138)

**p2p_vs_merchant**: merchant_or_other 0.86 (n=202), p2p 0.91 (n=40)
