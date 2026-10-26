### Track A — dilemma panel: between-persona separation and convergence (validation items)

| model | separation SD at base (null) | auto ratio [CI] | human ratio [CI] | neutral ratio [CI] |
|---|---|---|---|---|
| Qwen3.6 | 0.25 (0.15) | 0.92 [0.77, 0.99] | 0.92 [0.77, 0.98] | 0.80 [0.68, 0.87] |
| Qwen3.8 | 0.23 (0.14) | 0.96 [0.78, 1.14] | 0.91 [0.79, 1.14] | 0.99 [0.80, 1.29] |
| OLMo DPO | 0.15 (0.11) | 1.00 [0.78, 1.26] | 0.78 [0.62, 1.09] | 1.07 [0.80, 1.26] |
| OLMo final | 0.18 (0.13) | 0.93 [0.79, 1.11] | 0.90 [0.76, 1.07] | 1.01 [0.82, 1.14] |

Per dimension (auto / base mean |p − p_default| ratio, validation):

| model | risk | self_others | honesty | rules_outcomes | loyalty_fairness | autonomy_deference |
|---|---|---|---|---|---|---|
| Qwen3.6 | 1.21 [0.66, 1.85] | 0.94 [0.63, 1.06] | 0.88 [0.79, 0.95] | 0.88 [0.55, 1.06] | 0.74 [0.68, 0.80] | 1.00 [0.68, 1.14] |
| Qwen3.8 | 1.46 [0.58, 2.21] | 1.12 [0.88, 1.18] | 0.83 [0.64, 1.00] | 0.95 [0.68, 1.31] | 0.78 [0.58, 0.93] | 0.74 [0.61, 1.43] |
| OLMo DPO | 0.65 [0.42, 1.50] | 1.42 [0.57, 1.53] | 0.78 [0.65, 1.60] | 1.68 [0.71, 2.10] | 0.93 [0.81, 1.05] | 0.84 [0.48, 1.53] |
| OLMo final | 1.00 [0.45, 1.59] | 0.95 [0.87, 1.00] | 0.86 [0.68, 1.21] | 0.62 [0.38, 1.33] | 1.00 [0.87, 1.32] | 1.09 [0.67, 1.45] |

### Track A — landscape (dilemma prompts, layer 36): activation spread vs base, Mantel ρ (p)

| model | base | auto | human | neutral |
|---|---|---|---|---|
| Qwen3.6 | spread 1.00; ρ 0.31 (p 0.021) | spread 0.93; ρ 0.25 (p 0.069) | spread 0.94; ρ 0.26 (p 0.050) | spread 1.03; ρ 0.20 (p 0.102) |
| Qwen3.8 | spread 1.00; ρ 0.12 (p 0.221) | spread 0.93; ρ 0.13 (p 0.211) | spread 0.95; ρ 0.09 (p 0.280) | spread 0.98; ρ 0.08 (p 0.295) |
| OLMo DPO | spread 1.00; ρ 0.26 (p 0.099) | spread 0.98; ρ 0.31 (p 0.056) | spread 0.96; ρ 0.40 (p 0.023) | spread 1.12; ρ 0.33 (p 0.050) |
| OLMo final | spread 1.00; ρ 0.35 (p 0.039) | spread 0.96; ρ 0.42 (p 0.022) | spread 0.94; ρ 0.45 (p 0.012) | spread 1.09; ρ 0.34 (p 0.042) |

### Reporting task (secondary): misreport rate by pressure × context; refusals

| model | context | mild | strong | explicit | suggest | between-persona SD (null) |
|---|---|---|---|---|---|---|
| Qwen3.6 | base | 0.03 | 0.06 | 0.14 | 0.01 | 0.10 (null 0.03) |
| Qwen3.6 | auto | 0.01 | 0.03 | 0.16 | 0.01 | 0.05 (null 0.03) |
| Qwen3.6 | neutral | 0.01 | 0.03 | 0.07 | 0.01 | 0.05 (null 0.02) |
| Qwen3.8 | base | 0.00 | 0.00 | 0.01 | 0.00 | 0.01 (null 0.01) |
| Qwen3.8 | auto | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 (null 0.00) |
| Qwen3.8 | neutral | 0.00 | 0.00 | 0.01 | 0.00 | 0.01 (null 0.01) |
| OLMo DPO | base | 0.00 | 0.02 | 0.96 | 0.38 | 0.07 (null 0.03) |
| OLMo DPO | auto | 0.00 | 0.03 | 0.97 | 0.42 | 0.08 (null 0.03) |
| OLMo DPO | neutral | 0.00 | 0.03 | 0.97 | 0.24 | 0.06 (null 0.03) |
| OLMo final | base | 0.00 | 0.04 | 0.96 | 0.38 | 0.08 (null 0.03) |
| OLMo final | auto | 0.00 | 0.03 | 0.96 | 0.51 | 0.08 (null 0.03) |
| OLMo final | neutral | 0.00 | 0.05 | 0.95 | 0.28 | 0.08 (null 0.03) |

Stated (questionnaire) vs submitted (report) misreport, mean over personas:

* Qwen3.6: base: stated 0.06 vs submitted 0.06 (ρ 0.23); auto: stated 0.04 vs submitted 0.05 (ρ -0.01)
* Qwen3.8: base: stated 0.00 vs submitted 0.00 (ρ -0.17); auto: stated 0.00 vs submitted 0.00 (ρ -0.08)
* OLMo DPO: base: stated 0.02 vs submitted 0.34 (ρ 0.47); auto: stated 0.03 vs submitted 0.35 (ρ 0.82)
* OLMo final: base: stated 0.09 vs submitted 0.35 (ρ 0.56); auto: stated 0.08 vs submitted 0.37 (ρ 0.65)

### Track B — reporting task, pre-registered choice: misreport (default Assistant)

| model | L, s | context | none | trait +s (valid/ctrl) | trait −s (valid/ctrl) | range [CI] | random band | competent random band | interaction vs base [CI] |
|---|---|---|---|---|---|---|---|---|---|
| Qwen3.6 | L36, 0.3 | base | 0.01 | 0.01 (1.00/1.00) | 0.10 (0.97/0.94) | -0.09 [-0.15, -0.04] | 0.07 | 0.06 | – |
| Qwen3.6 | L36, 0.3 | auto | 0.01 | 0.02 (1.00/1.00) | 0.12 (0.98/1.00) | -0.10 [-0.18, -0.03] | 0.03 | 0.02 | -0.01 [-0.09, 0.06] |
| Qwen3.6 | L36, 0.3 | neutral | 0.00 | 0.02 (1.00/1.00) | 0.08 (0.99/1.00) | -0.06 [-0.11, -0.01] | 0.06 | 0.04 | 0.03 [-0.02, 0.10] |
| Qwen3.8 | L36, 0.6 | base | 0.00 | 0.00 (0.94/1.00) | 0.00 (1.00/1.00) | 0.00 [0.00, 0.00] | 0.01 | 0.00 | – |
| Qwen3.8 | L36, 0.6 | auto | 0.00 | 0.00 (0.98/1.00) | 0.00 (1.00/1.00) | 0.00 [0.00, 0.00] | 0.00 | 0.00 | 0.00 [0.00, 0.00] |
| Qwen3.8 | L36, 0.6 | neutral | 0.00 | 0.00 (0.94/1.00) | 0.01 (0.99/0.97) | -0.01 [-0.02, 0.00] | 0.01 | 0.01 | -0.01 [-0.02, 0.00] |
| OLMo DPO | L36, 1 | base | 0.28 | 0.24 (0.98/0.97) | 0.26 (1.00/1.00) | -0.01 [-0.06, 0.03] | 0.03 | 0.03 | – |
| OLMo DPO | L36, 1 | auto | 0.28 | 0.22 (0.99/1.00) | 0.30 (1.00/1.00) | -0.08 [-0.14, -0.02] | 0.08 | 0.05 | -0.06 [-0.13, 0.01] |
| OLMo DPO | L36, 1 | neutral | 0.23 | 0.16 (0.99/1.00) | 0.24 (1.00/1.00) | -0.08 [-0.13, -0.03] | 0.03 | 0.03 | -0.07 [-0.14, 0.00] |
| OLMo final | L36, 1 | base | 0.26 | 0.23 (1.00/1.00) | 0.28 (0.96/0.94) | -0.06 [-0.12, 0.01] | 0.03 | 0.02 | – |
| OLMo final | L36, 1 | auto | 0.29 | 0.22 (1.00/1.00) | 0.31 (0.96/0.94) | -0.09 [-0.15, -0.03] | 0.06 | 0.04 | -0.03 [-0.11, 0.04] |
| OLMo final | L36, 1 | neutral | 0.24 | 0.19 (1.00/1.00) | 0.26 (0.97/0.97) | -0.08 [-0.13, -0.02] | 0.02 | 0.01 | -0.02 [-0.09, 0.05] |

### Track B — reporting task, EXPLORATORY one-sided choice (PREREG 6.8): misreport (default Assistant)

| model | L, s | context | none | trait +s (valid/ctrl) | trait −s (valid/ctrl) | range [CI] | random band | competent random band | interaction vs base [CI] |
|---|---|---|---|---|---|---|---|---|---|
| Qwen3.6 | L32, 0.6 | base | 0.01 | 0.56 (1.00/0.97) | 0.11 (0.08/0.14) | 0.56 [0.20, 0.89] | 0.34 | 0.34 | – |
| Qwen3.6 | L32, 0.6 | auto | 0.01 | 0.53 (1.00/0.97) | 0.08 (0.09/0.08) | 0.38 [0.14, 0.67] | 0.38 | 0.40 | – [nan, nan] |
| Qwen3.6 | L32, 0.6 | neutral | 0.00 | 0.55 (1.00/1.00) | 0.06 (0.14/0.19) | 0.38 [0.16, 0.62] | 0.38 | 0.41 | -0.33 [-1.00, 0.00] |
| Qwen3.8 | L32, 1 | base | 0.00 | 0.36 (1.00/1.00) | 0.36 (0.52/0.42) | -0.03 [-0.16, 0.10] | 0.40 | 0.04 | – |
| Qwen3.8 | L32, 1 | auto | 0.00 | 0.37 (1.00/1.00) | 0.32 (0.55/0.56) | 0.02 [-0.11, 0.15] | 0.38 | 0.40 | 0.02 [-0.14, 0.19] |
| Qwen3.8 | L32, 1 | neutral | 0.00 | 0.35 (1.00/1.00) | 0.35 (0.53/0.56) | -0.03 [-0.15, 0.09] | 0.37 | 0.38 | 0.07 [-0.10, 0.24] |
| OLMo DPO | L40, 1 | base | 0.28 | 0.24 (0.93/0.86) | 0.27 (0.95/1.00) | -0.02 [-0.07, 0.02] | 0.03 | 0.04 | – |
| OLMo DPO | L40, 1 | auto | 0.28 | 0.22 (0.95/0.94) | 0.29 (0.96/0.97) | -0.07 [-0.13, -0.02] | 0.06 | 0.03 | -0.05 [-0.11, 0.01] |
| OLMo DPO | L40, 1 | neutral | 0.23 | 0.23 (0.98/1.00) | 0.21 (0.95/0.97) | 0.02 [-0.02, 0.08] | 0.04 | 0.03 | 0.04 [-0.03, 0.12] |
| OLMo final | L40, 0.6 | base | 0.26 | 0.24 (1.00/1.00) | 0.29 (0.98/0.97) | -0.05 [-0.11, 0.01] | 0.03 | 0.01 | – |
| OLMo final | L40, 0.6 | auto | 0.29 | 0.26 (1.00/1.00) | 0.28 (1.00/1.00) | -0.02 [-0.06, 0.02] | 0.04 | 0.03 | 0.03 [-0.04, 0.10] |
| OLMo final | L40, 0.6 | neutral | 0.24 | 0.23 (1.00/1.00) | 0.24 (0.99/1.00) | -0.01 [-0.06, 0.03] | 0.01 | 0.03 | 0.03 [-0.03, 0.10] |

### Track B — dilemma items (pre-registered choice): pooled pole-a range (+s − −s) over default + 4 a-priori roles

| model | dimension | context | range [CI] | random band | interaction vs base [CI] |
|---|---|---|---|---|---|
| Qwen3.6 | honesty | base | -0.06 [-0.13, 0.01] | 0.04 | – |
| Qwen3.6 | honesty | auto | -0.11 [-0.18, -0.03] | 0.04 | -0.05 [-0.13, 0.04] |
| Qwen3.6 | honesty | neutral | -0.05 [-0.10, 0.00] | 0.06 | 0.01 [-0.07, 0.09] |
| Qwen3.6 | rules_outcomes | base | -0.23 [-0.35, -0.12] | 0.06 | – |
| Qwen3.6 | rules_outcomes | auto | -0.16 [-0.23, -0.09] | 0.03 | 0.07 [-0.04, 0.17] |
| Qwen3.6 | rules_outcomes | neutral | -0.12 [-0.23, -0.03] | 0.06 | 0.10 [-0.00, 0.21] |
| Qwen3.8 | honesty | base | 0.00 [-0.06, 0.06] | 0.07 | – |
| Qwen3.8 | honesty | auto | -0.04 [-0.12, 0.06] | 0.06 | -0.04 [-0.14, 0.06] |
| Qwen3.8 | honesty | neutral | -0.02 [-0.07, 0.03] | 0.05 | -0.02 [-0.07, 0.03] |
| Qwen3.8 | rules_outcomes | base | -0.12 [-0.27, 0.02] | 0.06 | – |
| Qwen3.8 | rules_outcomes | auto | -0.08 [-0.23, 0.06] | 0.11 | 0.04 [-0.08, 0.14] |
| Qwen3.8 | rules_outcomes | neutral | -0.12 [-0.26, 0.02] | 0.07 | -0.00 [-0.11, 0.09] |
| OLMo DPO | honesty | base | -0.02 [-0.11, 0.07] | 0.08 | – |
| OLMo DPO | honesty | auto | -0.03 [-0.12, 0.04] | 0.05 | -0.02 [-0.12, 0.08] |
| OLMo DPO | honesty | neutral | -0.08 [-0.17, -0.01] | 0.03 | -0.07 [-0.16, 0.03] |
| OLMo DPO | rules_outcomes | base | -0.17 [-0.30, -0.05] | 0.09 | – |
| OLMo DPO | rules_outcomes | auto | -0.24 [-0.38, -0.10] | 0.04 | -0.07 [-0.17, 0.03] |
| OLMo DPO | rules_outcomes | neutral | -0.21 [-0.33, -0.08] | 0.07 | -0.03 [-0.16, 0.10] |
| OLMo final | honesty | base | -0.04 [-0.16, 0.09] | 0.05 | – |
| OLMo final | honesty | auto | -0.07 [-0.14, 0.00] | 0.06 | -0.03 [-0.13, 0.07] |
| OLMo final | honesty | neutral | 0.01 [-0.07, 0.07] | 0.04 | 0.05 [-0.07, 0.16] |
| OLMo final | rules_outcomes | base | -0.08 [-0.17, -0.01] | 0.12 | – |
| OLMo final | rules_outcomes | auto | -0.12 [-0.23, -0.02] | 0.09 | -0.04 [-0.11, 0.03] |
| OLMo final | rules_outcomes | neutral | -0.05 [-0.15, 0.03] | 0.07 | 0.03 [-0.05, 0.10] |

### Track B — dilemma items (EXPLORATORY): pooled pole-a range (+s − −s) over default + 4 a-priori roles

| model | dimension | context | range [CI] | random band | interaction vs base [CI] |
|---|---|---|---|---|---|
| Qwen3.6 | honesty | base | 0.08 [-0.23, 0.35] | 0.05 | – |
| Qwen3.6 | honesty | auto | 0.19 [-0.01, 0.41] | 0.09 | 0.11 [-0.14, 0.36] |
| Qwen3.6 | honesty | neutral | 0.36 [0.18, 0.58] | 0.12 | 0.23 [-0.08, 0.57] |
| Qwen3.6 | rules_outcomes | base | -0.04 [-0.12, 0.04] | 0.20 | – |
| Qwen3.6 | rules_outcomes | auto | 0.06 [-0.10, 0.26] | 0.26 | 0.08 [-0.09, 0.29] |
| Qwen3.6 | rules_outcomes | neutral | 0.18 [0.03, 0.38] | 0.20 | 0.19 [-0.01, 0.47] |
| Qwen3.8 | honesty | base | 0.04 [-0.12, 0.23] | 0.23 | – |
| Qwen3.8 | honesty | auto | 0.04 [-0.14, 0.24] | 0.35 | -0.00 [-0.18, 0.19] |
| Qwen3.8 | honesty | neutral | 0.06 [-0.10, 0.24] | 0.30 | 0.02 [-0.11, 0.15] |
| Qwen3.8 | rules_outcomes | base | 0.17 [-0.07, 0.39] | 0.49 | – |
| Qwen3.8 | rules_outcomes | auto | 0.12 [0.01, 0.23] | 0.46 | -0.05 [-0.29, 0.20] |
| Qwen3.8 | rules_outcomes | neutral | 0.19 [0.06, 0.32] | 0.50 | 0.02 [-0.22, 0.28] |
| OLMo DPO | honesty | base | -0.05 [-0.12, 0.03] | 0.05 | – |
| OLMo DPO | honesty | auto | -0.07 [-0.16, 0.01] | 0.06 | -0.03 [-0.11, 0.07] |
| OLMo DPO | honesty | neutral | -0.08 [-0.19, 0.01] | 0.04 | -0.03 [-0.13, 0.07] |
| OLMo DPO | rules_outcomes | base | -0.17 [-0.28, -0.05] | 0.04 | – |
| OLMo DPO | rules_outcomes | auto | -0.28 [-0.43, -0.13] | 0.03 | -0.12 [-0.24, 0.00] |
| OLMo DPO | rules_outcomes | neutral | -0.19 [-0.36, -0.03] | 0.04 | -0.03 [-0.08, 0.04] |
| OLMo final | honesty | base | -0.00 [-0.11, 0.10] | 0.03 | – |
| OLMo final | honesty | auto | -0.00 [-0.08, 0.08] | 0.07 | 0.00 [-0.11, 0.09] |
| OLMo final | honesty | neutral | -0.07 [-0.15, 0.01] | 0.06 | -0.07 [-0.19, 0.06] |
| OLMo final | rules_outcomes | base | -0.04 [-0.17, 0.07] | 0.08 | – |
| OLMo final | rules_outcomes | auto | -0.06 [-0.15, 0.03] | 0.08 | -0.02 [-0.11, 0.07] |
| OLMo final | rules_outcomes | neutral | -0.04 [-0.13, 0.06] | 0.07 | -0.00 [-0.08, 0.10] |

### Trait projection before the decision (layer 36, reporting prompts): shift vs base in persona-SD units

* Qwen3.6: auto: default 0.94, mean 1.22; neutral: default 2.21, mean 1.66
* Qwen3.8: auto: default 0.29, mean 0.94; neutral: default -1.29, mean -0.23
* OLMo DPO: auto: default -0.11, mean 0.89; neutral: default 2.47, mean 1.73
* OLMo final: auto: default 0.22, mean 1.56; neutral: default 3.73, mean 2.69
