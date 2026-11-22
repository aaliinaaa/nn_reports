# Lab 0 — experiment analysis

### E1. Repeatability

- runs: `24008b2e91ed_20261121132704_ca5a83_24628`, `24008b2e91ed_20261121141355_1850f1_35802`
- EXP_ID match: **True** (`24008b2e91ed` vs `24008b2e91ed`)
- first-batch checksum match: **True**
- relative final-metric gap: `0.000e+00`
- diff_runs (must be empty for same EXP_ID): `empty`

### E2. Variance across seeds

- runs: 5; seeds differ by +0..4 from base_seed
- m = `0.467660`, s_m = `0.013588`, 95% CI = `[0.455749, 0.479571]`
- minimal detectable effect (2*s_m): `0.027177` (improvements below this are not established)
- all EXP_ID distinct and differing only in S: **True**

### E3. Checkpoint/resume equivalence

- continuous run: `24008b2e91ed_20261121185253_077a4d_73014`; resumed run: `24008b2e91ed_20261121195929_074c2d_69277`
- d_theta = `8.505e-04` (threshold A: 1e-7)
- relative final-metric gap: `1.130e-02`
- global_step match: **True** (395 vs 395)

### E4. Performance

- param_count: 558538; device: cpu
- epoch_time_s: `437.238`

| batch | latency mean (ms) | latency p95 (ms) | throughput (sps) | peak GPU mem (B) |
|---|---|---|---|---|
| 1 | 9.623 | 10.605 | 104.1 | None |
| 8 | 64.409 | 113.990 | 129.6 | None |
| 32 | 282.225 | 374.603 | 115.4 | None |
| 64 | 544.360 | 579.198 | 117.9 | None |

- argmin latency batch = **1**; argmax throughput batch = **8** (counterfactual #8)

### Counterfactual runs

- CF2 (num_workers=2): runs `0b3aae6609d2_20261121202843_655563_23187`, `0b3aae6609d2_20261121211851_c7109b_31193`; EXP_ID match = True; first-batch checksum match = True
- CF3 (truncated checkpoint resume): run `24008b2e91ed_20261121220840_4371eb_20554` (see d_theta vs E3-A)


> Each claim above must be cross-checked against the referenced RUN_ID directories under `runs/` before going into the report.
