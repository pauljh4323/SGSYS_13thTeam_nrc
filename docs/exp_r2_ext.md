# exp/r2-ext: gru_v8p2z6a / gru_v8p2z6b (external clean speech + external noise) vs gru_v8p2z3e

**Status: 2026-11-28. Neither run is promoted. 2026-11-29: the WiSE-FT blend gru_v8p2zw80 (α 0.8 between z3e and z6a) is promoted; see the last section.** Both pass the primary round-2 rule, with a pooled gen-test gain of about
+0.23 PESQ (z ≈ 16). Both fail two secondary rules: validation (−0.035 / −0.044 against the −0.03 bound) and val_remix
(−0.027 / −0.025 against −0.02). r2-prep, configs/submission.yaml, the r2-ready tag and the submission repo are unchanged.
The candidate is still gru_v8p2z3e + rms_norm.

The runs, configs, data and weights are as follows:
- Configs: configs/dl/gru_v8p2z6a.yaml (no EMA, seed 7) and configs/dl/gru_v8p2z6b.yaml (weight EMA 0.999, seed 8).
  - Both continue from z3e with fresh Adam at lr 5e-5, for 300 × 81 steps.
  - The data is ext_mix: LibriSpeech subset clean with p 0.5, external train_pool noise with p 0.6, SNR U[−10, 20] dB,
    and a second noise with p 0.3.
- Data and the use of each split: docs/external_data.md.
- Training ran on Colab (notebooks/colab_train.ipynb). Logs: logs/train_log_v8p2z6{a,b}.csv.
- Weights: weights/gru_v8p2z6{a,b}.pt. Both load with `torch.load(..., weights_only=True)`. The SHA-256 hashes are below.
- Metrics: scripts/eval.py (pesq 0.0.4 WB, 16-bit, rms_norm target_rms 0.079130, `--jobs 1 --resume`), with clean-hash rows at
  26f2d9a in results/experiments.csv.
- Paired analysis: `python scripts/gen/r2_ext_paired.py` > logs/r2_ext_paired.txt.

Full SHA-256:
- z6a `4036924485853988dccf5f6dbd2746d991ef5f35c9f97b944a3d4c953dc22f94`
- z6b `3caede88857dae33718a32d42ce64c60d8ae9e31fea4213682cdec72654e30ca`

## Training summary
Epoch 0 is z3e itself, scored with eval.py:
- val 3.6534;
- gen_unseen_dev 2.5773 and gen_multi_dev 2.0938, both new rows from this run;
- selection = mean of the three = 2.7748.

| run | best epoch (sel) | stopped | resumed | sel | val | G_unseen_dev | G_multi_dev | train64 | gap (train64 − val) |
|---|---|---|---|---|---|---|---|---|---|
| z3e (epoch 0) | – | – | – | 2.7748 | 3.6534 | 2.5773 | 2.0938 | ≈ 3.99 † | ≈ 0.33 † |
| z6a | 298 | 300 (budget, no early stop) | no ‡ | 3.0809 (+0.306) | 3.6179 (−0.036) | 3.0393 (+0.462) | 2.5853 (+0.492) | 3.9149 | 0.297 |
| z6b (EMA) | 288 | 300 (budget, no early stop) | no ‡ | 3.0791 (+0.304) | 3.6099 (−0.044) | 3.0387 (+0.461) | 2.5888 (+0.495) | 3.9221 | 0.312 |

† From z6b at epoch 2: its EMA weights are still about 85 % z3e, with val 3.6525, train64 3.9863 and gap 0.334.
‡ Epochs 1–300 are contiguous, with step +81 on every row and no duplicate epochs. The logs show no sign of a resume.

The training validator agrees exactly with eval.py on validation (3.6179 / 3.6099). In both runs the selection metric was
still rising at the end: the best epoch is near 300 and patience never fired. train64 *falls* by about 0.07, together with
val, so the dev gain is a shift of the training distribution away from the challenge data. It is not over-fitting.

## Results (eval.py; PESQ / SNR dB; paired Δ vs z3e ± SE (z))
| set | z3e | z6a | Δ z6a | z6b | Δ z6b |
|---|---|---|---|---|---|
| validation | 3.6534 / 1.21 | 3.6179 / 1.02 | −0.0354 ± 0.0071 (−5.0) | 3.6099 / 0.99 | −0.0435 ± 0.0076 (−5.7) |
| val 0 / 5 / 10 / 15 dB | 3.009 / 3.431 / 3.714 / 3.918 | 2.939 / 3.428 / 3.687 / 3.861 | −0.071 / −0.003 / −0.027 / −0.057 | 2.915 / 3.410 / 3.680 / 3.863 | −0.094 / −0.021 / −0.034 / −0.055 |
| val_remix | 3.4961 / 1.15 | 3.4691 / 0.97 | −0.0270 ± 0.0076 (−3.5) | 3.4709 / 0.97 | −0.0252 ± 0.0076 (−3.3) |
| G_unseen_test (328) | 2.8349 / 1.01 | 3.1875 / 0.85 | +0.3526 ± 0.0247 (+14.3) | 3.1938 / 0.90 | +0.3589 ± 0.0245 (+14.6) |
| G_multi_test (164) | 2.2402 / 0.78 | 2.7556 / 0.67 | +0.5154 ± 0.0364 (+14.1) | 2.7546 / 0.73 | +0.5145 ± 0.0353 (+14.6) |
| G_low (328) | 2.3114 / 0.39 | 2.2833 / 0.29 | −0.0280 ± 0.0063 (−4.5) | 2.2853 / 0.32 | −0.0261 ± 0.0067 (−3.9) |
| **pooled gen test (820)** | | | **+0.2329 ± 0.0147 (+15.8)** | | **+0.2360 ± 0.0146 (+16.2)** |
| probe −30/−20/−10/0 dB | 3.653 × 4 | 3.6179 × 4 (range 0) | | 3.6099 × 4 (range 0) | |

There were 0 PESQ failures. The G_low noisy-reference crash (docs/r2_gen_suite.md) did not occur, and every eval finished on its first attempt.

### Rule (CLAUDE.md round 2)
| criterion | z6a | z6b |
|---|---|---|
| primary: pooled gen Δ > 0, z ≥ 2 | pass (+0.233, z 15.8) | pass (+0.236, z 16.2) |
| validation Δ ≥ −0.03 | **fail** (−0.035) | **fail** (−0.044) |
| val_remix Δ ≥ −0.02 | **fail** (−0.027) | **fail** (−0.025) |
| probe flat (within 0.02) | pass (0.000) | pass (0.000) |
| **decision** | not promoted | not promoted |

### Breakdowns
- **Per corpus (Δ vs z3e).** Two patterns stand out:
  - The large gains are on ESC-50 and MUSAN noise. DEMAND gains little, but its test environments (TBUS, TCAR, TMETRO)
    are easy for z3e already (3.51).
  - z6b is slightly better on G_unseen DEMAND and on E+M. z6a is slightly better on ESC-50 and MUSAN alone.

  | set | corpus | z3e | Δ z6a | Δ z6b |
  |---|---|---|---|---|
  | G_unseen_test | DEMAND | 3.511 | +0.037 (z 1.7) | +0.056 (z 2.7) |
  | G_unseen_test | ESC-50 | 2.431 | +0.464 (z 12.6) | +0.471 (z 12.5) |
  | G_unseen_test | MUSAN | 2.557 | +0.559 (z 11.7) | +0.553 (z 11.5) |
  | G_multi_test | D+E | 2.304 | +0.505 (z 8.0) | +0.499 (z 8.0) |
  | G_multi_test | D+M | 2.564 | +0.484 (z 6.9) | +0.478 (z 7.1) |
  | G_multi_test | E+M | 1.845 | +0.558 (z 9.8) | +0.568 (z 10.7) |

- **Per SNR.** On G_unseen the gain is flat over −5 / 0 / 5 / 10 dB (+0.32 to +0.37). On the challenge noise, the loss
  is largest at 0 dB and 15 dB on validation, at 15 dB on val_remix (−0.06), and at both −5 and −10 dB on G_low
  (−0.02 to −0.03). So the regression is on the *challenge noise types* (NOISEX-92 + Hu) at every level. It is not tied
  to one SNR range.
- **Per speaker (validation, 5 speakers).** The delta is negative for all 5 speakers in both runs:
  - z6a: −0.003 (speaker015) to −0.078 (speaker016);
  - z6b: −0.001 to −0.081.
- **EMA effect (z6b − z6a, paired).** EMA is a wash:
  - validation −0.0081 ± 0.0041 (z −2.0);
  - val_remix +0.002 (z 0.3), G_unseen +0.006 (z 1.0), G_multi −0.001 (z −0.1) and G_low +0.002 (z 0.4).

  EMA gave a smoother selection curve, but no better weights at a fixed budget, and it was slightly worse on validation.

## Reading
1. External data generalises strongly. On noise no model of ours had seen, PESQ rises by +0.35 to +0.52, which is the
   largest gain of round 2 so far. The hidden test "includes noise types not in train/validation", so this matters.
2. The cost is −0.03 to −0.04 on the challenge noise distribution: validation, val_remix and G_low. With 60 % external
   noise and 50 % LibriSpeech clean speech, the challenge distribution makes up only about 20 % of the training pairs.
3. The current rule protects validation and val_remix, so both runs fail. Options for a next run are below. None of
   them has been run, and none of them changes the rule.
   - Lower p_ext_noise or p_libri, for example 0.3, to keep more challenge pairs.
   - Interpolate weights between z3e and z6a (`w = (1 − α) w_z3e + α w_z6a`, choosing α on train or dev, never on val_remix).
   - Add a train-subset term to `select_sets`, so that selection penalises the challenge-distribution drop.
   - Train longer: the selection metric was still rising at epoch 300.

## WiSE-FT blend z3e ↔ z6a (2026-11-29): gru_v8p2zw80 promoted
Weight-space interpolation (WiSE-FT, Wortsman et al. 2022). There is no training:
`theta = (1 − α) · z3e + α · z6a` over every float tensor of model_state, which includes the parameters and the
feat_mean / feat_std / norm buffers. The checkpoints have no non-float tensors.
- Script: scripts/wise_ft.py. It asserts identical keys, shapes, dtypes and model_cfg, blends in float64 and saves a
  gru_mask_v1 checkpoint that loads with `weights_only=True`. Its metadata records α and the SHA-256 of both ends.
- Sanity check (logs/wise_ft_sanity.txt): α 0 and α 1 give weights equal to z3e / z6a and bit-identical outputs on
  20/20 validation files (stratified subset, rms_norm inference).
- Configs: configs/dl/gru_v8p2zw{20..80}.yaml. Only weights/gru_v8p2zw80.pt is committed (SHA-256
  `9dfa8c7b6cfc642963466928bf223de0594e8ce9946a6c92c66f1808e609b002`). The other α are rebuilt with
  `python scripts/wise_ft.py --a weights/gru_v8p2z3e.pt --b weights/gru_v8p2z6a.pt --alpha 0.2 ... --out "weights/gru_v8p2zw{a}.pt"`.
- Metrics: eval.py (`--jobs 1 --resume`, rms_norm), with rows at d1dc567. Paired analysis: `python scripts/gen/r2_wise_paired.py dev|test 80`
  > logs/r2_wise_dev.txt / logs/r2_wise_test.txt.

**α selection rule, fixed before the sweep:** among the α with validation paired Δ ≥ −0.02, take the highest mean of
gen_unseen_dev and gen_multi_dev. The test sets were not used in the choice.

### Dev sweep (PESQ; paired Δ vs z3e)
| α | validation | G_unseen_dev | G_multi_dev | dev mean | val Δ ≥ −0.02 |
|---|---|---|---|---|---|
| 0 (z3e) | 3.6534 | 2.5773 | 2.0938 | 2.3356 | – |
| 0.2 | 3.6595 (+0.0062, z +4.0) | 2.7135 (+0.136) | 2.2411 (+0.147) | 2.4773 | yes |
| 0.3 | 3.6597 (+0.0063, z +3.0) | 2.7824 (+0.205) | 2.3182 (+0.224) | 2.5503 | yes |
| 0.4 | 3.6594 (+0.0060, z +2.2) | 2.8431 (+0.266) | 2.3940 (+0.300) | 2.6185 | yes |
| 0.5 | 3.6563 (+0.0029, z +0.9) | 2.9035 (+0.326) | 2.4552 (+0.361) | 2.6794 | yes |
| 0.6 | 3.6507 (−0.0027, z −0.6) | 2.9580 (+0.381) | 2.5042 (+0.410) | 2.7311 | yes |
| 0.7 | 3.6470 (−0.0063, z −1.4) | 2.9936 (+0.416) | 2.5398 (+0.446) | 2.7667 | yes |
| **0.8** | 3.6393 (−0.0141, z −2.6) | 3.0173 (+0.440) | 2.5611 (+0.467) | **2.7892** | yes → **chosen** |
| 1 (z6a) | 3.6179 (−0.0354) | 3.0393 (+0.462) | 2.5853 (+0.492) | 2.8123 | no |

The dev gain is close to linear in α, but the validation cost is convex: it stays ≥ 0 up to α 0.5. So α 0.8 keeps
95 % of z6a's dev gain at 40 % of its validation cost.

### Chosen α 0.8 vs z3e (eval.py; PESQ / SNR dB; paired Δ ± SE (z))
| set | z3e | gru_v8p2zw80 | Δ | (z6a Δ) |
|---|---|---|---|---|
| validation | 3.6534 / 1.20 | 3.6393 / 1.06 | −0.0141 ± 0.0054 (−2.6) | −0.0354 |
| val_remix | 3.4961 / 1.15 | 3.4907 / 1.01 | −0.0054 ± 0.0059 (−0.9) | −0.0270 |
| G_unseen_test (328) | 2.8349 / 1.01 | 3.1729 / 0.90 | +0.3380 ± 0.0230 (+14.7) | +0.3526 |
| G_multi_test (164) | 2.2402 / 0.78 | 2.7413 / 0.71 | +0.5011 ± 0.0350 (+14.3) | +0.5154 |
| G_low (328) | 2.3114 / 0.39 | 2.3062 / 0.32 | −0.0051 ± 0.0061 (−0.8) | −0.0280 |
| **pooled gen test (820)** | | | **+0.2334 ± 0.0138 (+16.9)** | +0.2329 |
| probe −30/−20/−10/0 dB | 3.653 × 4 | 3.6393 × 4 (range 0) | | |

There were 0 PESQ failures. The gen_low eval hit the known native pesq crash on the noisy reference score
(docs/r2_gen_suite.md) once, at file 88. `--resume` finished it on the first retry.

Per SNR, the remaining cost is at the edges: validation 0 dB −0.055 and 15 dB −0.028, and val_remix 15 dB −0.034.
At 5–10 dB the blend is level with or above z3e.

### Rule (CLAUDE.md round 2)
| criterion | gru_v8p2zw80 |
|---|---|
| primary: pooled gen Δ > 0, z ≥ 2 | pass (+0.233, z 16.9) |
| validation Δ ≥ −0.03 | pass (−0.014) |
| val_remix Δ ≥ −0.02 | pass (−0.005) |
| probe flat | pass (range 0) |
| **decision** | **promoted**: r2-prep via promote/r2-v8p2zw80 (e2e and export: docs/submissions.md on r2-prep) |

The pooled gen-test gain equals z6a's (+0.233), and the blend gives back most of z6a's challenge-distribution loss:
validation −0.035 → −0.014, val_remix −0.027 → −0.005 and G_low −0.028 → −0.005.
