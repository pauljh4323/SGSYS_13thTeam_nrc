# exp/r2-ext: gru_v8p2z6a / gru_v8p2z6b (external clean speech + external noise) vs gru_v8p2z3e

**Status: 2026-11-28. Neither run is promoted.** Both pass the primary round-2 rule, with a pooled gen-test gain of about
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
