# SFFN Structure Microprobes - No Loop Promotion

Date: 2026-07-02

## Purpose

After attention-only variants failed to beat loop16, the next paper-fidelity check focused on the Spiking Feed-Forward Network (SFFN).

The paper defines:

- `FC(x) = SN(Wx + b)`
- `SFFN(x) = FC2(FC1(x))`

The current code uses a LIF neuron after `fc1`, then applies `fc2`, with the block-level output LIF applied after residual addition. Two cheap runtime-only microprobes tested whether this area should become loop36:

1. add an SFFN-internal LIF after `fc2`;
2. zero-initialize SFFN biases without changing the existing PyTorch linear weight initialization.

No repository code was changed for either probe.

## Setup

- Code baseline: `main` at `5ffe340`
- Current best baseline: loop16 (`9c0d64f`, W&B `cvxuw267` / refresh `1f8plmtv`)
- GPU: 1x H200 via `CUDA_VISIBLE_DEVICES=0`
- Dataset: FineWeb-Edu streaming, first 320 prefetched micro-batches reused for base and candidate
- Sequence length: 512
- Time steps: 4
- Per-GPU batch size: 2
- Gradient accumulation: 16
- Optimizer steps: 20
- Precision: bf16
- Seed: Python, NumPy, and Torch set to 0 before each variant

## SFFN `fc2_lif` Probe

| Variant | Step 20 hard | Step 20 soft | Last10 hard | Last10 soft | Step20 delta vs base | Last10 delta vs base |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| base | 7.9564 | 4.5028 | 7.8951 | 4.4773 | 0.0000 / 0.0000 | 0.0000 / 0.0000 |
| SFFN `fc2_lif` | 8.6260 | 4.6915 | 8.5706 | 4.7058 | +0.6696 / +0.1887 | +0.6755 / +0.2285 |

Decision: fail immediately. Adding a second SFFN-internal spike stage made primary output losses much worse.

## SFFN Bias-Only Probe

| Variant | Step 20 hard | Step 20 soft | Last10 hard | Last10 soft | Step20 delta vs base | Last10 delta vs base |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| base | 7.9564 | 4.5028 | 7.8951 | 4.4773 | 0.0000 / 0.0000 | 0.0000 / 0.0000 |
| SFFN zero bias | 7.9352 | 4.5201 | 7.9065 | 4.4954 | -0.0213 / +0.0174 | +0.0114 / +0.0181 |

Decision: fail. Hard loss improved only at the final step, while soft loss and last-10 hard/soft were worse than base.

## Overall Decision

Do not create loop36 from either SFFN candidate. Keep loop16 as the current best baseline.

Next candidates should avoid adding extra SFFN spike stages and should not isolate SFFN bias zeroing unless a stronger diagnostic appears.
