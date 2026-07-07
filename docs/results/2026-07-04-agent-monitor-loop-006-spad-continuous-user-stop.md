# Agent Monitor Loop 006: SpAD Continuous 1200-to-1900 User Stop

## Material Passport

- Date: 2026-07-04
- Timezone: Asia/Singapore
- Mode: experiment-agent / run monitor
- Status: STOPPED_BY_USER
- Workspace: `/mnt/data3/data_xingrui/lueq/SpikLLM_OPD/SpikLLM/.worktrees/loop-opd-sa-topd`
- Output directory: `output/spad-continuous-3xh200-step1200-to1900-ckpt100-20260704-135949`

## Command

Continuation was launched from checkpoint step 1200 with three H200 GPUs:

```bash
CUDA_VISIBLE_DEVICES=0,1,2 torchrun --standalone --nproc_per_node=3 \
  -m bispikclm.train.train_spad \
  --config configs/bispikclm_opt125m_spad_continue_ckpt100.toml \
  --resume-from output/spad-continuous-3xh200-step1000-to4000-ckpt100-20260704-103227/checkpoint-step-1200.pt \
  --output-dir output/spad-continuous-3xh200-step1200-to1900-ckpt100-20260704-135949 \
  --metrics-jsonl output/spad-continuous-3xh200-step1200-to1900-ckpt100-20260704-135949/metrics.jsonl \
  --max-steps 1900 \
  --scheduler-max-steps 4000 \
  --sequence-length 1024 \
  --time-steps 4 \
  --batch-size 4 \
  --gradient-accumulation-steps 64 \
  --seed 1000 \
  --train
```

## Result

- User requested stop at approximately `2026-07-04 15:22 +08`.
- Training was interrupted with `Ctrl-C` in tmux.
- Process exit signature in `train.log`: `KeyboardInterrupt` on ranks 0/1/2 and torch elastic `SignalException: signal 2`.
- This is a manual stop, not an OOM, NaN, gradient explosion, or checkpoint write failure.
- The planned 1900 checkpoint was not reached.
- The requested post-training 8-task evaluation was not started.

## Last Observed Training State

- Last metric row: `train/step=1328`.
- Last total loss: `2.4946317113935947`.
- Last grad norm: `0.18092317879199982`.
- Last teacher top-k mass: `0.8344190716743469`.
- Last silent layer count: `0.0`.
- Last overactive layer count: `0.0`.
- Last peak memory: `76.90716695785522` GB per worker report.

## Artifacts

- Valid new checkpoint written before stop:
  - `checkpoint-step-1300.pt`
  - Size observed: `1502657945` bytes
  - Load check: `step=1300`, optimizer present, scheduler present
- No checkpoint was written for steps 1400, 1500, 1600, 1700, 1800, or 1900.

## Evaluation

No 8-task evaluation was run for this continuation because the user stopped the run before the target checkpoint step 1900.

## Monitor Verdict

The continuation direction remained operationally stable until the manual stop:

- No silent or overactive spike layers were observed in the last monitored segment.
- Teacher top-k mass stayed in the normal range observed earlier in this run.
- Gradient norm stayed small and did not show an explosion.
- The only completed new checkpoint from this continuation is step 1300.

No method feasibility conclusion should be drawn from this interrupted run beyond "training from step 1200 can proceed and checkpoint at 1300 successfully."
