# Loop Iteration 88 - Three-GPU Baseline Launch Failure

Date: 2026-07-02

## Purpose

Attempted to launch the user's requested three-GPU best-baseline run after GPUs appeared free.

Intended setup:

- current `main` / loop16 source baseline;
- loop14 three-GPU geometry: `seq=1024`, `T=4`, per-rank batch `4`, gradient accumulation `64`;
- three GPUs: `CUDA_VISIBLE_DEVICES=0,1,2`;
- W&B enabled.

## Result

The run exited before worker initialization and did not occupy GPUs.

Run artifact:

- Run name: `loop88-loop16-baseline-seq1024-bs4-ga64-3xh200-1bt-20260702-091350`
- Log: `logs/loop88-loop16-baseline-seq1024-bs4-ga64-3xh200-1bt-20260702-091350/train.log`
- Output: `output/loop88-loop16-baseline-seq1024-bs4-ga64-3xh200-1bt-20260702-091350`
- Port: `29788`

Failure:

```text
torch.distributed.DistNetworkError: The server socket has failed to listen on any local network address.
port: 29788 ... EADDRINUSE ... address already in use
```

Decision: launch failure only. No W&B run, no optimizer steps, no checkpoint, and no baseline update. Relaunch with a different high port.
