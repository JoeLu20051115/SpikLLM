from scripts.analyze_wandb_probe import analyze_rows


def _row(step: int, hard: float, soft: float) -> dict[str, float]:
    return {
        "train/step": float(step),
        "loss/hard_loss": hard,
        "loss/soft_loss": soft,
        "loss/total_loss": (hard + soft) / 2,
        "train/lr": 1e-4,
    }


def test_probe_analysis_passes_when_hard_and_soft_below_five() -> None:
    rows = [_row(step, 4.0, 4.5) for step in range(1, 2001)]

    result = analyze_rows(rows, window=500)

    assert result["decision"] == "pass"
    assert result["last_step"] == 2000


def test_probe_analysis_extends_when_trend_is_strongly_down() -> None:
    rows = [_row(step, 12.0 - step * 0.003, 11.0 - step * 0.003) for step in range(1, 2001)]

    result = analyze_rows(rows, window=500)

    assert result["decision"] == "extend"
    assert result["hard_slope_per_100"] < -0.2
    assert result["soft_slope_per_100"] < -0.2


def test_probe_analysis_fails_high_oscillation() -> None:
    rows = [_row(step, 14.0 + (step % 3), 12.0 + (step % 2)) for step in range(1, 2001)]

    result = analyze_rows(rows, window=500)

    assert result["decision"] == "fail"
