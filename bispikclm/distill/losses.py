import math


def mean_squared_error(predictions: list[float], targets: list[float]) -> float:
    if len(predictions) != len(targets):
        raise ValueError("predictions and targets must have the same length")
    if not predictions:
        return 0.0
    return sum((pred - target) ** 2 for pred, target in zip(predictions, targets, strict=True)) / len(predictions)


def kl_divergence(student_probs: list[float], teacher_probs: list[float], epsilon: float = 1e-8) -> float:
    if len(student_probs) != len(teacher_probs):
        raise ValueError("student_probs and teacher_probs must have the same length")
    total = 0.0
    for student, teacher in zip(student_probs, teacher_probs, strict=True):
        safe_student = max(student, epsilon)
        safe_teacher = max(teacher, epsilon)
        total += safe_teacher * math.log(safe_teacher / safe_student)
    return total

