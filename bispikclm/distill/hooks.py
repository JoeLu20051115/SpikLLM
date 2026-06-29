from collections.abc import Callable


TeacherHook = Callable[[str], None]


def emit_teacher_event(name: str, hooks: list[TeacherHook] | None = None) -> list[str]:
    if not hooks:
        return [name]
    for hook in hooks:
        hook(name)
    return [name]

