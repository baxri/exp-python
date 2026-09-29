# exp-python

Python playground project managed with `uv`.

## Folders

- `Algorithms/` — for learning Python by solving coding problems (LeetCode-style).
  Files here are created by the `/give-me-task` skill with a problem statement and
  a stub function. The user solves them by hand; Claude should not solve the task
  or fill in the implementation unless explicitly asked to.

## Code style

- Project requires Python >= 3.12. Use built-in generic types (`list[int]`,
  `dict[str, int]`, etc.) instead of `typing.List`/`typing.Dict`/etc. — the
  `typing` versions are deprecated and ruff will flag them.
