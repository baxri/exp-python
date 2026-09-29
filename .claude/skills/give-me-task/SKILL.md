---
name: give-me-task
description: Generate a new easy LeetCode-style Python practice problem as a file in the Algorithms folder, for the user to solve themselves. Use when the user runs /give-me-task or asks for a new coding exercise/practice problem.
---

# give-me-task

Give the user a new, easy, LeetCode-style coding problem to practice Python, as a
file in `Algorithms/`. This is for learning — do NOT solve the problem or write the
implementation. Only create the problem file with a stub.

Beyond the algorithm itself, each task should also expose the user to a Python
language feature/syntax they may not use daily (see the feature list below), so
over time they get hands-on reps across the language, not just algorithmic logic.

## Steps

1. List existing files in `Algorithms/` to see which problems and which Python
   features have already been given (check each file's "Python feature spotlight"
   section, see below). Avoid repeating the same algorithm topic AND avoid repeating
   the same spotlighted feature.
2. Pick one easy-difficulty problem, in the general style of LeetCode "Easy" tier
   (e.g. arrays, strings, hash maps, two pointers, simple recursion, basic math,
   stacks/queues, linked lists). Vary the topic/pattern from what's already in the
   folder.
3. Pick one Python feature to spotlight for this task, rotating through areas such as:
   - comprehensions (list/dict/set) and generator expressions
   - `enumerate`, `zip`, unpacking (`a, *rest = ...`, multiple assignment)
   - `collections` (`Counter`, `defaultdict`, `deque`, `namedtuple`)
   - `itertools` (`combinations`, `permutations`, `groupby`, `accumulate`)
   - f-strings and string formatting/methods
   - slicing tricks, `sorted()` with `key=`/`reverse=`
   - `dataclasses`, `@property`, basic OOP (classes, `__init__`, dunder methods)
   - generators/`yield`, closures, decorators
   - `match` statement (structural pattern matching)
   - context managers (`with`, `contextlib`)
   - type hints beyond the basics (`Optional`, `Union`/`|`, `TypedDict`, generics)
   - recursion, `functools` (`reduce`, `lru_cache`, `partial`)
   - exception handling (`try`/`except`/`else`/`finally`, custom exceptions)
   Choose a feature that fits the chosen problem naturally — don't force it.
4. Create a new file in `Algorithms/` named `NN_snake_case_title.py`, where `NN` is
   the next zero-padded number after the highest existing one (start at `01`).
5. The file should contain:
   - A module docstring with: the problem title, a clear problem statement,
     1-2 examples with input/output, any constraints, and a short
     "Python feature spotlight: <feature>" line naming the feature to practice
     and one sentence on why/where it could help here (without giving away the
     full solution).
   - A function stub with a descriptive name, type hints, and a `pass` body
     (or `raise NotImplementedError`) — no solution logic.
   - A `if __name__ == "__main__":` block with a couple of example calls the user
     can uncomment/run to test their own implementation once they've written it,
     using `print(...)` or `assert` statements against the expected example outputs.
6. Report back to the user: the problem title, a one-line summary, the spotlighted
   Python feature, and the file path. Do not restate the full problem in chat — let
   them open the file.
7. Do not implement the solution, add hints beyond what's in the docstring, or edit
   the file further unless the user explicitly asks for help/a hint/a review.
