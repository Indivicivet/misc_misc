# Python Project Rules

## The Golden Rule (Scope Control)
- Zero Unsolicited Edits: You are strictly forbidden from editing code unrelated to the specific task. If you see "bad" code in a different function, ignore it unless it causes a breaking error for the current task. You may make any edits to fully achieve the goal requested and no further.
- No Extra Refactoring: Do not simplify logic, rename variables, or change structure unless explicitly asked to "Refactor" or the code is being heavily modified for the current task. This also means you should not add random comments to other bits of code.

## Python & Formatting Standards
- General code style: Try to keep consistent with the code nearby that you're modifying. Look in the current file, and if the current file is short or new, you can also pick a couple of adjacent files to compare style with. Style of these files should be your primary guidance.
- Formatter: Follow 'Black' formatting rules (88 chars per line, double quotes, trailing commas). Run black to check.
- Comments: Add comments with detailed explanation or "gotcha" notes where necessary. You do not generally need to add "self-explanatory" comments. Comments MUST NOT be included if they simply declare what a variable is in a way that could be otherwise made clear by a better choice of variable name.
- Typing: Only use type hints when adding or modifying function signatures if this helps to clarify the types involved; don't bother with them if the types are highly variable (e.g. any arithmetic types). Don't bother with them if the code is already clear. Do not add types to existing code unless requested.
- Avoid single use variables - inline variables where possible, unless declaring them as a separate variable is used for debugging, or it makes it significantly clearer what something is for - typically including only at the use site is better.
- Imports: Use absolute imports. Do not reorganize existing imports unless a new one is required. Avoid "import X as Y" except for explicit cases that there is justification for IN THE CODEBASE you're working on (you are allowed `import numpy as np` and matplotlib pyplot as plt).
- Avoid mutating arguments except in rare cases where this is the only clear way to implement an algorithm.

## Post-Edit Requirements:
- Verify any modifications you made - check that ALL modifications in the diff relate DIRECTLY to the task you are currently working on. Check there are no random other changes to unrelated or unmodified code.
- After modifying any Python files, you MUST immediately run `python -m black <file_path>` using the integrated terminal.
- Verification: If `black` produces changes, you must read the updated file and ensure the final state you present to me is the formatted version.
- Silent Execution: You do not need to ask for permission to run the formatter; consider it a mandatory part of the "Save" process.

## Restricted Actions
- Do not add or remove comments from *unrelated* parts of code. This does not prohibit you from, say, removing a "todo" comment when addressing that todo.
- Do not "cleanup" unused imports in files you aren't actively fixing.
