"""Deliberate failure #03: syntax error raised at runtime.

Uses exec() instead of a file-level syntax error so that lazy107's own
import scanner can still parse this entry. The runtime traceback is the
signature the debug command must recognize.

Expected log signature: SyntaxError: invalid syntax
sacct ExitCode: 1:0
"""

exec("def broken(: pass")  # noqa: E999

print("unreachable")
