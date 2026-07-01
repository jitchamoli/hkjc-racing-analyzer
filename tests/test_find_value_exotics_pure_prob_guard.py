"""AST guard: find_value_exotics must be fed pure_prob, not calibrated_prob.

Mirrors the quinella_bet_ev guard (test_audit_2026_06_09_pass3.py). CLAUDE.md
Rule 1: exotic edge math uses pure_prob (pre-market-blend independent opinion),
NOT calibrated_prob (market-blended) — else edge vs the market is mechanically
compressed. The docstring in exotic_calculator was wrong before 2026-07-01;
this pins the WIRING so a revert to calibrated_prob fails a test, not just a
code comment.

The call site reuses the name `model_probs` (an earlier assignment builds it
from calibrated_prob for conviction-tier shares), so we can't just check the
arg name — we verify the assignment NEAREST-PRECEDING the find_value_exotics
call subscripts 'pure_prob'.
"""
import ast
import inspect

import dashboard.generator as g


def _string_consts(node):
    return {n.value for n in ast.walk(node)
            if isinstance(n, ast.Constant) and isinstance(n.value, str)}


def test_find_value_exotics_fed_pure_prob():
    tree = ast.parse(inspect.getsource(g.build_race_data))
    calls = [n for n in ast.walk(tree)
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
             and n.func.id == "find_value_exotics"]
    assert calls, "find_value_exotics call not found in build_race_data"

    for call in calls:
        assert call.args, "find_value_exotics called with no positional args"
        first = call.args[0]
        assert isinstance(first, ast.Name), (
            f"expected a variable as first arg, got {ast.dump(first)}")
        target = first.id

        # Assignments to `target` that are list comprehensions, before the call.
        assigns = [n for n in ast.walk(tree)
                   if isinstance(n, ast.Assign)
                   and any(isinstance(t, ast.Name) and t.id == target for t in n.targets)
                   and isinstance(n.value, ast.ListComp)
                   and n.lineno < call.lineno]
        assert assigns, f"no list-comp assignment to `{target}` precedes the call"

        nearest = max(assigns, key=lambda n: n.lineno)
        keys = _string_consts(nearest.value)
        assert "pure_prob" in keys, (
            f"`{target}` fed to find_value_exotics must be built from 'pure_prob' "
            f"(CLAUDE.md Rule 1); nearest assignment uses {keys}")
        assert "calibrated_prob" not in keys, (
            "find_value_exotics fed calibrated_prob — edge vs market is "
            "mechanically compressed (CLAUDE.md Rule 1 violation)")
