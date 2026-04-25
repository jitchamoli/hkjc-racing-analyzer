"""Verify the rendered dashboard HTML's <script> block is syntactically valid JS.

Catches regressions like apostrophes/newlines inside JS string literals that
slip past pytest because pytest only string-searches the HTML — it doesn't
parse the JS.
"""
import json
import re
import shutil
import subprocess
import pytest

from dashboard.template import generate_html


def _make_minimal_race(num=1, tier='SKIP'):
    return {
        'race_number': num, 'venue': 'Sha Tin', 'time': '13:00',
        'distance': 1200, 'class_type': 'Class 4', 'going': 'Good',
        'course': 'A', 'conviction_tier': tier, 'has_edge': False,
        'top_2_share': 0.15, 'top_3_share': 0.22,
        'horses': [{'number': 1, 'name': 'Test', 'jockey': 'TJ', 'trainer': 'TT',
                    'draw': 1, 'win_odds': 5.0, 'rank': 1, 'pure_prob': 0.2,
                    'market_prob': 0.2, 'edge_pp': 0, 'ev_win': 0, 'overlay': 1.0,
                    'edge_fires': False, 'model_rank': 1, 'reasoning': [],
                    'calibrated_prob': 0.2, 'rule_score': 50, 'ml_prob': 0.2,
                    'expected_return': 0, 'bet_recommendation': 'SKIP',
                    'confidence': '', 'place_prob': 0.3, 'last_6': '', 'age': 4,
                    'horse_weight': 1100, 'days_since_last_run': 14,
                    'rating_change': 0, 'going_form': None,
                    'jockey_horse_stats': None, 'distance_pb': None,
                    'distance_record': None, 'odds_movement': {},
                    'steam_move': {'is_steam': False}, 'cold_horse': {'is_cold': False},
                    'weight': 130, 'gear': '', 'rating': 60, 'place_odds': 2.5}],
        'speed_map': [], 'alerts': [], 'tier_stats': {},
    }


@pytest.mark.skipif(shutil.which('node') is None, reason='node not installed')
def test_rendered_script_block_parses_in_node():
    races = [_make_minimal_race(i) for i in range(1, 12)]
    html = generate_html(json.dumps(races), '2026-04-27 13:00:00', '2026/04/27')

    # Extract the single <script> block (the dashboard has exactly one)
    m = re.search(r'<script>(.*?)</script>', html, re.DOTALL)
    assert m, 'Could not find the main <script> block'
    js_body = m.group(1)

    result = subprocess.run(
        ['node', '--check', '-'],
        input=js_body, text=True, capture_output=True, timeout=15,
    )
    assert result.returncode == 0, (
        f'node --check failed:\n{result.stderr}\n\nJS body excerpt:\n{js_body[:500]}'
    )


def test_no_raw_newlines_inside_js_string_literals():
    """Quick lexer-style smoke test that catches the most common regression mode."""
    races = [_make_minimal_race(1)]
    html = generate_html(json.dumps(races), '2026-04-27 13:00:00', '2026/04/27')
    m = re.search(r'<script>(.*?)</script>', html, re.DOTALL)
    assert m
    js = m.group(1)

    # Scan for single-quoted strings that span physical lines.
    # A JS string literal starts at ' and can be terminated by '; if a real \n
    # appears between them with no closing ', the parser fails.
    in_string = False
    quote = None
    for i, c in enumerate(js):
        if c == '\n' and in_string:
            # Find the surrounding ~80 chars for diagnostic
            start = max(0, i - 80)
            end = min(len(js), i + 40)
            pytest.fail(
                f'Raw newline inside a JS {quote}-quoted string at index {i}.\n'
                f'Context: ...{js[start:end]!r}...'
            )
        if not in_string and c in ("'", '"'):
            in_string = True
            quote = c
        elif in_string and c == quote and js[i - 1] != '\\':
            in_string = False
            quote = None
