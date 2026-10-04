"""Track what our GATES throw away.

2026/05/24 R2: the model's top 4 were exactly the first 4 home, and the ticket
was EMPTY — tier LOW, every horse SKIP. F4 BOX-4 would have returned $369 on
$10. One race proves nothing, but nothing in the repo was accumulating these,
so the question "do the gates help?" had no dataset behind it.

This logs, per race and per product, what the SAME model top-N box would have
returned, split by whether the gate put that product on the ticket. Paper only
— it changes no bet.
"""
import pytest

from scripts.track_gate_skips import PRODUCTS, products_on_ticket, score_product


TICKET = {"bets": [
    {"label": "BOX TRIO-5", "amount": 100, "legs": [{"number": n} for n in (1, 2, 3, 4, 5)]},
    {"label": "BOX QIN-4 PROTECTION", "amount": 60, "legs": [{"number": n} for n in (1, 2, 3, 4)]},
    {"label": "BOX TRIO-6", "amount": 200, "paper": True, "legs": [{"number": n} for n in range(1, 7)]},
]}


class TestProductsOnTicket:
    def test_reads_the_labels_that_were_actually_staked(self):
        assert products_on_ticket(TICKET) == {"TRIO-5", "QIN-4"}

    def test_paper_lines_do_not_count_as_staked(self):
        """BOX TRIO-6 is paper since 2026-09-06 — it is not a gate decision to bet."""
        assert "TRIO-6" not in products_on_ticket(TICKET)

    def test_empty_ticket_means_everything_was_skipped(self):
        assert products_on_ticket({"bets": []}) == set()


class TestScoreProduct:
    #  ranks 1..6 = horses 10,20,30,40,50,60
    RANKS = [10, 20, 30, 40, 50, 60]

    def test_trio5_hit_pays_the_trio_dividend_once(self):
        got = score_product(self.RANKS, [20, 30, 50, 99], {"TRIO": 175.0}, PRODUCTS["TRIO-5"])
        assert got == (100.0, 175.0, True)      # 10 combos x $10, podium inside top 5

    def test_trio5_miss_pays_nothing_but_still_costs(self):
        got = score_product(self.RANKS, [20, 30, 60, 99], {"TRIO": 175.0}, PRODUCTS["TRIO-5"])
        assert got == (100.0, 0.0, False)       # rank-6 horse on the podium

    def test_f4_box5_needs_all_four_finishers_in_the_box(self):
        assert score_product(self.RANKS, [10, 20, 30, 40], {"FIRST 4": 369.0}, PRODUCTS["F4-5"])[2] is True
        assert score_product(self.RANKS, [10, 20, 30, 60], {"FIRST 4": 369.0}, PRODUCTS["F4-5"])[2] is False

    def test_qin4_needs_the_first_two(self):
        assert score_product(self.RANKS, [30, 40, 99, 98], {"QUINELLA": 162.0}, PRODUCTS["QIN-4"])[2] is True
        assert score_product(self.RANKS, [30, 50, 99, 98], {"QUINELLA": 162.0}, PRODUCTS["QIN-4"])[2] is False

    def test_missing_dividend_is_not_scored_as_a_loss(self):
        """No dividend on record (overseas row, unpriced) → skip, don't book -$100."""
        assert score_product(self.RANKS, [20, 30, 50, 99], {}, PRODUCTS["TRIO-5"]) is None

    def test_short_field_cannot_fill_the_box(self):
        assert score_product([10, 20, 30], [10, 20, 30, 40], {"TRIO": 99.0}, PRODUCTS["TRIO-5"]) is None

    @pytest.mark.parametrize("key,combos", [("QIN-4", 6), ("TRIO-5", 10), ("F4-5", 5)])
    def test_stake_is_flat_ten_per_combination(self, key, combos):
        ranks = [1, 2, 3, 4, 5, 6]
        top4 = [90, 91, 92, 93]          # a clean miss, so only the stake matters
        stake, ret, hit = score_product(ranks, top4, {"TRIO": 1, "QUINELLA": 1, "FIRST 4": 1}, PRODUCTS[key])
        assert (stake, ret, hit) == (combos * 10.0, 0.0, False)
