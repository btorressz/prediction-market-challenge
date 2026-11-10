
"""Phase 1: collateral-aware, inventory-limited prediction market maker.

This single-file strategy is compatible with the public API of
``orderbook_pm_challenge``. It deliberately uses only observable StepState
fields; it cannot access the simulator's true probability or order-flow source.

Baseline behavior:
- Quote one passive bid and one passive ask when a valid competitor spread exists.
- Improve the competitor by one tick when the visible spread permits.
- Budget cash for both resting bids and uncovered asks before submitting.
- Cap YES, NO, and net directional inventory.
- Preserve order priority if existing quotes still match the desired levels.
- Cancel and replace only when quotes or risk limits change materially.

This is a baseline for benchmarking, not an optimized competition strategy.
"""

from __future__ import annotations

import math

from orderbook_pm_challenge.strategy import BaseStrategy
from orderbook_pm_challenge.types import (
    CancelAll,
    PlaceOrder,
    Side,
    StepState,
)


class Strategy(BaseStrategy):
    """Passive market maker using the official callback and action contracts."""

    # Exchange contract: 1...99 cent ticks; quantities in 0.01-share increments.
    MIN_TICK = 1
    MAX_TICK = 99
    QUANTITY_STEP = 0.01

    # Conservative initial parameters. Tune on held-out simulation seeds.
    QUOTE_SIZE = 6.0
    MAX_YES_INVENTORY = 80.0
    MAX_NO_INVENTORY = 80.0
    MAX_ABS_NET_INVENTORY = 70.0
    MIN_CASH_BUFFER = 25.0
    UNCOVERED_ASK_CASH_FRACTION = 0.50

    # Preserve an existing order until it drops below this fraction of the
    # currently desired quantity, unless its price/risk target has changed.
    MIN_REMAINING_FRACTION = 0.25

    @staticmethod
    def _round_quantity_down(value: float) -> float:
        """Respect the 0.01-share lot size; never round available funds up."""
        if value <= 0.0:
            return 0.0
        return round(math.floor((value + 1e-9) / 0.01) * 0.01, 10)

    @staticmethod
    def _needs_cancel(state: StepState) -> list[CancelAll]:
        """Avoid submitting a pointless cancel when no orders are resting."""
        return [CancelAll()] if state.own_orders else []

    def _quote_ticks(self, state: StepState) -> tuple[int, int] | None:
        """Derive legal, noncrossing passive prices from competitor quotes.

        The competitor book is a noisy, potentially stale reference, not an
        observable fair value. Phase 1 does NOT pretend to know true p_t.
        """
        bid = state.competitor_best_bid_ticks
        ask = state.competitor_best_ask_ticks
        if bid is None or ask is None:
            return None
        if not (self.MIN_TICK <= bid < ask <= self.MAX_TICK):
            return None

        gap = ask - bid

        # When at least two ticks are available inside the competitor spread,
        # offer a one-tick improvement on each side. For narrower spreads,
        # match the visible prices rather than crossing our own quotes.
        if gap >= 3:
            bid += 1
            ask -= 1

        if bid >= ask:
            return None

        return bid, ask

    def _target_quantities(
        self, state: StepState, bid_tick: int, ask_tick: int
    ) -> tuple[float, float]:
        """Compute safe quantities assuming any old orders are cancelled.

        YES bid fill: spends price * qty and increases YES inventory.
        Covered YES ask fill: sells existing YES with no new cash reserve.
        Uncovered YES ask fill: spends (1-price) * qty to create NO inventory.
        """
        yes = max(0.0, state.yes_inventory)
        no = max(0.0, state.no_inventory)
        net = yes - no

        # Worst-case single-side fill limits: neither side can push net
        # exposure beyond the maximum even if the opposite side never fills.
        max_buy = min(
            self.QUOTE_SIZE,
            max(0.0, self.MAX_YES_INVENTORY - yes),
            max(0.0, self.MAX_ABS_NET_INVENTORY - net),
        )
        max_sell = min(
            self.QUOTE_SIZE,
            max(0.0, yes + self.MAX_NO_INVENTORY - no),
            max(0.0, self.MAX_ABS_NET_INVENTORY + net),
        )

        bid_price = bid_tick / 100.0
        ask_price = ask_tick / 100.0
        spendable = max(0.0, state.cash - self.MIN_CASH_BUFFER)

        # Reserve cash for the uncovered portion of the ask first. Existing
        # YES shares can cover asks with no extra collateral. Reserve only
        # half the available buffer-adjusted cash for newly created NO when
        # both sides are enabled, leaving capacity for bids.
        covered_yes = min(yes, max_sell)
        ask_budget = (
            spendable * self.UNCOVERED_ASK_CASH_FRACTION
            if max_buy >= self.QUANTITY_STEP
            else spendable
        )
        ask_capacity = covered_yes + ask_budget / (1.0 - ask_price)
        sell_qty = self._round_quantity_down(min(max_sell, ask_capacity))

        uncovered_sell = max(0.0, sell_qty - yes)
        ask_reserve = (1.0 - ask_price) * uncovered_sell

        # The simulator cancels existing reservations before placing our new
        # quotes. Use total cash (not pre-cancel state.free_cash) and account
        # for the ask reserve above to avoid an insufficient-cash failure.
        remaining_budget = max(0.0, spendable - ask_reserve)
        buy_qty = self._round_quantity_down(
            min(max_buy, remaining_budget / bid_price)
        )

        return buy_qty, sell_qty

    def _orders_match(
        self,
        state: StepState,
        bid_tick: int,
        buy_qty: float,
        ask_tick: int,
        sell_qty: float,
    ) -> bool:
        """Keep safe orders resting to preserve FIFO position where useful."""
        targets = {}

        if buy_qty >= self.QUANTITY_STEP:
            targets[Side.BUY] = (bid_tick, buy_qty)

        if sell_qty >= self.QUANTITY_STEP:
            targets[Side.SELL] = (ask_tick, sell_qty)

        if len(state.own_orders) != len(targets):
            return False

        seen = set()

        for order in state.own_orders:
            if order.side not in targets or order.side in seen:
                return False

            seen.add(order.side)
            tick, target_qty = targets[order.side]
            remaining = order.remaining_quantity

            if order.price_ticks != tick:
                return False

            if not (
                target_qty * self.MIN_REMAINING_FRACTION - 1e-9
                <= remaining
                <= target_qty + 1e-9
            ):
                return False

        return True

    def on_step(self, state: StepState):
        """Produce only legal public actions; no network or engine access."""
        ticks = self._quote_ticks(state)

        if ticks is None:
            return self._needs_cancel(state)

        bid_tick, ask_tick = ticks
        buy_qty, sell_qty = self._target_quantities(
            state, bid_tick, ask_tick
        )

        if buy_qty < self.QUANTITY_STEP and sell_qty < self.QUANTITY_STEP:
            return self._needs_cancel(state)

        if self._orders_match(state, bid_tick, buy_qty, ask_tick, sell_qty):
            return []

        # Cancel first: the engine processes actions in order, releasing all
        # old cash and YES reservations before accepting new limit orders.
        actions = self._needs_cancel(state)

        if buy_qty >= self.QUANTITY_STEP:
            actions.append(
                PlaceOrder(
                    side=Side.BUY,
                    price_ticks=bid_tick,
                    quantity=buy_qty,
                )
            )

        if sell_qty >= self.QUANTITY_STEP:
            actions.append(
                PlaceOrder(
                    side=Side.SELL,
                    price_ticks=ask_tick,
                    quantity=sell_qty,
                )
            )

        return actions
