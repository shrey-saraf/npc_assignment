import logging
from decimal import Decimal
from typing import Dict, List

import pandas_ta as ta
from hummingbot.core.data_type.common import OrderType, PriceType, TradeType
from hummingbot.core.data_type.order_candidate import OrderCandidate
from hummingbot.core.event.events import OrderFilledEvent
from hummingbot.strategy.script_strategy_base import ScriptStrategyBase
from hummingbot.data_feed.candles_feed.candles_factory import CandlesFactory, CandlesConfig
from hummingbot.connector.connector_base import ConnectorBase

class PMMVolatilityTrendRisk(ScriptStrategyBase):
    trading_pair = "SOL-USDT"
    exchange = "binance_paper_trade"
    price_source = PriceType.MidPrice

    order_amount = Decimal("0.01")
    base_refresh_time = 15  # base refresh time for adaptive mechanism
    order_refresh_time = base_refresh_time
    create_timestamp = 0

    # Candles config
    candle_exchange = "binance"
    candles_interval = "1m"
    candles_length = 30
    max_records = 1000

    # Indicator Scalars (use Decimal for all)
    bid_spread_scalar = Decimal("0.05")
    ask_spread_scalar = Decimal("0.05")
    inventory_risk_scalar = Decimal("0.3")
    rsi_period = 14
    rsi_skew_scalar = Decimal("0.001")
    volatility_scalar = Decimal("30")  # for adaptive refresh

    # Volume spike config
    volume_spike_multiplier = Decimal("2")  # spike if current volume > 2x average
    volume_spike_active = False

    # Define which markets to connect to
    markets = {exchange: {trading_pair}}

    def __init__(self, connectors: Dict[str, ConnectorBase]):
        super().__init__(connectors)
        self.create_timestamp = 0
        self.bid_spread = Decimal("0.001")
        self.ask_spread = Decimal("0.001")
        self.rsi = Decimal("50")

        # Manual balance fallback
        self.manual_base_balance = Decimal("0")
        self.manual_quote_balance = Decimal("0")
        self.use_manual_balances = True

        try:
            self.candles = CandlesFactory.get_candle(CandlesConfig(
                connector=self.candle_exchange,
                trading_pair=self.trading_pair,
                interval=self.candles_interval,
                max_records=self.max_records
            ))
            self.candles.start()
        except Exception as e:
            self.logger().error(f"Failed to initialize candle feed: {e}")
            self.candles = None

    def on_stop(self):
        if self.candles:
            self.candles.stop()

    def on_tick(self):
        if not self.ready_to_trade or not self.candles or self.candles.candles_df.empty:
            return
        if self.create_timestamp <= self.current_timestamp:
            self.cancel_all_orders()
            self.update_indicators()
            proposal = self.create_proposal()
            adjusted = self.adjust_proposal_to_budget(proposal)
            self.place_orders(adjusted)
            self.create_timestamp = self.current_timestamp + self.order_refresh_time

    def update_indicators(self):
        df = self.candles.candles_df.copy()
        if df.empty:
            return

        df.ta.natr(length=self.candles_length, scalar=1, append=True)
        df.ta.rsi(length=self.rsi_period, append=True)

        latest = df.iloc[-1]
        self.bid_spread = Decimal(str(latest[f"NATR_{self.candles_length}"])) * self.bid_spread_scalar
        self.ask_spread = Decimal(str(latest[f"NATR_{self.candles_length}"])) * self.ask_spread_scalar
        self.rsi = Decimal(str(latest[f"RSI_{self.rsi_period}"]))

        natr = Decimal(str(latest[f"NATR_{self.candles_length}"]))
        self.order_refresh_time = float(self.base_refresh_time / (1 + self.volatility_scalar * natr))

        avg_volume = Decimal(str(df['volume'].iloc[-self.candles_length:].mean()))
        self.volume_spike_active = Decimal(str(latest['volume'])) > self.volume_spike_multiplier * avg_volume

    def create_proposal(self) -> List[OrderCandidate]:
        ref_price = Decimal(str(self.connectors[self.exchange].get_price_by_type(self.trading_pair, self.price_source)))
        base_asset, quote_asset = self.trading_pair.split("-")

        if self.use_manual_balances:
            base = self.manual_base_balance
            quote = self.manual_quote_balance
        else:
            base = Decimal(str(self.connectors[self.exchange].get_balance(base_asset)))
            quote = Decimal(str(self.connectors[self.exchange].get_balance(quote_asset)))

        total_value = base * ref_price + quote
        base_ratio = (base * ref_price / total_value) if total_value > 0 else Decimal("0.5")
        inventory_skew = (Decimal("0.5") - base_ratio) * self.inventory_risk_scalar * ref_price

        rsi_skew = Decimal("0")
        if self.rsi > 70:
            rsi_skew = self.rsi_skew_scalar * ref_price
        elif self.rsi < 30:
            rsi_skew = -self.rsi_skew_scalar * ref_price

        mid_price = ref_price + inventory_skew + rsi_skew

        self.logger().info(
            f"[Inventory] Base: {base:.4f}, Quote: {quote:.2f}, Base Ratio: {base_ratio:.4f}, "
            f"Inventory Skew: {inventory_skew:.4f}, Mid Price: {mid_price:.2f}"
        )

        best_bid = Decimal(str(self.connectors[self.exchange].get_price(self.trading_pair, False)))
        best_ask = Decimal(str(self.connectors[self.exchange].get_price(self.trading_pair, True)))

        buy_price = min(mid_price * (Decimal("1") - self.bid_spread), best_bid)
        sell_price = max(mid_price * (Decimal("1") + self.ask_spread), best_ask)

        if self.volume_spike_active:
            buy_price *= Decimal("1.002")
            sell_price *= Decimal("0.998")
            self.logger().info("Volume spike detected. Tightening spreads!")

        return [
            OrderCandidate(self.trading_pair, True, OrderType.LIMIT, TradeType.BUY, self.order_amount, buy_price),
            OrderCandidate(self.trading_pair, True, OrderType.LIMIT, TradeType.SELL, self.order_amount, sell_price),
        ]

    def adjust_proposal_to_budget(self, proposal: List[OrderCandidate]) -> List[OrderCandidate]:
        return self.connectors[self.exchange].budget_checker.adjust_candidates(proposal, all_or_none=True)

    def place_orders(self, orders: List[OrderCandidate]):
        for order in orders:
            direction = "BUY" if order.order_side == TradeType.BUY else "SELL"
            self.logger().info(f"Placing {direction} order: {order.amount} {order.trading_pair} at {order.price:.4f}")
            if order.order_side == TradeType.BUY:
                self.buy(self.exchange, order.trading_pair, order.amount, order.order_type, order.price)
            else:
                self.sell(self.exchange, order.trading_pair, order.amount, order.order_type, order.price)

    def cancel_all_orders(self):
        for order in self.get_active_orders(self.exchange):
            self.logger().info(f"Cancelling order: {order.client_order_id} at {order.price}")
            self.cancel(self.exchange, order.trading_pair, order.client_order_id)

    def did_fill_order(self, event: OrderFilledEvent):
        msg = f"{event.trade_type.name} {round(event.amount, 2)} {event.trading_pair} {self.exchange} at {round(event.price, 2)}"
        self.log_with_clock(logging.INFO, msg)
        self.notify_hb_app_with_timestamp(msg)

        if self.use_manual_balances:
            if event.trade_type == TradeType.BUY:
                self.manual_base_balance += Decimal(str(event.amount))
                self.manual_quote_balance -= Decimal(str(event.amount)) * Decimal(str(event.price))
            elif event.trade_type == TradeType.SELL:
                self.manual_base_balance -= Decimal(str(event.amount))
                self.manual_quote_balance += Decimal(str(event.amount)) * Decimal(str(event.price))

    def format_status(self) -> str:
        lines = []

        lines.extend(["", "  Balances:"])
        lines.extend(["    " + line for line in self.get_balance_df().to_string(index=False).split("\n")])

        try:
            df = self.active_orders_df()
            lines.extend(["", "  Orders:"])
            lines.extend(["    " + line for line in df.to_string(index=False).split("\n")])
        except ValueError:
            lines.append("\n  No active maker orders.")

        ref_price = Decimal(str(self.connectors[self.exchange].get_price_by_type(self.trading_pair, self.price_source)))
        best_bid = Decimal(str(self.connectors[self.exchange].get_price(self.trading_pair, False)))
        best_ask = Decimal(str(self.connectors[self.exchange].get_price(self.trading_pair, True)))

        bid_spread_bps = (ref_price - best_bid) / ref_price * Decimal("10000")
        ask_spread_bps = (best_ask - ref_price) / ref_price * Decimal("10000")

        lines.extend(["\n----------------------------------------------------------------------"])
        lines.extend([
            f"  Mid Price: {ref_price:.2f}",
            f"  RSI: {self.rsi:.2f}",
            f"  Bid Spread: {self.bid_spread * Decimal('10000'):.2f} bps | Best Bid Spread: {bid_spread_bps:.2f} bps",
            f"  Ask Spread: {self.ask_spread * Decimal('10000'):.2f} bps | Best Ask Spread: {ask_spread_bps:.2f} bps",
            f"  Adaptive Refresh Time: {self.order_refresh_time:.2f}s",
            f"  Volume Spike Active: {self.volume_spike_active}"
        ])
        lines.append("----------------------------------------------------------------------")

        if self.candles:
            candles_df = self.candles.candles_df.tail(self.candles_length)
            lines.append(f"\n  Candles: {self.candles.name} | Interval: {self.candles.interval}")
            lines.extend(["    " + line for line in candles_df[::-1].to_string(index=False).split("\n")])

        return "\n".join(lines)
