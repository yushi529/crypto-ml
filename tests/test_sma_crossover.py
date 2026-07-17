from __future__ import annotations

import unittest

import pandas as pd

from src.backtest.sma_crossover import BacktestConfig, run_backtest


class SmaCrossoverBacktestTest(unittest.TestCase):
    def setUp(self) -> None:
        self.index = pd.date_range("2025-01-01", periods=12, freq="h", tz="UTC")
        self.config = BacktestConfig(
            fast_window=2,
            slow_window=3,
            fee_rate=0.0,
            slippage_rate=0.0,
        )

    def test_signal_is_shifted_to_prevent_lookahead(self) -> None:
        prices = pd.DataFrame(
            {"close": [10, 10, 10, 20, 20, 20, 20, 20, 20, 20, 20, 20]},
            index=self.index,
        )

        result, _, _ = run_backtest(prices, self.config)

        first_signal = result.index[result["signal"] == 1][0]
        first_position = result.index[result["position"] == 1][0]
        self.assertEqual(first_position, first_signal + pd.Timedelta(hours=1))
        self.assertEqual(result.loc[first_signal, "strategy_return"], 0.0)

    def test_cost_is_charged_on_entry_and_exit(self) -> None:
        prices = pd.DataFrame(
            {"close": [10, 10, 11, 12, 11, 10, 10, 10, 10, 10, 10, 10]},
            index=self.index,
        )
        config = BacktestConfig(
            fast_window=2,
            slow_window=3,
            fee_rate=0.001,
            slippage_rate=0.0002,
        )

        result, _, metrics = run_backtest(prices, config)

        self.assertEqual(result["turnover"].sum(), 2.0)
        self.assertAlmostEqual(result["trading_cost"].sum(), 0.0024)
        self.assertGreater(
            result["gross_strategy_equity"].iloc[-1],
            result["strategy_equity"].iloc[-1],
        )
        self.assertLess(metrics["cost_drag"], 0)
        self.assertEqual(metrics["trade_count"], 1)

    def test_flat_market_has_zero_return_and_drawdown(self) -> None:
        prices = pd.DataFrame({"close": [10.0] * 12}, index=self.index)

        result, trades, metrics = run_backtest(prices, self.config)

        self.assertTrue(trades.empty)
        self.assertEqual(result["strategy_equity"].iloc[-1], 1.0)
        self.assertEqual(metrics["total_return"], 0.0)
        self.assertEqual(metrics["max_drawdown"], 0.0)


if __name__ == "__main__":
    unittest.main()
