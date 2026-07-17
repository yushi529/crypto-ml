"""単純移動平均クロス戦略をバックテストする。"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

HOURS_PER_YEAR = 365 * 24
REQUIRED_COLUMNS = {"timestamp", "close"}


@dataclass(frozen=True)
class BacktestConfig:
    """バックテスト条件。"""

    fast_window: int = 24
    slow_window: int = 168
    fee_rate: float = 0.001
    slippage_rate: float = 0.0002
    periods_per_year: int = HOURS_PER_YEAR

    def __post_init__(self) -> None:
        if self.fast_window <= 0:
            raise ValueError("fast_window は正の整数にしてください")
        if self.slow_window <= self.fast_window:
            raise ValueError("slow_window は fast_window より大きくしてください")
        if self.fee_rate < 0 or self.slippage_rate < 0:
            raise ValueError("取引コストは0以上にしてください")


def load_ohlcv(path: str | Path, *, exclude_last: bool = False) -> pd.DataFrame:
    """OHLCV CSVを読み込み、バックテスト用に検証・整形する。"""
    df = pd.read_csv(path, parse_dates=["timestamp"])
    missing = REQUIRED_COLUMNS.difference(df.columns)
    if missing:
        raise ValueError(f"必要な列がありません: {', '.join(sorted(missing))}")

    df = df.sort_values("timestamp").drop_duplicates("timestamp")
    if exclude_last:
        df = df.iloc[:-1]
    if df.empty:
        raise ValueError("バックテスト対象データがありません")
    if df["timestamp"].dt.tz is None:
        df["timestamp"] = df["timestamp"].dt.tz_localize("UTC")
    if df["close"].isna().any() or (df["close"] <= 0).any():
        raise ValueError("close は欠損のない正の値である必要があります")
    return df.set_index("timestamp")


def run_backtest(
    prices: pd.DataFrame, config: BacktestConfig
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """ロング／キャッシュ型SMAクロス戦略を実行する。

    時点tの終値で確定したシグナルを1本遅らせ、t+1のリターンに適用する。
    これにより終値を見て同じ終値で約定する先読みを避ける。
    """
    if len(prices) < config.slow_window + 2:
        raise ValueError(
            f"少なくとも {config.slow_window + 2} 行のデータが必要です"
        )

    result = prices.copy()
    result["fast_sma"] = result["close"].rolling(config.fast_window).mean()
    result["slow_sma"] = result["close"].rolling(config.slow_window).mean()
    result["signal"] = (result["fast_sma"] > result["slow_sma"]).astype(float)
    result["position"] = result["signal"].shift(1).fillna(0.0)
    result["market_return"] = result["close"].pct_change().fillna(0.0)
    result["turnover"] = result["position"].diff().abs().fillna(
        result["position"].abs()
    )
    result["trading_cost"] = result["turnover"] * (
        config.fee_rate + config.slippage_rate
    )
    result["gross_strategy_return"] = result["position"] * result["market_return"]
    result["strategy_return"] = (
        result["gross_strategy_return"] - result["trading_cost"]
    )
    result["gross_strategy_equity"] = (1 + result["gross_strategy_return"]).cumprod()
    result["strategy_equity"] = (1 + result["strategy_return"]).cumprod()
    result["benchmark_equity"] = (1 + result["market_return"]).cumprod()
    result["drawdown"] = (
        result["strategy_equity"] / result["strategy_equity"].cummax() - 1
    )

    trades = extract_trades(result)
    metrics = calculate_metrics(result, trades, config)
    return result, trades, metrics


def extract_trades(result: pd.DataFrame) -> pd.DataFrame:
    """ポジション系列から各トレードの損益と保有時間を抽出する。"""
    previous = result["position"].shift(1).fillna(0.0)
    entries = list(result.index[(result["position"] == 1) & (previous == 0)])
    exits = list(result.index[(result["position"] == 0) & (previous == 1)])
    rows: list[dict[str, Any]] = []

    for entry in entries:
        future_exits = [exit_time for exit_time in exits if exit_time > entry]
        is_open = not future_exits
        exit_time = future_exits[0] if future_exits else result.index[-1]
        trade_returns = result.loc[entry:exit_time, "strategy_return"]
        rows.append(
            {
                "entry_time": entry,
                "exit_time": exit_time,
                "return": (1 + trade_returns).prod() - 1,
                "holding_hours": (exit_time - entry).total_seconds() / 3600,
                "status": "open" if is_open else "closed",
            }
        )

    return pd.DataFrame(
        rows,
        columns=["entry_time", "exit_time", "return", "holding_hours", "status"],
    )


def calculate_metrics(
    result: pd.DataFrame, trades: pd.DataFrame, config: BacktestConfig
) -> dict[str, Any]:
    """戦略リターン、リスク、取引特性の指標を計算する。"""
    returns = result["strategy_return"]
    benchmark_returns = result["market_return"]
    elapsed_years = (
        (result.index[-1] - result.index[0]).total_seconds()
        / (365.25 * 24 * 3600)
    )
    final_equity = float(result["strategy_equity"].iloc[-1])
    gross_final_equity = float(result["gross_strategy_equity"].iloc[-1])
    benchmark_equity = float(result["benchmark_equity"].iloc[-1])
    total_return = final_equity - 1
    gross_total_return = gross_final_equity - 1
    benchmark_total_return = benchmark_equity - 1
    cagr = final_equity ** (1 / elapsed_years) - 1 if elapsed_years > 0 else math.nan
    benchmark_cagr = (
        benchmark_equity ** (1 / elapsed_years) - 1
        if elapsed_years > 0
        else math.nan
    )
    annualized_volatility = float(returns.std(ddof=1) * math.sqrt(config.periods_per_year))
    sharpe_ratio = _annualized_ratio(returns, returns.std(ddof=1), config)
    downside = returns[returns < 0]
    sortino_ratio = _annualized_ratio(returns, downside.std(ddof=1), config)
    max_drawdown = float(result["drawdown"].min())
    calmar_ratio = cagr / abs(max_drawdown) if max_drawdown < 0 else math.nan

    trade_returns = trades["return"] if not trades.empty else pd.Series(dtype=float)
    wins = trade_returns[trade_returns > 0]
    losses = trade_returns[trade_returns < 0]
    gross_profit = float(wins.sum())
    gross_loss = float(abs(losses.sum()))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else math.nan

    return {
        "start": result.index[0].isoformat(),
        "end": result.index[-1].isoformat(),
        "bars": int(len(result)),
        "total_return": total_return,
        "gross_total_return": gross_total_return,
        "cost_drag": total_return - gross_total_return,
        "cagr": cagr,
        "max_drawdown": max_drawdown,
        "sharpe_ratio": sharpe_ratio,
        "sortino_ratio": sortino_ratio,
        "annualized_volatility": annualized_volatility,
        "calmar_ratio": calmar_ratio,
        "benchmark_total_return": benchmark_total_return,
        "benchmark_cagr": benchmark_cagr,
        "excess_total_return": total_return - benchmark_total_return,
        "exposure": float(result["position"].mean()),
        "trade_count": int(len(trades)),
        "closed_trade_count": int((trades["status"] == "closed").sum())
        if not trades.empty
        else 0,
        "win_rate": float((trade_returns > 0).mean()) if len(trade_returns) else math.nan,
        "profit_factor": profit_factor,
        "average_trade_return": float(trade_returns.mean())
        if len(trade_returns)
        else math.nan,
        "average_holding_hours": float(trades["holding_hours"].mean())
        if not trades.empty
        else math.nan,
        "max_drawdown_duration_hours": _max_drawdown_duration(result["drawdown"]),
        "total_turnover": float(result["turnover"].sum()),
        "estimated_cost_sum": float(result["trading_cost"].sum()),
    }


def _annualized_ratio(
    returns: pd.Series, denominator: float, config: BacktestConfig
) -> float:
    if pd.isna(denominator) or denominator == 0:
        return math.nan
    return float(
        returns.mean() / denominator * math.sqrt(config.periods_per_year)
    )


def _max_drawdown_duration(drawdown: pd.Series) -> int:
    underwater = drawdown < 0
    groups = (~underwater).cumsum()
    durations = underwater.groupby(groups).sum()
    return int(durations.max()) if not durations.empty else 0


def save_outputs(
    result: pd.DataFrame,
    trades: pd.DataFrame,
    metrics: dict[str, Any],
    config: BacktestConfig,
    output_dir: str | Path,
    symbol: str,
) -> dict[str, Path]:
    """指標、明細、Markdownレポート、グラフを保存する。"""
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    paths = {
        "metrics": directory / "metrics.json",
        "trades": directory / "trades.csv",
        "equity": directory / "equity_curve.csv",
        "report": directory / "report.md",
        "figure": directory / "backtest.png",
    }

    payload = _json_safe({"strategy": asdict(config), "metrics": metrics})
    paths["metrics"].write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    trades.to_csv(paths["trades"], index=False)
    result[
        [
            "close",
            "fast_sma",
            "slow_sma",
            "position",
            "market_return",
            "gross_strategy_return",
            "strategy_return",
            "gross_strategy_equity",
            "strategy_equity",
            "benchmark_equity",
            "drawdown",
        ]
    ].to_csv(paths["equity"])
    paths["report"].write_text(
        build_markdown_report(metrics, config, symbol), encoding="utf-8"
    )
    plot_backtest(result, config, symbol, paths["figure"])
    return paths


def build_markdown_report(
    metrics: dict[str, Any], config: BacktestConfig, symbol: str
) -> str:
    """人が読みやすいバックテストレポートを生成する。"""
    rows = [
        ("累積リターン", _percent(metrics["total_return"])),
        ("取引コスト控除前の累積リターン", _percent(metrics["gross_total_return"])),
        ("取引コストによる累積リターン差", _percent(metrics["cost_drag"])),
        ("最大ドローダウン", _percent(metrics["max_drawdown"])),
        ("年率シャープレシオ", _number(metrics["sharpe_ratio"])),
        ("CAGR（年率複利リターン）", _percent(metrics["cagr"])),
        ("年率ボラティリティ", _percent(metrics["annualized_volatility"])),
        ("Sortinoレシオ", _number(metrics["sortino_ratio"])),
        ("Calmarレシオ", _number(metrics["calmar_ratio"])),
        ("バイ＆ホールド累積リターン", _percent(metrics["benchmark_total_return"])),
        ("対バイ＆ホールド超過リターン", _percent(metrics["excess_total_return"])),
        ("市場エクスポージャー", _percent(metrics["exposure"])),
        ("取引回数", str(metrics["trade_count"])),
        ("勝率", _percent(metrics["win_rate"])),
        ("プロフィットファクター", _number(metrics["profit_factor"])),
        ("平均トレードリターン", _percent(metrics["average_trade_return"])),
        ("平均保有時間", f'{metrics["average_holding_hours"]:.1f} 時間'),
        ("最大ドローダウン継続時間", f'{metrics["max_drawdown_duration_hours"]} 時間'),
        ("総売買回転量", f'{metrics["total_turnover"]:.0f} 回分'),
        ("単純合算した推定取引コスト", _percent(metrics["estimated_cost_sum"])),
    ]
    table = "\n".join(f"| {label} | {value} |" for label, value in rows)
    cost = config.fee_rate + config.slippage_rate
    return f"""# SMAクロス戦略 バックテストレポート

## 戦略

- 対象: `{symbol}` 1時間足
- ルール: {config.fast_window}時間SMAが{config.slow_window}時間SMAを上回る間はロング、それ以外は現金
- 約定: シグナルを1時間遅らせて反映（先読み防止）
- 取引コスト: 売買ごとに `{cost:.2%}`（手数料 `{config.fee_rate:.2%}` + スリッページ `{config.slippage_rate:.2%}`）
- 対象期間: {metrics['start']} ～ {metrics['end']}（{metrics['bars']:,}本）

## 結果

| 指標 | 結果 |
|---|---:|
{table}

![バックテスト結果](backtest.png)

## 結果の読み取り

- 戦略の累積リターンはバイ＆ホールドを `{abs(metrics['excess_total_return']) * 100:.2f}`ポイント{_comparison_word(metrics['excess_total_return'])}。
- コスト控除前でも累積リターンは `{metrics['gross_total_return']:.2%}` で、シグナル自体がこの期間に有効ではありませんでした。取引コストはさらに `{abs(metrics['cost_drag']) * 100:.2f}`ポイント成績を押し下げました。
- 勝率は `{metrics['win_rate']:.2%}`、プロフィットファクターは `{metrics['profit_factor']:.2f}` です。1未満のプロフィットファクターは、利益合計より損失合計が大きかったことを示します。
- 市場エクスポージャーを `{metrics['exposure']:.2%}` に抑えても最大ドローダウンは `{metrics['max_drawdown']:.2%}` であり、このルールだけでは十分な下落抑制ができませんでした。

## 注意点

- リスクフリーレートは0%としてシャープレシオを計算しています。
- 税金、スプレッド、資金調達コスト、市場インパクトは含みません。
- 現物のロング／キャッシュを想定し、ショートやレバレッジは使用しません。
- 過去の結果は将来の運用成績を保証しません。
"""


def plot_backtest(
    result: pd.DataFrame,
    config: BacktestConfig,
    symbol: str,
    output: str | Path,
) -> None:
    """価格・資産曲線・ドローダウンを1枚に描画する。"""
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(
        3,
        1,
        figsize=(16, 11),
        sharex=True,
        gridspec_kw={"height_ratios": [1.5, 1.3, 1.0]},
        constrained_layout=True,
    )

    axes[0].plot(result.index, result["close"], label="Close", linewidth=0.8)
    axes[0].plot(
        result.index,
        result["fast_sma"],
        label=f"SMA {config.fast_window}h",
        linewidth=0.9,
    )
    axes[0].plot(
        result.index,
        result["slow_sma"],
        label=f"SMA {config.slow_window}h",
        linewidth=1.0,
    )
    axes[0].set_title(f"{symbol} — SMA Crossover Backtest")
    axes[0].set_ylabel("Price (USDT)")
    axes[0].legend(loc="upper left", ncol=3)

    axes[1].plot(
        result.index,
        result["strategy_equity"],
        label="SMA strategy",
        linewidth=1.2,
        color="#2563eb",
    )
    axes[1].plot(
        result.index,
        result["gross_strategy_equity"],
        label="SMA before costs",
        linewidth=0.9,
        linestyle="--",
        color="#0f766e",
    )
    axes[1].plot(
        result.index,
        result["benchmark_equity"],
        label="Buy & hold",
        linewidth=1.0,
        color="#64748b",
    )
    axes[1].set_ylabel("Growth of 1 USDT")
    axes[1].legend(loc="upper left")

    axes[2].fill_between(
        result.index,
        result["drawdown"] * 100,
        0,
        color="#dc2626",
        alpha=0.5,
        linewidth=0,
    )
    axes[2].set_ylabel("Drawdown (%)")
    axes[2].set_xlabel("Time (UTC)")

    locator = mdates.AutoDateLocator(minticks=6, maxticks=12)
    axes[2].xaxis.set_major_locator(locator)
    axes[2].xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(True, color="#cbd5e1", alpha=0.55, linewidth=0.6)

    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _percent(value: float) -> str:
    return "N/A" if pd.isna(value) else f"{value:.2%}"


def _number(value: float) -> str:
    return "N/A" if pd.isna(value) else f"{value:.2f}"


def _comparison_word(excess_return: float) -> str:
    return "上回りました" if excess_return >= 0 else "下回りました"


def _json_safe(value: Any) -> Any:
    """NaN/InfinityをJSONのnullへ変換する。"""
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SMAクロス戦略をバックテストする")
    parser.add_argument("--input", required=True, help="入力するOHLCV CSV")
    parser.add_argument("--output-dir", required=True, help="結果の出力ディレクトリ")
    parser.add_argument("--symbol", default="BTC/USDT", help="レポート上の銘柄名")
    parser.add_argument("--fast-window", type=int, default=24)
    parser.add_argument("--slow-window", type=int, default=168)
    parser.add_argument("--fee-rate", type=float, default=0.001)
    parser.add_argument("--slippage-rate", type=float, default=0.0002)
    parser.add_argument(
        "--exclude-last",
        action="store_true",
        help="形成中の可能性がある末尾の足を除外する",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    config = BacktestConfig(
        fast_window=args.fast_window,
        slow_window=args.slow_window,
        fee_rate=args.fee_rate,
        slippage_rate=args.slippage_rate,
    )
    prices = load_ohlcv(args.input, exclude_last=args.exclude_last)
    result, trades, metrics = run_backtest(prices, config)
    paths = save_outputs(
        result, trades, metrics, config, args.output_dir, args.symbol
    )

    print("SMAクロス戦略 バックテスト結果")
    print(f"期間: {metrics['start']} ～ {metrics['end']}")
    print(f"累積リターン: {_percent(metrics['total_return'])}")
    print(f"最大ドローダウン: {_percent(metrics['max_drawdown'])}")
    print(f"年率シャープレシオ: {_number(metrics['sharpe_ratio'])}")
    print(f"CAGR: {_percent(metrics['cagr'])}")
    print(f"取引回数: {metrics['trade_count']}")
    print(f"レポート: {paths['report']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
