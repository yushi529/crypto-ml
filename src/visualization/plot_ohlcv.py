"""OHLCV CSV の終値・出来高・リターンを可視化する。"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

REQUIRED_COLUMNS = {"timestamp", "close", "volume"}


def load_market_data(path: str | Path) -> pd.DataFrame:
    """CSV を読み込み、時刻順に整列して1時間リターンを追加する。"""
    df = pd.read_csv(path, parse_dates=["timestamp"])
    missing = REQUIRED_COLUMNS.difference(df.columns)
    if missing:
        raise ValueError(f"必要な列がありません: {', '.join(sorted(missing))}")

    df = df.sort_values("timestamp").set_index("timestamp")
    df["return_pct"] = df["close"].pct_change() * 100
    return df


def plot_market_data(df: pd.DataFrame, output: str | Path) -> Path:
    """終値・出来高・1時間リターンを3段の時系列グラフとして保存する。"""
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(
        3,
        1,
        figsize=(16, 10),
        sharex=True,
        gridspec_kw={"height_ratios": [2.0, 1.0, 1.2]},
        constrained_layout=True,
    )

    axes[0].plot(df.index, df["close"], color="#2563eb", linewidth=0.9)
    axes[0].set_ylabel("Close (USDT)")
    axes[0].set_title("BTC/USDT (Binance) — 1-hour Market Overview")

    axes[1].fill_between(
        df.index,
        df["volume"],
        color="#64748b",
        alpha=0.65,
        linewidth=0,
    )
    axes[1].set_ylabel("Volume (BTC)")

    returns = df["return_pct"]
    axes[2].plot(df.index, returns, color="#f97316", linewidth=0.55)
    axes[2].axhline(0, color="#334155", linewidth=0.7)
    axes[2].set_ylabel("1h Return (%)")
    axes[2].set_xlabel("Time (UTC)")

    locator = mdates.AutoDateLocator(minticks=6, maxticks=12)
    axes[2].xaxis.set_major_locator(locator)
    axes[2].xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))

    for axis in axes:
        axis.grid(True, color="#cbd5e1", alpha=0.55, linewidth=0.6)
        axis.spines[["top", "right"]].set_visible(False)

    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="OHLCV CSV の終値・出来高・リターンを可視化する"
    )
    parser.add_argument("--input", required=True, help="入力する OHLCV CSV")
    parser.add_argument("--output", required=True, help="出力する画像ファイル")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    df = load_market_data(args.input)
    output_path = plot_market_data(df, args.output)
    print(f"{len(df)} 行を可視化しました: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
