"""取引所から OHLCV データを取得するモジュール。

ccxt を利用して、指定した取引所・シンボル・時間足の OHLCV
(Open/High/Low/Close/Volume) を取得する。1回のAPIコールで取得できる
本数には上限があるため、``since`` から ``until`` までページネーションで
繰り返し取得し、重複を除いた DataFrame として返す。

CLI としても実行できる:

    python -m src.data.fetch_ohlcv --symbol BTC/USDT --timeframe 1h \
        --since 2023-01-01 --output data/btc_usdt_1h.csv
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import ccxt
import pandas as pd

# OHLCV の列。ccxt は [timestamp(ms), open, high, low, close, volume] を返す。
OHLCV_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]


def parse_datetime_to_ms(value: str) -> int:
    """日付/日時文字列を UTC のエポックミリ秒に変換する。

    ``2023-01-01`` や ``2023-01-01T00:00:00Z`` のような ISO 8601 形式を受け付ける。
    タイムゾーン指定がない場合は UTC とみなす。
    """
    text = value.strip().replace("Z", "+00:00")
    dt = datetime.fromisoformat(text)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def fetch_ohlcv(
    symbol: str = "BTC/USDT",
    timeframe: str = "1h",
    since: str | int | None = None,
    until: str | int | None = None,
    exchange_id: str = "binance",
    limit: int = 1000,
    max_retries: int = 3,
) -> pd.DataFrame:
    """OHLCV を取得して DataFrame で返す。

    Args:
        symbol: 取引ペア（例: ``"BTC/USDT"``）。
        timeframe: 時間足（例: ``"1m"``, ``"1h"``, ``"1d"``）。
        since: 取得開始時刻。ISO 文字列またはエポックミリ秒。None なら
            取引所が返す最古付近から。
        until: 取得終了時刻（含まない）。None なら現在時刻まで。
        exchange_id: ccxt の取引所 ID（例: ``"binance"``, ``"bybit"``）。
        limit: 1回のАПIコールあたりの最大取得本数。
        max_retries: ネットワーク/レート制限エラー時のリトライ回数。

    Returns:
        ``timestamp`` (tz-aware UTC の DatetimeIndex) をインデックスに持ち、
        ``open/high/low/close/volume`` 列を持つ DataFrame。
    """
    exchange_class = getattr(ccxt, exchange_id, None)
    if exchange_class is None:
        raise ValueError(f"未知の取引所です: {exchange_id!r}")

    exchange = exchange_class({"enableRateLimit": True})

    if not exchange.has.get("fetchOHLCV"):
        raise RuntimeError(f"{exchange_id} は fetchOHLCV に対応していません")

    since_ms = _to_ms(since)
    until_ms = _to_ms(until)
    if until_ms is None:
        until_ms = exchange.milliseconds()

    timeframe_ms = exchange.parse_timeframe(timeframe) * 1000

    all_rows: list[list[float]] = []
    cursor = since_ms
    while True:
        batch = _fetch_batch(
            exchange, symbol, timeframe, cursor, limit, max_retries
        )
        if not batch:
            break

        # until を超えた行を除外する。
        batch = [row for row in batch if row[0] < until_ms]
        if not batch:
            break

        all_rows.extend(batch)

        last_ts = batch[-1][0]
        next_cursor = last_ts + timeframe_ms
        # 進捗がない/範囲を超えたら終了。
        if next_cursor <= (cursor or 0) or next_cursor >= until_ms:
            break
        cursor = next_cursor

        # 取得件数が limit 未満なら、これ以上のデータは無い。
        if len(batch) < limit:
            break

    return _to_dataframe(all_rows)


def _to_ms(value: str | int | None) -> int | None:
    if value is None or isinstance(value, int):
        return value
    return parse_datetime_to_ms(value)


def _fetch_batch(
    exchange,
    symbol: str,
    timeframe: str,
    since_ms: int | None,
    limit: int,
    max_retries: int,
) -> list[list[float]]:
    """1バッチ分を取得する（リトライ付き）。"""
    for attempt in range(1, max_retries + 1):
        try:
            return exchange.fetch_ohlcv(
                symbol, timeframe=timeframe, since=since_ms, limit=limit
            )
        except (ccxt.NetworkError, ccxt.ExchangeError) as exc:
            if attempt == max_retries:
                raise
            wait = min(2 ** attempt, 30)
            print(
                f"[warn] 取得失敗 ({attempt}/{max_retries}): {exc}. "
                f"{wait}秒後に再試行",
                file=sys.stderr,
            )
            time.sleep(wait)
    return []


def _to_dataframe(rows: list[list[float]]) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=OHLCV_COLUMNS)
    if df.empty:
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        return df.set_index("timestamp")

    df = df.drop_duplicates(subset="timestamp").sort_values("timestamp")
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    return df.set_index("timestamp")


def save(df: pd.DataFrame, output: str | Path) -> Path:
    """DataFrame を拡張子に応じて CSV / Parquet で保存する。"""
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".parquet":
        df.to_parquet(path)
    else:
        df.to_csv(path)
    return path


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="取引所から BTC/USDT などの OHLCV を取得する"
    )
    parser.add_argument("--symbol", default="BTC/USDT", help="取引ペア")
    parser.add_argument("--timeframe", default="1h", help="時間足 (1m, 1h, 1d ...)")
    parser.add_argument("--since", default=None, help="開始日時 (例: 2023-01-01)")
    parser.add_argument("--until", default=None, help="終了日時 (含まない)")
    parser.add_argument("--exchange", default="binance", help="ccxt の取引所 ID")
    parser.add_argument("--limit", type=int, default=1000, help="1コールの最大本数")
    parser.add_argument(
        "--output",
        default=None,
        help="保存先ファイル (.csv / .parquet)。未指定なら標準出力に先頭を表示",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    df = fetch_ohlcv(
        symbol=args.symbol,
        timeframe=args.timeframe,
        since=args.since,
        until=args.until,
        exchange_id=args.exchange,
        limit=args.limit,
    )
    print(f"{len(df)} 本の OHLCV を取得しました ({args.symbol}, {args.timeframe})")

    if args.output:
        path = save(df, args.output)
        print(f"保存しました: {path}")
    else:
        with pd.option_context("display.max_rows", 10):
            print(df)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
