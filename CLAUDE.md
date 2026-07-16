# CLAUDE.md

このファイルは、Claude Code (claude.ai/code) がこのリポジトリで作業する際のガイダンスです。

## プロジェクト概要

`crypto-ml` — 暗号資産（クリプト）の機械学習プロジェクト。
現状はセットアップ初期段階で、`.devcontainer` のみが存在します。実際のコード・依存関係・ディレクトリ構成が追加され次第、本ファイルを更新してください。

## 開発環境

- パッケージ/環境管理: **uv**（`pyproject.toml` + `uv.lock`）。Python 3.12。
- Dev Container: `mcr.microsoft.com/devcontainers/base:ubuntu24.04` をベースイメージとして使用（`.devcontainer/devcontainer.json`）。
- ホスト: WSL2 上の Linux。

## コマンド

- セットアップ（依存の同期）: `uv sync`
- 依存の追加: `uv add <package>`
- スクリプト実行: `uv run python -m <module>`
- OHLCV 取得の例:
  ```bash
  uv run python -m src.data.fetch_ohlcv \
      --symbol BTC/USDT --timeframe 1h \
      --since 2024-01-01 --output data/btc_usdt_1h.csv
  ```

## アーキテクチャ / 構成

想定するデータフロー: データ取得 → 前処理 → 特徴量 → 学習 → 評価 → 推論。

- `src/data/fetch_ohlcv.py` — ccxt 経由で取引所（既定 Binance）から OHLCV を
  取得。API の本数上限を超える範囲は `since`→`until` でページネーション取得し、
  重複除去・時刻順ソート済みの `pandas.DataFrame`（UTC の DatetimeIndex）で返す。
  CSV / Parquet 保存に対応。CLI (`python -m src.data.fetch_ohlcv`) 兼ライブラリ。
- `data/` — 取得データの保存先（`.gitignore` 済み、`.gitkeep` のみ追跡）。

## 運用メモ

- コミットは日本語・英語どちらでも可。作業者は Yushi (kamayushi529@gmail.com)。
- リモート: `origin` = https://github.com/yushi529/crypto-ml.git
- **変更を行った後は GitHub に push すること。**
