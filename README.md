# btc_trading_bot

Bitcoin 1-minute OHLCV data collection and time-series forecasting prototype for research/education. The project explores a PyTorch-based forecasting pipeline around BTC/USDT data, technical indicators, preprocessing, training, and future trading-system integration.

> **Status:** prototype / research scaffold. Some modules are placeholders and the repository is not a production trading system.

## What is included

- Binance BTC/USDT OHLCV collection through `ccxt`
- Technical-indicator feature engineering (moving averages, RSI, Bollinger Bands)
- Preprocessing and train/validation/test split utilities
- PyTorch training scaffold for sequence forecasting
- Example raw/processed BTC market-data snapshot and scaler artifacts
- Planning notes for future FastAPI/UI/trading automation work

## Repository layout

```text
btc_trading_bot/
├── data/
│   ├── raw/                  # sample BTC OHLCV CSV snapshot
│   └── processed/            # generated NumPy training arrays
├── models/                   # generated scalers/features; checkpoints ignored
├── src/
│   ├── configs/              # environment-driven configuration
│   ├── data_loader/          # fetch/preprocess/split data
│   ├── features/             # technical indicators
│   ├── trainer/              # training scaffold
│   ├── strategy/             # future strategy code
│   ├── trading/              # future exchange execution code
│   └── server/               # future API code
├── requirements.txt
└── 기획서.txt
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
cp .env.example .env
```

`BINANCE_API_KEY` and `BINANCE_SECRET_KEY` are optional for public-market-data experiments, but required for any authenticated exchange action. Never commit real API keys.

## Environment variables

See [.env.example](./.env.example).

| Variable | Default | Purpose |
| --- | --- | --- |
| `BINANCE_API_KEY` | empty | Binance API key, if authenticated requests are needed |
| `BINANCE_SECRET_KEY` | empty | Binance secret key |
| `PAIR_SYMBOL` | `BTC/USDT` | Market symbol for `ccxt` |
| `TIMEFRAME` | `1m` | OHLCV timeframe |
| `FETCH_LIMIT` | `1000` | Candles per fetch request |
| `FETCH_INTERVAL_SEC` | `0.6` | Delay between API requests |
| `YEARS_TO_FETCH` | `2` | Historical lookback length |

## Common commands

Fetch market data:

```bash
python -m src.data_loader.fetch_data
```

Preprocess data and write scaler artifacts:

```bash
python -m src.data_loader.preprocess_data
```

Create train/validation/test arrays:

```bash
python -m src.data_loader.prepare_data_for_training
```

Training scaffold:

```bash
python -m src.trainer.train
```

The training scaffold may require completing/importing the dataset and model classes referenced by `src.trainer.train` before it can run end-to-end.

## Data and artifact policy

This public repository currently includes a market-data snapshot and generated NumPy/scaler artifacts for reproducibility. Future generated data, checkpoints, and local model outputs should be kept out of git unless intentionally published.

Ignored by default:

- `.env`, `.env.*` except `.env.example`
- `models/checkpoints/`
- `models/*.pth`, `models/*.pt`
- local caches and Python bytecode

## Security notes

- Do not commit exchange API keys, account identifiers, `.env` files, or live trading logs.
- Use exchange API keys with the minimum required permissions. Disable withdrawal permissions for research/testing keys.
- Test all strategy logic in paper trading/backtesting before any real order execution.

## Financial disclaimer

This repository is for educational and research purposes only. It is **not financial advice** and does not guarantee trading performance. Cryptocurrency trading is high risk and may result in loss of capital.

## License

MIT License. See [LICENSE](./LICENSE).
