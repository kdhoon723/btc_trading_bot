# btc_trading_bot

비트코인 1분봉 OHLCV 데이터를 수집하고, 시계열 예측 모델을 실험하기 위한 연구/학습용 프로토타입입니다. BTC/USDT 데이터 수집, 보조지표 계산, 전처리, PyTorch 학습 스캐폴드, 향후 자동매매 시스템 연동 구상을 포함합니다.

> **상태:** 프로토타입 / 연구용 스캐폴드입니다. 일부 모듈은 아직 placeholder이며, 실전 자동매매 시스템으로 바로 사용할 수 없습니다.

## 포함된 내용

- `ccxt` 기반 Binance BTC/USDT OHLCV 수집
- 이동평균, RSI, Bollinger Bands 등 기술적 지표 생성
- 전처리 및 train/validation/test 분할 유틸리티
- PyTorch 기반 시계열 예측 학습 스캐폴드
- 예시 BTC 시장 데이터 스냅샷과 scaler 산출물
- FastAPI/UI/자동매매 연동을 위한 기획 메모

## 폴더 구조

```text
btc_trading_bot/
├── data/
│   ├── raw/                  # 예시 BTC OHLCV CSV 스냅샷
│   └── processed/            # 생성된 NumPy 학습 배열
├── models/                   # scaler/features 산출물; checkpoints는 ignore
├── src/
│   ├── configs/              # 환경변수 기반 설정
│   ├── data_loader/          # 데이터 수집·전처리·분할
│   ├── features/             # 기술적 지표
│   ├── trainer/              # 학습 스캐폴드
│   ├── strategy/             # 향후 전략 코드
│   ├── trading/              # 향후 거래소 실행 코드
│   └── server/               # 향후 API 코드
├── requirements.txt
└── 기획서.txt
```

## 설치

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
cp .env.example .env
```

`BINANCE_API_KEY`, `BINANCE_SECRET_KEY`는 공개 시세 데이터 실험만 할 때는 비워둘 수 있습니다. 인증이 필요한 거래소 기능을 사용할 때만 설정하세요. 실제 API 키는 절대 커밋하지 않습니다.

## 환경변수

자세한 예시는 [.env.example](./.env.example)을 참고하세요.

| 변수 | 기본값 | 설명 |
| --- | --- | --- |
| `BINANCE_API_KEY` | 빈 값 | 인증 요청이 필요할 때 사용할 Binance API key |
| `BINANCE_SECRET_KEY` | 빈 값 | Binance secret key |
| `PAIR_SYMBOL` | `BTC/USDT` | `ccxt` 수집 대상 심볼 |
| `TIMEFRAME` | `1m` | OHLCV 타임프레임 |
| `FETCH_LIMIT` | `1000` | 한 번에 요청할 candle 수 |
| `FETCH_INTERVAL_SEC` | `0.6` | API 요청 간 대기 시간 |
| `YEARS_TO_FETCH` | `2` | 과거 데이터 수집 기간 |

## 자주 쓰는 명령

시장 데이터 수집:

```bash
python -m src.data_loader.fetch_data
```

전처리 및 scaler 산출물 생성:

```bash
python -m src.data_loader.preprocess_data
```

학습/검증/테스트 배열 생성:

```bash
python -m src.data_loader.prepare_data_for_training
```

학습 스캐폴드 실행:

```bash
python -m src.trainer.train
```

`src.trainer.train`에서 참조하는 dataset/model 클래스는 아직 완성 전일 수 있으므로, end-to-end 학습 전에 해당 모듈 구현 상태를 확인해야 합니다.

## 데이터와 산출물 정책

현재 공개 저장소에는 재현성을 위한 시장 데이터 스냅샷과 NumPy/scaler 산출물이 포함되어 있습니다. 향후 새로 생성되는 데이터, checkpoint, 실험 로그, 로컬 모델 산출물은 의도적으로 공개할 때를 제외하고 git에 넣지 마세요.

기본적으로 ignore되는 항목:

- `.env`, `.env.*` — 단 `.env.example`은 포함
- `models/checkpoints/`
- `models/*.pth`, `models/*.pt`
- 로컬 cache, Python bytecode, 실험 로그

## 보안 메모

- 거래소 API key, 계정 식별자, `.env` 파일, 실거래 로그는 커밋하지 않습니다.
- 연구/테스트용 거래소 key에는 최소 권한만 부여하고, 출금 권한은 비활성화하세요.
- 모든 전략은 실주문 전에 paper trading과 backtest로 충분히 검증해야 합니다.

## 투자 유의사항

이 저장소는 교육·연구 목적입니다. 투자 조언이 아니며 수익을 보장하지 않습니다. 암호화폐 거래는 원금 손실 위험이 큽니다.

## 라이선스

MIT License. 자세한 내용은 [LICENSE](./LICENSE)를 참고하세요.
