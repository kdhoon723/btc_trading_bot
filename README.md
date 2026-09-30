# btc_trading_bot

BTC/USDT 1분봉 데이터를 수집하고, 기술적 지표를 계산해 시계열 학습용 배열로 만드는 연구·학습용 프로젝트입니다. 저장소에는 데이터 수집과 전처리 파이프라인, 예시 데이터와 산출물이 포함되어 있지만 **학습 모델과 자동매매 기능은 완성되어 있지 않습니다.**

## 현재 구현 범위

| 단계 | 상태 | 내용 |
| --- | --- | --- |
| 시장 데이터 수집 | 구현 | `ccxt`로 Binance BTC/USDT OHLCV 수집 |
| 특성 생성 | 구현 | OHLCV, 이동평균, RSI, Bollinger Bands 계산 |
| 정규화 | 구현 | 입력 특성과 종가를 각각 `MinMaxScaler`로 변환 |
| 데이터 분할 | 구현 | 시간 순서를 유지해 train 70%, validation 20%, test 10%로 분할 |
| 모델 학습 | 실행 불가 | 학습 스크립트가 참조하는 dataset·Transformer 모델 모듈이 저장소에 없음 |
| 추론·백테스트·자동 주문 | 미구현 | 빈 패키지 디렉터리만 있고 실행 코드는 없음 |
| API·UI | 미구현 | 빈 패키지 디렉터리만 있고 실행 코드는 없음 |

따라서 현재 저장소가 제공하는 실행 범위는 **수집 → 전처리 → 학습 배열 생성**까지입니다. 프로젝트 이름과 달리 실거래 봇으로 사용할 수 없습니다.

## 데이터 처리 흐름

```text
Binance OHLCV 또는 저장소의 CSV 스냅샷
  → 이동평균(5·20·60), RSI(14), Bollinger Bands 계산
  → 결측 구간 제거
  → 입력 특성과 종가 정규화
  → 시간 순서대로 train / validation / test 분할
  → NumPy 배열과 scaler·특성 목록 저장
```

기본 입력 CSV와 생성된 NumPy 배열, scaler, 특성 목록을 저장소에 함께 두어 현재 전처리 구조를 확인할 수 있습니다. 이 파일들은 모델 성능이나 투자 수익을 입증하는 결과가 아닙니다.

## 프로젝트 구조

```text
btc_trading_bot/
├── data/
│   ├── raw/                  # BTC OHLCV CSV 스냅샷
│   └── processed/            # train/validation/test NumPy 배열
├── models/                   # scaler와 특성 목록
├── src/
│   ├── configs/              # 환경변수와 데이터 경로
│   ├── data_loader/          # 수집, 전처리, 데이터 분할
│   ├── features/             # 기술적 지표 코드
│   ├── trainer/              # 미완성 학습 스크립트
│   ├── model/                # 현재 모델 구현 없음
│   ├── inference/            # 현재 실행 코드 없음
│   ├── strategy/             # 현재 실행 코드 없음
│   ├── trading/              # 현재 실행 코드 없음
│   ├── server/               # 현재 실행 코드 없음
│   └── ui/                   # 현재 실행 코드 없음
├── .env.example
├── requirements.txt
└── 기획서.txt
```

## 시작하기

```bash
git clone https://github.com/kdhoon723/btc_trading_bot.git
cd btc_trading_bot

python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env       # Windows에서는 파일을 직접 복사
```

### 저장소의 데이터 전처리

저장소에 포함된 `data/raw/btc_1m_2years.csv`를 읽어 scaler와 특성 목록을 생성합니다.

```bash
python -m src.data_loader.preprocess_data
```

전처리와 함께 train/validation/test 배열을 다시 생성하려면 다음 명령을 실행합니다.

```bash
python -m src.data_loader.prepare_data_for_training
```

### 시장 데이터 다시 수집

`.env`의 수집 조건을 확인한 뒤 실행합니다.

```bash
python -m src.data_loader.fetch_data
```

수집기는 Binance 선물 시장을 대상으로 과거 OHLCV를 여러 번 요청합니다. 실행 시점의 거래소 API 정책, 네트워크 상태와 요청 제한이 결과에 영향을 줄 수 있습니다.

### 모델 학습 상태

다음 명령은 **현재 저장소에서 실행되지 않습니다.**

```bash
python -m src.trainer.train
```

`src/trainer/train.py`가 아래 두 모듈을 import하지만 파일이 저장소에 없습니다.

- `src.trainer.dataset.TimeSeriesDataset`
- `src.model.transformer_model.TransformerForecastModel`

학습 루프 자체에는 데이터 로드, early stopping, checkpoint 저장과 테스트 손실 계산 구조가 작성되어 있습니다. 그러나 누락된 두 구현을 추가하기 전에는 end-to-end 학습을 시작할 수 없습니다.

## 환경변수

전체 예시는 [.env.example](./.env.example)을 참고하세요.

| 변수 | 기본값 | 설명 |
| --- | --- | --- |
| `BINANCE_API_KEY` | 빈 값 | 인증 요청에 사용할 Binance API 키 |
| `BINANCE_SECRET_KEY` | 빈 값 | Binance secret key |
| `PAIR_SYMBOL` | `BTC/USDT` | 수집 대상 심볼 |
| `TIMEFRAME` | `1m` | OHLCV 타임프레임 |
| `FETCH_LIMIT` | `1000` | 요청 한 번에 받을 candle 수 |
| `FETCH_INTERVAL_SEC` | `0.6` | 요청 사이의 대기 시간(초) |
| `YEARS_TO_FETCH` | `2` | 수집을 시작할 과거 시점(연 단위) |

공개 시세 데이터만 다룰 때는 Binance 키를 비워 둘 수 있습니다. 인증이 필요한 기능을 추가할 경우 실제 키를 `.env`에만 두고 저장소에 커밋하지 마세요.

## 데이터와 산출물 정책

현재 저장소에는 파이프라인 구조를 확인할 수 있도록 시장 데이터 스냅샷과 NumPy·scaler 산출물을 포함합니다. 새로 생성한 데이터, checkpoint, 실험 로그와 로컬 모델은 공개 여부를 확인한 뒤 관리하세요.

`.gitignore`는 다음 항목을 기본적으로 제외합니다.

- `.env`, `.env.*` (`.env.example` 제외)
- `models/checkpoints/`
- `models/*.pth`, `models/*.pt`, `models/*.onnx`
- `runs/`, `logs/`, Python cache와 가상환경

## 보안과 사용 범위

- 거래소 API 키, 계정 식별 정보, `.env`, 실거래 로그를 커밋하지 마세요.
- 연구용 API 키에는 최소 권한만 부여하고 출금 권한을 비활성화하세요.
- 이 저장소에는 주문 실행 코드가 없으며, 실거래 사용을 전제로 하지 않습니다.
- 거래 전략을 추가하더라도 실주문 전에 별도의 백테스트와 paper trading 검증이 필요합니다.

## 투자 유의사항

이 저장소는 교육·연구 목적입니다. 투자 조언이 아니며 수익을 보장하지 않습니다. 암호화폐 거래에는 원금 손실 위험이 있습니다.

## 라이선스

[MIT License](./LICENSE)
