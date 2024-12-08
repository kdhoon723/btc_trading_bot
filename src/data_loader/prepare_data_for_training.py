# F:\git\btc_trading_bot\src\data_loader\prepare_data_for_training.py

import os
import logging
import numpy as np
from src.data_loader.preprocess_data import load_and_preprocess_data

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def prepare_data(models_dir="F:\\git\\btc_trading_bot\\models",
                 processed_dir="F:\\git\\btc_trading_bot\\data\\processed",
                 recent_days=None):
    """
    preprocess_data.py에서 정의한 load_and_preprocess_data 함수를 호출해
    전처리를 마친 전체 데이터(X_data_array, Y_data_array)를 로드한 뒤,
    학습(Train), 검증(Validation), 테스트(Test) 세트로 나누고
    npy 파일로 저장한다.

    recent_days=None으로 전체 데이터 사용.

    분할 비율 예:
    - Train: 70%
    - Valid: 20%
    - Test: 10%

    저장 후 train.py 등에서 불러와 학습 진행.
    """
    os.makedirs(processed_dir, exist_ok=True)

    # 전처리 수행 (None으로 전체 사용)
    X_data_array, Y_data_array, features, idx = load_and_preprocess_data(
        data_path="F:\\git\\btc_trading_bot\\data\\raw\\btc_1m_2years.csv",
        recent_days=recent_days, 
        models_dir=models_dir
    )
    logging.info("Data loaded and preprocessed successfully.")

    N = len(X_data_array)
    train_size = int(0.7 * N)
    valid_size = int(0.2 * N)
    test_size = N - train_size - valid_size
    logging.info(f"Total Samples: {N}, Train: {train_size}, Valid: {valid_size}, Test: {test_size}")

    X_train, Y_train = X_data_array[:train_size], Y_data_array[:train_size]
    X_valid, Y_valid = X_data_array[train_size:train_size+valid_size], Y_data_array[train_size:train_size+valid_size]
    X_test, Y_test = X_data_array[train_size+valid_size:], Y_data_array[train_size+valid_size:]

    # npy로 저장
    np.save(os.path.join(processed_dir, "X_train.npy"), X_train)
    np.save(os.path.join(processed_dir, "Y_train.npy"), Y_train)
    np.save(os.path.join(processed_dir, "X_valid.npy"), X_valid)
    np.save(os.path.join(processed_dir, "Y_valid.npy"), Y_valid)
    np.save(os.path.join(processed_dir, "X_test.npy"), X_test)
    np.save(os.path.join(processed_dir, "Y_test.npy"), Y_test)

    # features 목록과 scaler는 이미 preprocess_data 단계에서 models_dir에 저장되었음.
    # 여기서는 추가로 features.txt를 읽어와서 processed_dir에도 복사할 수 있음(옵션).
    features_file = os.path.join(models_dir, "features.txt")
    if os.path.exists(features_file):
        with open(features_file, 'r') as f:
            saved_features = f.read().splitlines()
        # 필요한 경우 processed_dir에 다시 저장 가능
        # 여기서는 단순 로깅
        logging.info(f"Features loaded: {saved_features}")

    logging.info("Data split into train/valid/test and saved to npy files successfully.")

if __name__ == "__main__":
    # 실행 예시
    prepare_data(recent_days=None)  # 전체 데이터 사용
