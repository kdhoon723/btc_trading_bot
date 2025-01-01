# F:\git\btc_trading_bot\src\trainer\train.py

import os
import logging
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from src.trainer.dataset import TimeSeriesDataset
from src.model.transformer_model import TransformerForecastModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def train_model(
    processed_dir="F:\\git\\btc_trading_bot\\data\\processed",
    models_dir="F:\\git\\btc_trading_bot\\models",
    window_size=1440,
    batch_size=32,
    epochs=50,
    patience=5,
    learning_rate=1e-3
):
    """
    모델 학습:
    - processed_dir에서 X_train, Y_train, X_valid, Y_valid, X_test, Y_test 로드
    - TimeSeriesDataset으로 감싸고 DataLoader 생성
    - Transformer 모델 정의 및 학습
    - Early Stopping, Best Model 저장, Scheduler 적용
    - Test 세트 평가 후 로깅
    - Best Model: models_dir/best_model.pth
    """
    # 데이터 로드
    X_train = np.load(os.path.join(processed_dir, "X_train.npy"))
    Y_train = np.load(os.path.join(processed_dir, "Y_train.npy"))
    X_valid = np.load(os.path.join(processed_dir, "X_valid.npy"))
    Y_valid = np.load(os.path.join(processed_dir, "Y_valid.npy"))
    X_test = np.load(os.path.join(processed_dir, "X_test.npy"))
    Y_test = np.load(os.path.join(processed_dir, "Y_test.npy"))

    logging.info(f"Train samples: {len(X_train)}, Valid samples: {len(X_valid)}, Test samples: {len(X_test)}")

    # Dataset/Dataloader
    train_dataset = TimeSeriesDataset(X_train, Y_train, window_size=window_size)
    valid_dataset = TimeSeriesDataset(X_valid, Y_valid, window_size=window_size)
    test_dataset = TimeSeriesDataset(X_test, Y_test, window_size=window_size)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    input_dim = X_train.shape[1]

    # 모델 생성 (Transformer 기반)
    model = TransformerForecastModel(
        input_dim=input_dim,
        d_model=64,       # 필요시 튜닝
        nhead=4,
        num_layers=3,
        dim_feedforward=256,
        dropout=0.1
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    criterion = torch.nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=2, verbose=True)

    best_val_loss = float('inf')
    no_improve_count = 0

    os.makedirs(models_dir, exist_ok=True)
    best_model_path = os.path.join(models_dir, "best_model.pth")

    for epoch in range(1, epochs+1):
        # Train phase
        model.train()
        train_loss_sum = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            pred = model(X_batch)
            loss = criterion(pred, y_batch)
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item() * len(X_batch)
        train_loss = train_loss_sum / len(train_loader.dataset)

        # Validation phase
        model.eval()
        val_loss_sum = 0
        with torch.no_grad():
            for X_batch, y_batch in valid_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                pred = model(X_batch)
                loss = criterion(pred, y_batch)
                val_loss_sum += loss.item() * len(X_batch)
        val_loss = val_loss_sum / len(valid_loader.dataset)

        scheduler.step(val_loss)

        logging.info(f"Epoch {epoch}/{epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")

        # Early Stopping & Checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve_count = 0
            torch.save(model.state_dict(), best_model_path)
            logging.info("New best model saved.")
        else:
            no_improve_count += 1
            if no_improve_count >= patience:
                logging.info("Early stopping triggered.")
                break

    # Best 모델 로드
    model.load_state_dict(torch.load(best_model_path))
    model.eval()

    # Test 평가
    test_loss_sum = 0
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            pred = model(X_batch)
            loss = criterion(pred, y_batch)
            test_loss_sum += loss.item() * len(X_batch)
    test_loss = test_loss_sum / len(test_loader.dataset)
    logging.info(f"Test Loss: {test_loss:.6f}")

    return model, test_loss


if __name__ == "__main__":
    model, test_loss = train_model()
    print("Final Test Loss:", test_loss)
