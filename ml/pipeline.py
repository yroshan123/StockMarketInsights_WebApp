import numpy as np, torch
from ml.features import build_features_for_ticker
from ml.model import LSTMModel, train_model

def fit_and_predict_one_ticker(stock_df, sent_df, seq_len=20, epochs=50, batch=64, lr=1e-3, test_ratio=0.2, horizon=7):
    df_full, feats, targ, Xtr, ytr, Xte, yte, fsc, tsc = build_features_for_ticker(
        stock_df, sent_df, seq_len=seq_len, test_ratio=test_ratio, horizon=horizon
    )
    if len(Xtr) < 5 or len(Xte) < 1:
        raise ValueError("Not enough data for sequences (try shorter seq_len or smaller test split).")

    model = LSTMModel(input_dim=len(feats))
    model = train_model(model, Xtr, ytr, epochs=epochs, batch=batch, lr=lr)

    model.eval()
    with torch.no_grad():
        pred_scaled = model(Xte).squeeze().numpy()
    y_pred = tsc.inverse_transform(pred_scaled.reshape(-1,1)).ravel()
    y_true = tsc.inverse_transform(yte.numpy().reshape(-1,1)).ravel()
    rmse = float(np.sqrt(np.mean((y_pred - y_true)**2)))

    # Ahead-of-time prediction from the latest observed window
    last_scaled = df_full.copy()
    last_scaled[feats] = fsc.transform(last_scaled[feats])
    last_seq = torch.tensor(last_scaled[feats].iloc[-seq_len:].values, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        next_return_scaled = model(last_seq).item()
    next_return = float(tsc.inverse_transform([[next_return_scaled]])[0,0])

    last_close = float(df_full["Close"].iloc[-1])
    price_h_pred = float(last_close * (1.0 + next_return))

    return {
        "rmse_return": rmse,
        "pred_next_return": next_return,
        "last_close": last_close,
        "pred_price_in_h": price_h_pred,
    }

# Re-export for app plotting
def build_features_for_ticker(*args, **kwargs):
    from ml.features import build_features_for_ticker as _bf
    return _bf(*args, **kwargs)