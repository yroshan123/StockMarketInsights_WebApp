import torch, torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, output_dim=1, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.fc   = nn.Linear(hidden_dim, output_dim)
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

def train_model(model, X_train, y_train, epochs=50, batch=64, lr=1e-3):
    ds = torch.utils.data.TensorDataset(X_train, y_train)
    dl = torch.utils.data.DataLoader(ds, batch_size=batch, shuffle=True)
    crit = nn.MSELoss()
    opt  = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(epochs):
        model.train()
        for xb, yb in dl:
            opt.zero_grad()
            pred = model(xb)
            loss = crit(pred, yb.unsqueeze(1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
    return model