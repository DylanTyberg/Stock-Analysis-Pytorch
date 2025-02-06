#%%
import torch
from torch import nn
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

data = pd.read_csv("s&p Oct 2011 - Oct 2024.csv")
data['Date'] = pd.to_datetime(data['Date'])
data = data.sort_values(by='Date')

for column in data.columns:
    if data[column].dtype == 'object':
        data[column] = pd.to_numeric(data[column].str.replace(',', ''), errors='coerce')

num_days = 30
data["SMA"] = data["Close"].rolling(window=num_days).mean()
data["EMA"] = data["Close"].ewm(span=num_days, adjust=False).mean()
data = data.dropna()

def calculate_rsi(data, column="Close", window=14):
    delta = data[column].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    RS = gain / loss
    RSI = 100 - (100 / (1 + RS))
    return RSI

data["RSI"] = calculate_rsi(data, column="Close", window=7)
data = data.dropna()

scaler = MinMaxScaler()
original_close = data["Close"]
columns_to_scale = data.filter(regex="Close|SMA|EMA|RSI|Open|High|Low").columns
data[columns_to_scale] =  scaler.fit_transform(data[columns_to_scale])

def create_sequences(data, seq_len):
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i:(i + seq_len)])
        y.append(data[i + seq_len, 0])
    return np.array(X), np.array(y)

sequence_length = 10
X, y = create_sequences(data[columns_to_scale].values, sequence_length)

train_split = int(0.8* len(X))
X_train, y_train = X[:train_split], y[:train_split]
X_test, y_test = X[train_split:], y[train_split:]

X_train = torch.tensor(X_train, dtype=torch.float32).view(-1, sequence_length, 7)
X_test = torch.tensor(X_test, dtype=torch.float32).view(-1, sequence_length, 7)
y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
y_test = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

class LSTM(nn.Module):
    def __init__(self, input_shape, hidden_units, layer_dim, output_shape):
        super(LSTM, self).__init__()
        self.hidden_units = hidden_units
        self.layer_dim = layer_dim
        self.lstm = nn.LSTM(input_shape, hidden_units, layer_dim, batch_first=True)
        self.lstm = nn.LSTM(input_shape, hidden_units, layer_dim, batch_first=True, dropout=0.4)
        self.fc = nn.Linear(hidden_units, output_shape)

    def forward(self, X):
        h0 = torch.zeros(self.layer_dim, X.size(0), self.hidden_units).requires_grad_()
        c0 = torch.zeros(self.layer_dim, X.size(0), self.hidden_units).requires_grad_()
        out, _ = self.lstm(X, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

model = LSTM(7, 128, 1, 1)
loss_fn = nn.L1Loss()
optimizer = torch.optim.Adam(params=model.parameters(), lr=0.005, weight_decay=1e-5)

epochs = 700
for epoch in range(epochs):
    model.train()
    y_pred = model(X_train)
    loss = loss_fn(y_pred, y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    model.eval()
    test_pred = model(X_test)
    test_loss = loss_fn(test_pred, y_test)

    if epoch % 100  == 0:
        print(f"Epoch: {epoch}|Train Loss: {loss}|Test Loss: {test_loss}")

model.eval()
with torch.inference_mode():
    predictions = model(X_test)
    predictions = predictions.detach().numpy()
    zero_arrays = [np.zeros_like(predictions) for _ in range(6)]
    predictions = scaler.inverse_transform(np.concatenate([predictions] + zero_arrays, axis=1))


df = data.iloc[train_split:train_split + len(predictions)].copy()
df["Close"] = original_close[train_split:train_split + len(predictions)]
df["Predicted Close"] = predictions[:, 0]
print(df[["Date", "Close", "Predicted Close"]])


import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
plt.plot(df["Date"], df["Close"], label="Actual Close", color="blue")
plt.plot(df["Date"], df["Predicted Close"], label="Predicted Close", color="red")
plt.title("Actual vs Predicted Prices")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
# %%
seq = data[-10:].copy()
input = seq[columns_to_scale].values
input_tensor = torch.tensor(input, dtype=torch.float32).unsqueeze(0)
model.eval()
with torch.inference_mode():
    predicted_close = model(input_tensor).item()
    predicted_close_scaled  = scaler.inverse_transform([[predicted_close] + [0] * (len(columns_to_scale) - 1)])[0][0]
    print(f"Last Close: {original_close.iloc[-1]}")
    print("Raw predicted_close from model:", predicted_close_scaled)

# %%
#1. Prepare the last sequence of data
def update_features(data, new_close_value, num_days):
    new_open_value = data.iloc[-1]['Close']
    
    deviation = 0.01
    calculated_high = new_close_value * (1 + deviation)
    calculated_low = new_close_value * (1 - deviation)
    
    new_high_value = max(calculated_high, calculated_low)
    new_low_value = min(calculated_high, calculated_low)
    print("Current last date in data:", data["Date"].iloc[-1])

    # Calculate the next date
    next_date = data["Date"].iloc[-1] + pd.Timedelta(days=1)
    print("Next date:", next_date)
    
    new_row = {"Date":next_date, "Open": new_open_value, "High": new_high_value, "Low": new_low_value, "Close":new_close_value, "SMA": np.nan, "EMA":np.nan, "RSI":np.nan}
    new_row = pd.DataFrame([new_row])
    print(new_row)
    #new_row[columns_to_scale] = scaler.transform(new_row[columns_to_scale])
    #new_row["Open"] = new_open_value
    print(new_row)
    print(data.tail())
    data = pd.concat([data, new_row], ignore_index=True)
    print(data.tail())
    data['Date'] = pd.to_datetime(data['Date'])
    data["SMA"] = data["Close"].rolling(window=num_days).mean()
    data["EMA"] = data["Close"].ewm(span=num_days, adjust=False).mean()
    data["RSI"] = calculate_rsi(data, "Close", 14)
    
    return data

def forecast_future_prices(model, data, scaler, seq_len, num_steps, num_days):
    last_sequence = data[-seq_len:].copy()
    predictions = []
    for step in range(num_steps):
        print(f"last {last_sequence}")
        input_data = last_sequence[columns_to_scale].values
        print(input_data)
        input_tensor = torch.tensor(input_data, dtype=torch.float32).unsqueeze(0)
        print(input_tensor)
        
        model.eval()
        with torch.inference_mode():
            predicted_close = model(input_tensor).item()
            print("Raw predicted_close from model:", predicted_close)
            
        predicted_close_scaled  = scaler.inverse_transform([[predicted_close] + [0] * (len(columns_to_scale) - 1)])[0][0]
        predictions.append(predicted_close_scaled)
        
        data = update_features(data, predicted_close, num_days)
        last_sequence = data[-seq_len:]
    
    return predictions

seq_len = 10
num_steps = 10
num_days = 30

predictions = forecast_future_prices(model, data, scaler, seq_len, num_steps, num_days)
print("Updated data (last few rows):")
print(data.tail())
print("Predicted future prices:", predictions)

# %%
