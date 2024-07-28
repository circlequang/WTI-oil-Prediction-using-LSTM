import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Read CSV file
file_path = 'wti-20240213.csv'
data = pd.read_csv(file_path, delimiter=';')
data['Date-time'] = pd.to_datetime(data['Date-time'])
data = data.sort_values('Date-time')
close_prices = data['Close'].values

# Split data into training and testing sets
train_size = int(len(close_prices) * 0.9)
train, test = close_prices[:train_size], close_prices[train_size:]

# Function to create dataset
def create_dataset(data, time_step=1):
    X, y = [], []
    for i in range(len(data)-time_step-1):
        a = data[i:(i+time_step)]
        X.append(a)
        y.append(data[i + time_step])
    return np.array(X), np.array(y)

time_step = 10
X_train, y_train = create_dataset(train, time_step)
X_test, y_test = create_dataset(test, time_step)

print(X_train)
print(y_train)

# Reshape input to be [samples, time steps, features] for LSTM
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# Split data into training and testing sets
train_size = int(len(close_prices) * 0.9)
train, test = close_prices[:train_size], close_prices[train_size:]

# Build LSTM model
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(time_step, 1)))
model.add(LSTM(50, return_sequences=False))
model.add(Dense(1))

model.compile(loss='mean_squared_error', optimizer='adam')

model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100, batch_size=64, verbose=1)

# Predict
test_predict = model.predict(X_test)

# Print actual prices and predicted prices
print("Actual prices and predicted prices:")
for i in range(len(y_test)):
    print(f"Actual: {y_test[i]}, Predicted: {test_predict[i][0]}")

# Calculate MSE
mse = mean_squared_error(y_test, test_predict)
print(f"MSE: {mse}")

# Plot the chart
plt.plot(y_test, label='Actual Prices')
plt.plot(test_predict, label='Predicted Prices', alpha=0.7)
plt.xlabel("Time")
plt.ylabel("Closing Price")
plt.legend()
plt.show()
