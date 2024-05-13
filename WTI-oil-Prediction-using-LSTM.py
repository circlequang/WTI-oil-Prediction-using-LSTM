import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Đọc file CSV
file_path = 'wti-20240213.csv'
data = pd.read_csv(file_path, delimiter=';')
data['Date-time'] = pd.to_datetime(data['Date-time'])
data = data.sort_values('Date-time')
close_prices = data['Close'].values


# Chia dữ liệu thành tập huấn luyện và tập kiểm tra
train_size = int(len(close_prices) * 0.9)
train, test = close_prices[:train_size], close_prices[train_size:]

# Hàm để tạo dataset
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

# Reshape input để có thể đưa vào LSTM [samples, time steps, features]
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)


# Chia dữ liệu thành tập huấn luyện và tập kiểm tra
train_size = int(len(close_prices) * 0.9)
train, test = close_prices[:train_size], close_prices[train_size:]


# Xây dựng mô hình LSTM
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(time_step, 1)))
model.add(LSTM(50, return_sequences=False))
model.add(Dense(1))

model.compile(loss='mean_squared_error', optimizer='adam')


model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100, batch_size=64, verbose=1)


# Dự đoán
test_predict = model.predict(X_test)

# In giá thực tế và giá dự đoán
print("Giá thực tế và giá dự đoán:")
for i in range(len(y_test)):
    print(f"Thực tế: {y_test[i]}, Dự đoán: {test_predict[i][0]}, , Dự đem đoán: {X_test[i][0]}")

# Tính MSE
mse = mean_squared_error(y_test, test_predict)
print(f"MSE: {mse}")

# Vẽ biểu đồ
plt.plot(y_test, label='Giá Thực Tế')
plt.plot(test_predict, label='Giá Dự Đoán', alpha=0.7)
plt.xlabel("Thời gian")
plt.ylabel("Giá Đóng Cửa")
plt.legend()
plt.show()
