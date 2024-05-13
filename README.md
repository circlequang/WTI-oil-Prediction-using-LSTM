   
**Hướng dẫn sơ lược:** 
1. Dataset: lấy giá đóng cửa từ 2002-02-07 đến 2024-02-13, tổng cộng 5518 dòng dữ liệu. Đây là số lượng data tương đối phù hợp huấn luyện
2. Mô hình dự đoán: dùng mô hình Long Short Term Memory (LSTM) là một kiến trúc của mạng nơ-ron hồi quy (Recurrent Neural Network - RNN), và nó thuộc về lĩnh vực học sâu (Deep Learning), một nhánh của học máy (Machine Learning).
3. Mô hình dùng 10 bước giá gần nhất để dự đoán giá phiên kế tiếp, dùng 2 mạng LSMT xếp chồng lên nhau.
   
**Chú ý:**  Mô hình dự đoán tốt trên  giá dầu WTI, không có nghĩa là sẽ dự đoán tốt trên các forex/ crypto hay mã chứng khoán khác. Do mỗi loại sẽ có đặc thù riêng.

Xem thêm thông tin tại https://dudoangia.com/du-doan-gia-dau-wti-bang-thuat-toan-lsmt/ 
