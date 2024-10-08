# house_price_prediction.py
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# 示例數據
data = {
    'area': [50, 60, 70, 80, 90, 100, 110, 120, 130, 140],
    'rooms': [1, 2, 2, 3, 3, 3, 4, 4, 4, 5],
    'price': [150, 200, 250, 300, 350, 400, 450, 500, 550, 600]
}

# 創建 DataFrame
df = pd.DataFrame(data)

# 分割數據集
X = df[['area', 'rooms']]
y = df['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 創建並訓練模型
model = LinearRegression()
model.fit(X_train, y_train)

# 預測
y_pred = model.predict(X_test)

# 繪圖
plt.scatter(X_test['area'], y_test, color='black')
plt.scatter(X_test['area'], y_pred, color='blue')
plt.xlabel('Area')
plt.ylabel('Price')
plt.title('House Price Prediction')
plt.show()