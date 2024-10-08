# digit_recognition.py
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.datasets import load_digits

# 加載數據集
digits = load_digits()
X = digits.data
y = digits.target

# 分割數據集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 創建並訓練模型
model = SVC()
model.fit(X_train, y_train)

# 預測
y_pred = model.predict(X_test)

# 評估
print(classification_report(y_test, y_pred))