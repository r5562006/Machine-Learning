# spam_detection.py
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# 示例數據
data = {
    'feature1': np.random.randn(100),
    'feature2': np.random.randn(100),
    'is_spam': np.random.randint(0, 2, 100)
}

# 創建 DataFrame
df = pd.DataFrame(data)

# 分割數據集
X = df[['feature1', 'feature2']]
y = df['is_spam']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 創建並訓練模型
model = LogisticRegression()
model.fit(X_train, y_train)

# 預測
y_pred = model.predict(X_test)

# 評估
print(classification_report(y_test, y_pred))