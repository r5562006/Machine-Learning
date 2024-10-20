### [README_English_version](readme_env.md)

# 機器學習

此專案涵蓋了機器學習中的多個領域，包括監督學習、非監督學習和強化學習，並且對每個領域下的各種算法進行了實作和應用。特別是在非監督學習和監督學習方面，專案詳細展示了多種算法的實現及其應用場景，從數據預處理、特徵工程到模型訓練與優化。

### 專案目標
掌握機器學習的各種算法理論與實踐應用，涵蓋分類、回歸、聚類、降維、異常檢測及強化學習等主題。
演示各種常用機器學習算法的實際應用，如監督學習中的分類和回歸，非監督學習中的降維和聚類，強化學習中的智能體訓練等。
提供詳細的代碼示例，方便用戶了解如何應用這些算法來解決實際問題。

### 專案結構
```
├── supervised_learning/   # 監督學習
├── unsupervised_learning/ # 非監督學習
├── reinforcement_learning/ # 強化學習
```

### 1. 監督學習 (Supervised Learning)
監督學習通過已知的標籤來訓練模型，學習數據中的模式，從而在新數據上進行預測。這一部分涵蓋了多種分類和回歸算法，並展示了它們在多個常見數據集上的應用。

使用的算法：
- 線性回歸 (Linear Regression)：用於回歸任務，預測連續數據變量，特別適合線性關係明顯的數據集。

- 邏輯回歸 (Logistic Regression)：適用於二元分類問題，使用線性模型來預測結果的概率。

- K-近鄰算法 (K-Nearest Neighbors, KNN)：一種基於距離度量的分類與回歸算法，通過比較數據點之間的距離，選取最接近的 K 個鄰居來進行預測。該算法簡單直觀，適合小規模數據集。其主要挑戰在於高維數據上，因為距離度量在高維空間中變得不太準確。

  應用場景：分類任務，如影評情感分析、手寫數字識別。
- 決策樹 (Decision Trees)：通過將數據集逐層劃分來進行分類和回歸，基於屬性選擇標準（如基尼係數或信息增益）來決定每個節點的劃分。決策樹具有很好的可解釋性，但容易過擬合，特別是在數據集較小或具有較多特徵的情況下。

  應用場景：分類和回歸任務，如癌症預測、客戶流失預測。
- 支持向量機 (Support Vector Machines, SVM)：該算法通過尋找能夠最大化類間距的超平面來進行分類，適合於高維數據集。SVM 通常在數據集上表現出色，但對於大型數據集訓練時間較長。

  應用場景：文本分類、圖像分類、醫療診斷。
- 隨機森林 (Random Forest)：基於決策樹的集成算法，通過多個隨機選擇特徵的決策樹來進行分類或回歸。隨機森林具有較強的穩健性和泛化能力，並能有效減少過擬合。

  應用場景：分類和回歸任務，如信用評分、風險評估。
- 梯度提升機 (Gradient Boosting Machines, GBM)：該算法通過順序地構建一組弱學習器（通常是決策樹），每個新的樹都修正前一組樹的錯誤預測。GBM 可以用於分類和回歸，並且通常在複雜數據集上表現出色。其變體包括 XGBoost、LightGBM 和 CatBoost，這些算法具有更好的速度和性能。

  應用場景：分類和回歸任務，如銷售預測、金融市場分析、顧客購買行為預測。


### 示例項目：
房價預測：基於波士頓房價數據集，使用線性回歸模型預測房價。
手寫數字識別：基於 MNIST 數據集，使用 KNN、SVM 和神經網絡進行圖像分類。
客戶流失預測：使用決策樹和隨機森林模型來預測客戶是否會流失。
銷售數據預測：應用梯度提升機 (GBM) 模型來預測產品銷售數據，並進行趨勢分析。

### 專案結構

```
supervised_learning/
├── linear_regression/  # 線性回歸
├── logistic_regression/ # 邏輯回歸
├── knn/                # K 近鄰算法
├── decision_tree/      # 決策樹
├── svm/                # 支持向量機
├── random_forest/      # 隨機森林
├── gradient_boosting/  # 梯度提升機
```

### 2. 非監督學習 (Unsupervised Learning)
非監督學習在沒有標籤的情況下，尋找數據中的模式和結構。這部分主要集中在聚類、降維及其他應用領域，如異常檢測和主題建模。

### 使用的算法：
聚類算法：
- K-Means 聚類：將數據分成 K 個簇，每個簇由其中心點（質心）表示，適用於常見的聚類任務。
  
- 層次聚類 (Hierarchical Clustering)：構建一個層次結構的聚類樹，逐步合併或分裂數據點，適合探索數據的多層次結構。
  
- DBSCAN (Density-Based Spatial Clustering of Applications with Noise)：基於密度的聚類方法，能夠識別任意形狀的簇，並且能有效處理噪聲數據。
  
- Gaussian Mixture Models (GMM)：基於概率模型的聚類方法，假設數據是由多個高斯分佈組成，適合應用於模糊邊界的聚類任務。
  
降維算法：
- 主成分分析 (PCA)：將高維數據投影到低維空間，保留數據中的主要變異性，適合於數據降維和可視化。
  
- 線性判別分析 (LDA)：有監督的降維方法，旨在最大化類間距離和最小化類內距離，常用於分類任務中的降維。
  
- t-SNE (t-Distributed Stochastic Neighbor Embedding)：非線性降維方法，特別適合高維數據的可視化。
  
- UMAP (Uniform Manifold Approximation and Projection)：另一種非線性降維方法，適合於高維數據的可視化，通常比 t-SNE 快。
  
關聯規則學習：
- Apriori 算法：用於挖掘頻繁項集和關聯規則，特別適用於市場籃分析。
  
- FP-Growth (Frequent Pattern Growth)：另一種挖掘頻繁項集的方法，效率高於 Apriori。
  
密度估計：
- 核密度估計 (KDE)：用於估計數據的概率密度函數，特別適合於連續數據的分佈估計。
  
- 高斯混合模型 (GMM)：除了用於聚類外，還可應用於密度估計，建模多模態數據的概率分佈。
  
自組織映射 (SOM)：
- 自組織映射 (Self-Organizing Map)：一種神經網絡算法，將高維數據映射到低維（通常是二維）網格中，用於聚類和降維。
  
異常檢測：
- 孤立森林 (Isolation Forest)：基於隨機森林的異常檢測方法，適合大規模高維數據的異常檢測。
- 局部離群因子 (Local Outlier Factor, LOF)：基於密度的異常檢測方法，用於識別數據集中密度較低的異常點。
  
主題建模：
- 潛在狄利克雷分配 (LDA)：用於文本數據的主題建模，從大量文檔中識別潛在主題。
  
自編碼器 (Autoencoder)：
- 自編碼器 (Autoencoder)：一種神經網絡，用於學習數據的低維表示，常應用於數據降維和異常檢測。

### 3.半監督學習（Semi-Supervised Learning）

一種機器學習方法，它結合了有標籤數據和無標籤數據進行訓練。這種方法在標籤數據稀缺但無標籤數據豐富的情況下特別有用。半監督學習的目標是利用無標籤數據來提高模型的性能，從而在標籤數據有限的情況下仍能獲得較好的學習效果。

### 專案結構

```
Semi-supervised_learning/
├── Self-Training/                            # 自訓練
├── Co-training/                              # 共訓練
├── Label Propagation/                        # 標籤傳播
├── Semi-Supervised GANs/                     # 半監督生成對抗網絡
├── Label Spreading/                          # 標籤傳遞
├── Semi-Supervised Variational Autoencoders, VAE/      # 半監督變分自編碼器
├── Semi-Supervised SVM/                      # 半監督支持向量機
```

### 半監督學習的基本概念

- 有標籤數據：這些數據包含輸入特徵和對應的目標標籤，用於指導模型學習。

- 無標籤數據：這些數據僅包含輸入特徵，沒有對應的目標標籤。無標籤數據通常比有標籤數據多得多。

- 模型訓練：半監督學習算法利用有標籤數據進行初始訓練，然後使用無標籤數據進行進一步的學習和改進。

### 半監督學習的常見算法

- 自訓練（Self-Training）：首先使用有標籤數據訓練一個初始模型，然後使用該模型對無標籤數據進行預測，將高置信度的預測結果作為新的標籤加入訓練集中，重複這一過程。

- 共訓練（Co-Training）：使用兩個或多個分類器在不同的特徵子集上進行訓練，並互相標記無標籤數據。

- 標籤傳播（Label Propagation）：將標籤信息從有標籤數據點傳播到無標籤數據點，通常使用圖形基於的方法。

- 標籤傳遞（Label Spreading）：標籤傳播的變體，使用核方法來傳播標籤，並且在傳播過程中進行正則化。

- 半監督生成對抗網絡（Semi-Supervised GANs）：生成對抗網絡通過生成器和判別器的對抗訓練來學習數據分佈，並使用判別器進行分類。

- 半監督變分自編碼器（Semi-Supervised Variational Autoencoders, VAE）：變分自編碼器通過學習數據的潛在表示來進行分類和生成。

- 半監督支持向量機（Semi-Supervised SVM）：通過在無標籤數據上進行約束來改進分類邊界。
