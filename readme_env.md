## Machine Learning
This project covers multiple areas of machine learning, including supervised learning, unsupervised learning, and reinforcement learning, with practical implementations and applications of various algorithms in each field. Particularly in unsupervised and supervised learning, the project provides detailed demonstrations of the implementation of multiple algorithms and their application scenarios, from data preprocessing, feature engineering to model training and optimization.

## Project Goals:
Master theoretical and practical applications of various machine learning algorithms, covering topics such as classification, regression, clustering, dimensionality reduction, anomaly detection, and reinforcement learning.
Demonstrate the practical application of commonly used machine learning algorithms, such as classification and regression in supervised learning, dimensionality reduction and clustering in unsupervised learning, and agent training in reinforcement learning.
Provide detailed code examples to help users understand how to apply these algorithms to solve real-world problems.

### Project Structure:

 ```
├── supervised_learning/   # Supervised Learning
├── unsupervised_learning/ # Unsupervised Learning
├── reinforcement_learning/ # Reinforcement Learning
└── README.md
```

## 1. Supervised Learning
Supervised learning trains models using known labels to learn patterns in data, enabling predictions on new data. This section covers various classification and regression algorithms and demonstrates their application on common datasets.

### Algorithms Used:

- Linear Regression: Used for regression tasks to predict continuous data variables, particularly suitable for datasets with an obvious linear relationship.

- Logistic Regression: Suitable for binary classification problems, it uses a linear model to predict the probability of outcomes.

- K-Nearest Neighbors (KNN): A distance-based classification and regression algorithm that predicts by comparing distances between data points and selecting the K nearest neighbors. Simple and intuitive, KNN works well for small datasets but struggles with high-dimensional data as distance metrics become less accurate.

- Application scenarios: Classification tasks like sentiment analysis of reviews, handwritten digit recognition.

- Decision Trees: Classifies and regresses by recursively dividing the dataset based on attribute selection criteria (e.g., Gini coefficient or information gain). Decision trees are highly interpretable but prone to overfitting, especially on small datasets with many features.

### Application scenarios: Classification and regression tasks like cancer prediction, customer churn prediction.

- Support Vector Machines (SVM): This algorithm finds a hyperplane that maximizes the margin between classes, suitable for high-dimensional datasets. SVM often performs well but can be slow to train on large datasets.

### Application scenarios: Text classification, image classification, medical diagnosis.

- Random Forest: An ensemble algorithm based on decision trees that performs classification or regression by constructing multiple trees with random feature selection. Random forests are robust, generalize well, and reduce overfitting.

### Application scenarios: Classification and regression tasks like credit scoring, risk assessment.

- Gradient Boosting Machines (GBM): This algorithm sequentially builds a set of weak learners (usually decision trees), with each new tree correcting the errors of the previous set. GBM works well on complex datasets and has variants like XGBoost, LightGBM, and CatBoost, which offer better speed and performance.

### Application scenarios: Classification and regression tasks like sales forecasting, financial market analysis, customer purchasing behavior prediction.

### Example Projects:

Housing Price Prediction: Predicting housing prices using linear regression on the Boston housing dataset.
Handwritten Digit Recognition: Using KNN, SVM, and neural networks for image classification on the MNIST dataset.
Customer Churn Prediction: Predicting customer churn using decision tree and random forest models.
Sales Data Prediction: Applying GBM models to forecast product sales and analyze trends.
File Structure:

```
supervised_learning/
├── linear_regression/  # Linear Regression
├── logistic_regression/ # Logistic Regression
├── knn/                # K-Nearest Neighbors
├── decision_tree/      # Decision Tree
├── svm/                # Support Vector Machines
├── random_forest/      # Random Forest
├── gradient_boosting/  # Gradient Boosting Machines
```

## 2. Unsupervised Learning
Unsupervised learning seeks patterns and structures in data without labels. This section focuses on clustering, dimensionality reduction, and other application areas such as anomaly detection and topic modeling.

### Algorithms Used:

- Clustering Algorithms:

- K-Means Clustering: Divides data into K clusters, with each cluster represented by its centroid, suitable for common clustering tasks.

- Hierarchical Clustering: Constructs a hierarchical clustering tree by iteratively merging or splitting data points, useful for exploring the multi-level structure of data.

- DBSCAN (Density-Based Spatial Clustering of Applications with Noise): A density-based clustering method that can identify clusters of arbitrary shapes and handle noisy data well.

- Gaussian Mixture Models (GMM): A probability-based clustering method that assumes the data is composed of multiple Gaussian distributions, suitable for tasks with fuzzy boundaries between clusters.

### Dimensionality Reduction Algorithms:

- Principal Component Analysis (PCA): Projects high-dimensional data into a low-dimensional space, preserving the major variance in the data, ideal for dimensionality reduction and visualization.

- Linear Discriminant Analysis (LDA): A supervised dimensionality reduction method that maximizes inter-class distance and minimizes intra-class distance, commonly used for dimensionality reduction in classification tasks.

- t-SNE (t-Distributed Stochastic Neighbor Embedding): A non-linear dimensionality reduction method particularly suited for visualizing high-dimensional data.

- UMAP (Uniform Manifold Approximation and Projection): Another non-linear dimensionality reduction method, faster than t-SNE, suitable for high-dimensional data visualization.


### Association Rule Learning:

### Apriori Algorithm: Used for mining frequent itemsets and association rules, especially applicable to market basket analysis.
- FP-Growth (Frequent Pattern Growth): Another method for mining frequent itemsets, more efficient than Apriori.

 Density Estimation:

- Kernel Density Estimation (KDE): Used to estimate the probability density function of data, suitable for estimating the distribution of continuous data.

- Gaussian Mixture Model (GMM): In addition to clustering, it can also be used for density estimation, modeling the probability distribution of multi-modal data.

 Self-Organizing Map (SOM):

- Self-Organizing Map: A neural network algorithm that maps high-dimensional data into a low-dimensional (usually 2D) grid, used for clustering and dimensionality reduction.
Anomaly Detection:

 Isolation Forest: An anomaly detection method based on random forests, suitable for large-scale high-dimensional data.

- Local Outlier Factor (LOF): A density-based anomaly detection method used to identify data points with relatively lower densities.

 Topic Modeling:

- Latent Dirichlet Allocation (LDA): Used for topic modeling of text data, identifying latent topics from large collections of documents.

### Autoencoders:

- Autoencoders: A type of neural network used for learning low-dimensional representations of data, commonly applied for dimensionality reduction and anomaly detection.
