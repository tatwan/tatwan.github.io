# ML Algorithm Decision Tree - Interactive Guide

## Overview

An interactive decision tree to help practitioners choose the right machine learning algorithm based on their problem type, data characteristics, and constraints. Covers scikit-learn algorithms and statsmodels time series methods.

---

## Decision Tree Structure

### Level 1: Primary Task Type

```
What is your primary goal?
├── Prediction/Inference → [Level 2A: Supervised Learning]
├── Pattern Discovery → [Level 2B: Unsupervised Learning]
├── Forecasting Future Values → [Level 2C: Time Series]
├── Limited Labels Available → [Level 2D: Semi-Supervised]
├── Learning from Interaction → [Level 2E: Reinforcement Learning]
└── Anomaly/Outlier Detection → [Level 2F: Anomaly Detection]
```

---

## Level 2A: Supervised Learning

### Question: What type of output are you predicting?

```
├── Discrete Classes → [Classification]
├── Continuous Values → [Regression]
└── Ordered Items → [Ranking/Learning to Rank]
```

---

## Classification Branch

### Question: How many classes?

```
├── Binary (2 classes) → [Binary Classification]
├── Multi-class (3+ mutually exclusive) → [Multi-class Classification]
└── Multi-label (multiple labels per instance) → [Multi-label Classification]
```

### Binary/Multi-class Classification - Question: What's your priority?

```
├── Interpretability Required → [Interpretable Models]
│   ├── Logistic Regression
│   ├── Decision Tree
│   ├── Naive Bayes
│   └── K-Nearest Neighbors (small data)
│
├── Maximum Accuracy → [High-Performance Models]
│   ├── Random Forest
│   ├── Gradient Boosting (XGBoost, LightGBM, CatBoost)
│   ├── Support Vector Machine (SVM)
│   └── Neural Networks (large data)
│
├── Fast Training/Prediction → [Efficient Models]
│   ├── Logistic Regression
│   ├── Naive Bayes
│   ├── SGD Classifier
│   └── Linear SVM
│
└── Handling Imbalanced Data → [Imbalanced-aware]
    ├── Random Forest (class_weight='balanced')
    ├── Gradient Boosting (scale_pos_weight)
    ├── SMOTE + Any classifier
    └── EasyEnsemble / BalancedRandomForest
```

### Classification Algorithms Detail

| Algorithm | Best For | Data Size | Interpretability | Handles Non-linear | Scikit-learn |
|-----------|----------|-----------|------------------|-------------------|--------------|
| Logistic Regression | Linear boundaries, baseline | Any | High | No (needs feature eng.) | `LogisticRegression` |
| Decision Tree | Interpretable rules | Small-Medium | Very High | Yes | `DecisionTreeClassifier` |
| Random Forest | General purpose, robust | Medium-Large | Medium | Yes | `RandomForestClassifier` |
| Gradient Boosting | Tabular data, competitions | Medium-Large | Low | Yes | `GradientBoostingClassifier` |
| XGBoost | Tabular data, speed | Medium-Large | Low | Yes | `xgboost.XGBClassifier` |
| LightGBM | Large data, categorical features | Large | Low | Yes | `lightgbm.LGBMClassifier` |
| CatBoost | Categorical features | Medium-Large | Low | Yes | `catboost.CatBoostClassifier` |
| SVM (RBF) | Small-medium, high-dim | Small-Medium | Low | Yes | `SVC` |
| Linear SVM | High-dimensional, text | Any | Medium | No | `LinearSVC` |
| Naive Bayes | Text, fast baseline | Any | High | No | `GaussianNB`, `MultinomialNB` |
| KNN | Small data, simple | Small | High | Yes | `KNeighborsClassifier` |
| AdaBoost | Weak learners ensemble | Medium | Low | Depends | `AdaBoostClassifier` |
| Extra Trees | Similar to RF, more random | Medium-Large | Medium | Yes | `ExtraTreesClassifier` |
| Voting Classifier | Ensemble diverse models | Any | Low | Depends | `VotingClassifier` |
| Stacking | Meta-learning | Medium-Large | Low | Depends | `StackingClassifier` |

### Multi-label Classification Algorithms

| Algorithm | Description | Scikit-learn |
|-----------|-------------|--------------|
| Binary Relevance | Independent binary classifier per label | `MultiOutputClassifier` |
| Classifier Chains | Chain of classifiers, uses previous predictions | `ClassifierChain` |
| Label Powerset | Treats each label combination as single class | `sklearn.preprocessing.MultiLabelBinarizer` + classifier |
| Multi-label KNN | KNN adapted for multi-label | `sklearn.neighbors` |

---

## Regression Branch

### Question: What relationship do you expect?

```
├── Linear Relationship → [Linear Models]
│   ├── Linear Regression (baseline)
│   ├── Ridge Regression (L2 regularization)
│   ├── Lasso Regression (L1, feature selection)
│   ├── ElasticNet (L1 + L2)
│   ├── Bayesian Ridge
│   └── Huber Regressor (robust to outliers)
│
├── Non-linear Relationship → [Non-linear Models]
│   ├── Polynomial Regression
│   ├── Decision Tree Regressor
│   ├── Random Forest Regressor
│   ├── Gradient Boosting Regressor
│   ├── SVR (Support Vector Regression)
│   ├── KNN Regressor
│   └── Gaussian Process Regression
│
├── High-dimensional / Sparse → [Regularized Linear]
│   ├── Lasso (sparse solutions)
│   ├── ElasticNet
│   ├── SGD Regressor
│   └── LARS / LARS-Lasso
│
└── Outliers Present → [Robust Models]
    ├── Huber Regressor
    ├── RANSAC Regressor
    ├── Theil-Sen Estimator
    └── Quantile Regression
```

### Regression Algorithms Detail

| Algorithm | Best For | Assumption | Handles Non-linear | Scikit-learn |
|-----------|----------|------------|-------------------|--------------|
| Linear Regression | Baseline, linear data | Linear relationship | No | `LinearRegression` |
| Ridge | Multicollinearity | Linear | No | `Ridge` |
| Lasso | Feature selection | Linear, sparse | No | `Lasso` |
| ElasticNet | Mixed L1/L2 | Linear | No | `ElasticNet` |
| Polynomial | Curved relationships | Polynomial | Yes (explicit) | `PolynomialFeatures` + `LinearRegression` |
| Decision Tree | Interpretable, non-linear | None | Yes | `DecisionTreeRegressor` |
| Random Forest | General purpose | None | Yes | `RandomForestRegressor` |
| Gradient Boosting | High accuracy | None | Yes | `GradientBoostingRegressor` |
| XGBoost | Tabular, speed | None | Yes | `xgboost.XGBRegressor` |
| LightGBM | Large data | None | Yes | `lightgbm.LGBMRegressor` |
| SVR | Small-medium data | Kernel choice | Yes | `SVR` |
| KNN Regressor | Local patterns | Local similarity | Yes | `KNeighborsRegressor` |
| Gaussian Process | Uncertainty estimation | GP prior | Yes | `GaussianProcessRegressor` |
| Huber | Outliers present | Robust | No | `HuberRegressor` |
| RANSAC | Many outliers | Inlier model | Depends | `RANSACRegressor` |
| Theil-Sen | Outliers, small data | Robust median | No | `TheilSenRegressor` |
| Bayesian Ridge | Uncertainty, small data | Bayesian prior | No | `BayesianRidge` |
| ARD Regression | Automatic relevance | Sparse Bayesian | No | `ARDRegression` |

---

## Level 2B: Unsupervised Learning

### Question: What do you want to discover?

```
├── Group Similar Items → [Clustering]
├── Reduce Dimensions → [Dimensionality Reduction]
├── Find Unusual Points → [Anomaly Detection]
├── Discover Associations → [Association Rules]
└── Learn Data Distribution → [Density Estimation]
```

---

## Clustering Branch

### Question: Do you know the number of clusters?

```
├── Yes, K is known → [Partitioning Methods]
│   ├── K-Means (spherical clusters)
│   ├── K-Medoids (robust to outliers)
│   ├── Mini-Batch K-Means (large data)
│   └── Bisecting K-Means
│
├── No, discover K → [Automatic K Methods]
│   ├── DBSCAN (density-based, finds outliers)
│   ├── HDBSCAN (varying density)
│   ├── OPTICS (density, hierarchical)
│   ├── Mean Shift (mode finding)
│   └── Affinity Propagation (exemplars)
│
├── Want Hierarchy → [Hierarchical Methods]
│   ├── Agglomerative Clustering
│   ├── BIRCH (large data)
│   └── Divisive Clustering
│
└── Special Structure → [Specialized Methods]
    ├── Spectral Clustering (non-convex shapes)
    ├── Gaussian Mixture Models (soft clustering)
    └── DBSCAN/HDBSCAN (arbitrary shapes)
```

### Clustering Algorithms Detail

| Algorithm | Shape | Scalability | Handles Noise | Auto K | Scikit-learn |
|-----------|-------|-------------|---------------|--------|--------------|
| K-Means | Spherical | Very High | No | No | `KMeans` |
| Mini-Batch K-Means | Spherical | Highest | No | No | `MiniBatchKMeans` |
| K-Medoids | Spherical | Medium | Better | No | `sklearn_extra.cluster.KMedoids` |
| DBSCAN | Arbitrary | High | Yes (labels as -1) | Yes | `DBSCAN` |
| HDBSCAN | Arbitrary | High | Yes | Yes | `hdbscan.HDBSCAN` |
| OPTICS | Arbitrary | Medium | Yes | Yes | `OPTICS` |
| Mean Shift | Arbitrary | Low | No | Yes | `MeanShift` |
| Agglomerative | Any (linkage) | Medium | No | No | `AgglomerativeClustering` |
| BIRCH | Spherical | Very High | No | No | `Birch` |
| Spectral | Non-convex | Low-Medium | No | No | `SpectralClustering` |
| Gaussian Mixture | Elliptical | Medium | No | No (use BIC) | `GaussianMixture` |
| Affinity Propagation | Exemplar-based | Low | No | Yes | `AffinityPropagation` |

### Clustering Decision Factors

| Factor | Recommendation |
|--------|----------------|
| Millions of points | Mini-Batch K-Means, BIRCH |
| Unknown number of clusters | DBSCAN, HDBSCAN, Mean Shift |
| Non-spherical clusters | DBSCAN, Spectral, HDBSCAN |
| Outliers present | DBSCAN, HDBSCAN |
| Need soft assignments | Gaussian Mixture |
| Need hierarchy | Agglomerative, BIRCH |
| High dimensions | K-Means + PCA, Spectral |

---

## Dimensionality Reduction Branch

### Question: What's your goal?

```
├── Visualization (2-3D) → [Visualization Methods]
│   ├── PCA (linear, fast)
│   ├── t-SNE (non-linear, local structure)
│   ├── UMAP (non-linear, global + local)
│   └── MDS (distance preservation)
│
├── Feature Extraction → [Feature Methods]
│   ├── PCA (variance maximization)
│   ├── Kernel PCA (non-linear)
│   ├── Incremental PCA (large data)
│   ├── Sparse PCA (interpretable)
│   ├── Factor Analysis (latent factors)
│   └── ICA (independent components)
│
├── Preserve Distances → [Distance Methods]
│   ├── MDS (metric)
│   ├── Isomap (geodesic)
│   └── Locally Linear Embedding (LLE)
│
└── Supervised Reduction → [Label-aware]
    ├── LDA (Linear Discriminant Analysis)
    └── Neighborhood Components Analysis (NCA)
```

### Dimensionality Reduction Algorithms Detail

| Algorithm | Linear | Preserves | Scalability | Scikit-learn |
|-----------|--------|-----------|-------------|--------------|
| PCA | Yes | Global variance | Very High | `PCA` |
| Incremental PCA | Yes | Global variance | Highest | `IncrementalPCA` |
| Kernel PCA | No | Kernel similarity | Medium | `KernelPCA` |
| Sparse PCA | Yes | Sparse components | Medium | `SparsePCA` |
| t-SNE | No | Local structure | Low | `TSNE` |
| UMAP | No | Local + Global | High | `umap.UMAP` |
| MDS | Yes | Distances | Low | `MDS` |
| Isomap | No | Geodesic distances | Low | `Isomap` |
| LLE | No | Local geometry | Low | `LocallyLinearEmbedding` |
| Factor Analysis | Yes | Latent factors | High | `FactorAnalysis` |
| ICA | Yes | Independence | High | `FastICA` |
| LDA | Yes | Class separation | High | `LinearDiscriminantAnalysis` |
| NCA | No | Neighbor distance | Medium | `NeighborhoodComponentsAnalysis` |
| Truncated SVD | Yes | Variance (sparse) | Very High | `TruncatedSVD` |
| Random Projection | Yes | Random (fast) | Highest | `GaussianRandomProjection` |

---

## Level 2C: Time Series

### Question: How many variables are you forecasting?

```
├── Single Variable → [Univariate Time Series]
├── Multiple Variables → [Multivariate Time Series]
└── Classification/Clustering of Series → [Time Series ML]
```

---

## Univariate Time Series Branch

### Question: Does your data have seasonality?

```
├── Yes, Seasonal → [Seasonal Models]
│   ├── Is it complex/multiple seasonalities?
│   │   ├── Yes → Prophet, TBATS, Multiple Seasonal Decomposition
│   │   └── No → SARIMA, Seasonal Exponential Smoothing
│   │
│   └── Data characteristics?
│       ├── Additive seasonality → Holt-Winters Additive
│       └── Multiplicative seasonality → Holt-Winters Multiplicative
│
├── No Seasonality → [Non-Seasonal Models]
│   ├── Is there a trend?
│   │   ├── Yes → Holt's Linear, ARIMA with differencing
│   │   └── No → Simple Exponential Smoothing, ARIMA
│   │
│   └── Stationary?
│       ├── Yes → ARMA, Simple ES
│       └── No → ARIMA (with differencing)
│
└── Not Sure → [Automatic Selection]
    ├── Auto ARIMA (pmdarima)
    ├── Prophet (Facebook)
    └── ETS (automatic exponential smoothing)
```

### Univariate Time Series Algorithms

| Algorithm | Trend | Seasonality | Complexity | Library |
|-----------|-------|-------------|------------|---------|
| **Exponential Smoothing Family** |
| Simple ES | No | No | Low | `statsmodels.tsa.holtwinters.SimpleExpSmoothing` |
| Holt's Linear | Yes | No | Low | `statsmodels.tsa.holtwinters.Holt` |
| Holt-Winters | Yes | Yes | Medium | `statsmodels.tsa.holtwinters.ExponentialSmoothing` |
| ETS (Auto) | Auto | Auto | Medium | `statsmodels.tsa.exponential_smoothing.ets.ETSModel` |
| **ARIMA Family** |
| AR | Stationary | No | Low | `statsmodels.tsa.ar_model.AutoReg` |
| MA | Stationary | No | Low | Part of ARIMA |
| ARMA | Stationary | No | Medium | `statsmodels.tsa.arima.model.ARIMA` (d=0) |
| ARIMA | Non-stationary | No | Medium | `statsmodels.tsa.arima.model.ARIMA` |
| SARIMA | Non-stationary | Yes | High | `statsmodels.tsa.statespace.sarimax.SARIMAX` |
| SARIMAX | Non-stationary | Yes + Exog | High | `statsmodels.tsa.statespace.sarimax.SARIMAX` |
| Auto ARIMA | Auto | Auto | Medium | `pmdarima.auto_arima` |
| **Modern Methods** |
| Prophet | Yes | Multiple | Medium | `prophet.Prophet` |
| TBATS | Yes | Multiple/Complex | High | `tbats.TBATS` |
| Theta | Yes | No | Low | `statsmodels.tsa.forecasting.theta.ThetaModel` |
| **Decomposition** |
| STL | - | Yes | Medium | `statsmodels.tsa.seasonal.STL` |
| Seasonal Decompose | - | Yes | Low | `statsmodels.tsa.seasonal.seasonal_decompose` |

### ARIMA Parameter Guide

| Parameter | Meaning | How to Determine |
|-----------|---------|-----------------|
| p (AR order) | Autoregressive terms | PACF plot (significant lags) |
| d (Differencing) | Integration order | ADF test, differencing until stationary |
| q (MA order) | Moving average terms | ACF plot (significant lags) |
| P (Seasonal AR) | Seasonal AR terms | Seasonal PACF |
| D (Seasonal diff) | Seasonal differencing | Usually 0 or 1 |
| Q (Seasonal MA) | Seasonal MA terms | Seasonal ACF |
| s (Season length) | Seasonal period | Domain knowledge (12=monthly, 7=daily, etc.) |

---

## Multivariate Time Series Branch

### Question: What's your goal?

```
├── Forecast Multiple Series Together → [Vector Models]
│   ├── VAR (Vector Autoregression)
│   ├── VARMA (Vector ARMA)
│   ├── VARMAX (VAR with exogenous)
│   └── VECM (Cointegrated series)
│
├── One Target with Multiple Inputs → [Transfer Function Models]
│   ├── ARIMAX / SARIMAX
│   ├── Dynamic Regression
│   └── Distributed Lag Models
│
└── Complex Dependencies → [State Space / Advanced]
    ├── Structural Time Series
    ├── Dynamic Factor Models
    └── Kalman Filter approaches
```

### Multivariate Time Series Algorithms

| Algorithm | Use Case | Assumptions | Library |
|-----------|----------|-------------|---------|
| VAR | Multiple related series | Stationary | `statsmodels.tsa.api.VAR` |
| VARMA | VAR + MA errors | Stationary | `statsmodels.tsa.statespace.varmax.VARMAX` |
| VARMAX | VAR with exogenous | Stationary | `statsmodels.tsa.statespace.varmax.VARMAX` |
| VECM | Cointegrated series | Non-stationary, cointegrated | `statsmodels.tsa.vector_ar.vecm.VECM` |
| SARIMAX | Target + exogenous | SARIMA assumptions | `statsmodels.tsa.statespace.sarimax.SARIMAX` |
| Dynamic Factor | Latent factors | Factor structure | `statsmodels.tsa.statespace.dynamic_factor.DynamicFactor` |
| Unobserved Components | Trend + Seasonal + Cycle | State space | `statsmodels.tsa.statespace.structural.UnobservedComponents` |

---

## Level 2D: Semi-Supervised Learning

### Question: How much labeled data do you have?

```
├── Very Little (< 10%) → [Self-Training Methods]
│   ├── Self-Training Classifier
│   ├── Label Propagation
│   └── Label Spreading
│
├── Some Labels (10-50%) → [Consistency Methods]
│   ├── Label Propagation
│   ├── Label Spreading
│   └── Semi-supervised SVM
│
└── Active Learning → [Query Selection]
    ├── Uncertainty Sampling
    ├── Query by Committee
    └── Expected Model Change
```

### Semi-Supervised Algorithms

| Algorithm | Method | Scikit-learn |
|-----------|--------|--------------|
| Self-Training | Iterative pseudo-labeling | `SelfTrainingClassifier` |
| Label Propagation | Graph-based propagation | `LabelPropagation` |
| Label Spreading | Soft graph-based | `LabelSpreading` |

---

## Level 2F: Anomaly Detection

### Question: What type of anomaly detection?

```
├── Point Anomalies → [Outlier Detection]
│   ├── One-Class SVM
│   ├── Isolation Forest
│   ├── Local Outlier Factor (LOF)
│   ├── Elliptic Envelope (Gaussian)
│   └── DBSCAN (cluster-based)
│
├── Contextual Anomalies → [Context-aware]
│   ├── LOF with features
│   └── Conditional density estimation
│
└── Collective Anomalies → [Sequence/Group]
    ├── Time series methods
    └── Graph-based methods
```

### Anomaly Detection Algorithms

| Algorithm | Assumption | Scalability | Scikit-learn |
|-----------|------------|-------------|--------------|
| Isolation Forest | Anomalies isolate easily | Very High | `IsolationForest` |
| One-Class SVM | Data in single class | Low-Medium | `OneClassSVM` |
| LOF | Local density difference | Medium | `LocalOutlierFactor` |
| Elliptic Envelope | Gaussian distribution | High | `EllipticEnvelope` |
| DBSCAN | Density-based | High | `DBSCAN` (noise = -1) |

---

## Feature Engineering & Preprocessing

### Question: What preprocessing do you need?

```
├── Numeric Scaling → [Scalers]
│   ├── StandardScaler (mean=0, std=1)
│   ├── MinMaxScaler (range [0,1])
│   ├── RobustScaler (outlier-resistant)
│   ├── MaxAbsScaler (sparse data)
│   └── Normalizer (unit norm)
│
├── Categorical Encoding → [Encoders]
│   ├── OneHotEncoder (nominal)
│   ├── OrdinalEncoder (ordinal)
│   ├── LabelEncoder (target)
│   └── TargetEncoder (high cardinality)
│
├── Missing Values → [Imputers]
│   ├── SimpleImputer (mean/median/mode)
│   ├── KNNImputer
│   └── IterativeImputer (MICE)
│
├── Feature Selection → [Selectors]
│   ├── VarianceThreshold (low variance)
│   ├── SelectKBest (univariate stats)
│   ├── RFE (recursive elimination)
│   ├── SelectFromModel (model-based)
│   └── SequentialFeatureSelector
│
└── Feature Creation → [Transformers]
    ├── PolynomialFeatures
    ├── SplineTransformer
    ├── FunctionTransformer
    └── FeatureUnion
```

---

## Model Selection & Validation

### Cross-Validation Strategies

| Strategy | Use Case | Scikit-learn |
|----------|----------|--------------|
| K-Fold | Standard, IID data | `KFold` |
| Stratified K-Fold | Classification, imbalanced | `StratifiedKFold` |
| Group K-Fold | Grouped data (users, etc.) | `GroupKFold` |
| Time Series Split | Temporal data | `TimeSeriesSplit` |
| Leave-One-Out | Small datasets | `LeaveOneOut` |
| Shuffle Split | Random sampling | `ShuffleSplit` |
| Repeated K-Fold | More stable estimates | `RepeatedKFold` |

### Hyperparameter Tuning

| Method | When to Use | Scikit-learn |
|--------|-------------|--------------|
| Grid Search | Small param space | `GridSearchCV` |
| Random Search | Large param space | `RandomizedSearchCV` |
| Halving Grid | Many params, limited time | `HalvingGridSearchCV` |
| Halving Random | Large space, limited time | `HalvingRandomSearchCV` |
| Bayesian Optimization | Expensive evaluations | `scikit-optimize`, `optuna` |

---

## Quick Reference Decision Matrix

### By Data Size

| Data Size | Recommended Algorithms |
|-----------|----------------------|
| < 1,000 | KNN, SVM, Gaussian Process, Decision Tree |
| 1,000 - 100,000 | Random Forest, Gradient Boosting, SVM |
| 100,000 - 1M | XGBoost, LightGBM, SGD, Neural Networks |
| > 1M | LightGBM, SGD, Mini-Batch K-Means, Online methods |

### By Interpretability Requirement

| Interpretability | Algorithms |
|-----------------|------------|
| Very High | Decision Tree, Logistic/Linear Regression, Naive Bayes |
| High | Rule-based, GAMs, Single Decision Trees |
| Medium | Random Forest (feature importance), Lasso |
| Low | Gradient Boosting, SVM, Neural Networks |

### By Speed Requirement

| Speed | Training | Prediction | Algorithms |
|-------|----------|------------|------------|
| Fastest | Very Fast | Very Fast | Naive Bayes, Linear/Logistic Regression |
| Fast | Fast | Fast | Decision Tree, KNN (small data) |
| Medium | Medium | Fast | Random Forest, LightGBM |
| Slow | Slow | Medium | SVM, Gradient Boosting |
| Slowest | Very Slow | Slow | Neural Networks, Gaussian Process |

---

## Special Cases & Recommendations

### Text Classification
1. **Simple/Fast**: Naive Bayes + TF-IDF
2. **Better**: Linear SVM + TF-IDF
3. **Best (traditional)**: Logistic Regression + n-grams
4. **Deep Learning**: Transformers (BERT, etc.)

### Image Classification
1. **Small data**: Transfer Learning (pretrained CNNs)
2. **Traditional**: HOG/SIFT features + SVM
3. **Deep Learning**: CNNs (ResNet, EfficientNet)

### Tabular Data
1. **Baseline**: Logistic/Linear Regression
2. **General**: Random Forest
3. **Competition**: XGBoost, LightGBM, CatBoost
4. **Ensemble**: Stacking multiple models

### Time Series Classification
1. **Feature-based**: Extract features + standard classifiers
2. **Distance-based**: DTW + KNN
3. **Specialized**: Rocket, Time Series Forest

---

## Algorithm Complexity Cheat Sheet

| Algorithm | Training Complexity | Prediction Complexity |
|-----------|--------------------|-----------------------|
| Linear Regression | O(np^2) | O(p) |
| Logistic Regression | O(np^2) | O(p) |
| Decision Tree | O(n * p * log(n)) | O(log(n)) |
| Random Forest | O(k * n * p * log(n)) | O(k * log(n)) |
| KNN | O(1) or O(n log n) | O(n * p) or O(log n) |
| SVM | O(n^2 * p) to O(n^3) | O(sv * p) |
| Naive Bayes | O(n * p) | O(p) |
| K-Means | O(n * k * p * i) | O(k * p) |
| DBSCAN | O(n^2) or O(n log n) | - |
| PCA | O(p^2 * n + p^3) | O(p * k) |

Where: n = samples, p = features, k = clusters/trees, i = iterations, sv = support vectors

---

## Version History

- v1.0 - Initial comprehensive guide
- Covers: scikit-learn, statsmodels, and popular extensions

---

*This guide is designed for the interactive ML Algorithm Decision Tree at tatwan.github.io*
