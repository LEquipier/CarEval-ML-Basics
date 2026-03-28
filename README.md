# UIC 数据挖掘项目

## 项目简介

本项目是北京师范大学-香港浸会大学联合国际学院（UIC）数据挖掘课程的实践项目，使用**汽车评估数据集（Car Evaluation Dataset）**，分别基于 K 近邻（KNN）、逻辑回归（Logistic Regression）和随机森林（Random Forest）三种算法对汽车综合评级进行多分类预测。

---

## 数据集说明

数据集来源于经典的汽车评估数据集，共包含以下 6 个特征和 1 个目标变量：

| 特征 | 描述 | 取值范围 |
|------|------|----------|
| `buying` | 购买价格 | low / med / high / vhigh |
| `maint` | 维护费用 | low / med / high / vhigh |
| `doors` | 车门数量 | 2 / 3 / 4 / 5more |
| `persons` | 载客人数 | 2 / 4 / more |
| `lug_boot` | 行李箱大小 | small / med / big |
| `safety` | 安全性 | low / med / high |
| `evaluation` | **评估结果（目标变量）** | unacc / acc / good / vgood |

- 训练集：`training.csv`，共 **1330** 条样本
- 测试集：`test.csv`，共 **333** 条样本

所有特征均为类别型变量，预处理阶段使用 `LabelEncoder` 进行数值编码。

---

## 项目结构

```
.
├── training.csv              # 训练数据集
├── test.csv                  # 测试数据集
├── KNN.ipynb                 # K 近邻算法实现（含从零手写版本）
├── Logistic Regression.ipynb # 逻辑回归模型训练与调优
├── Random_Forest.ipynb       # 随机森林模型训练与调优
└── README.md
```

---

## 模型介绍

### 1. K 近邻（KNN）
- 包含从零实现的 KNN 算法逻辑（纯 Python，不依赖 sklearn）
- 实现了数据集加载、距离计算（欧氏距离）、投票分类等核心模块

### 2. 逻辑回归（Logistic Regression）
- 使用 `sklearn.linear_model.LogisticRegression`，求解器为 `newton-cg`，多分类策略为 `multinomial`
- 绘制**学习曲线**与**验证曲线**分析模型表现
- 使用 `GridSearchCV`（5 折交叉验证）对超参数 `C`、`solver` 进行网格搜索调优

### 3. 随机森林（Random Forest）
- 使用 `sklearn.ensemble.RandomForestClassifier`
- 依次对 `n_estimators`（树的数量）和 `max_features`（特征数量）绘制验证曲线
- 使用 `GridSearchCV`（10 折交叉验证）进行超参数调优
- 评估指标包括**准确率（Accuracy）**和**宏平均 F1 分数（Macro F1-Score）**

---

## 数据可视化

项目在探索性数据分析（EDA）阶段包含以下可视化内容：

- 目标变量 `evaluation` 的类别分布（`sns.countplot`）
- 各特征与目标变量交叉的分组柱状图
- 特征相关性热力图（`sns.heatmap`）

---

## 环境依赖

```bash
Python >= 3.6
numpy
pandas
matplotlib
seaborn
scikit-learn
```

安装依赖：

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

---

## 运行方式

使用 Jupyter Notebook 依次打开并运行以下文件：

```bash
jupyter notebook
```

推荐运行顺序：
1. `Logistic Regression.ipynb` — 数据探索与基础建模
2. `Random_Forest.ipynb` — 集成方法调优
3. `KNN.ipynb` — 算法从零实现
