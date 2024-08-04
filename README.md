# 🌟 MNIST Dataset Analysis: Gradient Boosting for Regression

Welcome to the MNIST Dataset project transforming the classification problem into a regression task. This repository contains code and documentation to perform gradient boosting using absolute loss, with a focus on decision stumps and PCA for dimensionality reduction. Let's get started! 🚀

## 📁 Dataset

Use the [MNIST dataset](https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz) for the following tasks, focusing on digits 0 and 1, labeled as -1 and 1.

## 🧩 Task 1: Data Preparation

1. **Divide the Dataset**:
   - Split the train set into train and validation sets.
   - Keep **1000 samples** from each class for validation.
   - Validation set should be used to evaluate the performance of the classifier and must not be used in obtaining the PCA matrix.

## 📊 Task 2: Principal Component Analysis (PCA)

1. **Apply PCA** and reduce the dimension to **p = 5**.
2. Use the training set of the two classes to obtain the PCA matrix.
3. For the remaining tasks, use the reduced dimension dataset.

## 🌲 Task 3: Decision Stump Learning

1. **Learn a Decision Stump** using the training set:
   - For each dimension, find the unique values and sort them in ascending order.
   - Evaluate splits at the midpoint of two consecutive unique values.
   - Find the best split by minimizing the sum of squared residuals (SSR).
   - Denote this as **h1(x)**.

## 🔄 Task 4: Gradient Boosting Implementation

1. **Compute Residue** using \( y - 0.01 \times h1(x) \).
2. Build another tree **h2(x)** using the training set with updated labels (negative gradients).
3. Compute residue using \( y - 0.01 \times h1(x) - 0.01 \times h2(x) \).
4. Repeat to grow **300 such stumps**, updating labels every iteration based on negative gradients.

## 📈 Task 5: Evaluation

1. After every iteration, find the **MSE on validation set** and report.
2. Show a **plot of MSE on validation set vs. number of trees**.
3. Use the tree that gives the lowest MSE to evaluate on the test set.
4. **Report test MSE**.
## 🤝 Contributing
Feel free to open issues or submit pull requests if you have any suggestions or improvements.

## 📄 License
This project is licensed under the MIT License.
