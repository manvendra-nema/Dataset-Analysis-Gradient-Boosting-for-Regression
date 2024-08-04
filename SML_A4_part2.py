#!/usr/bin/env python
# coding: utf-8

# In[15]:


from tqdm import tqdm
import numpy as np
from sklearn.decomposition import PCA
import random  # Add import for random module
random.seed(10)

class Node:
    def __init__(self, data_indices, depth):
        self.data_indices = data_indices
        self.depth = depth
        self.left = None
        self.right = None
        self.split_dim = None
        self.split_value = None
        self.label = None

def create_data_matrix( X_train ):
    X = X_train.reshape(-1, 784)
    X = X.T
    return X


def center_data(X, mean=None):
    if mean is None:
        mean = np.mean(X, axis=1, keepdims=True)
    X_centered = X - mean
    return X_centered, mean

def apply_pca(X_centered, p):
    covariance_matrix = np.matmul(X_centered, X_centered.T) / (X_centered.shape[1] - 1)
    V, U = np.linalg.eigh(covariance_matrix)

    sorted_indices = np.argsort(V)[::-1]
    U_sorted = U[:, sorted_indices][:, :p]

    Y = U_sorted.T @ X_centered
    return U_sorted, Y

def reconstruct_data(U_sorted, Y):
    X_recon = U_sorted @ Y
    return X_recon

def calculate_mse(X_centered, X_recon):
    mse = np.sum((X_centered - X_recon) ** 2) / X_centered.size
    return mse

def plot_reconstructed_images(X_recon_p, p):
    fig, axes = plt.subplots(10, 5, figsize=(10, 10))
    for i in range(10):
        for j in range(5):
            axes[i, j].imshow(X_recon_p[:, i * 100 + j].reshape(28, 28), cmap='cubehelix_r')
            axes[i, j].axis('off')

    plt.suptitle(f"Reconstructed Images with p={p}")
    plt.show()

def calculate_class_accuracy(y_true, y_pred):

    num_classes = 3
    accuracy_per_class = np.zeros(num_classes)
    total_per_class = np.zeros(num_classes)

    for true_label, pred_label in zip(y_true, y_pred):
        total_per_class[true_label] += 1
        if true_label == pred_label:
            accuracy_per_class[true_label] += 1

    accuracy_per_class = accuracy_per_class / total_per_class
    return accuracy_per_class



def print_tree(root, level=0, prefix="Root:"):
    if root is not None:
        print(" " * (level * 4) + prefix, root.label)
        if root.left is not None or root.right is not None:
            if root.left is not None:
                print_tree(root.left, level + 1, prefix="L--")
            else:
                print(" " * ((level + 1) * 4) + "L--None")
            if root.right is not None:
                print_tree(root.right, level + 1, prefix="R--")
            else:
                print(" " * ((level + 1) * 4) + "R--None")



def SSR(y):
    mean_y = np.mean(y)
    return np.sum((y - mean_y)**2 )

def find_best_split(X, y, data_indices):
    n_samples, n_features = X[data_indices].shape
    
    best_loss = float('inf')
    best_split_dim = None
    best_split_value = None
    
    for dim in range(n_features):
        unique_values = np.unique(X[data_indices, dim])
        
        for i in range(len(unique_values) - 1):
            value = (unique_values[i] + unique_values[i + 1]) / 2  # Midpoint split
            left_indices = data_indices[X[data_indices, dim] <= value]
            right_indices = data_indices[X[data_indices, dim] > value]
            
            # Calculate SSR for left and right splits
            SSR_left = SSR(y[left_indices])
            SSR_right = SSR(y[right_indices])
            
            # Calculate total SSR for the split
            total_SSR = SSR_left + SSR_right
            
            if total_SSR < best_loss:
                best_loss = total_SSR
                best_split_dim = dim
                best_split_value = value
                
    return best_split_dim, best_split_value, best_loss
    

def assign_label_for_node(node, y):
    node.label = np.mean(y[node.data_indices])
    return node

def check_stopping_criteria(depth, max_depth):
    if depth >= max_depth:
        return True
    return False

def grow_tree(X, y, data_indices=None, depth=0, max_depth=1):
    global total_leaf_nodes
    
    if data_indices is None:
        data_indices = np.arange(X.shape[0])

    n_samples, n_features = X[data_indices].shape

    node= Node(data_indices=data_indices, depth=depth)

    if check_stopping_criteria(depth, max_depth):
        assign_label_for_node(node, y)
        total_leaf_nodes += 1
        return node, None

    best_split_dim, best_split_value,best_loss = find_best_split(X, y, data_indices)
    
    left_indices = data_indices[X[data_indices, best_split_dim] <= best_split_value]
    right_indices = data_indices[X[data_indices, best_split_dim] > best_split_value]

    node.split_dim = best_split_dim
    node.split_value = best_split_value

    node.left, _ = grow_tree(X, y, left_indices, depth=depth + 1, max_depth=max_depth)
    node.right, _ = grow_tree(X, y, right_indices, depth=depth + 1, max_depth=max_depth)

    return node, best_loss

def predict(x, node):
    if node.label is not None:
        return node.label
    if x[node.split_dim] <= node.split_value:
        return predict(x, node.left)
    else:
        return predict(x, node.right)

# # Load MNIST dataset
# mnist_data = np.load(r"D:\Downloads\mnist.npz")
# x_train, y_train = mnist_data['x_train'], mnist_data['y_train']

# # Select classes 0, 1
# selected_indices = np.where((y_train == 0) | (y_train == 1))[0]
# x_selected = x_train[selected_indices]
# y_selected = y_train[selected_indices]

# # Select 1000 samples from each class for validation
# class_0_indices = np.where(y_selected == 0)[0][:500]
# class_1_indices = np.where(y_selected == 1)[0][:500]
# val_indices = np.concatenate([class_0_indices, class_1_indices])

# x_val = x_selected[val_indices]
# y_val = y_selected[val_indices]

# # Remove validation samples from the training set
# x_train = np.delete(x_selected, val_indices, axis=0)
# y_train = np.delete(y_selected, val_indices)

# test_indices = []
# for i in range(2):  # For each class
#     class_indices = np.where(y_train == i)[0]
#     remaining_indices = np.setdiff1d(class_indices, val_indices)
#     test_indices.extend(remaining_indices[:500])  # Select 500 from each class

# x_test = x_train[test_indices]
# y_test = y_train[test_indices]

# # Remove test samples from the training set
# x_train = np.delete(x_train, test_indices, axis=0)
# y_train = np.delete(y_train, test_indices)

# # Verify shapes
# print("Shapes:")
# print("Training set:", x_train.shape, y_train.shape)
# print("Validation set:", x_val.shape, y_val.shape)
# print("Test set:", x_test.shape, y_test.shape)


# X = create_data_matrix(x_train)
# X_centered,X_train_mean = center_data(X)
# p = 5
# U_sorted, x_reduced = apply_pca(X_centered, p)
# x_reduced = x_reduced.T
# print(x_reduced.shape)


# X = create_data_matrix(x_val)
# X_centered_val,X_val_mean = center_data(X,X_train_mean)
# p = 5
# x_reduced_val = U_sorted.T @ X_centered_val
# x_reduced_val = x_reduced_val.T
# print(x_reduced_val.shape)



# X = create_data_matrix(x_test)
# X_centered_test,X_test_mean = center_data(X,X_train_mean)
# p = 5
# x_reduced_test = U_sorted.T @ X_centered_test
# x_reduced_test = x_reduced_test.T
# print(x_reduced_test.shape)

import numpy as np

# Load MNIST dataset
mnist_data = np.load(r"D:\Downloads\mnist.npz")
x_train_all, y_train_all = mnist_data['x_train'], mnist_data['y_train']
x_test_all, y_test_all = mnist_data['x_test'], mnist_data['y_test']

# Select classes 0 and 1
selected_train_indices = np.where((y_train_all == 0) | (y_train_all == 1))[0]
selected_test_indices = np.where((y_test_all == 0) | (y_test_all == 1))[0]

x_selected_train = x_train_all[selected_train_indices]
y_selected_train = y_train_all[selected_train_indices]
x_selected_test = x_test_all[selected_test_indices]
y_selected_test = y_test_all[selected_test_indices]

# Sample 1000 samples randomly from each class for validation
num_val_samples_per_class = 1000

# Initialize empty lists to store indices
val_indices = []

# For each class (0 and 1)
for class_label in [0, 1]:
    # Find indices of samples belonging to the current class in the training set
    class_indices_train = np.where(y_selected_train == class_label)[0]

    # Randomly select validation samples
    val_indices.extend(np.random.choice(class_indices_train, size=num_val_samples_per_class, replace=False))

# Convert list to numpy array
val_indices = np.array(val_indices)

# Remove validation samples from the training set
x_train = np.delete(x_selected_train, val_indices, axis=0)
y_train = np.delete(y_selected_train, val_indices)

# Separate validation set
x_val = x_selected_train[val_indices]
y_val = y_selected_train[val_indices]

# Verify shapes
print("Shapes:")
print("x_train:", x_train.shape)
print("y_train:", y_train.shape)
print("x_val:", x_val.shape)
print("y_val:", y_val.shape)

# x_test should contain all samples from the original x_test_all
x_test = x_selected_test
y_test = y_selected_test

# Verify shapes
print("x_test:", x_test.shape)
print("y_test:", y_test.shape)

X = create_data_matrix(x_train)
X_centered,X_train_mean = center_data(X)
p = 5
U_sorted, x_reduced = apply_pca(X_centered, p)
x_reduced = x_reduced.T
print(x_reduced.shape)


X = create_data_matrix(x_val)
X_centered_val,X_val_mean = center_data(X,X_train_mean)
p = 5
x_reduced_val = U_sorted.T @ X_centered_val
x_reduced_val = x_reduced_val.T
print(x_reduced_val.shape)



X = create_data_matrix(x_test)
X_centered_test,X_test_mean = center_data(X,X_train_mean)
p = 5
x_reduced_test = U_sorted.T @ X_centered_test
x_reduced_test = x_reduced_test.T
print(x_reduced_test.shape)




# def reg_boost(X, y, X_val, y_val, num_iterations):
    
#     classifiers = []  # Store the decision trees
#     validation_mse = []  # Store validation MSE
#     l = 0.01
#     global total_leaf_nodes
#     y_cap = np.array(y[:],dtype=np.float32)
#     y_val = np.array(y_val[:],dtype=np.float32)
#     for _ in tqdm(range(num_iterations)):
#         # Create a decision tree using the dataset
#         total_leaf_nodes = 0    
#         tree, best_loss = grow_tree(X, y_cap)
#         predictions = np.array([predict(x, tree) for x in X],dtype=np.float32)
#         y_cap = np.sign(y_cap - predictions)
        
#         classifiers.append(tree)

#         train_predictions = reg_boost_predict(classifiers,X)
#         # Calculate MSE on the validation set
#         mse = np.mean((y - train_predictions) ** 2)
#         print(f"Iteration {_+1}: Training MSE Regres Boost= {mse}")
#         validation_mse.append(mse)

        
#         # Calculate predictions on the validation set
#         val_predictions = reg_boost_predict(classifiers,X_val)
#         # Calculate MSE on the validation set
#         mse = np.mean((val_predictions - y_val) ** 2)
#         print(f"Iteration {_+1}: Validation MSE Regres Boost= {mse}")
#         validation_mse.append(mse)
        
#     return classifiers, validation_mse


# def reg_boost_predict(classifiers, X):
#     l = 0.01
#     predictions = np.zeros(len(X))
    
#     for tree in classifiers:
#         prediction = np.array([predict(x, tree) for x in X],dtype=np.float32)
#         # prediction[prediction == 0] = -1  # Change 0 to -1
#         predictions += (l * (prediction ))
    
#     return prediction
def reg_boost(X, y, X_val, y_val, num_iterations):
    
    classifiers = []  # Store the decision trees
    validation_mse = []  # Store validation MSE
    l = 0.01  # Learning rate
    global total_leaf_nodes
    y_cap = np.array(y[:], dtype=np.float32)
    y_val = np.array(y_val[:], dtype=np.float32)
    
    for idx in tqdm(range(num_iterations)):
        # Create a decision tree using the dataset
        total_leaf_nodes = 0    
        tree, _ = grow_tree(X, y_cap)
        predictions = np.array([predict(x, tree) for x in X], dtype=np.float32)
        # Update y_cap using gradient descent with absolute loss
        y_cap -= l * np.sign( y_cap-predictions )
        
        classifiers.append(tree)

        # Calculate train MSE
        train_predictions = reg_boost_predict(classifiers, X)
        train_mse = np.mean((y - train_predictions) ** 2)
        print(f"Iteration {idx+1}: Training MSE Regres Boost = {train_mse}")

        # Calculate validation MSE
        val_predictions = reg_boost_predict(classifiers, X_val)
        val_mse = np.mean((val_predictions - y_val) ** 2)
        print(f"Iteration {idx+1}: Validation MSE Regres Boost = {val_mse}")
        validation_mse.append(val_mse)
        
    return classifiers, validation_mse

def reg_boost_predict(classifiers, X):
    l = 0.01
    predictions = np.zeros(len(X))
    
    for tree in classifiers:
        prediction = np.array([predict(x, tree) for x in X], dtype=np.float32)
        predictions += l * prediction
    
    return predictions


# Example usage:
# classifiers = ada_boost(x_reduced, y_train, num_iterations,)
# Example usage:
num_iterations = 300
classifiers, validation_accuracies = reg_boost(x_reduced, y_train, x_reduced_val,y_val, num_iterations)

# Example usage:
# predictions = ada_boost_predict(classifiers, x_val)


# In[30]:


import matplotlib.pyplot as plt
plt.plot( validation_accuracies)
plt.xlabel('Number of Trees')
plt.ylabel('MSE')
plt.title('Validation MSE vs Number of Trees')
plt.grid(True)
plt.show()


# In[28]:


tree= classifiers[np.argmax(validation_accuracies)]
prediction = np.array([predict(x, tree) for x in x_reduced_test],dtype = np.int64)
print("Test MSE on best Tree",np.mean((prediction- y_test)**2))


# In[29]:


test_predictions = reg_boost_predict(classifiers, x_reduced_test)
print("Test error of Gradient Boosting",np.mean((test_predictions- y_test)**2))

