# Importing required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
from IPython.display import display, Image  # For displaying images in Colab

# Step 1: Load the Iris dataset
# Iris dataset is built into scikit-learn, with 150 samples, 4 features, and 3 classes
iris = load_iris()
X = iris.data  # Features: sepal length, sepal width, petal length, petal width
y = iris.target  # Target: 0 (Setosa), 1 (Versicolor), 2 (Virginica)
feature_names = iris.feature_names
class_names = iris.target_names

# Step 2: Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Training set size:", X_train.shape)
print("Test set size:", X_test.shape)

# Step 3: Normalize the features using StandardScaler
# Normalization ensures all features are on the same scale, which is crucial for KNN
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 4: Experiment with different K values and evaluate the model
# We'll try K values from 1 to 15 and store accuracies
k_values = range(1, 16)
accuracies = []

for k in k_values:
    # Initialize KNN classifier with current K
    knn = KNeighborsClassifier(n_neighbors=k)
    
    # Train the model
    knn.fit(X_train_scaled, y_train)
    
    # Make predictions on the test set
    y_pred = knn.predict(X_test_scaled)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    accuracies.append(accuracy)
    print(f"Accuracy for K={k}: {accuracy:.4f}")

# Step 5: Plot accuracy vs K and display inline in Colab
plt.figure(figsize=(8, 6))
plt.plot(k_values, accuracies, marker='o')
plt.title('Accuracy vs. K Value')
plt.xlabel('K')
plt.ylabel('Accuracy')
plt.grid(True)
plt.savefig('accuracy_vs_k.png')  # Save the plot
plt.show()  # Display the plot inline in Colab
print("Accuracy vs. K plot saved as 'accuracy_vs_k.png'")

# Step 6: Train the final model with the best K
# Choose K with highest accuracy (or a reasonable value like K=5)
best_k = k_values[np.argmax(accuracies)]
print(f"Best K: {best_k}")
knn_final = KNeighborsClassifier(n_neighbors=best_k)
knn_final.fit(X_train_scaled, y_train)

# Step 7: Evaluate the final model
# Make predictions
y_pred_final = knn_final.predict(X_test_scaled)

# Calculate accuracy
final_accuracy = accuracy_score(y_test, y_pred_final)
print(f"Final model accuracy with K={best_k}: {final_accuracy:.4f}")

# Generate confusion matrix
cm = confusion_matrix(y_test, y_pred_final)
print("Confusion Matrix:")
print(cm)

# Visualize confusion matrix using seaborn and display inline
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.savefig('confusion_matrix.png')  # Save the plot
plt.show()  # Display the plot inline in Colab
print("Confusion Matrix plot saved as 'confusion_matrix.png'")

# Print classification report for detailed metrics
print("\nClassification Report:")
print(classification_report(y_test, y_pred_final, target_names=class_names))

# Step 8: Visualize decision boundaries (using only 2 features for simplicity)
# We'll use petal length and petal width (features 2 and 3) for 2D visualization
X_subset = X[:, [2, 3]]  # Petal length and petal width
X_subset_scaled = scaler.fit_transform(X_subset)
X_train_sub, X_test_sub, y_train_sub, y_test_sub = train_test_split(X_subset_scaled, y, test_size=0.2, random_state=42)

# Train KNN on the subset
knn_subset = KNeighborsClassifier(n_neighbors=best_k)
knn_subset.fit(X_train_sub, y_train_sub)

# Create a mesh grid for plotting decision boundaries
x_min, x_max = X_subset_scaled[:, 0].min() - 1, X_subset_scaled[:, 0].max() + 1
y_min, y_max = X_subset_scaled[:, 1].min() - 1, X_subset_scaled[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))

# Predict on the mesh grid
Z = knn_subset.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plot decision boundaries and data points, display inline
plt.figure(figsize=(10, 8))
plt.contourf(xx, yy, Z, alpha=0.4, cmap='RdYlBu')
plt.scatter(X_subset_scaled[:, 0], X_subset_scaled[:, 1], c=y, s=50, edgecolor='k', cmap='RdYlBu')
plt.title(f'Decision Boundaries with K={best_k} (Petal Length vs Petal Width)')
plt.xlabel('Petal Length (scaled)')
plt.ylabel('Petal Width (scaled)')
plt.savefig('decision_boundaries.png')  # Save the plot
plt.show()  # Display the plot inline in Colab
print("Decision Boundaries plot saved as 'decision_boundaries.png'")

# Optional: Display saved PNGs as images in Colab (for confirmation)
print("Displaying saved PNG files:")
display(Image(filename='accuracy_vs_k.png'))
display(Image(filename='confusion_matrix.png'))
display(Image(filename='decision_boundaries.png'))
