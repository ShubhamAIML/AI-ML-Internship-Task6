# K-Nearest Neighbors (KNN) Classification 🌟  

## Overview 📋  
This repository contains the solution for **Task 6** of the **AI & ML Internship**, implementing a **K-Nearest Neighbors (KNN) classifier** on the **Iris dataset**. The goal was to understand and apply KNN for classification, normalize features, experiment with different K values, evaluate the model using accuracy and confusion matrix, and visualize decision boundaries. The project was developed in **Google Colab**, with visualizations displayed inline and saved as **PNGs** for submission.  

---

## Dataset 📊  
**Iris Dataset**: Built into scikit-learn, it includes:  
- **150 samples**  
- **4 features**: Sepal Length, Sepal Width, Petal Length, Petal Width  
- **3 classes**: Setosa, Versicolor, Virginica  

> **Note**: No external dataset file is included, as the dataset is loaded directly from scikit-learn.  

---

## Objectives 🎯  
✅ Load and preprocess the Iris dataset, normalizing features.  
✅ Implement KNN using `KNeighborsClassifier` from scikit-learn.  
✅ Experiment with K values from **1 to 15**.  
✅ Evaluate the model with **accuracy, confusion matrix, and classification report**.  
✅ Visualize **accuracy vs. K, confusion matrix, and decision boundaries**.  
✅ Submit a **GitHub repository** with code, visualizations, and a **README**.  

---

## Steps 🛠️  

### 1. Load and Preprocess Data  
- Loaded the Iris dataset using **scikit-learn**.  
- Split data into **80% training (120 samples)** and **20% testing (30 samples)** with `random_state=42`.  
- Normalized features using `StandardScaler` to ensure equal contribution to distance calculations.  

### 2. Train KNN Model  
- Used `KNeighborsClassifier` to train the model.  
- Experimented with **K values from 1 to 15**, calculating accuracy for each.  

### 3. Evaluate Model  
- Computed accuracy for each K (all achieved **100% accuracy**).  
- Selected the best K (**K=1**, as it was the first with maximum accuracy).  
- Generated a **confusion matrix** and **classification report** for the final model.  
  > **Note**: 100% accuracy across all K values suggests the Iris dataset is well-separated, and the test set may be particularly easy due to the random split.  

### 4. Visualize Results  
- Plotted **Accuracy vs. K** to show performance across K values.  
- Created a **Confusion Matrix heatmap** to visualize classification results.  
- Plotted **Decision Boundaries** for Petal Length vs. Petal Width to show class separation.  
- Visualizations were displayed inline in **Google Colab** using `plt.show()` and saved as **PNGs**.  

---

## Results 📈  

### Accuracy  
- **100% for all K values (1 to 15)**, indicating excellent model performance on the test set.  
- **Best K**: K=1 (chosen as the first K with maximum accuracy).  

### Confusion Matrix  
```
[[10  0  0]  
 [ 0  9  0]  
 [ 0  0 11]]  
```  
- **Perfect diagonal**: 10 Setosa, 9 Versicolor, 11 Virginica correctly classified.  

### Classification Report  
|            | Precision | Recall | F1-Score | Support |  
|------------|-----------|--------|----------|---------|  
| setosa     | 1.00      | 1.00   | 1.00     | 10      |  
| versicolor | 1.00      | 1.00   | 1.00     | 9       |  
| virginica  | 1.00      | 1.00   | 1.00     | 11      |  
| accuracy   |           |        | 1.00     | 30      |  
| macro avg  | 1.00      | 1.00   | 1.00     | 30      |  
| weighted avg | 1.00    | 1.00   | 1.00     | 30      |  

### Analysis  
- The **100% accuracy** across all K values is likely due to the Iris dataset’s small size and clear class separation, especially for Setosa.  
- The test set (`random_state=42`) may be particularly easy, leading to perfect performance.  
- In real-world scenarios, accuracy would vary with K, and **cross-validation** could provide a more robust evaluation.  
- **K=1** may overfit in noisier datasets, so **K=3 or K=5** could be more robust choices.  

---

## Visualizations 🖼️  

### 1. Accuracy vs. K 📉  
Shows accuracy (1.0) for K=1 to 15, indicating consistent performance.  

### 2. Confusion Matrix 🌡️  
Heatmap of the confusion matrix, with a perfect diagonal (10, 9, 11).  

### 3. Decision Boundaries 🗺️  
2D plot showing class separation for **Petal Length vs. Petal Width** with K=1.  

---

## Folder Structure 📁  
```
Task6-KNN-Classification/  
├── knn_classification_iris.py    # Python code  
├── accuracy_vs_k.png            # Accuracy plot  
├── confusion_matrix.png         # Confusion matrix heatmap  
├── decision_boundaries.png      # Decision boundaries plot  
├── README.md                    # This file  
└── requirements.txt             # Dependencies  
```  

---

## Files 📄  
- **`knn_classification_iris.py`**: Python script implementing KNN classification, including data preprocessing, model training, evaluation, and visualization.  
- **`accuracy_vs_k.png`**: Plot of accuracy vs. K values.  
- **`confusion_matrix.png`**: Confusion matrix heatmap.  
- **`decision_boundaries.png`**: Decision boundaries for Petal Length vs. Petal Width.  
- **`requirements.txt`**: List of required Python libraries.  
- **`README.md`**: This documentation file.  

---

## Libraries Used 🛠️  
- **scikit-learn**: For dataset loading, preprocessing, KNN model, and evaluation.  
- **pandas**: For data handling (minimal use since Iris is loaded via scikit-learn).  
- **numpy**: For numerical operations.  
- **matplotlib**: For plotting visualizations.  
- **seaborn**: For enhanced confusion matrix heatmap.  
- **IPython.display**: For displaying images inline in Google Colab.  

**Listed in `requirements.txt`:**  
```
scikit-learn  
pandas  
numpy  
matplotlib  
seaborn  
```  

---

## How to Run 🚀  

### 1. Clone the Repository  
```bash
git clone https://github.com/your-username/Task6-KNN-Classification.git  
cd Task6-KNN-Classification  
```  

### 2. Set Up Environment  
Install dependencies:  
```bash
pip install -r requirements.txt  
```  
> **Alternatively**, use **Google Colab** (no installation needed for most libraries).  

### 3. Run the Code  
- **In a local environment**:  
  ```bash
  python knn_classification_iris.py  
  ```  
- **In Google Colab**:  
  - Upload `knn_classification_iris.py` to Colab.  
  - Run all cells.  
  - Visualizations will display inline, and PNGs will be saved in `/content/`.  

### 4. Download Visualizations (Colab)  
```python
from google.colab import files  
files.download('accuracy_vs_k.png')  
files.download('confusion_matrix.png')  
files.download('decision_boundaries.png')  
```  

---

## Learnings 💡  
- **Instance-Based Learning**: KNN is a lazy learning algorithm that stores data and classifies based on nearest neighbors.  
- **Euclidean Distance**: Understood how KNN uses distance metrics to find neighbors.  
- **Normalization**: Learned why normalizing features is critical for distance-based algorithms like KNN.  
- **K Selection**: Explored the trade-off between small K (overfitting) and large K (underfitting).  
- **Evaluation**: Gained experience with accuracy, confusion matrix, and classification report for model evaluation.  
- **Visualization**: Mastered plotting decision boundaries and interpreting model behavior.  

---

Here's a clean separation of questions and answers in a structured format:

---

# K-Nearest Neighbors (KNN) Interview Q&A

## **1. How does the KNN algorithm work?**
### Question:
Explain the working mechanism of the KNN algorithm for classification tasks.

### Answer:
KNN operates in 3 key steps:
1. **Distance Calculation**: Computes distances (Euclidean/Manhattan) between the test point and all training samples
2. **Neighbor Selection**: Identifies K closest training points (neighbors)
3. **Majority Voting**: Assigns the class most frequent among neighbors

For regression, it averages the neighbors' values instead.

---

## **2. How do you choose the right K value?**
### Question:
What strategies help determine the optimal K in KNN?

### Answer:
Optimal K selection involves:
- **Odd numbers** (3,5,7) to avoid tie votes
- **Elbow Method**: Plot accuracy vs K; choose K where accuracy plateaus
- **Cross-validation**: Test multiple K values (1-20 typical)
- **Domain knowledge**: Larger K for noisy data, smaller K for clear patterns

---

## **3. Why is normalization important in KNN?**
### Question:
Why must features be normalized before applying KNN?

### Answer:
Normalization is critical because:
1. **Distance Bias**: Features with larger scales dominate the distance metric
2. **Equal Contribution**: Ensures all features weigh equally (e.g., sepal length vs petal width)
3. **Performance Impact**: Unnormalized data leads to poor accuracy

Common methods: StandardScaler (z-score) or MinMax scaling.

---

## **4. What is the time complexity of KNN?**
### Question:
Analyze KNN's computational efficiency during training and prediction.

### Answer:
- **Training**: O(1) - No model training, just stores data (lazy learner)
- **Prediction**:
  - Brute-force: O(n) per query (scans all points)
  - Optimized (KD-trees/Ball trees): O(log n) per query
- **Memory**: O(n) - Stores entire training set

---

## **5. What are pros and cons of KNN?**
### Question:
List key advantages and limitations of KNN.

### Answer:
**Pros:**
- Simple implementation
- No training phase
- Naturally handles multi-class problems
- Adapts to new data easily

**Cons:**
- Slow prediction for large datasets
- Sensitive to irrelevant features
- Requires careful feature scaling
- Struggles with high-dimensional data (curse of dimensionality)

---

## **6. Is KNN sensitive to noise?**
### Question:
How does noise affect KNN's performance?

### Answer:
Yes, KNN is noise-sensitive because:
- **Small K values** (K=1): Single noisy neighbor can misclassify
- **Impact Mitigation**:
  - Increase K to dilute noise effects
  - Remove outliers during preprocessing
  - Use weighted voting (closer neighbors matter more)

---

## **7. How does KNN handle multi-class problems?**
### Question:
Explain KNN's approach to classifying multiple classes.

### Answer:
KNN handles multi-class natively by:
1. Calculating distances to all class samples
2. Selecting K nearest neighbors across all classes
3. Assigning the majority class among neighbors

No need for One-vs-Rest or One-vs-One strategies like SVM requires.

---

## **8. What's the role of distance metrics in KNN?**
### Question:
How do distance metrics influence KNN's performance?

### Answer:
Distance metrics define similarity measurement:
- **Euclidean** (L2): Default for continuous features
- **Manhattan** (L1): Robust to outliers
- **Cosine**: Ideal for text/High-D data
- **Hamming**: Categorical data
- **Minkowski**: Generalized form (p=1: Manhattan, p=2: Euclidean)

Choice impacts model accuracy and neighbor selection.

---

This format:
1. Clearly separates questions from answers
2. Uses bold headers for quick scanning
3. Maintains consistent structure
4. Highlights key terms for emphasis
5. Organizes complex information into digestible points

---

## Future Improvements 🔧  
- **Cross-Validation**: Implement 5-fold cross-validation to get a more robust estimate of model performance.  
- **Different Distance Metrics**: Experiment with Manhattan or Minkowski distances to compare performance.  
- **Robust K Selection**: Prefer K=3 or K=5 over K=1 when accuracies are tied to reduce overfitting risk.  
- **Enhanced Visualizations**: Add legends or colorbars to plots for better clarity.  

---

**Happy Learning! 😄**
