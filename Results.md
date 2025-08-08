## 🧪 Results (Accuracy and Confusion Matrix)

The final `RandomForestClassifier` achieved an overall accuracy of **75.45%** on the test set.

---

### 📊 Confusion Matrix

The confusion matrix shows the number of correct and incorrect predictions for each class:

- **Class 0**: Non-Duplicate  
- **Class 1**: Duplicate  

|                        | Predicted: Non-Duplicate | Predicted: Duplicate |
|------------------------|--------------------------|-----------------------|
| **Actual: Non-Duplicate** | 41,496                   | 9,574                 |
| **Actual: Duplicate**     | 10,268                   | 19,492                |

---

### 📋 Classification Report

The classification report provides a detailed breakdown of the model's performance — including precision, recall, and F1-score for each class.

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| **0 (Non-Duplicate)** | 0.80      | 0.81   | 0.81     | 51,070  |
| **1 (Duplicate)**     | 0.67      | 0.65   | 0.66     | 29,760  |
|                       |           |        |          |         |
| **Accuracy**          |           |        | **0.75** | 80,830  |
| **Macro Avg**         | 0.74      | 0.73   | 0.73     | 80,830  |
| **Weighted Avg**      | 0.75      | 0.75   | 0.75     | 80,830  |

