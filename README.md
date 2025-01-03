# MNIST Dataset Digits Classification

This project implements a multi-layer neural network to classify handwritten digits using a MNIST-like dataset. It includes steps for data preprocessing, model design, training, and evaluation to achieve high accuracy in classifying digits from 0 to 9.

---

## **Objective**
- Build a multi-layer neural network to classify handwritten digits.
- Utilize gradient-based optimization and backpropagation to train the model.

---

## **Implementation Steps**

### **1. Data Preprocessing**
- **Normalization:** Scaled pixel values to a range between 0 and 1.
- **One-Hot Encoding:** Transformed labels into a binary matrix for classification.

### **2. Model Design**
- **Activation Functions:**
  - Hidden layers: Logistic sigmoid function.
  - Output layer: Softmax function for multi-class classification.
- **Loss Function:** Cross-entropy loss for optimization.

### **3. Training**
- **Forward Pass:** Compute activations for hidden and output layers.
- **Backpropagation:** Update weights and biases using gradients.
- **Optimization:** Gradient descent.

### **4. Evaluation**
- Tested the model on unseen data and computed accuracy.

---

## **Setup Instructions**

### **Requirements**
- Python 3.x
- Libraries: `numpy`, `pandas`, `sklearn`

### **Steps to Run**
1. Clone the repository:
   ```bash
   git clone https://github.com/NoellaButi/MNIST_Classification.git
   cd MNIST_Classification
   ```
2. Install required libraries:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the script to train and evaluate the model:
   ```bash
   python train_and_evaluate.py
   ```

---

## **Results**
### **Training Progress**
| Epoch | Loss    |
|-------|---------|
| 1     | 0.5075  |
| 2     | 0.3613  |
| 3     | 0.3202  |
| 4     | 0.3007  |
| 5     | 0.2892  |
| 6     | 0.2814  |
| 7     | 0.2757  |
| 8     | 0.2711  |
| 9     | 0.2674  |
| 10    | 0.2643  |

### **Final Evaluation**
- **Correct Classifications:** 9170
- **Incorrect Classifications:** 830
- **Accuracy:** 91.70%

---

## **Key Functions**

### **Activation Functions**
- **Sigmoid Function:**
  ```python
  def sigmoid(z):
      return 1 / (1 + np.exp(-z))
  ```
- **Softmax Function:**
  ```python
  def softmax(z):
      exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
      return exp_z / np.sum(exp_z, axis=1, keepdims=True)
  ```

### **Loss Function**
- **Cross-Entropy Loss:**
  ```python
  def cross_entropy_loss(y_pred, y_true):
      return -np.sum(y_true * np.log(y_pred + 1e-8)) / y_true.shape[0]
  ```

---

## **Acknowledgments**
- Dataset: Custom MNIST-like dataset for handwritten digit recognition.
- Techniques inspired by neural network implementations in academic research.
