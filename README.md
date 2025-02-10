# CS461 - Program 3  
## Neural Network for Heart Disease Classification  
**Author:** Alex Dowell  

## Data Preparation  
To prepare the dataset for analysis, I first read in the **Cleveland Clinic** dataset and separated the input parameters from the output (column 14). Incomplete observations were removed, and the categorical output variables were **one-hot encoded**.  

One-hot encoding converts categorical values into multiple binary columns, making it easier for the model to understand relationships between categories. Since the target variable had **five unique values (0-4)**, I created **five binary columns** to represent them.  

Next, I applied **standardization** using `StandardScaler()` from `sklearn`, which scales numerical values to have a **mean of 0** and **standard deviation of 1**. This helps improve the convergence speed and prevents features with larger scales from dominating the model.  

## Neural Network Configuration  
The model is a **fully connected feedforward neural network** with **three hidden layers**:  
- **Layer 1:** 26 neurons (ReLU activation)  
- **Layer 2:** 13 neurons (ReLU activation)  
- **Layer 3:** 7 neurons (ReLU activation)  
- **Output Layer:** 5 neurons (Softmax activation for one-hot encoded classification)  

I chose this architecture as it effectively maps relationships in the dataset. No nonlinear features (e.g., skip connections) were included since they are not necessary for this problem.  

## Validation Strategy  
I split the dataset into **training (80%), testing (10%), and validation (10%)** sets.  
Additionally, I implemented **8-fold cross-validation** to further validate model performance.  

To prevent overfitting, I used **early stopping**, which monitors the test loss and stops training when performance begins to degrade. The training process was limited to **15 epochs**.  

## Results  
| Dataset   | Accuracy |  
|-----------|----------|  
| Training  | 62%      |  
| Testing   | 57%      |  
| Validation| 50%      |  

Adding more layers and neurons **negatively impacted performance**, while reducing the learning rate **also decreased accuracy**.  

## Comments & Future Improvements  
The model showed **some predictive capability** but is far from optimal.  
Possible improvements include:  
- **Trying different network architectures** (e.g., deeper networks, dropout layers)  
- **Applying advanced optimization techniques**  
- **Using a larger dataset** to improve generalization  

One challenge was the **small dataset size**, making it difficult for the model to learn complex patterns. A larger dataset would likely result in **higher accuracy**.  

## References  
- Keras Documentation: [The Model Class](https://keras.io/api/models/model/)  

## Sample Output  
### **Dataset Shapes:**  
`x_train.shape, y_train.shape, x_test.shape, y_test.shape, x_val.shape, y_val.shape`  
`(237, 13) (237, 5) (30, 13) (30, 5) (30, 13) (30, 5)`  

### **Training Progress (First 5 Epochs):**  
Epoch 1/15 - loss: 1.6495 - accuracy: 0.1772 - val_loss: 1.6560 - val_accuracy: 0.1667
Epoch 2/15 - loss: 1.5889 - accuracy: 0.2447 - val_loss: 1.6017 - val_accuracy: 0.2333
Epoch 3/15 - loss: 1.5356 - accuracy: 0.3418 - val_loss: 1.5580 - val_accuracy: 0.4000
Epoch 4/15 - loss: 1.4858 - accuracy: 0.4304 - val_loss: 1.5186 - val_accuracy: 0.4333
Epoch 5/15 - loss: 1.4383 - accuracy: 0.4979 - val_loss: 1.4768 - val_accuracy: 0.4667

### **Validation Accuracy:**  
`Validation dataset accuracy: 50.0%`  
