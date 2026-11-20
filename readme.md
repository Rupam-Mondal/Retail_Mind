# RetailMind AI

RetailMind AI is an incremental AI project built to explore customer churn prediction using machine learning and deep learning. Each increment improves the previous solution through model comparison, tuning, validation, and neural networks.

The Telco Customer Churn dataset is used to predict whether a customer is likely to leave the service.

---

## Day 1 - Churn Prediction Baseline

Day 1 focused on building the initial end-to-end machine learning solution for customer churn prediction.

The dataset was loaded and cleaned before separating the customer features from the churn target. Numerical and categorical features were handled separately using preprocessing techniques such as missing-value imputation, feature scaling, and categorical encoding.

A reusable Scikit-learn pipeline was created so that preprocessing and model training could be performed together. The dataset was then divided into training and testing sets, and an initial classification model was trained and evaluated.

The model was evaluated using classification metrics such as accuracy, precision, recall, and F1-score.

This increment established the basic machine learning workflow that was reused and improved in the following days.

---

## Day 2 - Multi-Model Comparison

Day 2 extended the baseline solution by training and comparing multiple machine learning algorithms using the same preprocessing pipeline.

Instead of relying on a single classifier, different models were trained under the same conditions and evaluated on the test dataset.

The comparison considered:

- Accuracy
- Precision
- Recall
- F1-score

For the churn use case, **recall was given higher importance** because failing to identify a customer who is actually likely to churn can result in a missed retention opportunity.

The results from each model were stored and compared. Based on recall, **SVM emerged as the strongest candidate**, while Random Forest was the next strongest model.

These models were carried forward to the next increment for further optimization.

---

## Day 3 - Hyperparameter Tuning and Model Validation

Day 3 focused on improving the strongest models identified during Day 2.

SVM and Random Forest were selected as the main candidates and tuned using `GridSearchCV`. Different combinations of model hyperparameters were evaluated automatically to determine which configuration performed best.

Both **3-fold and 5-fold cross-validation** were used to make the evaluation more reliable and reduce dependence on a single train-validation split.

Recall remained the main optimization metric because churn detection was the primary objective.

For SVM, parameters such as the following were explored:

- `C`
- Kernel
- Gamma
- Class weight

For Random Forest, tuning included parameters such as:

- Number of trees
- Maximum tree depth
- Minimum samples required for splitting
- Minimum samples per leaf
- Class weight

The tuned models were evaluated again using accuracy, precision, recall, F1-score, classification reports, and confusion matrices.

Learning curves were also used to compare training and validation performance. This helped identify potential high-bias or high-variance behaviour and provided insight into whether additional data, feature improvements, or regularization could help.

The final tuned champion was selected primarily using recall and persisted using `joblib` so that the complete trained pipeline could be reused without retraining.

---

## Day 4 - Deep Learning Churn Model

Day 4 introduced deep learning into the churn prediction workflow.

A feed-forward Artificial Neural Network was developed using TensorFlow/Keras and trained using the same Telco Customer Churn problem.

Before training the neural network, the input data was prepared so that all features were numerical and appropriately scaled. Categorical variables were encoded and numerical features were standardized before being provided to the network.

The ANN used a small feed-forward architecture consisting of:

- Input features
- Dense hidden layers with ReLU activation
- Dropout for regularization
- A single output neuron with sigmoid activation

Sigmoid activation was used because churn prediction is a binary classification problem.

The network was trained using the Adam optimizer and binary cross-entropy loss.

Early stopping monitored validation loss during training. Training automatically stopped when validation performance stopped improving, which helped reduce unnecessary training and limit overfitting.

Training and validation loss were also examined to understand how well the ANN generalized beyond the training data.

---

## Day 5 - ANN Evaluation and Classical ML Benchmark

Day 5 focused on evaluating the neural network and determining whether introducing deep learning actually improved the churn prediction system.

The trained ANN produced churn probabilities, which were converted into class predictions using a decision threshold.

Performance was evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- Classification report
- Confusion matrix

The ANN results were then compared against the classical machine learning champion from the previous phase.

Recall remained an important comparison metric because the main objective is to correctly identify customers who are likely to churn.

The purpose of this comparison was not to assume that the ANN would automatically outperform classical machine learning. Instead, the measured test results were used to determine whether the additional model complexity provided a meaningful improvement.

The trained ANN was saved using the Keras model format, while the fitted preprocessing pipeline was saved separately so that future customer data can be transformed in exactly the same way before inference.

This comparison provides a practical basis for deciding whether a classical machine learning model or an ANN is more suitable for the current churn prediction workload.
---

## Day 6 - CNN-Based Fashion Product Classification

Day 6 introduced computer vision into RetailMind AI by developing a Convolutional Neural Network (CNN) for fashion product classification.

The increment started with a small Fashion-MNIST prototype to validate the CNN architecture before moving to the actual fashion product dataset.

The Fashion Product Images dataset was then used to build the real product classification pipeline. Images were loaded, resized, normalized, and prepared for training. Data augmentation was also introduced to improve the model's ability to generalize across different product images.

A custom CNN architecture was developed using the following components:

**-** Convolutional layers for extracting image features  
**-** Max pooling layers for reducing spatial dimensions  
**-** Dropout for regularization  
**-** Dense layers for product category classification  
**-** Softmax output for predicting product categories

The model was designed to classify fashion products into their respective categories using the prepared image dataset.

Performance evaluation included accuracy, classification reports, and a confusion matrix to understand which product categories were correctly classified and where the model made mistakes.

The trained classifier was designed to serve as the first computer vision module of RetailMind AI, extending the project beyond customer churn prediction into image-based product understanding.

---

## Day 7 - Transfer Learning with MobileNetV2

Day 7 focused on improving the fashion product classification system using Transfer Learning.

Instead of training an entire CNN architecture from scratch, a pretrained MobileNetV2 model was introduced as the backbone for extracting image features.

MobileNetV2 is a lightweight Convolutional Neural Network pretrained on the ImageNet dataset. Its learned visual features were reused to build a product classification model for RetailMind AI.

The Fashion Product Images dataset from Day 6 was reused so that both approaches could be evaluated on the same classification problem.

The transfer learning pipeline included:

**-** Loading MobileNetV2 with pretrained ImageNet weights  
**-** Removing the original classification layer using `include_top=False`  
**-** Freezing the pretrained backbone during initial training  
**-** Adding a custom classification head for fashion product categories  
**-** Training the new classification layers on the fashion product dataset

The model was trained to recognize product categories using the features extracted by MobileNetV2.

The main objective was to compare the transfer learning approach against the custom CNN developed in Day 6.

Both models were evaluated using accuracy, classification reports, and confusion matrices. Training performance and model accuracy were also considered to understand the benefits of reusing pretrained image features instead of learning everything from scratch.

The upgraded classifier was designed to be saved as the RetailMind vision model so that it could be reused in future computer vision workflows.

This increment demonstrated how transfer learning can be applied to image classification while reducing the need to train a deep neural network entirely from scratch.

---