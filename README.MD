# Heart Failure Prediction Dataset

## Dataset Description

Source:
https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction

This dataset is used for predicting heart failure based on clinical features.

## Dataset Attributes

- Age: Age of the patient [years]
- Sex: Sex of the patient
  - M: Male
  - F: Female
- ChestPainType: Type of chest pain
  - TA: Typical Angina
  - ATA: Atypical Angina
  - NAP: Non-Anginal Pain
  - ASY: Asymptomatic
- RestingBP: Resting blood pressure [mm Hg]
- Cholesterol: Serum cholesterol [mm/dl]
- FastingBS: Fasting blood sugar
  - 1: FastingBS > 120 mg/dl
  - 0: Otherwise
- RestingECG: Resting electrocardiogram results
  - Normal: Normal
  - ST: ST-T wave abnormality (T wave inversions and/or ST elevation or depression > 0.05 mV)
  - LVH: Probable or definite left ventricular hypertrophy by Estes' criteria
- MaxHR: Maximum heart rate achieved [60–202]
- ExerciseAngina: Exercise-induced angina
  - Y: Yes
  - N: No
- Oldpeak: ST depression induced by exercise (numeric value)
- ST_Slope: Slope of the peak exercise ST segment
  - Up: Upsloping
  - Flat: Flat
  - Down: Downsloping
- HeartDisease: Target variable
  - 1: Heart disease
  - 0: Normal

## Model Performance Metrics

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
|--------------|----------|-----|-----------|--------|----------|-----|
| Logistic Regression | 0.85326087 | 0.927418376 | 0.9 | 0.841121495 | 0.869565217 | 0.704402229 |
| Decision Tree | 0.804347826 | 0.80810778 | 0.865979381 | 0.785046729 | 0.823529412 | 0.608869779 |
| KNN | 0.8641304347826086 | 0.9415584415584415 | 0.91 | 0.8504672897196262 | 0.8792270531400966 | 0.7265199781462983 |
| Naive Bayes | 0.8641304347826086 | 0.9243840271877656 | 0.9270833333333334 | 0.8317757009345794 | 0.8768472906403941 | 0.7316448515998614 |
| Random Forest (Ensemble) | 0.8804347826086957 | 0.9433183638791116 | 0.9047619047619048 | 0.8878504672897196 | 0.8962264150943396 | 0.7554174210796076 |
| XGBoost (Ensemble) | 0.875 | 0.9339725694865881 | 0.9038461538461539 | 0.8785046728971962 | 0.8909952606635071 | 0.7449814354126474 |

## Model Performance Observations

| ML Model Name | Observation about model performance |
|--------------|------------------------------------|
| Logistic Regression | Shows strong overall performance with good accuracy and a high AUC, indicating excellent class separability. Precision is high, meaning it makes few false positive predictions. Recall is slightly lower, so it may miss some positive cases. MCC suggests solid but not top-tier balanced performance. 
Observation: A reliable and interpretable baseline model with good discrimination power, but slightly weaker in capturing all positives compared to ensemble methods. |
| Decision Tree | Lowest accuracy and lowest AUC among all models. Precision is relatively high, but recall is noticeably lower. F1 score and MCC indicate weaker overall and balanced performance. 
Observation: The Decision Tree underperforms compared to other models and likely suffers from overfitting or limited generalization. |
| KNN | High accuracy and very strong AUC. Precision and recall are well balanced. F1 score and MCC reflect strong classification capability. Observation: KNN performs very well overall, particularly in distinguishing classes, but may be computationally expensive for large datasets. |
| Naive Bayes | Accuracy comparable to KNN, with a strong AUC. Highest precision among all models, meaning very few false positives. Recall is slightly lower, indicating some missed positives. MCC suggests good balanced performance. Observation: Naive Bayes is highly effective when minimizing false positives is important, though it trades off some recall. |
| Random Forest (Ensemble) | Best overall accuracy and highest AUC. Strong balance between precision and recall. Highest F1 score and MCC, indicating excellent and well-balanced performance. Observation: Random Forest is the best-performing model overall, providing robustness, strong generalization, and balanced predictions. |
| XGBoost (Ensemble) | Accuracy slightly lower than Random Forest but still very high. AUC indicates strong discriminative ability. Precision and recall are well balanced. MCC shows strong but slightly inferior performance to Random Forest. Observation: XGBoost delivers high and consistent performance, closely rivaling Random Forest, with the advantage of being highly tunable. |
