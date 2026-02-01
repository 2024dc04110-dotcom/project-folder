Heart Failure Prediction Dataset
Dataset Description

Source:
Heart Failure Prediction Dataset – Kaggle

This dataset is used for predicting heart failure based on clinical features.

Dataset Attributes

Age: Age of the patient 
𝑦
𝑒
𝑎
𝑟
𝑠
years

Sex: Sex of the patient

M: Male

F: Female

ChestPainType: Type of chest pain

TA: Typical Angina

ATA: Atypical Angina

NAP: Non-Anginal Pain

ASY: Asymptomatic

RestingBP: Resting blood pressure 
𝑚
𝑚
𝐻
𝑔
mmHg

Cholesterol: Serum cholesterol 
𝑚
𝑚
/
𝑑
𝑙
mm/dl

FastingBS: Fasting blood sugar

1: FastingBS > 120 mg/dl

0: Otherwise

RestingECG: Resting electrocardiogram results

Normal: Normal

ST: ST-T wave abnormality (T wave inversions and/or ST elevation or depression > 0.05 mV)

LVH: Probable or definite left ventricular hypertrophy by Estes' criteria

MaxHR: Maximum heart rate achieved 
60
–
202
60–202

ExerciseAngina: Exercise-induced angina

Y: Yes

N: No

Oldpeak: ST depression induced by exercise (numeric value)

ST_Slope: Slope of the peak exercise ST segment

Up: Upsloping

Flat: Flat

Down: Downsloping

HeartDisease: Target variable

1: Heart disease

0: Normal

Model Performance Metrics
ML Model Name	Accuracy	AUC	Precision	Recall	F1 Score	MCC
Logistic Regression	0.8533	0.9274	0.9000	0.8411	0.8696	0.7044
Decision Tree	0.8043	0.8081	0.8660	0.7850	0.8235	0.6089
KNN	0.8641	0.9416	0.9100	0.8505	0.8792	0.7265
Naive Bayes	0.8641	0.9244	0.9271	0.8318	0.8768	0.7316
Random Forest (Ensemble)	0.8804	0.9433	0.9048	0.8879	0.8962	0.7554
XGBoost (Ensemble)	0.8750	0.9340	0.9038	0.8785	0.8910	0.7450
Model Performance Observations
Logistic Regression

Strong overall performance with good accuracy and high AUC, indicating excellent class separability.

High precision, meaning few false positives.

Slightly lower recall, so some positive cases may be missed.

MCC indicates solid but not top-tier balanced performance.

Observation:
A reliable and interpretable baseline model with strong discrimination power, but slightly weaker in capturing all positive cases compared to ensemble methods.

Decision Tree

Lowest accuracy and AUC among all models.

Relatively high precision, but noticeably lower recall.

F1 score and MCC indicate weaker overall and balanced performance.

Observation:
The Decision Tree underperforms compared to other models and likely suffers from overfitting or limited generalization.

K-Nearest Neighbors (KNN)

High accuracy and very strong AUC.

Well-balanced precision and recall.

Strong F1 score and MCC reflect robust classification capability.

Observation:
KNN performs very well overall, particularly in class separation, but may be computationally expensive for large datasets.

Naive Bayes

Accuracy comparable to KNN with a strong AUC.

Highest precision among all models, resulting in very few false positives.

Slightly lower recall, meaning some positives are missed.

Good MCC indicates balanced performance.

Observation:
Naive Bayes is highly effective when minimizing false positives is important, though it trades off some recall.

Random Forest (Ensemble)

Best overall accuracy and highest AUC.

Strong balance between precision and recall.

Highest F1 score and MCC, indicating excellent and well-balanced performance.

Observation:
Random Forest is the best-performing model overall, offering robustness, strong generalization, and balanced predictions.

XGBoost (Ensemble)

Accuracy slightly lower than Random Forest but still very high.

Strong AUC indicating excellent discriminative ability.

Well-balanced precision and recall.

MCC slightly lower than Random Forest.

Observation:
XGBoost delivers consistently high performance, closely rivaling Random Forest, with the added advantage of being highly tunable.
