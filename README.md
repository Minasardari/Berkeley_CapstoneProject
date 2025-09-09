### 🩸 Project : Diabetes Risk Prediction

## 🎯 Objective
The goal of this project is to **analyze health survey data** to identify key risk factors associated with diabetes and build a predictive model to classify individuals as **Diabetic** or **Non-Diabetic**.  
This project follows a **CRISP-DM framework**: data cleaning, exploratory data analysis (EDA), feature engineering, modeling, and evaluation.

**Author** : Mina Sardari

Early detection of diabetes is crucial to preventing serious complications such as kidney failure, heart disease, and stroke.  This project aims to leverage machine learning to predict diabetes risk based on demographic, medical, and symptom data. By identifying high-risk individuals earlier, we can enable timely lifestyle changes or medical intervention, ultimately improving patient outcomes and reducing healthcare costs. Moreover, the use of data-driven models promotes equitable, scalable, and personalized care as this work highly relevant to today’s public health challenges.

#### Research Question
Can we accurately predict early-stage diabetes risk using patient demographic and life style choices and  medical, and symptom data?

---
## 📂 Dataset
- Source: Public health survey (binary target = Diabetic).  
- Size: ~225,000 respondents.  
- Target variable: **Diabetic (1 = Yes, 0 = No)**.  
- Feature categories:
  - **Numeric:** BMI, MentHlth, PhysHlth  
  - **Binary:** HighBP, HighChol, Smoker, Stroke, HeartDiseaseorAttack, PhysActivity, Fruits, Veggies, HvyAlcoholConsump, AnyHealthcare, NoDocbcCost, DiffWalk  
  - **Categorical:** Age group, Education, Income, General Health (GenHlth)  

Kaggle Diabetes Prediction Dataset
LinkLinks to an external site.
Includes demographic and biomedical measurements data.

- [https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset]()

This dataset contains 3 files:
diabetes _ 012 _ health _ indicators _ BRFSS2015.csv is a clean dataset of 253,680 survey responses to the CDC's BRFSS2015
diabetes _ binary _ 5050split _ health _ indicators _ BRFSS2015.csv is a clean dataset of 70,692 survey responses to the CDC's BRFSS2015
diabetes _ binary _ health _ indicators _ BRFSS2015.csv is a clean dataset of 253,680 survey responses to the CDC's BRFSS2015

#### Methodology
Preprocessing & EDA
CRISP-DM, Data Preparation (cleaning missing values, encoding categorical features, scaling numerical attributes),
Correlation analysis and visualization
Feature engineering 
Model comparison and selecting a model (Logistic Regression, Decision Tree, KNN, SVC/SVM)
Also anticipate to use some other concept related to upcoming modules Random Forest and AdaBoost or Neural Networks
Evaluation: Accuracy, Precision, Recall,F1-Score ,Log Loss, Confusion Matrix

 ## 🧹 Data Preparation
- Removed duplicates and handled missing values.  
- Outliers reviewed; extreme values retained when clinically meaningful.  
- Feature engineering:
  - Created BMI groups (Normal, Overweight, Obese).  
  - Encoded GenHlth as ordinal (Poor→Excellent).  
  - Built interaction features: HighBP×HighChol, BMI×PhysHlth, Education×Income.  

---
## 🔍 EDA Key Findings
- **Imbalance:** 82.7% non-diabetic vs 17.3% diabetic.  
# 📊 Key Findings from EDA
<img width="1489" height="311" alt="image" src="https://github.com/user-attachments/assets/f5597726-9c5d-48b5-aacd-1cd79c83a15c" />

## 1. Numeric Features
- **BMI**
  - Centered in the overweight range (27–28).
  - Many respondents obese (BMI > 30); extreme cases (BMI > 40) likely true severe obesity rather than noise.
  - ✅ Key Insight: Obesity is a major predictor; prevalence rises from 9% (normal weight) → 29% (obese).

- **MentHlth & PhysHlth**
  - Highly skewed: majority report 0 unhealthy days.
  - Meaningful subgroup reports 10–30 days → chronic illness group.
  - ✅ Key Insight: PhysHlth correlates more strongly with diabetes than MentHlth. Outliers kept since they represent real health burdens.

---
<img width="1478" height="2555" alt="image" src="https://github.com/user-attachments/assets/d26c7cba-0b3f-4a21-94d8-43aa01519ac1" />

## 2. Binary Features
- **HighBP (45%) & HighChol (44%)**: Almost half the sample at cardiovascular risk; both show strong association with diabetes (~25–27% prevalence when present).
- **HeartDisease/Attack (10%) & Stroke (4.5%)**: Smaller groups but very high diabetes prevalence (~30–39%).
- **PhysActivity (73%)**: Protective factor — inactive individuals show ~25% prevalence vs 16% for active.
- **Diet (Fruits 61%, Veggies 79%)**: Mild protective effect; weak correlations.
- **DiffWalk (19%)**: One of the strongest single predictors — diabetics much more likely to report walking difficulties (~33% vs ~14%).
- **Healthcare Access**
  - AnyHealthcare (95%) → almost universal, not discriminative.
  - NoDocbcCost (9%) → higher diabetes prevalence when cost prevents care (~22%).

✅ Combined Insight: Cardiovascular factors (HighBP, HighChol, HeartDisease, Stroke) and mobility limitations (DiffWalk) are the most powerful binary predictors. PhysActivity is the clearest protective factor.

---

## 3. Categorical Features (Unfiltered Population View)
<img width="1980" height="1189" alt="image" src="https://github.com/user-attachments/assets/6cfa0994-8ca9-4f48-9468-22d44eaf49e3" />

- **GenHlth**: Majority rate health as Poor/Fair. Strongest categorical predictor — diabetes prevalence drops from 41% (Poor) → 4% (Excellent).
- **Age**: Skews older (65+ heavily represented). Diabetes prevalence increases sharply with age, especially 55+.
- **Education**: Lower education overrepresented; prevalence decreases with higher education.
- **Income**: Skewed toward lower income; prevalence decreases with higher income.

✅ Key Insight: The dataset is **not population-representative** (older, lower-income, lower-education bias), which must be considered in modeling and interpretation.

---

## 4. Target Variable
- **Non-Diabetic:** 82.7%
- **Diabetic:** 17.3%
- ✅ Key Insight: Dataset is imbalanced. Accuracy is misleading; evaluation must emphasize **Recall, F1, ROC-AUC, PR-AUC**.

---

## 5. Correlation Analysis
<img width="1989" height="790" alt="image" src="https://github.com/user-attachments/assets/480e6f25-9ba7-46aa-869c-f5117fa5a104" />
<img width="1111" height="790" alt="image" src="https://github.com/user-attachments/assets/b3778722-74ac-4869-af38-e330c5363f45" />

- **Top positive correlations with Diabetes:**
  - HighBP (0.26), BMI (0.21), DiffWalk (0.21), HighChol (0.20), HeartDisease/Attack (0.17), PhysHlth (0.16).
- **Protective correlations:**
  - PhysActivity (−0.10), Veggies (−0.04), Fruits (−0.02).
- **Weak/noisy correlations:**
  - MentHlth (0.06), NoDocbcCost (0.03), AnyHealthcare (0.02).

✅ Key Insight: No single feature dominates → multivariate models (logistic regression with interactions, tree-based methods) will outperform univariate predictors.

---

## 6. Interaction Effects
- **BMI × PhysHlth**: Strongest signal; diabetics cluster at high BMI + high PhysHlth.
- **HighBP × HighChol**: Prevalence rises to ~34% when both present (synergistic effect).
- **HighBP × Age**: Older adults with hypertension show much higher prevalence than younger hypertensives.
- **Education × Income**: Clear socioeconomic gradient — low education + low income = highest risk.
<img width="1119" height="590" alt="image" src="https://github.com/user-attachments/assets/06a98d49-62b5-46b3-9885-74863af22d54" />

✅ Key Insight: Diabetes risk is multifactorial and **driven by combined effects**. Interaction terms should be included in logistic regression, or use tree-based models which capture them automatically.

---
<img width="1589" height="4190" alt="image" src="https://github.com/user-attachments/assets/1114e177-c5a0-404f-adac-f46ddf66db73" />

## 7. Sex-Specific Patterns
- **BMI & PhysHlth**: Differences between diabetics and non-diabetics more pronounced in women.
- **MentHlth**: Diabetic women report more poor mental health days; effect weak in men.
- ✅ Key Insight: Diabetes has a heavier health burden in women → consider `Sex × BMI` and `Sex × PhysHlth` interactions.

---

# 🎯 Overall Takeaways
- **Core predictors**: HighBP, HighChol, BMI, DiffWalk, PhysHlth, GenHlth, Age, Income, Education.
- **Protective factors**: Physical activity, healthier diet (weak effect).
- **Critical interactions**: BMI × PhysHlth, HighBP × HighChol, Age × HighBP, Education × Income.
- **Modeling Implications**:
- <img width="501" height="393" alt="image" src="https://github.com/user-attachments/assets/e8e4de8e-add6-4905-a6f8-d01750ad6a50" />

  - Handle imbalance ( class weights).
  - Use **Recall, F1, ROC-AUC/PR-AUC** instead of Accuracy.
  - Include interaction features for logistic regression.
  - Will use **tree-based models** (Desion Tree or RandomForest, XGBoost) for capturing nonlinear + interaction effects automatically.


---
# 🎯 Business Takeaways
1. **Prevention Levers**: Target obesity, hypertension, and cholesterol control — the most cost-effective interventions.  
2. **High-Risk Segments**: Older adults, low-income groups, and those with mobility issues or poor self-rated health.  
3. **Access Barriers**: Address affordability gaps (9% skip care due to cost) to reduce long-term disease burden.  
4. **Modeling Focus**: Use advanced ML models that balance precision and recall, ensuring at-risk members are identified early for **preventive outreach and resource allocation**.  




## 🤖 Modeling
- **Baseline:** Dummy Classifier
  - Result: Predicted only Non-Diabetic cases (Recall = 0).  
  - Accuracy (82.9%) misleading due to imbalance.  
- **Next Steps:**
  - Address imbalance with **SMOTE resampling** or **class weights**.  
  - Evaluate models with **ROC-AUC, PR-AUC, Recall, and F1** (not accuracy).  
  - Test tree-based models (Random Forest, XGBoost) for capturing nonlinearities and interactions.  

---

## ✅ Expected Impact
- Provide insights into **key health, lifestyle, and socioeconomic risk factors**.  
- Develop a predictive model that balances sensitivity (recall for diabetics) with precision.  
- Support public health strategies by identifying high-risk groups (e.g., obese, hypertensive, low income/education).


#### Results
What did your research find?

#### Next steps
What suggestions do you have for next steps?

#### Outline of project

- [https://github.com/Minasardari/Berkeley_CapstoneProject/blob/main/MinaSardari_Capstone_DiabetePrediction.ipynb]()


##### Contact and Further Information
linkedin :[Mina Sardari](www.linkedin.com/in/mina-s-3b728651)


