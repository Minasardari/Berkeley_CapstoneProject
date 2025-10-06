# 🩸 Project : Diabetes Risk Prediction
## Outline of project

Link to my notebook: https://github.com/Minasardari/Berkeley_CapstoneProject/blob/main/MinaSardari_Capstone_DiabetePrediction.ipynb

### 🎯 Objective
The goal of this project is to **analyze health survey data** to identify key risk factors associated with diabetes and build a predictive model to classify individuals as **Diabetic** or **Non-Diabetic**.  
This project follows a **CRISP-DM framework**: data cleaning, exploratory data analysis (EDA), feature engineering, modeling, and evaluation.

**Author** : Mina Sardari

Early detection of diabetes is crucial to preventing serious complications such as kidney failure, heart disease, and stroke.  This project aims to leverage machine learning to predict diabetes risk based on demographic, medical, and symptom data. By identifying high-risk individuals earlier, we can enable timely lifestyle changes or medical intervention, ultimately improving patient outcomes and reducing healthcare costs. Moreover, the use of data-driven models promotes equitable, scalable, and personalized care as this work highly relevant to today’s public health challenges.

#### Research Question
- Can we accurately predict early-stage diabetes risk using patient demographic and life style choices and  medical, and symptom data?
- What risk factors are most helpfull to predict the diabetes risk, corolation?

### Model Outcomes or Predictions

- #### Type of Learning
    This project applies **Supervised Machine Learning**, where the model is trained on labeled data indicating whether a person is **diabetic (1)** or **non-diabetic (0)**.

- #### Learning Objective
    The task is a **classification problem**, aiming to predict the likelihood that an individual has diabetes based on demographic, lifestyle, and health-related features such as **BMI, blood pressure, physical activity, and general health**.

- #### Expected Model Output
  **Output Type:** Binary (0 or 1)  
  - **0 → Non-Diabetic**  
  - **1 → Diabetic**
---
## 📂 Dataset
- Source: Public health survey (binary target = Diabetic).  
- Size: ~225,000 respondents.  
- Target variable: **Diabetic (2=Pre Diabetic  1 = Diabetic , 0 = Non_Diabetic)**.  
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
  - Careted BMI groups (Normal, Overweight, Obese) for visual and better understanding but not used in model.  
  - Encoded GenHlth as ordinal (Poor→Excellent).
  - Encoded Education as ordinal (Higer→ Basic).
  - Truncate noisy data on BMI as over 60 was not stable and were outliers
  - Built interaction features: HighBP×HighChol, BMI×PhysHlth, Education×Income visuals.
  - In the dataset, individuals labeled as **2 (Prediabetic)** were reclassified as **1 (Diabetic)**.  
      This transformation simplifies the target variable into a **binary classification problem**, distinguishing between:
      - **0 → Non-Diabetic**
      - **1 → Diabetic (including Prediabetic)**

      ✅ This approach is justified because prediabetic individuals exhibit **similar risk factors and medical characteristics** as diabetic patients.  
      Combining these groups enhances model stability and ensures more **clinically meaningful predictions** for early risk detection.

---
## 🔍 EDA Key Findings
- **Imbalance:** 82.7% non-diabetic vs 17.3% diabetic.  
# 📊 Key Findings from EDA

## 1. EDA – Univariate Analysis

### A. Numeric Features
<img width="1489" height="311" alt="image" src="https://github.com/user-attachments/assets/f5597726-9c5d-48b5-aacd-1cd79c83a15c" />
<img width="1489" height="311" alt="image" src="https://github.com/user-attachments/assets/7514ceb7-b8e2-43f3-8b33-b88a5e523c9a" />

- **BMI**
  - Centered in the overweight range (27–28).
  - Many respondents obese (BMI > 30); extreme cases (BMI > 40) likely true severe obesity rather than noise.
  - ✅ Key Insight: Obesity is a major predictor; prevalence rises from 9% (normal weight) → 29% (obese).

- **MentHlth & PhysHlth**
  - Highly skewed: majority report 0 unhealthy days.
  - Meaningful subgroup reports 10–30 days → chronic illness group.
  - ✅ Key Insight: PhysHlth correlates more strongly with diabetes than MentHlth. Outliers kept since they represent real health burdens.

---

### B. Binary Features
<img width="1107" height="662" alt="image" src="https://github.com/user-attachments/assets/9b4e410b-2664-40cf-9d24-1831887a8abf" />
<img width="1111" height="676" alt="image" src="https://github.com/user-attachments/assets/1756216d-a91e-46aa-8a37-357778a3470f" />
<img width="1106" height="361" alt="image" src="https://github.com/user-attachments/assets/87192cdf-f582-4715-afd3-25d67904350f" />



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

### C. Categorical Features (Unfiltered Population View)
<img width="1980" height="1189" alt="image" src="https://github.com/user-attachments/assets/6cfa0994-8ca9-4f48-9468-22d44eaf49e3" />
<img width="1980" height="1189" alt="image" src="https://github.com/user-attachments/assets/4b4057e5-b9f1-47af-8f22-84b57e2ac54a" />
<img width="1959" height="844" alt="image" src="https://github.com/user-attachments/assets/a3ed26e4-872e-4982-8296-84cc1e450bba" />

- **GenHlth**: Majority rate health as Poor/Fair. Strongest categorical predictor — diabetes prevalence drops from 41% (Poor) → 4% (Excellent).
- **Age**: Skews older (65+ heavily represented). Diabetes prevalence increases sharply with age, especially 55+.
- **Education**: Lower education overrepresented; prevalence decreases with higher education.
- **Income**: Skewed toward lower income; prevalence decreases with higher income.

✅ Key Insight: The dataset is **not population-representative** (older, lower-income, lower-education bias), which must be considered in modeling and interpretation.

---

### D. Target Variable
- **Non-Diabetic:** 82.7%
- **Diabetic:** 17.3%
- ✅ Key Insight: Dataset is imbalanced. Accuracy is misleading; evaluation must emphasize **Recall, F1, ROC-AUC, PR-AUC**.

---

## 2. EDA Bivariate
  ### A. Numeric vs Diabetic
![Untitled](https://github.com/user-attachments/assets/13337116-05be-48ec-bfae-33702095e6c6)
- **Interpretation**

Rising Risk with BMI up to ~60

BMI 11.9–29.2 → ~11.7% diabetic

BMI 29.2–46.4 → ~26.3% diabetic

BMI 46.4–63.6 → ~41.5% diabetic 

✅ As BMI increases, diabetes prevalence rises sharply — consistent with obesity being a major risk factor.
Intresting : Drop-off in very high BMI bins
BMI 63.6–80.8 → ~16.1% BMI 80.8–98.0 → ~15.4% ⚠️ This looks counterintuitive. The likely reason is small sample sizes in these extreme BMI ranges. With very few cases, percentages become unstable.
The main signal is clear: obesity strongly increases diabetes risk. But the extreme outliers distort the tail end and we better drop outliers.

 ### B. Binary vs Diabetic
<img width="1498" height="2555" alt="image" src="https://github.com/user-attachments/assets/d22640f9-6f23-470f-8944-47e37c6fbbe6" /> <img width="1498" height="2555" alt="image" src="https://github.com/user-attachments/assets/8f090797-5ee4-4561-a58a-9121bcc590f7" />
| Feature | % (1) in Diabetic = 1 | % (1) in Diabetic = 0 | Δ (percentage points) | Key Insight |
|----------|-----------------------|-----------------------|------------------------|--------------|
| **HighBP** | 73.8 % | 25.8 % | **+48.0 pp** | Strongest predictor – hypertension highly associated with diabetes |
| **HighChol** | 66.4 % | 33.1 % | **+33.3 pp** | High cholesterol strongly linked to diabetes |
| **CholCheck** | 99.2 % | 99.2 % | 0.0 pp | Nearly universal – low variance |
| **Smoker** | 51.6 % | 51.5 % | +0.1 pp | Similar across groups – weak signal |
| **Stroke** | 8.9 % | 9.0 % | −0.1 pp | No difference – not predictive |
| **HeartDisease/Attack** | 21.4 % | 21.5 % | −0.1 pp | Comparable prevalence |
| **PhysActivity** | 63.4 % | 64.3 % | −0.9 pp | Slightly higher among non-diabetics |
| **Fruits** | 58.6 % | 59.2 % | −0.6 pp | Nearly identical – weak signal |
| **Veggies** | 75.7 % | 76.0 % | −0.3 pp | Nearly identical – weak signal |
| **HvyAlcoholConsump** | 2.6 % | 2.7 % | −0.1 pp | Very low prevalence in both |
| **AnyHealthcare** | 95.8 % | 95.3 % | +0.5 pp | Almost everyone has access – not predictive |
| **NoDocbcCost** | 10.9 % | 11.1 % | −0.2 pp | Minimal difference |
| **DiffWalk** | 36.2 % | 35.9 % | +0.3 pp | Similar – weak indicator |

 **Interpretation**

- **Most predictive binary features:**  
  🟥 `HighBP` and 🟧 `HighChol` show large differences between groups → strong predictors.  

- **Low-value / low-variance features:**  
  `CholCheck`, `AnyHealthcare`, and `HvyAlcoholConsump` — nearly universal or rare, so they contribute little.  

- **Neutral / non-predictive:**  
  Lifestyle features (`Smoker`, `PhysActivity`, `Fruits`, `Veggies`) show minimal class differences.  

---

✅ **Modeling Recommendation**
- **Keep:** `HighBP`, `HighChol` (strong signal)  
- **Consider dropping:** `CholCheck`, `AnyHealthcare`, `HvyAlcoholConsump` (low variance)  
- **Optionally test:** `DiffWalk`, `HeartDisease/Attack` (may interact with numeric features like BMI or Age)
  
### C.  Categoric vs Diabetic
<img width="1959" height="844" alt="image" src="https://github.com/user-attachments/assets/76bba5e9-d75d-4c1d-87d1-c05a7285617e" />
<img width="2084" height="820" alt="image" src="https://github.com/user-attachments/assets/9a28f96d-2059-4eb3-9dbf-4919c8b5a536" />

**Categorical Feature Analysis (Diabetic vs Non-Diabetic)**

| **Feature** | **Pattern / Trend Observed** | **Key Insights** |
|--------------|------------------------------|------------------|
| **GenHlth (General Health)** | As self-rated health improves from *Poor → Excellent*, diabetes prevalence drops sharply (41% → 3.9%). | Poor perceived health is strongly associated with higher diabetes rates. |
| **Sex** | Males (17.9%) and females (15.8%) show similar diabetic proportions, slightly higher in males. | Gender shows only mild variation in diabetes prevalence. |
| **Age** | Diabetes prevalence increases with age — from 2% (ages 18–24) to ~25% (ages 70–74). | Clear age-related risk: older adults are far more likely to have diabetes. |
| **Education** | Diabetes rate decreases with higher education — from 35% (no schooling) → 12.6% (college 4+). | Education may correlate with health literacy and preventive behavior. |
| **Income** | Lower income brackets show higher diabetes rates (29% for < $15k vs 10.7% for ≥ $75k). | Financial constraints likely impact access to healthcare and diet quality. |

🧩 **Summary**
- **Strongest associations:** Age ↑, Income ↓, Education ↓ ,General Health ↓
- **Weakest associations:** Sex and minor differences in self-reported health categories(“GenHlth” looks strong descriptively, but weak as an independent feature once objective factors (BMI, PhysHlth, Age) are included.
- **Overall pattern:** Socioeconomic and lifestyle factors — especially **low income**, **limited education**, and **older age** — are strongly tied to higher diabetes prevalence.  

**Overall Insight:**  
Age, general health, and socioeconomic factors (education and income) are **key demographic predictors** of diabetes.  
These insights highlight the importance of **targeted public health interventions** focusing on older, lower-income populations with poorer self-reported health.


---
### 3. EDA Multi variate Analysis
<img width="1989" height="790" alt="image" src="https://github.com/user-attachments/assets/480e6f25-9ba7-46aa-869c-f5117fa5a104" />
<img width="1111" height="790" alt="image" src="https://github.com/user-attachments/assets/b3778722-74ac-4869-af38-e330c5363f45" />
<img width="2133" height="533" alt="image" src="https://github.com/user-attachments/assets/823fab97-4251-4305-8732-0aeb670047eb" />


- **Top positive correlations with Diabetes:**
  - HighBP (0.26), BMI (0.21), DiffWalk (0.21), HighChol (0.20), HeartDisease/Attack (0.17), PhysHlth (0.16).
  - Rising Risk with BMI up to ~60 , BMI 11.9–29.2 → ~11.7% diabetic , BMI 29.2–46.4 → ~26.3% diabetic, BMI 46.4–63.6 → ~41.5% diabetic
- **Protective correlations:**
  - PhysActivity (−0.10), Veggies (−0.04), Fruits (−0.02).
- **Weak/noisy correlations:**
  - MentHlth (0.06), NoDocbcCost (0.03), AnyHealthcare (0.02).

✅ Key Insight: No single feature dominates → multivariate models (logistic regression with interactions, tree-based methods) will outperform univariate predictors.

 **Feature Correlation with Diabetes**

This chart and table summarize how strongly each feature correlates with the **Diabetic** outcome.  
Positive correlations (🔵) indicate a higher likelihood of diabetes, while negative correlations (🔴) suggest protective or inverse effects.

| Rank | Feature | Correlation | Interpretation | Decision |
|------|----------|-------------|----------------|----------------|
| 1️⃣ | **HighBP** | **+0.26** | Strongest correlation — hypertension is a major diabetes predictor | ✅ Keep |
| 2️⃣ | **DiffWalk** | **+0.21** | Walking difficulty often linked to obesity and metabolic risk | ✅ Keep |
| 3️⃣ | **HighChol** | **+0.21** | Elevated cholesterol — strong metabolic signal | ✅ Keep |
| 4️⃣ | **BMI** | **+0.19** | Higher BMI strongly associated with diabetes risk | ✅ Keep |
| 5️⃣ | **HeartDisease/Attack** | **+0.17** | Moderate comorbidity with diabetes | ✅ Keep |
| 6️⃣ | **PhysHlth** | **+0.17** | Poor physical health more common in diabetics | ✅ Keep |
| 7️⃣ | **Stroke** | **+0.10** | Mild relationship — secondary complication | ❌ Drop |
| 8️⃣ | **CholCheck** | **+0.07** | Nearly universal; limited variation | ❌ Drop |
| 9️⃣ | **MentHlth** | **+0.07** | Slight positive correlation; possible stress factor | ⚠️ Test |
| 🔟 | **Smoker** | **+0.05** | Minimal effect; similar rates in both groups | ❌ Drop |
| 11 | **NoDocbcCost** | **+0.03** | Cost barriers not strongly related | ❌ Drop |
| 12 | **AnyHealthcare** | **+0.01** | Almost everyone has healthcare — low variance | ❌ Drop |
| 13 | **Fruits** | **−0.03** | Slightly protective, weak signal | ❌ Drop |
| 14 | **Veggies** | **−0.05** | Weak inverse link — healthier lifestyle | ❌ Drop |
| 15 | **HvyAlcoholConsump** | **−0.06** | Low prevalence, minor effect | ❌ Drop |
| 16 | **PhysActivity** | **−0.11** | Moderate *protective* effect; active individuals less likely diabetic | ✅ Keep |

---

 **🧠 Summary**
- **Top Predictors:** `HighBP`, `HighChol`, `BMI`, `DiffWalk` — these drive most of the predictive signal.  
- **Moderate Predictors:** `HeartDisease/Attack`, `PhysHlth`, `Stroke` — useful in tree-based models or interactions.  
- **Low-Variance / Weak:** `CholCheck`, `AnyHealthcare`, `NoDocbcCost` — drop or downweight before modeling.  
- **Protective / Lifestyle Factors:** `PhysActivity`, `Veggies`, `Fruits` — retain for completeness and interpretability.

---

✅ Modeling Strategy
1. **Keep:** `HighBP`, `HighChol`, `BMI`, `DiffWalk`, `PhysActivity`  
2. **Test (Moderate):** `HeartDisease/Attack`, `PhysHlth`, `Stroke`, `Fruits`, `Veggies`, `Smoker`  
3. **Drop (Low Variance):** `CholCheck`, `AnyHealthcare`, `NoDocbcCost`, `HvyAlcoholConsump`
4. **Optionally test:** `DiffWalk`, `HeartDisease/Attack` (may interact with numeric features like BMI or Age)


<img width="2133" height="533" alt="image" src="https://github.com/user-attachments/assets/646dcc87-5c8c-4fd0-b648-c9a07f04d898" />


<img width="1189" height="829" alt="image" src="https://github.com/user-attachments/assets/368894da-ff6c-4251-a3bc-e4d0b0bf8ffc" />

The boxplots reveal that worse general health corresponds with higher BMI and poorer physical condition,
especially among diabetic individuals.
Mental-health effects exist but are less pronounced, suggesting physical and lifestyle factors are stronger diabetes predictors.



---

### 4. Interaction Effects

- **BMI × PhysHlth**: Strongest signal; diabetics cluster at high BMI + high PhysHlth.
- **HighBP × HighChol**: Prevalence rises to ~34% when both present (synergistic effect).
<img width="501" height="393" alt="image" src="https://github.com/user-attachments/assets/7c73989d-aa6d-469c-95bb-939be57371ac" />

- **HighBP × Age**: Older adults with hypertension show much higher prevalence than younger hypertensives.
- **Education × Income**: Clear socioeconomic gradient — low education + low income = highest risk.  
- Interpretation Insight: 

Individuals with limited education and low income exhibit up to 5× 
higher diabetes prevalence than their wealthier, better-educated counterparts.
This highlights how education and income together drive health disparities, likely via differences in nutrition, preventive care, and stress exposure.
<img width="1100" height="590" alt="image" src="https://github.com/user-attachments/assets/c6f17c27-4614-4e74-a49b-0f5295acfe7a" />


✅ Key Insight: Diabetes risk is multifactorial and **driven by combined effects**. Interaction terms should be included in logistic regression, or use tree-based models which capture them automatically.


---
### 5. Explore more on data
 Sex-Specific Patterns
- **BMI & PhysHlth**: Differences between diabetics and non-diabetics more pronounced in women.
- **MentHlth**: Diabetic women report more poor mental health days; effect weak in men.
- ✅ Key Insight: Diabetes has a heavier health burden in women → consider `Sex × BMI` and `Sex × PhysHlth` interactions.

 - also tried BMI Group for displaying better visual for affect on diabets but decided to remove this and keep number BMI to have less complex Model
<img width="989" height="390" alt="image" src="https://github.com/user-attachments/assets/ba5b4547-ebba-4a55-be10-819fe1710a64" />
- Scree Plot for numeric Plot
  - PC1 + PC2 explain ~78% of variance → Most structure is captured here.
  - PC3 adds only ~22% → marginal gain,
  -> safe to retain the first 2 components for visualization or modeling.
---

###6. 🎯 Overall Takeaways EDA
- **Core predictors**: HighBP, HighChol, BMI,  GenHlth, DiffWalk, PhysHlth, Age, Income, Education.
- **Protective factors**: Physical activity, healthier diet (weak effect).
- **Critical interactions**: BMI × PhysHlth, HighBP × HighChol, Age × HighBP, Education × Income.
- **Modeling Implications**:
  - Handle imbalance ( class weights).
  - Use **Recall, F1, ROC-AUC/PR-AUC** instead of Accuracy
      - A model could predict *everyone as non-diabetic* and still achieve **>80% accuracy**, even though it **completely fails to detect actual diabetics**. as data is imbalance
      - High accuracy in this case is misleading — it looks good, but it’s *clinically useless*.).
      - ✅ Better Metrics for Medical Screening

                  | **Metric** | **What It Measures** | **Why It Matters for Diabetes** |
                  |-------------|----------------------|----------------------------------|
                  | **Recall (Sensitivity)** | % of actual diabetics correctly identified | Missing diabetics (false negatives) can delay treatment — recall ensures we **catch as many diabetics as possible**. |
                  | **Precision** | % of predicted diabetics who are actually diabetic | Avoids unnecessary alarms or testing for healthy people. |
                  | **F1 Score** | Harmonic mean of precision & recall | Balances false negatives and false positives — ideal for **imbalanced data**. |
                  | **ROC-AUC** | Ability to rank diabetics higher than non-diabetics | Measures **overall discriminative power** — higher = better class separation. |
                  | **PR-AUC (Precision–Recall AUC)** | Focuses on performance for the diabetic class | More informative than ROC-AUC on **imbalanced datasets**. |
        -🩺 In a Diabetes Screening Context

                  | **Model Behavior** | **Real-World Meaning** |
                  |---------------------|------------------------|
                  | **High Recall** | Catches most diabetics → ideal for early detection. |
                  | **Low Recall** | Misses real diabetics → risky for public health screening. |
                  | **High Precision** | Fewer false alarms → more efficient for follow-up testing. |
                  | **High Accuracy but Low Recall** | Looks “good” statistically but **fails medically**. |

  - Include interaction features for logistic regression.
  - Will use **tree-based models** (Desion Tree or RandomForest, XGBoost) for capturing nonlinear + interaction effects automatically.
  - BMI Outlier Handling
      To ensure data stability, BMI values were **capped between 10 and 60** based on the interquartile range (IQR) method.  
      Extremely high BMI values (above 60) are considered **outliers or data-entry errors**, as they can distort statistical summaries and bias model training.  
      By filtering these unrealistic values, we maintain a more **robust and reliable distribution** that reflects real-world population health patterns.
---
## 🎯 Business Takeaways
1. **Prevention Levers**: Target obesity, hypertension, and cholesterol control — the most cost-effective interventions.  
2. **High-Risk Segments**: Older adults, low-income groups, and those with mobility issues or poor self-rated health.  
3. **Access Barriers**: Address affordability gaps (9% skip care due to cost) to reduce long-term disease burden.  
4. **Modeling Focus**: Use advanced ML models that balance precision and recall, ensuring at-risk members are identified early for **preventive outreach and resource allocation**.  

---
## 🛠️ Feature Engineering



## 🤖 Modeling
- **Baseline:** Dummy Classifier and Linesr Regression
results:
**Dummy Classifier:**
``accuracy train: 0.8292672193281148``

``accuracy test: 0.829273659427465``

``roc_auc: 0.5``

``f1_positive: 0.0``

``pr_auc: 0.17072634057253505``
``recall_positive: 0``


**Linear baseline (LogisticRegression):**

``accuracy train: 0.7000``

``accuracy test: 0.6994``

``roc_auc: 0.7856``

``f1_positive: 0.4540``

``pr_auc: 0.4178``

``recall_positive: 0.7321``
- The dummy classifier, which always predicts the majority group, gave us a deceptively high accuracy (~83%) but provided no real value for decision-making since it failed to identify any high-risk patients (ROC-AUC = 0.5, F1 = 0). In contrast, when we established Logistic Regression as our linear baseline, the model demonstrated meaningful predictive power: while overall accuracy dropped to ~70%, it successfully distinguished between patients at higher and lower risk (ROC-AUC ≈ 0.79, PR-AUC ≈ 0.42). This shows that, unlike the dummy model, Logistic Regression offers actionable insights and can serve as a solid starting point for building more advanced predictive models.

 ###📊 Linear Baseline (Logistic Regression) – Key Results

- **Strong sensitivity**: `recall_positive = 0.7321` → the model correctly identifies ~73% of actual diabetic cases.  
- **Meaningful ranking power**: `roc_auc ≈ 0.786` and `pr_auc ≈ 0.418` show the model can effectively separate high-risk from low-risk patients.  
- **Trade-off visible**: `f1_positive ≈ 0.454` with overall accuracy around 70% — expected when prioritizing recall of positive cases over general accuracy.
- 
## 📏 Evaluation Metric

### Selected Metric: **Recall (Sensitivity)**

#### ✅ Valid Interpretation
- A **high recall** means the model successfully identifies most diabetic individuals and 73 percent is considered high.  
- A **low recall** means the model misses many diabetics (high false negatives), which we observed in the baseline logistic regression 0.  

---

#### ✅ Rationale
- The dataset is **imbalanced** (≈83% Non-Diabetic, 17% Diabetic).  
- Accuracy is misleading: a model predicting “everyone = non-diabetic” achieves ~83% accuracy but 0% recall for diabetics for Dummy classifier but Linear Regression 73%.  
- From a **business and healthcare perspective**, missing diabetics (false negatives) is far more costly than false positives:
  - Missed cases = delayed treatment, higher future claims cost, worse patient outcomes.
  - False positives = additional screening cost, but far less severe impact.  

---

#### 🔑 Supporting Metrics
- **F1 Score**: Balances Recall and Precision, useful for capturing both the detection rate and correctness of positive predictions.  
- **ROC-AUC & PR-AUC**: Provide a broader view of model discrimination under imbalance.  

---

## 🎯 Key Insight
- For this project, **Recall is prioritized as the primary evaluation metric**, supported by **F1** and **ROC-AUC/PR-AUC**.  
- This ensures the model delivers **clinical and business value** by correctly identifying the maximum number of high-risk diabetic individuals, even if it means tolerating more false positives.
- **Next Steps:**
  - Address imbalance with **SMOTE resampling** or **class weights**.  
  - Evaluate models with **ROC-AUC, PR-AUC, Recall, and F1** (not accuracy).  
  - Test more models (KNN, or SVC or Decision Tree,  possible random forest) for capturing nonlinearities and interactions.  

---

## ✅ Expected Impact
- Provide insights into **key health, lifestyle, and socioeconomic risk factors**.  
- Develop a predictive model that balances sensitivity (recall for diabetics) with precision.  
- Support public health strategies by identifying high-risk groups (e.g., obese, hypertensive, low income/education).


#### Results
What did your research find?

#### Next steps
What suggestions do you have for next steps?



##### Contact and Further Information
linkedin :[Mina Sardari](www.linkedin.com/in/mina-s-3b728651)


