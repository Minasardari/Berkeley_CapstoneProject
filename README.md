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
 
- #### **Methodology used**
  reprocessing & EDA CRISP-DM, Data Preparation (cleaning missing values, encoding categorical features, scaling numerical attributes), Correlation analysis and visualization Feature engineering Model comparison and selecting a model (Logistic Regression, Decision Tree, KNN, SVC/SVM) Also anticipate to use some other concept related to upcoming modules Random Forest and AdaBoost or Neural Networks Evaluation: Accuracy, Precision, Recall,F1-Score ,Log Loss, Confusion Matrix
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

**Features (Inputs – 21 variables)**
1.   HighBP – High blood pressure
2.   HighChol – High cholesterol
3.   CholCheck – Cholesterol check in 5 years
4.   BMI – Body Mass Index
5.   Smoker – Ever smoked 100 cigarettes
6.   Stroke – Ever had a stroke
7.   HeartDiseaseorAttack – Coronary heart disease or myocardial infarction
8.   PhysActivity – Physical activity in past 30 days

9.   PhysActivity – Physical activity in past 30 days

10.   Fruits – Consume fruits 1+ times per day
11.   Veggies – Consume vegetables 1+ times per day

12.   HvyAlcoholConsump – Heavy alcohol consumption
13.   AnyHealthcare – Have any kind of health care coverage
14.   NoDocbcCost – Couldn’t see a doctor because of cost
15.   GenHlth – Self-rated general health (1=excellent → 5=poor)
16.   MentHlth – Days of poor mental health (past 30)
17.   PhysHlth – Days of poor physical health (past 30)
18.   DiffWalk – Serious difficulty walking or climbing stairs
19.   Sex – Biological sex (0=Female, 1=Male)
20.   Age – Age category (13 levels)
21.   Education – Education level (1–6 scale)
22.   Income – Household income level (1–8 scale)


**Output (Target – differs by dataset)**
Diabetes_binary (3 classes, imbalanced, full population)

0 = no diabetes
1 = diabetes
2 = prediabetes 

---

 ## 🧹 Data Preparation
- Removed duplicates and handled missing values.  
- Outliers reviewed; extreme values retained when clinically meaningful.  
- Feature engineering:
  - Careted BMI groups (Normal, Overweight, Obese) for visual and better understanding but not used in model.  
  - Encoded GenHlth as ordinal (Poor→Excellent).
  - Encoded Education as ordinal (Higer→ Basic).
  - Truncate noisy data on BMI as over 60 was not stable and were outliers
  - Built interaction features: HighBP×HighChol, BMI×PhysHlth, Education×Income visuals.
  - Individuals labeled as **2 (Prediabetic)**  has been reclassified as **1 (Diabetic)**.  
      This transformation simplifies the target variable into a **binary classification problem**, distinguishing between:
      - **0 → Non-Diabetic**
      - **1 → Diabetic (including Prediabetic)**

      ✅ This approach is justified because prediabetic individuals exhibit **similar risk factors and medical characteristics** as diabetic patients.  
      Combining these groups enhances model stability and ensures more **clinically meaningful predictions** for early risk detection.

---
## 🔍 EDA Key Findings
- **Imbalance:** 82.7% non-diabetic vs 17.3% diabetic.  

### 1. EDA – Univariate Analysis

#### A. Numeric Features
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



#### B. Binary Features
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



#### C. Categorical Features (Unfiltered Population View)
<img width="1980" height="1189" alt="image" src="https://github.com/user-attachments/assets/6cfa0994-8ca9-4f48-9468-22d44eaf49e3" />
<img width="1980" height="1189" alt="image" src="https://github.com/user-attachments/assets/4b4057e5-b9f1-47af-8f22-84b57e2ac54a" />
<img width="1959" height="844" alt="image" src="https://github.com/user-attachments/assets/a3ed26e4-872e-4982-8296-84cc1e450bba" />

- **GenHlth**: Majority rate health as Poor/Fair. Strongest categorical predictor — diabetes prevalence drops from 41% (Poor) → 4% (Excellent).
- **Age**: Skews older (65+ heavily represented). Diabetes prevalence increases sharply with age, especially 55+.
- **Education**: Lower education overrepresented; prevalence decreases with higher education.
- **Income**: Skewed toward lower income; prevalence decreases with higher income.

✅ Key Insight: The dataset is **not population-representative** (older, lower-income, lower-education bias), which must be considered in modeling and interpretation.

---

#### D. Target Variable
- **Non-Diabetic:** 82.7%
- **Diabetic:** 17.3%
- ✅ Key Insight: Dataset is imbalanced. Accuracy is misleading; evaluation must emphasize **Recall, F1, ROC-AUC, PR-AUC**.

---

### 2. EDA Bivariate
  #### A. Numeric vs Diabetic
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

 #### B. Binary vs Diabetic
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

✅ **Modeling Recommendation**
- **Keep:** `HighBP`, `HighChol` (strong signal)  
- **Consider dropping:** `CholCheck`, `AnyHealthcare`, `HvyAlcoholConsump` (low variance)  
- **Optionally test:** `DiffWalk`, `HeartDisease/Attack` (may interact with numeric features like BMI or Age)
  
#### C.  Categoric vs Diabetic
<img width="1959" height="844" alt="image" src="https://github.com/user-attachments/assets/76bba5e9-d75d-4c1d-87d1-c05a7285617e" />
<img width="2084" height="820" alt="image" src="https://github.com/user-attachments/assets/9a28f96d-2059-4eb3-9dbf-4919c8b5a536" />

**Categorical Feature Analysis (Diabetic vs Non-Diabetic)**
<img width="1189" height="829" alt="image" src="https://github.com/user-attachments/assets/368894da-ff6c-4251-a3bc-e4d0b0bf8ffc" />

The boxplots reveal that worse general health corresponds with higher BMI and poorer physical condition,
especially among diabetic individuals.
Mental-health effects exist but are less pronounced, suggesting physical and lifestyle factors are stronger diabetes predictors.
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
 **Sex-Specific Patterns**
- **BMI & PhysHlth**: Differences between diabetics and non-diabetics more pronounced in women.
- **MentHlth**: Diabetic women report more poor mental health days; effect weak in men.
- ✅ Key Insight: Diabetes has a heavier health burden in women → consider `Sex × BMI` and `Sex × PhysHlth` interactions.

 - also tried BMI Group for displaying better visual for affect on diabets but decided to remove this and keep number BMI to have less complex Model
<img width="989" height="390" alt="image" src="https://github.com/user-attachments/assets/ba5b4547-ebba-4a55-be10-819fe1710a64" />

<img width="819" height="770" alt="image" src="https://github.com/user-attachments/assets/7839b6cf-c1e8-4b0d-8eed-452bdf5f1e33" />
This visual basically prove with more poor days as mental or physical health and higher BMI the Diabete is more likely. 

**Scree Plot for numeric Plot**
<img width="691" height="470" alt="image" src="https://github.com/user-attachments/assets/49047c54-8725-4f7c-bce3-b4b722f33794" />
features = ["BMI", "PhysHlth", "MentHlth"] 
  - PC1 + PC2 explain ~78% of variance → Most structure is captured here.
  - PC3 adds only ~22% → marginal gain,
  -> safe to retain the first 2 components for visualization or modeling.
---

### 6. Overall Takeaways for EDA
#### 🎯 Modeling Takeaways
- **Core predictors**: HighBP, HighChol, BMI,  GenHlth, DiffWalk, PhysHlth, Age, Income, Education.
- **Protective factors**: Physical activity, healthier diet (weak effect).
- **Critical interactions**: BMI × PhysHlth, HighBP × HighChol, Age × HighBP, Education × Income.
    
#### 🎯 Business Takeaways
1. **Prevention Levers**: Target obesity, hypertension, and cholesterol control — the most cost-effective interventions.  
2. **High-Risk Segments**: Older adults, low-income groups, and those with mobility issues or poor self-rated health.  
3. **Access Barriers**: Address affordability gaps (9% skip care due to cost) to reduce long-term disease burden.  
4. **Modeling Focus**: Use advanced ML models that balance precision and recall, ensuring at-risk members are identified early for **preventive outreach and resource allocation**.  
---
## ✂️ Train/Test split
To prepare the dataset for model training and evaluation, a stratified train-test split was applied. The data was divided into 80% for training and 20% for testing, using a fixed random state (42) to ensure reproducibility. The stratification parameter was included to maintain the same proportion of diabetic and non-diabetic cases in both subsets.
This approach helps prevent class imbalance issues during model evaluation and ensures that performance metrics accurately reflect the model’s ability to generalize across both classes.

---
## 🛠️ Feature Engineering
 - **1. Drop features:**
     According to EDA following Columns will be dropped as they don't have not strong assiciation with target. ['HvyAlcoholConsump', 'AnyHealthcare', 'NoDocbcCost','CholCheck','Diabetes_binary_str','BMI_group','Smoker','Stroke', 'Fruits','Veggies' ]
 - **2. Feature Transform**
<img width="1687" height="394" alt="image" src="https://github.com/user-attachments/assets/95c77cd0-009e-4205-af0d-aaaa364bcab3" />

**ColumnTransformer** pipeline was implemented to preprocess the dataset before modeling.  
Each feature type was handled using an appropriate transformation to improve consistency, interpretability, and model performance.

### 🔹 Numerical Features
Features such as `BMI`, `MentHlth`, `PhysHlth`, `HighBP`, `HighChol`, `HeartDiseaseorAttack`, `PhysActivity`, and `DiffWalk` were standardized using **`StandardScaler`** to ensure all numeric inputs share a similar range and distribution.

### 🔹 Ordinal Categorical Features

Encoded using **`OrdinalEncoder`** to preserve the natural order of categories:

| Feature | Encoding Order |
|----------|----------------|
| `GenHlth` | Excellent → Very Good → Good → Fair → Poor |
| `Education` | Higher → High School → Basic |
| `Income` | Ordered by increasing income ranges |

### 🔹 Nominal Categorical Features

Non-ordered variables (such as `Sex`,'Age', etc.) were encoded using **`OneHotEncoder(drop='first')`** to prevent multicollinearity while retaining interpretability.

### 🔹  **3. BMI**
   BMI Outlier Handling
   <img width="452" height="230" alt="image" src="https://github.com/user-attachments/assets/409f01dd-044a-475e-bb20-6f96ef1b73db" />

      To ensure data stability, BMI values were **capped between 10 and 60** based on the interquartile range (IQR) method.  
      Extremely high BMI values (above 60) are considered **outliers or data-entry errors**, as they can distort statistical summaries and bias model training.  
      By filtering these unrealistic values, we maintain a more **robust and reliable distribution** that reflects real-world population health patterns.
---

## 🤖 Modeling

- **1. Modeling Implications**:
  - Should Handle imbalance ( class weights).
  - **Recall, F1, ROC-AUC/PR-AUC** instead of Accuracy
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
      
      - 🩺 In a Diabetes Screening Context
      | **Model Behavior** | **Real-World Meaning** |
      |---------------------|------------------------|
      | **High Recall** | Catches most diabetics → ideal for early detection. |
      | **Low Recall** | Misses real diabetics → risky for public health screening. |
      | **High Precision** | Fewer false alarms → more efficient for follow-up testing. |
      | **High Accuracy but Low Recall** | Looks “good” statistically but **fails medically**. |

 -  **2.Baseline:** Dummy Classifier and Linesr Regression
 
            **Dummy Classifier:**
            ``accuracy train: 0.83``
            ``accuracy test: 0.83``
            ``roc_auc: 0.5``
            ``f1_positive: 0.0``
            ``pr_auc: 0.17``
            ``recall_positive: 0``
  The dummy classifier, which always predicts the majority group, gave us a deceptively high accuracy (~83%) but provided no real value for decision-making since it failed to identify any high-risk patients (ROC-AUC = 0.5, F1 = 0). In contrast, when we established Logistic Regression as our linear baseline, the model demonstrated meaningful predictive power: while overall accuracy dropped to ~70%, it successfully distinguished between patients at higher and lower risk (ROC-AUC ≈ 0.80, PR-AUC ≈ 0.43). This shows that, unlike the dummy model, Logistic Regression offers actionable insights and can serve as a solid starting point for building more advanced predictive models.

 - **3. Linear Baseline (Logistic Regression)**

| Metric | Score | Interpretation |
|:--------------------------|:------:|:--------------------------------------------------------------|
| **Accuracy (Train)**      | 0.71   | Model fits training data moderately well — no major overfitting detected. |
| **Accuracy (Test)**       | 0.71   | Generalizes similarly on unseen data, confirming stable performance. |
| **ROC-AUC**               | 0.80   | Strong discriminative ability between diabetic and non-diabetic classes. |
| **F1 (Positive Class)**   | 0.47   | Moderate balance between precision and recall for diabetic detection. |
| **PR-AUC**                | 0.43   | Moderate precision-recall trade-off, suitable for imbalanced data. |
| **Recall (Positive Class)** | 0.75 | Captures 75 % of true diabetic cases — good sensitivity for early-risk screening. |

- **Strong sensitivity**: `recall_positive = 0.71` → the model correctly identifies ~71% of actual diabetic cases.  
- **Meaningful ranking power**: `roc_auc ≈ 0.80` and `pr_auc ≈ 0.43` show the model can effectively separate high-risk from low-risk patients.  
- **Trade-off visible**: `f1_positive ≈ 0.47` with overall accuracy around 70% — expected when prioritizing recall of positive cases over general accuracy.
<img width="605" height="432" alt="image" src="https://github.com/user-attachments/assets/6183c6ea-4b43-4f1b-8218-9f51ce28bc21" />

- The model correctly identifies **26,243 non-diabetic** and **5,662 diabetic** cases.
- **Recall (Sensitivity)** is strong — most diabetic patients are detected (75 %).
- **False positives (11,075)** show the model sometimes flags healthy individuals as at risk,
  but this trade-off is acceptable in **early screening**, where missing diabetic cases (false negatives) is more critical.  
---
### 🤖 Model Performance Comparison
This report summarizes the performance of six machine learning models on the **Diabetes Prediction** dataset.  
- **XGBoost**
- **SVC**
- **Decision Tree** 
- **Random Forest**
- **Logistic Regression**
-  **KNN**
Each model was evaluated using **Accuracy**, **ROC-AUC**, **F1**, and **Recall** metrics.  
Timing metrics for **training**, **prediction**, and **scoring** were also recorded to assess efficiency.


**🩺 Diabetes Risk Prediction — Model Evaluation Report**
This project evaluates multiple machine learning models for predicting **diabetes risk** using health and demographic data.  
The analysis compares models on accuracy, ROC-AUC, recall, and runtime performance to determine the best fit for real-world healthcare applications.

| Rank | Model | Accuracy (Train) | Accuracy (Test) | ROC-AUC | F1 (Positive) | Recall (Positive) | Total Time (s) | Notes |
|------|--------|------------------|-----------------|----------|----------------|-------------------|----------------|-------|
| 🥇 1 | **XGBoost** | 0.85 | **0.84** | 0.80 | 0.27 | 0.18 | **0.92** | ✅ Best trade-off between accuracy, generalization, and speed. Efficient for deployment. |
| 🥈 2 | **SVC** | 0.84 | **0.84** | **0.80** | 0.26 | 0.17 | 7.76 | High accuracy but slower; useful if runtime is less critical. |
| 🥉 3 | **Logistic Regression** | 0.71 | 0.71 | **0.80** | **0.47** | **0.75** | 3.75 | Excellent recall; interpretable model for screening use-case. |
| 4 | **Random Forest** | **0.97** | 0.82 | 0.75 | 0.28 | 0.21 | 22.36 | Overfitted (train ≫ test); strong but inefficient. |
| 5 | **KNN** | 0.87 | 0.82 | 0.70 | 0.30 | 0.23 | 15.46 | Acceptable results, but very slow at prediction time. |
| 6 | **Decision Tree** | **0.97** | 0.77 | 0.59 | 0.31 | 0.31 | 1.35 | Highly overfitted; poor ROC-AUC and generalization. |

### 💡 Analysis Summary

- **Generalization:**  
  XGBoost shows minimal gap between training (0.85) and testing (0.84) accuracy → excellent bias–variance balance.  
  Random Forest and Decision Tree clearly overfit.

- **Sensitivity (Recall):**  
  Logistic Regression detects the most positive (diabetic) cases — critical for medical screening.

- **Speed:**  
  XGBoost completes training + scoring in under **1 second**, far faster than SVC or KNN.

#### All Model Insights
##### 1️⃣ Logistic Regression
- **Strengths:** Strong recall (0.75), good F1, easy to interpret.  
- **Weaknesses:** Lower accuracy; limited to linear separability.  
- **Best for:** Early screening where identifying true positives is critical.
##### 2️⃣ XGBoost
- **Strengths:** Excellent accuracy and ROC-AUC; fastest training time.  
- **Weaknesses:** Lower recall may miss minority-class cases.  
- **Best for:** High-performance production use.

##### 3️⃣ SVC
- **Strengths:** Balanced accuracy and AUC similar to XGBoost.  
- **Weaknesses:** Training slower; recall limited.  
- **Best for:** Medium-sized datasets needing robust generalization.

##### 4️⃣ Random Forest
- **Strengths:** High training accuracy (0.98) and stable performance.  
- **Weaknesses:** Overfitting (train-test gap) and long runtime.  
- **Best for:** Feature importance analysis and interpretability in ensemble form.

##### 5️⃣ KNN
- **Strengths:** Good recall; simple to implement.  
- **Weaknesses:** Computationally expensive for prediction; sensitive to feature scaling.  
- **Best for:** Baseline model or low-dimensional datasets.
##### 6️⃣ Decision Tree
- **Strengths:** Fast and easy to visualize.  
- **Weaknesses:** Overfits easily; poor generalization (AUC = 0.59).  
- **Best for:** Quick prototypes or explainable decision rules.

#### ⏱️ Runtime Overview

| Stage | Description | Notes |
|--------|--------------|-------|
| **Train Time** | Time taken to fit the model. | RF and SVC were the slowest. |
| **Predict Time** | Time taken to generate predictions. | KNN was the most time-intensive. |
| **Score Time** | Evaluation time for metrics. | Minimal variance across models. |
| **Total Time** | Aggregate runtime. | XGBoost provided best efficiency-performance ratio. |
  

#### 📋 Business Perspective

From a healthcare analytics standpoint:

- **XGBoost** is the best choice for production — **accurate, stable, and fast**.  
  It can power large-scale diabetes risk screening efficiently.

- **Logistic Regression** remains valuable for clinical decision support where **interpretability** and **high recall** (catching at-risk patients) are essential.

- overfitted models like **Decision Tree** and **Random Forest** for deployment should be avoided.

#### 🚀 Recommendation Summary

| Objective | Recommended Model | Rationale |
|------------|------------------|------------|
| **Accurate & Fast Screening Tool** | 🥇 **XGBoost** | High accuracy + minimal overfitting + runtime < 1 s |
| **Clinical Decision Support** | 🥉 **Logistic Regression** | Transparent coefficients, top recall |
| **Exploratory / Research** | 🌳 **Random Forest** | Good for feature-importance exploration only  (train-test gap noticeable)   |
| **Fastest Model:** |⚡ **Decision Tree** |(lightweight but weak AUC)|
---
### 🧩 Hyperparameter tuning

Now that we’ve identified the most promising models (XGBoost, SVC, and Logistic Regression), the next step is to perform hyperparameter tuning to optimize their performance.
This process will help fine-tune parameters such as learning rate, regularization strength, and tree depth — aiming to improve accuracy, ROC-AUC, and recall while preventing overfitting.

<img width="783" height="702" alt="image" src="https://github.com/user-attachments/assets/31d90b16-b186-4d44-a7d5-743417489677" />

---

### 🧠 Selecting best model

#### 🏁 Best Tuned Models — Hyperparameter Optimization Results
 
After identifying the top-performing models, we conducted **hyperparameter tuning** using `GridSearchCV`.  
This step optimized parameters such as learning rate, regularization, and tree depth to improve overall model generalization and recall.

| Rank | Model | Best Params | Train Time (s) | CV ROC-AUC | Train Accuracy | Test Accuracy | Test ROC-AUC | Test PR-AUC | Test F1@0.5 |
|------|--------|--------------|----------------|-------------|----------------|----------------|---------------|--------------|--------------|
| 🥇 1 | **XGBoost** | {'colsample_bytree': 0.8, 'learning_rate': 0.1, 'max_depth': 3, 'n_estimators': 300, 'subsample': 0.8} | **312.33** | **0.805** | **0.838** | **0.838** | **0.810** | **0.468** | **0.295** |
| 🥈 2 | **SVC (Logistic Loss)** | {'alpha': 0.0001, 'loss': 'log_loss', 'max_iter': 1000, 'penalty': 'l2', 'tol': 0.0001} | 833.41 | 0.800 | 0.834 | 0.835 | 0.804 | 0.443 | 0.264 |
| 🥉 3 | **Logistic Regression** | {'C': 10, 'solver': 'lbfgs'} | 67.34 | 0.800 | 0.713 | 0.716 | 0.804 | 0.443 | **0.475** |
| 4 | **Random Forest** | {'class_weight': 'balanced', 'max_depth': 10, 'min_samples_split': 10, 'n_estimators': 300} | 1145.17 | 0.799 | 0.720 | 0.718 | 0.804 | 0.458 | **0.476** |
| 5 | **Decision Tree** | {'criterion': 'entropy', 'max_depth': 10, 'min_samples_split': 10} | 40.88 | 0.779 | **0.839** | 0.835 | 0.789 | 0.427 | 0.262 |
| 6 | **KNN** | {'algorithm': 'brute', 'n_jobs': -1, 'n_neighbors': 9, 'p': 2, 'weights': 'uniform'} | 470.92 | 0.738 | **0.850** | 0.826 | 0.741 | 0.357 | 0.300 |

 
#### 🔧 Interpretation

- **XGBoost** again achieved the **highest overall ROC-AUC (0.81)** and balanced performance with a moderate training time (~312 s).  
- **SVC** delivered strong results but was **significantly slower (~833 s)**, making it less practical for large-scale deployment.  
- **Logistic Regression** remained the **most interpretable model** and offered **top F1 (0.475)**, making it valuable for recall-sensitive screening.  
- **Random Forest** performed similarly in recall/F1 but with heavy computation time (>1100 s).  
- **KNN** showed strong training accuracy but generalization dropped — confirming it is less efficient and prone to overfitting.  


#### 💼 Business Recommendation

- **XGBoost (Tuned)** → ✅ **Best overall** for production: strong ROC-AUC, stable accuracy, fast inference.  
- **Logistic Regression (Tuned)** → ❤️ **Best for medical interpretation** and recall-focused decision support.  
- **Random Forest (Tuned)** → ⚙️ **Good secondary model** for feature importance exploration and ensemble stacking.  

--- 
### 🤖 Select Best Model

Based on tuning results, we selected XGBoost as the final model — it provides the best balance of ROC-AUC, accuracy, and runtime efficiency.
Next, we will tune the decision threshold on a validation slice and calibrate the predicted probabilities (using techniques like Platt scaling or Isotonic regression) to further refine classification performance.
This calibration step often boosts PR-AUC and F1-score, ensuring more reliable probability estimates and better alignment with real-world diabetes risk detection.
#### ⚙️🎯**Threshold optimization**

- #### 1. ⚙️ Threshold Tuning & Probability Calibration

After selecting XGBoost as the final model, I performed decision-threshold optimization on a validation slice to maximize the F1-score.
The procedure uses the precision-recall curve to identify the optimal cutoff (thr_star) where the harmonic mean of precision and recall is highest.
This approach refines the model’s classification boundary beyond the default 0.5 threshold.

Once the best threshold was found, we re-evaluated the tuned model on the held-out test set to measure F1, PR-AUC, and ROC-AUC at this optimal decision point.
Additionally, XGBoost’s predicted probabilities can be further calibrated (e.g., with Platt Scaling or Isotonic Regression) to improve probability reliability—often resulting in better PR-AUC and F1 metrics in medical screening contexts.

Best VAL F1: 0.48361356511779735 @ threshold = 0.23107977
Test F1 (tuned): 0.4957191780821918
Test PR-AUC: 0.4679131858620371
Test ROC-AUC: 0.8102984023395083

- #### 2. 🎯Calibrate Probabilities + Tune Threshold (Validation-Driven)

Calibrated XGBoost probabilities with isotonic regression (via CalibratedClassifierCV) and then tune the decision threshold on a held-out validation slice to maximize F1 along the precision–recall curve. Calibration improves probability reliability; threshold tuning aligns the classifier with our operational objective (higher F1/PR-AUC in screening). The calibrated model is finally evaluated once on the untouched test set.

Best VAL F1 (calibrated): 0.4843773509198275 @ threshold = 0.2559741705656052
Calibrated Test PR-AUC: 0.4671227630649516
Calibrated Test F1@τ* : 0.4965815403177157


#### 🧠 Next Step

Continue with **model explainability** using **SHAP** or **Permutation Importance** to interpret top features influencing predictions.  
This will enhance **transparency** and **trust** in the diabetes risk prediction pipeline.

---
## 📈 Feature Importance Analysis

<img width="500" height="500" alt="image" src="https://github.com/user-attachments/assets/442532a9-7a49-40b1-8fa3-02772a43f8b2" />
<img width="500" height="500" alt="image" src="https://github.com/user-attachments/assets/8aca29e2-06d9-49db-9203-6d77869ef579" />
<img width="289" height="300" alt="image" src="https://github.com/user-attachments/assets/9da7e24f-e44d-49ed-8ec2-0007bf881a15" />

#### 🔹 Top Predictors Identified (XGBoost)

| Rank | Feature | Interpretation |
|------|----------|----------------|
| 1️⃣ | **GenHlth (General Health)** | Strongest predictor — poor self-reported health correlates strongly with diabetes risk. |
| 2️⃣ | **BMI (Body Mass Index)** | High BMI strongly increases diabetes likelihood. |
| 3️⃣ | **HighBP (High Blood Pressure)** | Hypertension is a major co-morbidity of diabetes. |
| 4️⃣ | **HighChol (High Cholesterol)** | Metabolic disorder marker, common among diabetic individuals. |
| 5️⃣ | **Income** | Lower income levels often correlate with limited access to preventive care. |
| 6️⃣ | **Age 65–69 / 70–74** | Age group with higher risk; lifestyle and insulin resistance factors. |
| 7️⃣ | **Sex (Male)** | Males show slightly higher risk in this dataset. |
| 8️⃣ |**MentHlth (Mental Health Days)** | Chronic stress or poor mental health can contribute to metabolic issues. |
| 9️⃣ | **HeartDiseaseorAttack** |Cardiovascular history aligns with higher diabetes risk predictions. |
| 🔟 |**DiffWalk / PhysHlth** | Reduced mobility and poor physical health contribute modestly but meaningfully. |

#### 🩺 Insights
- **Self-perceived health (GenHlth)** is the single most predictive variable — a strong proxy for multiple underlying risks.  
-  Lifestyle and metabolic factors**BMI**, and **Blood Pressure/Cholesterol** dominate the model’s decision process.  
- **Socioeconomic factors** (Income, Education) appear as moderate contributors.  
- **Age and Sex** reflect biological and lifestyle risk stratification.  

---
## 📏 Evaluation 

#### 📊 Population Overview
- **Total Patients Evaluated**: 46,285
- **Diabetic Cases Identified**: 7,945  
  → **Prevalence**: ~17.2%

#### 📈 Model Outcomes
- **True Positives (TP)**: 5,015 — correctly flagged diabetic patients
- **False Positives (FP)**: 8,340 — non-diabetic patients incorrectly flagged
- **False Negatives (FN)**: 2,930 — diabetic patients missed by the model
- **True Negatives (TN)**: ~30,000 — correctly identified non-diabetic patients

#### 📈 Key Metrics (Approximate)
- **Prevalence**: 7,945 / 46,285 ≈ **17.2%**  
  Indicates the baseline rate of diabetes in the population.

- **Precision (PPV)**: 5,252 / (5,252 + 7,957) ≈ **0.40**  
  Of those flagged as diabetic, ~40% were truly diabetic.  
  → *Implication*: Moderate risk of false alarms; may strain follow-up resources.

- **Recall (Sensitivity/TPR)**: 5,252 / (5,252 + 2,693) ≈ **0.66**  
  The model catches ~66% of actual diabetic cases.  
  → *Implication*: Stronger emphasis on minimizing missed diagnoses.

- **Specificity (TNR)**: 30,000 / (30,000 + 7,957) ≈ **0.79**  
  ~79% of non-diabetic patients were correctly ruled out.

- **False Positive Rate (FPR)**: 1 − specificity ≈ **0.21**  
  ~21% of non-diabetics were incorrectly flagged.

- **Negative Predictive Value (NPV)**: 30,000 / (30,000 + 2,693) ≈ **0.918**  
  When the model predicts "not diabetic," it's correct ~91.1% of the time.  
  → *Implication*: Strong confidence in ruling out low-risk patients.

- **Accuracy**: (30,000 + 5,015) / 46,285 ≈ **0.757**  
  Overall correctness of predictions is ~75.7%.

- **F1 Score**: ≈ **0.48**  
  Balances precision and recall. Indicates moderate overall effectiveness.

<img width="605" height="437" alt="image" src="https://github.com/user-attachments/assets/ccfca38c-1560-4de3-9fe1-9a159d00694e" />

#### 📊 Confusion Matrix Counts
- **True Negatives (TN)**: ~30,000 — correctly identified non-diabetic patients
- **False Positives (FP)**: 7,957 — patients incorrectly flagged as diabetic
- **False Negatives (FN)**: 2,693 — missed diabetic cases
- **True Positives (TP)**: 5,252 — correctly identified diabetic patients
- **Total Samples**: 46,285

#### 🧠 Strategic Takeaways
- The model is **recall-oriented**, prioritizing detection of diabetic cases.
- Precision is modest, suggesting a need for **threshold tuning** or **post-model triage** to reduce false positives.
- High NPV and specificity support safe exclusion of low-risk patients.
- F1 score reflects a reasonable trade-off, but further optimization could improve clinical utility.

---




### Results
#### 🧾 Main Takeaways, Business Recommendations & Future Work

### 🧩 **Key Findings**

- The best-performing model, **XGBoost**, achieved a **Test ROC-AUC of 0.81** and strong generalization across folds.  
- The most influential predictors of diabetes risk were:
  1. **General Health (GenHlth)**
  2. **Body Mass Index (BMI)**
  3. **High Blood Pressure (HighBP)**
  4. **High Cholesterol (HighChol)**
  5. **Income Level**
- These features align strongly with clinical evidence — confirming that **lifestyle, cardiovascular health, and socioeconomic status** are central to diabetes risk.
- SHAP analysis showed clear interpretability:
  - Poor self-rated health and high BMI sharply increase predicted risk.
  - Low income and aging also raise likelihood.
  - Male individuals have a slightly higher predicted risk.

### 🩺 Strategic Implications
- The model is **recall-oriented**, prioritizing detection over precision—appropriate for high-risk domains like diabetes screening.
- **Follow-up protocols** may be needed to manage false positives efficiently.
- **Threshold optimization** and **post-model triage** could improve precision without compromising recall.
- High NPV supports **safe exclusion**, enabling confident decisions for low-risk patients.

### ✅ Expected Impact
- Provide insights into **key health, lifestyle, and socioeconomic risk factors**.  
- Develop a predictive model that balances sensitivity (recall for diabetics) with precision.  
- Support public health strategies by identifying high-risk groups (e.g., obese, hypertensive, low income/education).

## Next steps

| Focus Area | Description |
|-------------|--------------|
| **Feature Engineering** | Incorporate new predictors such as dietary habits, physical activity frequency, sleep quality, or genetic predisposition. |
| **Deployment Pipeline** | Build a app or website for Predicting based on User provided survey values. |
| **Model Refinement** | Explore **Ensemble Stacking (XGB + LR)** or **LightGBM/CatBoost** for improved precision and faster training. |
| **Class Imbalance Handling** | Apply **SMOTE** to improve recall for minority diabetic cases. |
| **Temporal Analysis** | Study longitudinal health data to identify early warning signals before diabetes onset. |
| **Fairness & Bias Testing** | Evaluate model performance across gender, age, and income groups to ensure equitable outcomes. |

---

## 🏁 **Conclusion**

This project demonstrates that **machine learning models—especially XGBoost—can effectively predict diabetes risk** using publicly available health survey data.  
Beyond prediction, **explainable AI (via SHAP)** provided clear, interpretable insights that align with real-world medical understanding.

By applying these findings, healthcare organizations can:
- Prioritize early detection,
- Optimize resource allocation,
- and promote **data-driven public health strategies** to reduce diabetes prevalence over time.

### 🩺Clinical takeaway:

What this tool does :
- Uses routine survey‐style information (age, BMI, blood pressure, cholesterol, activity level, general health, etc.) to estimate a patient’s risk of diabetes( The model catches ~66% of actual diabetic cases)  .
- Returns a risk score (0–1) and a flag (Diabetic vs Non-Diabetic) based on a tuned threshold chosen for screening.

- The tool is tuned to miss fewer true diabetics (higher recall) even if that creates some false alarms. That’s appropriate for screening. It should augment, not replace, clinical judgment and confirmatory testing (A1C, FPG, OGTT).
Follow-up steps:
   Order confirmatory labs (A1C / fasting glucose).
   Offer brief lifestyle counseling (diet, activity, weight management).
   Address BP/lipid control and mobility barriers where relevant.
   Prioritize care navigation for lower-income or low-education patients.

### Leasson Learned:
“My first SHAP run flagged a red flag—key signals like High Blood Pressure and High Cholesterol were ‘missing.’ Turns out I’d accidentally excluded the binary features from my preprocessing. After adding them back into the ColumnTransformer, the models slowed a bit but became reliable, and the feature attributions and outcomes finally made clinical sense.”
“I also discovered that using Python’s gc.collect() (garbage collector) between GridSearchCV loops prevented the process from getting stuck indefinitely. This simple fix helped free up memory, allowing my hyperparameter tuning to run smoothly and complete successfully.”

##### Contact and Further Information
linkedin :[Mina Sardari](www.linkedin.com/in/mina-s-3b728651)


