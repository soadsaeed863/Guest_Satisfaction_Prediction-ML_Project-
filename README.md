#  Guest Satisfaction Prediction

Machine learning pipeline that predicts guest satisfaction for vacation-rental listings from listing metadata and text descriptions. Built as a two-milestone team project at the Faculty of Computers and Information, Ain Shams University (Scientific Computing dept.).

##  Overview
- **Dataset:** 8,724 listings × 69 columns (metadata + text: summary, amenities, house rules, etc.)
- **Milestone 1 (Regression):** predict `review_scores_rating` (0–100)
- **Milestone 2 (Classification):** classify `guest_satisfaction` level

##  Pipeline
1. **Data cleaning:** duplicates check, dropping columns with >80% missing values, removing redundant/ID columns, stripping `$` and `%`, converting dates to numeric
2. **Imputation:** median (numeric), mode (categorical), group-based imputation for zipcode/state
3. **Outliers:** IQR method + clipping extreme values
4. **Encoding:** binary mapping, manual dictionaries, factorization for high-cardinality columns, **TF-IDF** (500 features) for text columns
5. **Feature selection:** SelectKBest (f_regression / f_classif), Mutual Information, XGBoost/Random Forest importance, LassoCV, PCA experiments
6. **No data leakage (Milestone 2):** stratified 80/20 split; all imputation stats, encoders and the TF-IDF vectorizer fitted on the training set only, then serialized to `preprocessing.pkl`
7. **Tuning:** GridSearchCV (3-fold CV) for all models

##  Results

### Milestone 1: Regression
| Model | MSE | R² |
|---|---|---|
| **LightGBM** | **6.04** | **0.606** |
| Random Forest | 6.33 | 0.587 |
| XGBoost | 8.48 | 0.447 |
| CatBoost | 9.72 | 0.366 |
| Decision Tree | 11.96 | 0.220 |
| Linear Regression | 12.13 | 0.209 |
| Lasso | 12.17 | 0.206 |
| ElasticNet | 12.79 | 0.166 |

Adding feature selection, outlier clipping and a union of four selection methods (SelectKBest-F, Mutual Information, XGBoost importance, LassoCV) improved R² from ~0.24 to ~0.61.

### Milestone 2: Classification
| Model | Train Acc. | Test Acc. |
|---|---|---|
| **SVM (RBF/Poly)** | 0.938 | **0.953** |
| Random Forest | 0.791 | 0.786 |
| LightGBM | 0.705 | 0.713 |
| Decision Tree | 0.598 | 0.612 |
| Logistic Regression | 0.579 | 0.589 |
| Gaussian Naive Bayes | 0.525 | 0.520 |

##  Key Findings
- Ensemble/boosting models clearly outperformed linear models in regression.
- Combining TF-IDF text features with listing metadata improved performance.
- Hybrid feature selection (statistical tests + model-based importance) gave the most robust feature set.
- Most influential features: `host_is_superhost`, `number_of_reviews`, `number_of_stays`, `host_since`, `host_total_listings_count`.

##  Tech Stack
Python · Pandas · NumPy · Scikit-learn · XGBoost · LightGBM · CatBoost · Matplotlib · Seaborn · Jupyter

##  Repository Structure
| File | Description |
|---|---|
| `ML_Project.ipynb` | Main notebook (Milestone 1 & 2) |
| `ML_Project_1.ipynb` | Additional notebook |
| `GuestSatisfactionPrediction.csv` | Dataset |
| `Final_Report.pdf` | Full project report |

##  Team (SC_6)
Supervised by **Dr. Dina Khattab**

- Soad Saeed — [@soadsaeed863](https://github.com/soadsaeed863)
- Ibrahem Mohamed
- Menna Hassan
- Raghad Sami
- Fatma Elzahraa Atef
- Hossam Eldin Ahmed

> Original team repository: https://github.com/Guest-Satisfaction-Prediction/Guest-Satisfaction-Prediction-ML-Project-

## My Contribution
- **Feature Selection:** worked on selecting the most informative features (SelectKBest with f_classif, Random Forest importance, and their union) to reduce dimensionality and overfitting.
- **Modeling:** trained, tuned (GridSearchCV, 3-fold CV) and evaluated **LightGBM** and **Gaussian Naive Bayes**, and compared their accuracy and training/inference time against the other team models.
