<h1 align="center"><strong>Lab Animal Growth Prediction</strong></h1>

## Overview
This project presents a reproducible machine learning framework for predicting laboratory mouse body weight from biological and environmental factors using regression models. The current implementation and validation are performed exclusively on mice using synthetic [data](https://github.com/Ibrahim-El-Khouli/Lab-Animal-Growth-Prediction/blob/main/data/mouse_growth.csv) and a fully documented [notebook](https://github.com/Ibrahim-El-Khouli/Lab-Animal-Growth-Prediction/blob/main/lab-animal-growth-prediction.ipynb). While the framework is technically adaptable to other laboratory species, such extension would require retraining on appropriate datasets.

---

## Objective
The objective of this project is to develop and evaluate regression-based models that predict mouse body weight (in grams) from a combination of biological (Age, Sex, Strain) and environmental (Diet) factors. The goal is to demonstrate a computational workflow that remains interpretable to veterinary and biomedical professionals while emphasizing reproducibility and biological realism.

---

## Why Mice
Mice were selected as the initial species for model development due to their extensive use in laboratory research and the availability of well-characterized biological and environmental parameters. Key advantages include:

- Well-documented genetics and commonly used strains (C57BL/6, BALB/c, DBA/2, 129/Sv, FVB/N)
- Short life cycle and rapid growth, suitable for longitudinal studies
- Extensive historical data that facilitates the generation of realistic synthetic datasets
- High translational relevance to biomedical research

Focusing on mice allows for a reproducible and interpretable modeling framework, which could later be retrained and adapted for other laboratory species if appropriate datasets are available.

---

## Task Type
**Regression** — predicting a continuous outcome (body weight in grams) based on biological and environmental features.

---

## Dataset
A synthetic dataset of approximately 1,000 mice was generated for this project. The dataset includes the following variables:

| Variable | Description |
|-----------|-------------|
| AnimalID | Unique identifier for each mouse |
| Age_weeks | Age of the mouse in weeks (1–12 weeks) |
| Sex | Male or Female |
| Strain | C57BL/6, BALB/c, DBA/2, 129/Sv, FVB/N |
| Diet | Low, Medium, or High |
| Weight_g | Mouse body weight in grams (target variable) |

The dataset is saved as `mouse_growth.csv` for full reproducibility.

---

## Methods and Tools
This framework was developed in Python using widely adopted libraries for data handling, visualization, and regression modeling.

- **Data handling:** pandas, numpy  
- **Visualization:** matplotlib, seaborn  
- **Machine learning:** scikit-learn (Linear Regression, Decision Tree Regressor, cross-validation)  
- **Model persistence:** joblib (for saving and reloading trained models)  

Each stage of the workflow—from dataset generation to model evaluation—is implemented in a reproducible manner within the accompanying Jupyter notebook.

---

## Veterinary Relevance
Predicting body weight trajectories is an essential aspect of laboratory animal management and research. Understanding how growth is influenced by genetic and environmental factors can assist in:

- Monitoring and interpreting growth patterns across experimental groups  
- Evaluating strain–diet interactions  
- Identifying non-linear age-related effects (e.g., early rapid growth followed by a plateau)  

By combining regression modeling with biological interpretability, this project demonstrates how machine learning can complement veterinary and biomedical data analysis in controlled experimental settings.

---

## Reproducibility
All datasets and trained models are preserved to enable complete reproduction of the workflow:

- Trained regression models are saved using `joblib`, allowing predictions without retraining.  
- The synthetic dataset (`mouse_growth.csv`) ensures full transparency of preprocessing and analysis.  
- Data generation parameters introduce controlled random variability to approximate biological noise while maintaining interpretability.

This approach aligns with reproducible research principles and facilitates educational use, demonstration, and benchmarking.

---

## Step 1: Data and Dependencies
The project uses standard Python scientific libraries for numerical computation, data management, visualization, and machine learning. These tools collectively enable data generation, analysis, and model evaluation within a single reproducible framework.

---

## Step 2: Dataset Fields
The dataset consists of the following fields:

| Field | Description |
|--------|-------------|
| AnimalID | Unique identifier for each subject |
| Age_weeks | Age in weeks (1–12) |
| Sex | Male or Female |
| Strain | C57BL/6, BALB/c, DBA/2, 129/Sv, FVB/N |
| Diet | High, Medium, or Low |
| Weight_g | Mouse body weight in grams (target variable) |

**Notes for Researchers:**  
Although this dataset simulates 1,000 mice, the framework is easily extendable. Researchers can adjust the number of animals, strains, or diets, or include additional features (e.g., environmental enrichment, physiological markers) as needed. The design allows adaptation to other species through retraining with relevant data, not direct reuse of the current model.

---

## Step 3: Synthetic Dataset Generation
The synthetic dataset represents 1,000 mice characterized by the following features:

- Age_weeks: 1–12 weeks  
- Sex: Male or Female  
- Strain: C57BL/6, BALB/c, DBA/2, 129/Sv, FVB/N  
- Diet: Low, Medium, or High  
- Weight_g: Computed target variable representing body weight  

Weights were generated to reflect realistic biological patterns influenced by age, strain, sex, and diet. This synthetic design allows controlled demonstration of regression methods while maintaining biological plausibility.

---

## Step 4: Weight Calculation
Body weight values were generated using the following relationship:

**Weight_g = Base_weight_for_age + Strain_effect + Sex_effect + Diet_effect + Random_noise**

Where:

| Component | Description |
|------------|-------------|
| Base_weight_for_age | Linear increase from approximately 5 g at week 1 to 30 g at week 12 |
| Strain_effect | Small additive or subtractive adjustments representing genetic differences |
| Sex_effect | Males +1–2 g; females baseline (0 g adjustment) |
| Diet_effect | High diet: +2 g, Medium diet: 0 g, Low diet: –2 g |
| Random_noise | Random variation simulating individual biological differences |

This formulation generates biologically coherent growth patterns while avoiding unrealistic uniformity or excessive noise.

---

## Step 5: Data Visualization
To examine the behavior of the generated data and validate its biological realism, several exploratory visualizations were performed:

1. Scatter plot: Age (weeks) vs. Weight (g) for all mice  
2. Scatter plot: Age vs. Weight grouped by Strain  
3. Scatter plot: Age vs. Weight grouped by Sex  
4. Scatter plot: Age vs. Weight grouped by Diet  

**Purpose of visualization:**
- Assess general growth trends across age  
- Identify expected differences between strains, sexes, and diets  
- Detect any unrealistic values or excessive variability  
- Confirm that age, sex, strain, and diet collectively influence body weight in a plausible manner  

Visual examination confirms that the synthetic dataset behaves in a biologically realistic manner, providing a suitable foundation for regression modeling and veterinary interpretation.

---

## Step 6: Data Preprocessing for Machine Learning

Before model training, the dataset is prepared for numerical analysis. Categorical variables must be encoded in a form interpretable by regression algorithms, while continuous variables remain numeric.

### Preprocessing Steps
1. **Categorical Encoding:**  
   Convert `Sex`, `Strain`, and `Diet` into numerical representations using one-hot encoding.
2. **Continuous Feature:**  
   Retain `Age_weeks` as a numeric predictor.
3. **Target Variable:**  
   Define `Weight_g` as the dependent (target) variable.
4. **Data Validation:**  
   Inspect dataset structure and confirm preprocessing integrity.

### Baseline Categories
To enable biological interpretability, specific baseline categories are defined as reference levels:

| Variable | Baseline Category |
|-----------|------------------|
| Strain | 129/Sv |
| Diet | High |
| Sex | Female |

All other categories are interpreted relative to these baselines.  
For instance, if `Strain_BALB/c = –1.8`, it indicates that BALB/c mice weigh approximately 1.8 g less than 129/Sv mice when all other factors are constant.  
This encoding preserves interpretability for veterinarians when examining regression coefficients.

---

## Step 7: Linear Regression Model

### Objective
Train a **Linear Regression** model to predict mouse body weight from biological and environmental features and assess its predictive accuracy.

### Procedure
1. Split the dataset into **training (80%)** and **testing (20%)** subsets.  
2. Fit a Linear Regression model using the training data.  
3. Generate predictions on both sets.  
4. Evaluate model performance using:
   - Coefficient of Determination (**R²**)  
   - Mean Absolute Error (**MAE**)  
   - Root Mean Squared Error (**RMSE**)  
5. Inspect coefficients to interpret the biological meaning of each feature.  
6. Visualize actual vs. predicted weights and residual distributions.

### Interpretation of Coefficients
Dropped (baseline) categories act as reference points:

- Example: `Strain_BALB/c = –1.1` → BALB/c mice weigh ~1.1 g less than 129/Sv.
- Example: `Diet_Low = –3.9` → low-diet mice weigh ~3.9 g less than high-diet mice.
- Example: `Sex_M = +1.0` → males weigh ~1 g more than females.

These coefficients provide a quantitative interpretation of biological differences consistent with empirical expectations.

---

## Step 8: Residual and Coefficient Interpretation

### Residuals
Residuals measure the difference between actual and predicted weights:

\[
\text{Residual} = \text{Actual Weight} - \text{Predicted Weight}
\]

- **Residual ≈ 0:** Model prediction is accurate.  
- **Positive Residual:** Mouse is heavier than expected.  
- **Negative Residual:** Mouse is lighter than expected.  
- **Residual Spread (~±σ):** Reflects natural biological variability.

A random residual distribution indicates an adequate model fit, while patterned residuals may suggest nonlinearity or missing interactions.

### Coefficients
- **Age_weeks:** Represents expected weekly weight gain.  
- **Categorical Coefficients:** Quantify differences relative to baseline categories.  
  - Positive → increases predicted weight.  
  - Negative → decreases predicted weight.

Interpretation of coefficients provides direct biological insights into growth dynamics.

---

## Step 9: Decision Tree Regression

### Objective
Implement a **Decision Tree Regressor** to capture nonlinear and interaction effects in mouse growth patterns.

### Methodology
1. Train a Decision Tree model using the training data.  
2. Predict body weights for both training and test sets.  
3. Evaluate using **R²** and **MAE** for direct comparison with Linear Regression.  
4. Examine **feature importances** to identify which variables most influence weight predictions.  
5. Optionally visualize actual vs. predicted values and residual distributions.

### Biological Relevance
Decision trees can naturally represent biological nonlinearities, such as:
- Strain-specific growth plateaus.  
- Diet effects that vary with age.  
- Complex interdependencies between biological and environmental factors.

This flexibility enables data-driven exploration of biologically meaningful patterns without manually introducing polynomial or interaction terms.

---

## Step 10: Interpreting Actual vs. Predicted Scatter Plots

When evaluating model performance visually:

- **Points near the dashed line:** Accurate predictions.  
- **Above the line:** Overestimation (predicted weight > actual).  
- **Below the line:** Underestimation (predicted weight < actual).  

Such plots provide a rapid visual method for assessing prediction quality and systematic bias from a veterinary research perspective.

---

## Step 11: Cross-Validation and Model Comparison

### Objective
Apply **K-Fold Cross-Validation** to evaluate model robustness and generalizability across multiple subsets of data.

### Rationale
In laboratory research, animal cohorts are typically tested in multiple batches. Cross-validation replicates this by iteratively training and validating models on different data splits, ensuring consistent performance across age, strain, and diet combinations.

### Procedure
1. Select a number of folds (e.g., K = 5).  
2. For each fold:
   - Split data into training and validation sets.  
   - Train the model on the training subset.  
   - Predict weights for the validation subset.  
   - Record R² and MAE.  
3. Compute mean ± standard deviation for each metric across folds.  
4. Compare performance between Linear Regression and Decision Tree models.

This process helps detect overfitting and identifies which model generalizes better to unseen animals.

---

## Step 12: Polynomial and Interaction Features (Advanced Analysis)

To explore potential nonlinear and interactive effects:

- **Polynomial Features (degree = 2):**  
  Include squared terms such as `Age_weeks²` to capture accelerated or plateauing growth phases.

- **Interaction Features:**  
  Combine variables, e.g., `Age_weeks × Diet_Low`, to evaluate how age-related growth depends on diet.

These features allow the regression model to describe more complex biological relationships without manual adjustment.

**Interpretation:**  
Although advanced models marginally improved metrics, results indicated that mouse growth in this dataset remains predominantly linear with predictable contributions from strain and diet.  
This finding aligns with controlled laboratory conditions and consistent husbandry practices.

---

## Step 13: Saving Outputs for Reproducibility

### Objective
Preserve datasets and trained models to ensure transparency and reproducibility across future analyses or collaborations.

### Saved Files
| File | Description |
|------|--------------|
| `mouse_growth.csv` | Synthetic dataset used for training and evaluation |
| `linear_regression_mouse_model.pkl` | Trained Linear Regression model |
| `decision_tree_mouse_model.pkl` | Trained Decision Tree model |

Intermediate feature-engineered datasets are regenerated as needed and therefore not stored separately.  
Saved models can be reloaded for direct prediction on new mouse datasets, ensuring consistent, reproducible workflows.

---

## Conclusion

This project established a reproducible computational framework to predict **mouse body weight** based on age, sex, strain, and diet.  

**Key Findings:**
- **Age** is the dominant predictor of growth (~2 g per week on average).  
- **Diet and strain** introduce expected weight shifts consistent with laboratory observations.  
- Linear Regression explained most variance (R² ≈ 0.98), with residual errors corresponding to biological variability.  
- **Decision Tree Regression** captured subtle nonlinearities but did not significantly outperform the linear model.

**Interpretation:**  
Mouse growth under standardized laboratory conditions follows a largely linear trajectory influenced by dietary and genetic factors. Residual deviations reflect natural inter-individual variability rather than model deficiency.

---

## Flexibility of the Framework

While demonstrated exclusively in **laboratory mice**, this computational structure is **technically species-agnostic**.  
Researchers may adapt it to other models—such as rats, rabbits, or non-human primates—by retraining on species-specific datasets and adjusting relevant biological features (e.g., breed, housing conditions, physiological markers).

The framework can also scale to:
- Larger sample sizes or additional experimental arms.  
- Inclusion of environmental enrichment, microbiome composition, or metabolic parameters.  
- Integration of non-linear or time-series modeling for longitudinal analyses.

---

## Summary
This project provides a reproducible, interpretable, and extensible foundation for laboratory animal growth modeling.  
It bridges machine learning with veterinary insight, emphasizing both methodological rigor and biological meaning—offering a transparent template for researchers studying growth, physiology, and experimental variation in controlled animal settings.

---

## **GitHub Repositories for Other Work**

- [PostOpPainGuard™](https://github.com/Ibrahim-El-Khouli/PostOpPainGuard.git)
- [LECI - Lab Environmental Comfort Index](https://github.com/Ibrahim-El-Khouli/LECI-Lab-Environmental-Comfort-Index.git)  
- [Lab Animal Health Risk Prediction](https://github.com/Ibrahim-El-Khouli/Lab-Animal-Health-Risk-Prediction.git)  
- [Lab Animal Growth Prediction](https://github.com/Ibrahim-El-Khouli/Lab-Animal-Growth-Prediction.git)

---

## **License**

**Lab Animal Health Risk Prediction** is released under the **MIT License** — free for academic, research, and non-commercial use.
