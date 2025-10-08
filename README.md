# Lab Animal Growth Prediction 

A reproducible machine learning framework for predicting laboratory animal growth trajectories using regression models.  
While demonstrated using synthetic mouse data, the framework is intentionally **species-adaptable** and can be adapted to other lab animals.  
This project emphasizes **veterinary interpretability**, with embedded vet notes, pseudocode guidance, and an integrated "Vet Interpretation" column in datasets.

---

## Project Goal
The objective of this project is to build and evaluate regression models that predict body weight in laboratory animals from **biological** (age, sex, strain) and **environmental** (diet) factors.  
The broader aim is to provide a **flexible and interpretable framework** for veterinary and biomedical researchers studying growth patterns, treatment effects, and inter-strain variability.

---

## Why Mice?
Mice were chosen because they are **the most widely used laboratory animals**:

- Well-characterized genetics and strains (C57BL/6, BALB/c, DBA/2, FVB/N, etc.)  
- Short lifecycle and rapid growth, ideal for longitudinal studies  
- Extensive historical data, which makes modeling easier and reproducible  
- High translational value for biomedical research  

> Starting with mice allows the creation of a **reproducible template** that can be extended to rats, guinea pigs, rabbits, non-human primates, or other lab animals.

---

## Methods
- **Data Source:** Synthetic dataset (~1,000 mice, 1–12 weeks of age)  
- **Preprocessing:** One-hot encoding of categorical features (`drop_first=True`)  
- **Models:**  
  - Linear Regression → interpretable baseline  
  - Decision Tree Regression → captures non-linear patterns  
- **Evaluation Metrics:**  
  - Coefficient of Determination (R²)  
  - Mean Absolute Error (MAE)  
  - Root Mean Squared Error (RMSE)  
- **Visualization & Vet Interpretability:**  
  - Growth curves across age, diet, and strain  
  - Actual vs. predicted weight scatter plots  
  - Residual analysis for model assessment  
  - Vet notes in Markdown and pseudocode cells  
  - Integrated "Vet Interpretation" column to annotate feature effects biologically

---

## Key Results
- **Age** emerged as the strongest predictor (~2 g/week)  
- **Diet and strain** contributed additional variance, consistent with laboratory observations  
- Linear Regression explained most of the variance (**R² ≈ 0.98, MAE ≈ 0.8 g, RMSE ≈ 1.0 g**)  
- Visualizations confirmed good—but not perfect—fit, reflecting natural biological variability  

**Vet Interpretation Example:**  
- `Age_weeks = +2 g/week`  
- `Strain_BALB/c ≈ -1.1 g` vs baseline (129/Sv)  
- `Diet_Low ≈ -3.99 g` vs baseline (High diet)  
- `Sex_M ≈ +0.96 g` vs baseline (Female)  

> Values indicate expected change **compared to baseline categories**.

---

## Flexibility of the Framework
This repository is **species-adaptable**:

- Works for any animal model (rats, guinea pigs, rabbits, non-human primates)  
- Scales easily to larger datasets, multiple strains, or experimental arms  
- Customizable features: breed, genetic line, husbandry conditions, environmental enrichment, microbiome data, clinical or physiological markers  
- Vet-focused: annotations, pseudocode, and integrated interpretation columns make results interpretable for veterinary professionals

> In short: a reusable template for researchers to tailor to their own experimental designs **with biological/veterinary perspective in mind**.

---

## Reproducibility & Notes
- Trained models are **saved for future use**, enabling predictions without retraining  
- The CSV dataset ensures others can **recreate feature engineering, train/test splits, and analyses**  
- Synthetic data is generated with controlled noise to mimic natural biological variation:  
```python
# Add biological noise
# σ = 0.5 g → closer to real mouse variability (< ±1.5 g), but risks overfitting (R² ≈ 1.0)
# σ = 1.0 g → broader spread (~±3 g), prevents artificial perfection
weight += np.random.normal(0, 1)  # Simulate natural biological variation
For the complete workflow, predictions, and visualizations, see the see the [Jupyter notebook](https://github.com/Ibrahim-El-Khouli/Lab-Animal-Growth-Prediction/blob/main/lab-animal-growth-prediction.ipynb).
