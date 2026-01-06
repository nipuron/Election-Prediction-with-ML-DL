# Election-Prediction-with-ML-DL

This repository contains a neural network model designed to predict the outcomes of the Swedish general election at the municipal level. By using socio-economic and demographic data, the model learns to estimate the voting percentage for each political party.

## Project Description
The core of this project is a **Multi-Layer Perceptron (MLP)** built using **PyTorch**. It analyzes the relationship between municipality-specific statistics (such as income levels, education, and employment) and how those factors influence the distribution of votes among the 8 major Swedish political parties.

## Features
- **Flexible Neural Network:** Easily adjustable hidden layers, activation functions, and dropout rates via a configuration object.
- **Preprocessing:** Integrated pipeline using `Scikit-learn` for feature scaling (`StandardScaler`) and data splitting.
- **Softmax Output:** The model uses a Softmax activation in the final layer to ensure that the predicted vote shares always sum to 100%.
- **Analysis Tools:** Includes functions to compare predictions against ground truth and test hypothetical demographic "profiles."

## Example Output (Swedish)
--- Analys för 1282 Landskrona
              Parti  Prediktion (%)  Faktiskt (%)  Skillnad (p.e.)
        Moderaterna       17.959999          17.3             0.66
      Centerpartiet        6.440000           3.8             2.64
        Liberalerna        3.680000           6.1            -2.42
  Kristdemokraterna        6.320000           3.6             2.72
       Miljöpartiet        3.570000           3.1             0.47
 Socialdemokraterna       29.590000          31.1            -1.51
     Vänsterpartiet        4.920000           4.8             0.12
Sverigedemokraterna       25.450001          27.4            -1.95
     övriga partier        2.070000           2.8            -0.73

## Future Improvements
-**Feature Importance:** Integrate SHAP values to identify which demographic factors (e.g., age or income) drive specific party votes.
-**Visualization:** Add Matplotlib scripts to graph training/validation loss curves.
-**Cross-Validation:** Implement K-Fold cross-validation to better handle the limited number of municipalities (290).

### Prerequisites
You will need Python 3.8 or higher. Install the necessary dependencies via pip:

```bash
pip install torch pandas numpy scikit-learn openpyxl
