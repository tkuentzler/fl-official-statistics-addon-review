# Federated Learning for distributed databases


## 2023-03-21 Meeting

- Julius im Nachgang
    - centralized cross validation instead of train test split?
    - Test: Funktioniert das Zentralisiert fuer die einzelnen regions?
        - liegt es an der FL aggregation oder an den individuellen Modellen?
            - letzter Datensatz geht schlecht

## Remarks (Oli Hauke)

### Notes

- **Medical Insurance**
    - data (< 30 MB) no need for distributing (Dask/Spark or Federated)
    - ml: dask can boost the calculations, tensorflow connection?

### Qs@UniHamburg

**General**
- planning: paper release? 
- collaboration?


**Medical Insurance**
- EDA: 
    - only feature "smoker" has a high impact on the target (s. correlations)
    - smoker has low variation (2 class), charges has high variation in each subclass -> trees stable, but bad for DNNs?
    - charges has 2nd highest correlation (0.3) to feature "age", explainable by increasing baseline charges with increasing age; lower bound has low variance;  (s. correlation/scatter)
    - **inhomogenity** between charges and bmi (variance proportional, s. scatter)
    - **inhomogenity** between charges and children (variance inversely proportional, s. scatter)
- ML
    - Performance improveble (centralized, no DNN), DNN has problems
    - AutoML 
        - Pycaret: Slightly worse performance (variance?), best $R^2= 0.85$ (5-fold cross variation). The best model was **Gradient Boosting** (very close to random forest).
        - TPOT: best performance $R^2 = 0.8861$ with `RandomForestRegressor(input_matrix, bootstrap=True, max_features=0.75, min_samples_leaf=11, min_samples_split=9, n_estimators=100)`
- DNN has also centralized problems (s. loss very high variance, adapted stopping). 
    - No real tuning, only 3 Hyperparameter combinations
    - Test über Train?
    - DNN no good model for the problem? 
    - Missing complexity? Even stronger decentralized.
    - Reduce learning rate or batch Size? S. [Problem with high Variance](https://www.quora.com/When-training-a-neural-network-what-does-it-mean-if-the-loss-on-the-validation-set-has-high-variance-e-g-it-goes-back-and-forth-each-epoch-between-good-and-bad-loss-How-do-I-know-when-to-stop-training-the-network)
- FL: 
    - no activation function (centrally is relu used)?  Why another model?
    - why has each element in  federated_insurance_data only 20 obs?

**LTE Umlaut**
- is the available data synthetic or real sample?

**PM Bejing**
- ...

