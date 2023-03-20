# Federated Learning for distributed databases


## Remarks (Oli Hauke)


### Notes

- **Medical Insurance**
    - data (< 30 MB) no need for distributing (Dask/Spark or Federated)
    - ml: dask can boost the calculations, tensorflow connection?

### Qs@UniHamburg

**Medical Insurance**
- EDA: 
    - only feature "smoker" has a high impact on the target (s. correlations)
    - smoker has low variation (2 class), charges has high variation in each subclass -> trees stable, but bad for DNNs?
    - charges has 2nd highest correlation (0.3) to feature "age", explainable by increasing baseline charges with increasing age; lower bound has low variance;  (s. correlation/scatter)
    - **inhomogenity** between charges and bmi (variance proportional, s. scatter)
    - **inhomogenity** between charges and children (variance inversely proportional, s. scatter)
- ML
    -AutoML (Pycaret): 
        - slightly worse performance, best $R^2= 0.85$ (5-fold cross variation)
        - the close best model was **Gradient Boosting**
    - TPOT: best performance $R^2 = 0.8861$ with `RandomForestRegressor(input_matrix, bootstrap=True, max_features=0.75, min_samples_leaf=11, min_samples_split=9, n_estimators=100)`
- DNN has also centralized problems (s. loss very high variance, adapted stopping). 
    - DNN no good model for the problem? 
    - Missing complexity? Even stronger decentralized.

**LTE Umlaut**
- is the available data synthetic or real sample?

**PM Bejing**
- ...

## Original Work
This repository is divided into three main parts. These can be found in the top folders.

### PM2.5 prediction for Beijing

The pm-beijing folder contains data with weather and air pollution data and can be used to predict the PM2.5 concentration in the air. The folder contains the following subfolders and files:
* data: The pm25 beijing dataset
* models: The trained models are saved here, it contains already trained models
* tensorboard-logs: The log files used by tensorboard
* pm25_beijing.py: The python module containing all the functions used in the notebooks
* pm25_central_model.ipynb: The single model, trained on all the data
* pm25_cross_validation.ipynb: The above model but trained with cross validation
* pm25_feature_comparison.ipynb: A simple model to compare how time and wind direction affect the prediction
* pm25_federated.ipynb: A federated model for PM2.5 prediction
* pm25_federated_cross_val.ipynb: The above federated model with cross validation
* pm25_hyperparametersearch.ipynb: A hyperparametersearch for optimising the model
* pm25_local_models.ipynb: A local model for each station is trained here

### LTE data from Umlaut

This folder contains everything related to Umlaut and their data. As the original data from Umlaut is not accessible it contains only short dummy data to test the models.
* data: The data files, both files hold the same kind of information, however be aware that the name of the features differ slightly
* lte_federated.ipynb: The federated model for the LTE data and centralised models for comparison reasons
* lte_hyperparametersearch.ipynb: A hyperparametersearch for different models
* umlaut_lte.py: A python module containing some functions used by the notebooks

### Medical insurance data

Everything concerning the medical insurance data is stored here. This dataset is used to calculate charges a person has to pay and contains data as age, sex, BMI, smoker, number of children and region.
* data: The medical insurance data
* med_insurance.ipynb: Some general data processing steps and a centralised model for medical insurance charge prediction
* med_insurance_federated.ipynb: The federated model for medical insurance charge prediction


