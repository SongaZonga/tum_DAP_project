# tum_DAP_project
Our attempt at forecasting demand 18 months in advance for infineon with a local model for each product.

## Questions to tackle (from infineon readme, subpoint is how we plan to address the question):
- From a methodological standpoint what time series forecasting options are available, what are advantages and disadvantages (forecasting new products – cold start, …)?
 - Different models e.g. ARIMA/LSTM....
 - Different libraries statsmodels/Tensorflow/Darts/Sktime
 - Exogenous problems:
    - Cold Start - Find similar products with features similar to new product and predict based on that (maybe based on a certain match threshold etc.) 
- Which additional features can you extract from the provided data for model training (e.g., window features, lag features, seasonality, trend, …)?
    - We need to perform more EDA
- Which of the provided external indicators has a good correlation with the demand of specific products (or product groups)?
    - Can pick external indicators based on a certain correlation threshold
    - If there are correlated external variables, drop the lower one.
- Which of the provided external indicators has a leading characteristic in the demand of specific products (or product groups) --> cross correlation?
    - More EDA is required

**Bonus question**
Find other macroeconomic & semiconductor market indicators and check their performance regarding correlation significance & leading characteristic on product demand (e.g., stock market developments, GDP (gross domestic product), per capita GDP)?


## Data Cleaning Steps:
1. Remove everything with the "no Plan" tag in the column "planning_method_latest"
2. We also remove all products that have had less than **18** months of demand records.

## Data Modeling :
- Local models will be local to product names.
- Columns such as "product_main_family", "product_marketing_name", and "product_basic_type" will be transformed into dummy variables/factors to be used in prediction. We intend to use these to create a match with cold start products.

## Models used
### Baseline models to be compared against:
1. [ARIMA](https://www.linkedin.com/advice/3/what-advantages-disadvantages-arima-models-forecasting#:~:text=Advantages%20of%20ARIMA%20models&text=ARIMA%20models%20can%20account%20for,a%20few%20parameters%20and%20assumptions.)
    - ARIMA models can account for various patterns, such as linear or nonlinear trends, constant or varying volatility, and seasonal or non-seasonal fluctuations. ARIMA models are also easy to implement and interpret, as they only require a few parameters and assumptions. 
    - ARIMA may not be suitable as we want to create a local model and tuning and extracting information from an ACF/PACF plot is not possible. We could implement a new function to do it, but due to the time constraints and papers showing many other models performing better than ARIMA, we did not think this was a good use of time.
2. [LSTM] (https://www.quora.com/Why-is-LSTM-good-for-time-series-prediction)
    - LSTM can provide better results than parametric models and standard RNNs when dealing with complex autocorrelation sequences (long memory), large datasets, and the probability distribution of the underlying process is unknown or not replicable using standard parametric methods like ARIMA. 

### Other models we plan to explore:
1. RNN
2. Facebook Prophet
3. VARIMA
4. Kats
5. XGBoost
6. DeepAR

Time Series CV will be performed on all models and then be ranked by their SMAPE Score 

## Multi-Step Prediction:
We intend to use multiple step prediction to predict 18 months ahead [using the following methods](https://arxiv.org/pdf/1108.3259.pdf)
- Recursive
- Direct
- DirRec
- MIMO (requires multivariate predictions)
- DIRMO (requires multivariate predictions)

## Metric used
**SMAPE** - To ensure that models are easily compared to Infineon's models

## Modeling Libraries Used:
1. Sklearn
2. Statsmodel
3. Tensorflow
4. prophet
5. GluonTS
6. Darts
