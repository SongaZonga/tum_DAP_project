# tum_DAP_project
Our attempt at forecasting demand 18 months in advance for infineon with a local model for each product.

## Questions to tackle (from infineon readme, subpoint is how we plan to address the question):
- From a methodological standpoint what time series forecasting options are available, what are advantages and disadvantages (forecasting new products – cold start, …)?
 - Exogenous problems:
    - Cold Start 
- Which additional features can you extract from the provided data for model training (e.g., window features, lag features, seasonality, trend, …)?
- Which of the provided external indicators has a good correlation with the demand of specific products (or product groups)?
- Which of the provided external indicators has a leading characteristic in the demand of specific products (or product groups) --> cross correlation?

**Bonus question**
Find other macroeconomic & semiconductor market indicators and check their performance regarding correlation significance & leading characteristic on product demand (e.g., stock market developments, GDP (gross domestic product), per capita GDP)?

## Data Modeling :
- Local models will be local to product names.
- Columns such as "product_main_family", "product_marketing_name", and "product_basic_type" will be transformed into dummy variables/factors to be used in prediction. We intend to use these to create a match with cold start products.
- Files:
  - 20230411_SummerTerm23_Data_Challenge_Infineon_Data.csv 
  - IFX_DE.csv
  - eda.ipynb
  - full_df.csv (modified version of orginal dataset, with stock prices, and only with products with more than or equal to 50 datapoints)
  - yfinance_data.ipynb

## Models used
### Baseline models to be compared against:
1. [ARIMA](https://www.linkedin.com/advice/3/what-advantages-disadvantages-arima-models-forecasting#:~:text=Advantages%20of%20ARIMA%20models&text=ARIMA%20models%20can%20account%20for,a%20few%20parameters%20and%20assumptions.)
    - ARIMA models can account for various patterns, such as linear or nonlinear trends, constant or varying volatility, and seasonal or non-seasonal fluctuations. ARIMA models are also easy to implement and interpret, as they only require a few parameters and assumptions. 
    - ARIMA may not be suitable as we want to create a local model and tuning and extracting information from an ACF/PACF plot is not possible. We could implement a new function to do it, but due to the time constraints and papers showing many other models performing better than ARIMA, we did not think this was a good use of time.
    - Files:
      - arima_loop_nocov.ipynb
      - Arima_score.csv
      - arima_loop_cov.ipynb
      - Arima_score_cov.csv
      - arima_loop_18mnth.ipynb
      - Arima_score_cov_18mnth.csv
2. [LSTM](https://www.quora.com/Why-is-LSTM-good-for-time-series-prediction)
    - LSTM can provide better results than parametric models and standard RNNs when dealing with complex autocorrelation sequences (long memory), large datasets, and the probability distribution of the underlying process is unknown or not replicable using standard parametric methods like ARIMA. 
    - Files:
      - Multivariate LSTM.ipynb
      - Infineon.ipynb
3. Random Forest
   - Random Forest (RF) is a versatile machine-learning algorithm commonly used for classification and regression tasks. It operates by combining multiple decision trees to make predictions. In a random forest, each decision tree is trained on a random subset of the training data and features, which helps prevent overfitting and improves generalization.
     - Files:
       - Random forest.ipynb

## Metric used
**SMAPE** - To ensure that models are easily compared to Infineon's models

Cold Start forecasting:
Cold Start - Find similar products with features similar to new product and predict
   - Files:
     - clustering.ipynb
     - cold_start_LSTM.ipynb
     - cold_start_LSTM_clus.ipynb
     - cold_start.csv
     - cold_start_clus.csv

## Modeling Libraries Used:
- Check requirements.txt
