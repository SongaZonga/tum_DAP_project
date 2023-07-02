import numpy as np
import pandas as pd
from darts.models import AutoARIMA
from darts.metrics import smape
from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler
import matplotlib.pyplot as plt

input_file="20230411_SummerTerm23_Data_Challenge_Infineon_Data.csv"
date_column_name = 'reporting_time'
no_months = 50

df = pd.read_csv(input_file, parse_dates=[date_column_name]).set_index(date_column_name)
df = df[df['planning_method_latest']!='no Plan']
sub_df = df[df.groupby('product_name').product_name.transform('count')>=no_months].copy()

all_prod_names = list(sub_df['product_name'].unique())

metrics_df = pd.DataFrame({'product_name' : all_prod_names})

train = 0.7
val = 0.3

metric_list = []

print('start of loop')
for name in all_prod_names:
    print(f'working on product: {name}')
    scaler_train = Scaler()
    scaler_val_all = Scaler()
    scaler_val_uni = Scaler()
    scaler_val_cov = Scaler()
    print('variables scaled')
    
    prod_df = sub_df[sub_df['product_name']==f'{name}'][['demand', 'demand_lb', 'demand_ub']]
    darts_df = TimeSeries.from_dataframe(prod_df, freq='MS', fillna_value=prod_df['demand'].mean())
    train_set, val_set= darts_df.split_before(train)

    train_set = scaler_train.fit_transform(train_set)
    val_set_all = scaler_val_all.fit_transform(val_set)
    val_set_uni = scaler_val_uni.fit_transform(val_set['demand'])
    val_set_cov = scaler_val_cov.fit_transform(val_set[['demand', 'demand_lb', 'demand_ub']])

    model_aarima = AutoARIMA(information_criterion = 'hqic')
    model_aarima.fit(series = train_set['demand'])
    forecast = scaler_val_uni.inverse_transform(model_aarima.predict(len(val_set)))
    forecast = TimeSeries.pd_dataframe(forecast)
    forecast['demand'] = forecast['demand'].apply(lambda x: x if x>0 else 0.0001)
    forecast = TimeSeries.from_dataframe(forecast)
    metric_score = 100 - smape(val_set['demand'], forecast)
    metric_list.append(metric_score)


    plt.plot(TimeSeries.pd_series(scaler_train.inverse_transform(train_set)['demand']), label = 'historical')
    plt.plot(TimeSeries.pd_series(val_set['demand']), label = 'val_set')
    plt.plot(TimeSeries.pd_series(forecast), label = 'forecast')
    plt.legend()
    plt.title(f'Product {name}, Accuracy: {metric_score}')
    plt.savefig(f'output/AutoArima/graphs/{name}.png')
    plt.close()

    model_aarima.save(f'output/AutoArima/models/{name}.pkl')

metrics_df['Arima_SMAPE'] = metric_list


metrics_df.to_csv('output/AutoArima/Arima_score.csv')
print('completed')