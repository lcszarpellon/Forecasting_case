# -*- coding: utf-8 -*-
"""
Created on Sun Jun  8 15:52:34 2025

@author: lucas
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.tsa.stattools import adfuller
import statsmodels.api as sm
import time
import warnings
import sys
import os
from datetime import datetime
sys.path.append(os.path.abspath(r"..\code_forecast_model\classes"))
from Data_ETL_Class import ETL_Model
from SARIMA_Class import SARIMA_Model
warnings.filterwarnings('ignore')
plt.rcParams['figure.figsize'] = [10,5]

#%%
current_date = datetime.now().strftime("%Y-%m-%d")
path = r'..\code_forecast_model\exports\\'
file_name_perform = f'model_performance_{current_date}.xlsx'
file_name_result = f'forecast_results_{current_date}.xlsx'
#%%
original_df = pd.read_excel(r"..\data\toy_dataset.xlsx")
original_df.columns = original_df.columns.str.strip().str.lower()
original_df['id'] = original_df['id'].astype('string')
#%%
Actuals_df = original_df.copy()
#%%
Actuals_df_plot= Actuals_df.groupby(Actuals_df['date'])['forecast'].sum().to_frame()
sns.lineplot(x=Actuals_df_plot.index,y=Actuals_df_plot['forecast'])
plt.savefig(path+'plot_original_trend.jpg', format='jpg', dpi=300, bbox_inches='tight')
#%%
date_col = 'date'
pred_col = 'forecast'
level = 'level_key'
pattern = ['X','Y','Z']
Pattern_col = 'Pattern(I)'
forecast_steps = 6
#%%
decompose = Actuals_df.groupby(date_col)[pred_col].sum()
model_df = decompose.to_frame()
result = sm.tsa.seasonal_decompose(decompose, model = 'additive')
result.plot()
plt.savefig(path+'plot_seasonal_decompose.jpg', format='jpg', dpi=300, bbox_inches='tight')
#%%
plot_pacf(decompose.values.squeeze(), lags=10,method='ywm')
plt.plot()
plt.savefig(path+'plot_pacf.jpg', format='jpg', dpi=300, bbox_inches='tight')
#%%

adf_test = adfuller(decompose)
print(f'\nADF Statistics: {adf_test[0]}')
print(f'\np-value: {adf_test[1]}')  
if adf_test[1] > 0.5:
    print('\nThe series IS Stationaty')
else:
    print('\nThe series is NOT Stationaty')


#%% 
### Single Level SARIMA

start_time = time.time()

group_columns = ['id']
etl_instance = ETL_Model(Actuals_df, pred_col, date_col, group_columns,path)
df_formated = etl_instance.format_data()
df_segmentation = etl_instance.demand_sensing(df_formated, 'id')
df_pred_ready = etl_instance.pattern_selection(df_formated, df_segmentation, Pattern_col, pattern)
df_pred_ready = df_pred_ready.groupby(df_pred_ready.index)[pred_col].sum().to_frame()

sarima_model = SARIMA_Model(df_pred_ready, pred_col, forecast_steps, seasonal_period=12)
forecast_df, performance_df = sarima_model.sarima_forecast_single_level()

sns.lineplot(forecast_df, x= date_col, y= pred_col, hue = 'Type')
plt.savefig(path+'plot_forecast.jpg', format='jpg', dpi=300, bbox_inches='tight')


end_time = time.time()
execution_time = round((end_time - start_time) * 0.0166667, 0)
print(f'\nFINAL - Execution time: {execution_time} \n')

#%%
forecast_df.to_excel(path + file_name_result)

performance_df.to_excel(path + file_name_perform)

