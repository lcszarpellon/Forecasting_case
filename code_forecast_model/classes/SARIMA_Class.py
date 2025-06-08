# -*- coding: utf-8 -*-
"""
Created on Wed Jun  4 10:08:05 2025

@author: l019868
Version 2:
    - Added outlier replacer with seasonal mean
    - replace Z pattern with seasonal mean

"""



from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import optuna
import pandas as pd
import numpy as np
import time

class SARIMA_Model:
    def __init__(self, df, pred_col, forecast_steps, seasonal_period=12):
        self.df = df
        self.pred_col = pred_col
        self.forecast_steps = forecast_steps
        self.seasonal_period = seasonal_period

    def forecast_sarima(self, y, order, seasonal_order):
        model = SARIMAX(y, order=order, seasonal_order=seasonal_order, enforce_stationarity=False, enforce_invertibility=False)
        model_fit = model.fit(disp=False)
        forecast = model_fit.forecast(steps=self.forecast_steps)
        return forecast, model_fit

    def optimize_sarima_order(self, y, max_p=5,max_d=3,max_q=5, max_P=3, max_D=1, max_Q=3, n_trials=35):
        def objective(trial):
            p = trial.suggest_int('p', 0, max_p)
            d = trial.suggest_int('d', 0, max_d)
            q = trial.suggest_int('q', 0, max_q)
            P = trial.suggest_int('P', 0, max_P)
            D = trial.suggest_int('D', 0, max_D)
            Q = trial.suggest_int('Q', 0, max_Q)

            try:
                size = int(len(y) * 0.7)
                train, test = y[:size], y[size:]
                history = list(train)
                predictions = []

                for t in range(len(test)):
                    model = SARIMAX(history, order=(p, d, q), seasonal_order=(P, D, Q, self.seasonal_period),
                                    enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
                    yhat = model.forecast()[0]
                    predictions.append(yhat)
                    history.append(test[t])

                rmse = mean_squared_error(test, predictions, squared=False)
                mape = mean_absolute_percentage_error(test, predictions)
                trial.set_user_attr('rmse', rmse)
                trial.set_user_attr('mape', mape)
                return rmse
            except:
                return float('inf')

        study = optuna.create_study(direction='minimize'
                                    #pruner = optuna.pruners.MedianPruner()
                                    )
        study.optimize(objective, n_trials=n_trials)
        best_trial = study.best_trial
        best_order = tuple(best_trial.params[k] for k in ['p', 'd', 'q'])
        best_seasonal_order = tuple(best_trial.params[k] for k in ['P', 'D', 'Q']) + (self.seasonal_period,)
        return best_order, best_seasonal_order, best_trial.user_attrs['rmse'], best_trial.user_attrs['mape']

    def sarima_forecast_by_level(self, level_col, Pattern_col):
        start_time = time.time()
        all_predictions = pd.DataFrame()
        model_performance = []
    
        for l in self.df[level_col].unique():
            print(f'\n===== Results for {l} =====\n')
            df_level = self.df[self.df[level_col] == l].copy()
            df_level_l = df_level.copy()
            average = df_level[self.pred_col].replace(0, np.nan).mean()
            df_level['month'] = df_level.index.month
            df_level[self.pred_col] = df_level.groupby('month')[self.pred_col].transform(
                lambda x: x.replace(0, np.nan).fillna(x.mean())
                )
            
            y = df_level[self.pred_col].values
            pattern = df_level[Pattern_col].iloc[0]
    
            if pattern in ['X', 'Y']:
                best_order, best_seasonal_order, rmse, mape = self.optimize_sarima_order(y)
                forecast_values, model_fit = self.forecast_sarima(y, best_order, best_seasonal_order)
            else:
                
                monthly_means = df_level.groupby('month')[self.pred_col].mean()
                last_month = df_level['month'].iloc[-1]

                forecast_months = [(last_month + i - 1) % 12 + 1 for i in range(1, self.forecast_steps + 1)]
                
                forecast_values = [monthly_means.get(month, monthly_means.mean())
                                   for month in forecast_months
                                   ]

                
                #forecast_values = [average] * self.forecast_steps
                best_order = 'mean'
                best_seasonal_order = 'mean'
                rmse = 0
                mape = 0
    
            last_date = df_level.index[-1]
            forecast_index = pd.date_range(last_date, periods=self.forecast_steps + 1, freq='M')[1:]
    
            original_df = df_level_l[[self.pred_col]].copy()
            original_df.reset_index(inplace=True)
            original_df.rename(columns={'index': 'date'}, inplace=True)
            original_df.rename(columns={'Date': 'date'}, inplace = True)
            original_df[level_col] = l
            original_df['Type'] = 'original'
            original_df['SARIMA_order'] = 'NA'
            original_df['Pattern'] = pattern
    
            forecast_df = pd.DataFrame({
                'date': forecast_index,
                level_col: l,
                self.pred_col: forecast_values,
                'Type': 'forecast',
                'SARIMA_order': [f"{best_order}, {best_seasonal_order}"] * self.forecast_steps,
                'Pattern': pattern
            })
    
            forecast_df = pd.concat([forecast_df, original_df])
            all_predictions = pd.concat([all_predictions, forecast_df], ignore_index=True)
    
            model_performance.append({
                level_col: l,
                'SARIMA_order': f"{best_order}, {best_seasonal_order}",
                'RMSE': round(rmse, 2),
                'Model ACC': 1 - round(mape, 2),
                'Pattern': pattern
            })
    
        exec_time = round(time.time() - start_time, 2)
        print(f"Final Execution Time: {exec_time} seconds")
        model_performance_df = pd.DataFrame(model_performance)
        return all_predictions, model_performance_df
    
    
    def sarima_forecast_single_level(self):
        start_time = time.time()
        all_predictions = pd.DataFrame()
        model_performance = []
    
        
        print('\n===== sarima_forecast_single_level =====\n')
        df_level = self.df.copy()
        df_level_l = df_level.copy()
        average = df_level[self.pred_col].replace(0, np.nan).mean()
        df_level['month'] = df_level.index.month
        df_level[self.pred_col] = df_level.groupby('month')[self.pred_col].transform(
            lambda x: x.replace(0, np.nan).fillna(x.mean())
            )

        y = df_level[self.pred_col].values
        

 
        best_order, best_seasonal_order, rmse, mape = self.optimize_sarima_order(y)
        forecast_values, model_fit = self.forecast_sarima(y, best_order, best_seasonal_order)


        last_date = df_level.index[-1]
        forecast_index = pd.date_range(last_date, periods=self.forecast_steps + 1, freq='M')[1:]

        original_df = df_level_l[[self.pred_col]].copy()
        original_df.reset_index(inplace=True)
        original_df.rename(columns={'index': 'date'}, inplace=True)
        original_df.rename(columns={'visual_date': 'date'}, inplace = True)
        original_df['Type'] = 'original'
        original_df['SARIMA_order'] = 'NA'
       

        forecast_df = pd.DataFrame({
            'date': forecast_index,
            self.pred_col: forecast_values,
            'Type': 'forecast',
            'SARIMA_order': [f"{best_order}, {best_seasonal_order}"] * self.forecast_steps,
            
        })

        forecast_df = pd.concat([forecast_df, original_df])
        all_predictions = pd.concat([all_predictions, forecast_df], ignore_index=True)

        model_performance.append({
            'SARIMA_order': f"{best_order}, {best_seasonal_order}",
            'RMSE': round(rmse, 2),
            'Model ACC': 1 - round(mape, 2),
            
        })
    
        exec_time = round(time.time() - start_time, 2)
        print(f"Final Execution Time: {exec_time} seconds")
        model_performance_df = pd.DataFrame(model_performance)
        return all_predictions, model_performance_df
