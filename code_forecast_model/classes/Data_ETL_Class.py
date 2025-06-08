# -*- coding: utf-8 -*-
"""
Created on Tue Apr 15 10:09:04 2025

@author: l019868
"""

# Importing necessary libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
import seaborn as sns
import warnings
from datetime import date
from dateutil.relativedelta import relativedelta
warnings.filterwarnings('ignore')
plt.rcParams['figure.figsize'] = [10,5]

class ETL_Model:
    def __init__(self,df,pred_col,date_col,group_columns, path):
        self.df = df
        self.pred_col = pred_col
        self.date_col = date_col
        self.group_columns = group_columns
        self.path = path
        
        
    def format_df (self):
        df_new = self.df.set_index(self.date_col)
        df_new = df_new.dropna(subset = [self.pred_col])
        return df_new
    
    def agg_data(self):
        df_agg = self.df.groupby(self.group_columns + [self.df.index])[self.pred_col].sum().reset_index()
        df_agg = df_agg.set_index(self.date_col)
        return df_agg
    
        
    def format_data(self):
        format_df = self.df.copy()
        format_df['level_key'] = format_df[self.group_columns].apply(
            lambda row: '_'.join(row.values.astype(str)),axis = 1)
        categorical_col = 'level_key'
        format_df[categorical_col] = format_df[categorical_col].astype(str)
        format_df[self.pred_col]=format_df[self.pred_col].astype(float)
        format_df[self.date_col] = pd.to_datetime(format_df[self.date_col])
       
        format_df = format_df[[self.date_col,self.pred_col,categorical_col]].dropna()
  
        df_month = format_df.groupby([
            categorical_col,
            pd.Grouper(key=self.date_col, freq='M')
        ])[self.pred_col].sum().reset_index()

        df_month = df_month.set_index(self.date_col)
        df_pivot_dates = pd.pivot_table(df_month,values = self.pred_col, index=categorical_col,
                               columns=df_month.index,aggfunc=sum).fillna(0)
        df_pivot_dates['Sum']=df_pivot_dates[list(df_pivot_dates.columns)].sum(axis=1)
        df_pivot_dates = df_pivot_dates[df_pivot_dates['Sum'] != 0]
        df_pivot_dates = df_pivot_dates.drop(columns = ['Sum'])
        df_unpivoted_dates = df_pivot_dates.reset_index().melt(id_vars = 'level_key',
                                                               var_name=self.date_col,
                                                               value_name=self.pred_col)
        df_unpivoted_dates
        df_unpivoted_dates.set_index('date',inplace=True)
        return df_unpivoted_dates
    
    
    def pareto_graph(self,df,level,num_col):
        df_pareto = df.groupby(level)[num_col].sum().reset_index()
        df_pareto = df_pareto.sort_values(by = num_col, ascending = False)
        df_pareto['Total_Percentage'] = (df_pareto[num_col]/df_pareto[num_col].sum())*100
        df_pareto['Cumulative_Sum'] = df_pareto[num_col].cumsum()
        df_pareto['Cumulative_Percentage'] = df_pareto['Total_Percentage'].cumsum()
        fig, ax1 = plt.subplots()
        sns.barplot(x=df_pareto['Pattern(I)'], y='Sum',data = df_pareto, ax=ax1, color = 'b')
        ax2 = ax1.twinx()
        sns.lineplot(x=df_pareto['Pattern(I)'], y='Cumulative_Percentage', data = df_pareto, 
                     ax= ax2, color = 'r', marker = 'o')
        ax1.set_xlabel('Pattern')
        ax1.set_ylabel('Invoiced Volume(MT)')
        ax2.set_ylabel('Cumulative %')
        plt.title('Pareto chart count per Pattern')
        plt.show()
        plt.savefig(self.path+'plot_pareto.jpg', format='jpg', dpi=300, bbox_inches='tight')
        return
    
    def demand_sensing(self,df_sales,title = ' '):
        categorical_col = 'level_key'
        pivot_df = pd.pivot_table(df_sales,values=self.pred_col,index=categorical_col,
                                  columns=df_sales.index,aggfunc = sum).fillna(0)
        pivot_sense = pivot_df.copy()
        pivot_sense['ADI'] = len(pivot_df.columns)/pivot_df[list(pivot_df.columns)].gt(0).sum(axis=1)
        pivot_sense['Avg']=pivot_df[list(pivot_df.columns)].mean(axis=1)
        pivot_sense['StdDev']=pivot_df[list(pivot_df.columns)].std(axis=1)
        pivot_sense['CV'] = pivot_sense['StdDev']/pivot_sense['Avg']
        pivot_sense['CV2'] = pivot_sense['CV']**2
        pivot_sense['ADI/CV2'] = pivot_sense['ADI'] / pivot_sense['CV2']
        pivot_sense['Pattern(I)'] = 'Z' 
        #'Lumpy'
        pivot_sense.loc[(pivot_sense['CV2'] <= 0.6) & (pivot_sense['ADI'] <= 1.32),
                        'Pattern(I)'] = 'X'
        #'Smooth'
        pivot_sense.loc[(pivot_sense['CV2'] <= 0.6) & (pivot_sense['ADI'] > 1.32),
                        'Pattern(I)'] = 'Y'
        #'Intermittent'
        pivot_sense.loc[(pivot_sense['CV2'] > 0.6) & (pivot_sense['ADI'] <= 1.32),
                        'Pattern(I)'] = 'Y'
        #'Erratic'
        pivot_sense['Sum']=pivot_df[list(pivot_df.columns)].sum(axis=1)
        pivot_sense = pivot_sense.sort_values(by = 'Sum', ascending = False)
        pivot_sense['Total_Percentage'] = (pivot_sense['Sum']/pivot_sense['Sum'].sum())*100
        pivot_sense['Cumulative_Sum'] = pivot_sense['Sum'].cumsum()
        pivot_sense['Cumulative_Percentage'] = pivot_sense['Total_Percentage'].cumsum()
        pivot_sense['Pattern(ABC)'] = 'C' 
        pivot_sense.loc[(pivot_sense['Cumulative_Percentage'] <= 80),'Pattern(ABC)'] = 'A'
        pivot_sense.loc[(pivot_sense['Cumulative_Percentage'] > 80) & 
                     (pivot_sense['Cumulative_Percentage'] <= 95),
                     'Pattern(ABC)'] = 'B'
        pivot_sense['Segmentation(ABCXYZ)'] = pivot_sense['Pattern(I)']+'-'+pivot_sense['Pattern(ABC)']
        pivot_sense['Pattern(I)'].value_counts().plot(kind='bar')
        plt.grid()
        plt.title(f'Bar plot for {title} {self.group_columns}')
        plt.show()
        ETL_Model.pareto_graph(self,pivot_sense,'Pattern(I)','Sum')
        sns.scatterplot(data = pivot_sense, x='Sum', y='ADI/CV2', hue ='Segmentation(ABCXYZ)')
        plt.title(f'Scatter plot for {self.group_columns}')
        plt.show()
        plt.savefig(self.path+'plot_Scatter.jpg', format='jpg', dpi=300, bbox_inches='tight')
        return pivot_sense
    
    def pattern_selection(self,df_formated,df_pattern,Pattern_col,pattern):
        df = df_pattern[df_pattern[Pattern_col].isin(pattern)]
        df_filter = df[Pattern_col].reset_index()
        key_list = list(df_filter['level_key'])
        final_df = df_formated[df_formated['level_key'].isin(key_list)]
        original_index = final_df.index.copy()

        final_df = final_df.merge(df_filter[['level_key', Pattern_col]], 
                                      on = 'level_key',
                                      how = 'left'
                                      ) 
        final_df.index = original_index
        return final_df
    
    def outlier_removal(df,pred_col,threshold_z = 0.2):
        df_no_out = df
        df_no_out['zscore'] = scipy.stats.zscore(df_no_out[pred_col])
        df_filtered = df_no_out[df_no_out['zscore'].abs() <= threshold_z]
        df_filtered = df_filtered.drop(columns=['zscore'])
        return df_filtered
    
    def outlier_removal_iqr(df,pred_col,factor = 1.5):
        Q1 = df[pred_col].quantile(0.25)
        Q3 = df[pred_col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - factor * IQR
        upper = Q3 + factor * IQR
        df_filtered = df[(df[pred_col] >= lower) & (df[pred_col] <= upper)]
        return df_filtered
    
    def level_plot(self,dfx,level, title = ''):
        if dfx[level].nunique() > 10:
            sns.lineplot(x = dfx.index,y=dfx[self.pred_col] 
                     )
            plt.title(f'Line plot for {title} {self.group_columns}')
            plt.show()
        else:
            sns.lineplot(x = dfx.index,y=dfx[self.pred_col] 
             , hue = dfx[level])
            plt.title(f'Line plot  for {title} {self.group_columns}')
            plt.show()
            sns.boxplot(x = dfx[self.pred_col],y=dfx[level])
            plt.title(f'Box plot for {title} {self.group_columns}')
            plt.show()
        return
    
    def disaggregate (self,df,original_df,forecast_df,level_col,disag_col):
        start_date = date.today() - relativedelta(years = 1)
        df = df[df.index >= start_date]
        original_df = original_df[original_df[self.date_col] >= start_date]
        total_values = df.groupby(level_col)[self.pred_col].sum().reset_index()
        groupby_col = self.group_columns + disag_col
        select_col = groupby_col + [self.pred_col]
        drop_col = self.group_columns + ['total']
        df_weight = original_df[select_col]
        df_weight = df_weight.groupby(groupby_col)[self.pred_col].sum().reset_index()
        df_weight = df_weight.rename(
            columns={self.pred_col: 'total'})
        df_weight['key'] = df_weight[self.group_columns].agg('_'.join, axis = 1)
        df_weight = pd.merge(df_weight,total_values,left_on='key', 
                         right_on=level_col,how = 'inner')
        df_weight['weight'] = np.where(
            df_weight[self.pred_col] != 0,
            df_weight['total']/df_weight[self.pred_col],
            0
        )
        df_weight = df_weight.rename(
            columns={self.pred_col: 'value_original'})
        df_weight = df_weight.drop(drop_col,axis = 1)
        df_product_dis = pd.merge(forecast_df,df_weight,left_on=level_col, 
                         right_on=level_col,how = 'inner')
        df_product_dis = df_product_dis.drop(['value_original','ARIMA_order'], axis = 1)
        df_product_dis['vol_dist'] = df_product_dis[self.pred_col]*df_product_dis['weight']
        return df_product_dis 