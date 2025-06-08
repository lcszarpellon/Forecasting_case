import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import seaborn as sns
from sklearn.feature_selection import chi2 
sns.set_theme(rc={'figure.figsize':(12,8)})
#%%
path = r"..\data\code_data_analysis\exports\\"
#%%
df = pd.read_excel(r"..\data\toy_dataset.xlsx")
df.columns = df.columns.str.strip().str.lower()
df.set_index("date", inplace = True)
#%%
df.head()
df.columns
print(df.dtypes)
null_cols = (df.isnull().sum())
null_percent = (null_cols/len(df))*100
print(f'\n{null_cols}')
print(f'\n{null_percent}')

#%%

cat = df.select_dtypes(exclude = [int,float,np.datetime64])
num = df.select_dtypes(exclude = ['object', np.datetime64])
num = num.drop(['id'], axis = 1)

print(cat.columns)
print(num.columns)
#%%

df_plot= df.groupby(df.index)['forecast'].sum().to_frame()

sns.lineplot(x=df_plot.index,y=df_plot['forecast'])
plt.savefig(path+'plot_original_trend.jpg', format='jpg', dpi=300, bbox_inches='tight')

#%%
df_plot_bar = df.groupby(['state', 'dealer'])['forecast'].sum().reset_index()
#%%
sns.barplot(data=df_plot_bar, x='state', y='forecast')
plt.title("Sum of Forecasts per State")
plt.xlabel("State")
plt.ylabel("Forecast Sum")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
plt.savefig(path+'plot_bar_state.jpg', format='jpg', dpi=300, bbox_inches='tight')
#%%
sns.barplot(data=df_plot_bar, x='dealer', y='forecast')
plt.title("Sum of Forecasts per Dealer")
plt.xlabel("Dealer")
plt.ylabel("Forecast Sum")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
plt.savefig(path+'plot_bar_dealer.jpg', format='jpg', dpi=300, bbox_inches='tight')
#%%
sns.barplot(data=df_plot_bar, x='state', y='forecast', hue='dealer')
plt.title("Sum of Forecasts per State by Dealer")
plt.xlabel("State")
plt.ylabel("Forecast Sum")
plt.xticks(rotation=45)
plt.legend(title="Dealer")
plt.tight_layout()
plt.show()
plt.savefig(path+'plot_bar_state_dealer.jpg', format='jpg', dpi=300, bbox_inches='tight')

#%%

for k in num:
    if k != 'forecast': 
        sns.lmplot(x=k,y='forecast', data=num, order = 1)
        plt.title(f'linear relation {k}')
        plt.ylabel('forecast')
        plt.xlabel(k)
        plt.show()
        plt.savefig(path+f'plot_numerical_relation_{k}.jpg', format='jpg', dpi=300, bbox_inches='tight')

#%%
for k in num:
    if k != 'forecast':
        sns.displot(num[k],kde= True)
        plt.title(f'distribution_plot {k}')
        plt.show()
        plt.savefig(path+f'plot_distribution_{k}.jpg', format='jpg', dpi=300, bbox_inches='tight')
        
#%%
for k in num:
    if k != 'forecast':
        sns.boxplot(x=df[k],color = 'y')
        plt.title(f'box-plot {k}')
        plt.show()
        plt.savefig(path+f'plot_box_{k}.jpg', format='jpg', dpi=300, bbox_inches='tight')
#%%
#Replacing null with mean
num_adj = num.copy()
for i in num_adj:
    if i != "forecast":
        num_adj[i] = num_adj[i].fillna(num_adj[i].mean())
#%%
df_t = df.copy()
for k in cat:
    ordinal = {j:i for i,j in enumerate(df_t[k].unique(),0)}
    df_t[k] = df_t[k].map(ordinal)
#%%
cat_cols = cat.columns 
print(cat_cols)
cat_df = df_t.drop(num.columns,axis=1)

y = df_t['forecast'].astype(int)
#%%
chi = chi2(cat_df,y)
p_values = pd.Series(chi[1])
p_values.index = cat_df.columns
cat_corr = p_values


print(f'\n Correlation: \n {cat_corr}')
#%%
corr_num=num_adj.corr().round(2)
sns.heatmap(data=corr_num, annot = True,cmap="crest")
plt.savefig(path+'plot_heat_map.jpg', format='jpg', dpi=300, bbox_inches='tight')

#%%
X = num_adj.drop('forecast', axis = 1)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)

sns.barplot(x=importances.values, y=importances.index)
plt.title('Variable relevance (Random Forest)')
plt.xlabel('Relevance')
plt.ylabel('Variable')
plt.tight_layout()
plt.show()
plt.savefig(path+'plot_var_relevance.jpg', format='jpg', dpi=300, bbox_inches='tight')
#%%
df_pivot = df.reset_index()
df_pivot = pd.pivot_table(
    df_pivot,
    values='forecast',
    index='id',
    columns=df_pivot['date'].dt.to_period('M'),
    aggfunc='sum',
    fill_value=0
)

#%%

scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_pivot)

kmeans = KMeans(n_clusters=3, random_state=42)
df_pivot['Cluster'] = kmeans.fit_predict(X_scaled)
#%%
print(df_pivot['Cluster'])

df_pivot.groupby('Cluster').mean().T.plot(figsize=(10, 6))
plt.title('Average Cluster per ID')
plt.xlabel('Month')
plt.ylabel('Average Forecast')
plt.legend(title='Cluster')
plt.show()
plt.savefig(path+'plot_id_cluster.jpg', format='jpg', dpi=300, bbox_inches='tight')




