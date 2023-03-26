import pandas as pd
df=pd.read_excel('supermarket_sales.xlsx')
print(df.columns)
df_new=df[['Gender','Product line','Total']]

pivot_table=df_new.pivot_table(index='Gender',columns='Product line',values='Total'
                   ,aggfunc='sum')
print(pivot_table)
pivot_table.to_excel('pivot_table.xlsx','Report',startrow=4)