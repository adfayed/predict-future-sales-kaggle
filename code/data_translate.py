# Basic packages
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import random as rd # generating random numbers
from datetime import datetime # manipulating date formats
import pdb # python debugger
# Viz
import matplotlib.pyplot as plt # basic plotting
#import seaborn as sns # for prettier plots
# Google Translate package
from mtranslate import translate

# TIME SERIES
# from statsmodels.tsa.arima_model import ARIMA
# from statsmodels.tsa.statespace.sarimax import SARIMAX
# from pandas.plotting import autocorrelation_plot
# from statsmodels.tsa.stattools import adfuller, acf, pacf,arma_order_select_ic
# import statsmodels.formula.api as smf
# import statsmodels.tsa.api as smt
# import statsmodels.api as sm
# import scipy.stats as scs

# ----------FUNCTIONS----------
def downcast_dtypes(df):
    float_cols = [c for c in df if df[c].dtype == "float64"]
    int_cols = [c for c in df if df[c].dtype in ["int64", "int32"]]
    df[float_cols] = df[float_cols].astype(np.float32)
    df[int_cols] = df[int_cols].astype(np.int16)
    return df

# IMPORT ALL INPUT DATA
#pdb.set_trace()
sales=pd.read_csv("../data/sales_train.csv")
item_cat=pd.read_csv("../data/item_cat_en.csv")
item=pd.read_csv("../data/item_en.csv")
shops=pd.read_csv("../data/shops_en.csv")
sample_output=pd.read_csv("../data/sample_submission.csv")
test=pd.read_csv("../data/test.csv")

# Correct date format
#sales.date = sales.date.apply(lambda x:datetime.strptime(x,'%d.%m.%Y'))

# Convert Shop names to English
#shops.shop_name = shops.shop_name.apply(lambda x:translate(x, "en", "auto"))

# Convert Item Cat names to English
#item_cat.item_category_name = item_cat.item_category_name.apply(lambda x:translate(x, "en", "auto"))

# Convert Item names to English
#item.item_name = item.item_name.apply(lambda x:translate(x, "en", "auto"))

# Save English versions to a csv
# shops.to_csv("../data/shops_en.csv", index=False)
# item_cat.to_csv("../data/item_cat_en.csv", index=False)
# item.to_csv("../data/item_en.csv", index=False)


# Let's merge all the data into one DataFrame
# sales_temp = pd.merge(sales, shops, how='left', on=['shop_id']) 
# sales_temp = pd.merge(sales_temp, item, how='left', on=['item_id'])
# sales_new = pd.merge(sales_temp, item_cat, how='left', on=['item_category_id'])

# Finally save it to a csv file
# sales_new.to_csv("../data/sales_train_en_complete.csv", index=False)


sales = downcast_dtypes(sales)
#print(sales.info())


# IDEA 1
# Convert russian names to english

# IDEA 2
# Downcast data

# IDEA 3
# Colate all data in one big dataframe

# IDEA 4
# Find interesting ways to aggregate and plot the data

# IDEA 5
# Delete outliers (price, sales volume)

# IDEA 6 https://www.kaggle.com/kyakovlev/1st-place-solution-part-1-hands-on-data/
# Remove outdated items