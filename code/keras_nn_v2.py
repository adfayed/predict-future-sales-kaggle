from hyperopt import hp, fmin, tpe, STATUS_OK, STATUS_FAIL, Trials
from itertools import product
import pdb

from sklearn import preprocessing
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

import numpy as np
import pandas as pd
import math
from xgboost import XGBRegressor

pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 100)

import gc
import time
import pickle
import re

# Neural net requirements below
from keras import regularizers, backend
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, Dropout
from keras.callbacks import EarlyStopping

# Plotting
import matplotlib.pyplot as plt

def downcast_dtypes(df):
    float_cols = [c for c in df if df[c].dtype == "float64"]
    int_cols = [c for c in df if df[c].dtype in ["int64", "int32"]]
    df[float_cols] = df[float_cols].astype(np.float32)
    df[int_cols] = df[int_cols].astype(np.int16)
    return df


def clean_data():
    # Load the data
    sales_data = pd.read_csv('../data/sales_train.csv')
    test_data = pd.read_csv('../data/test.csv')
    items = pd.read_csv('../data/items.csv')
    item_categories = pd.read_csv('../data/item_categories.csv')
    shops = pd.read_csv('../data/shops.csv')

    # Update the date format
    sales_data['date'] = pd.DataFrame(pd.to_datetime(sales_data['date'], format='%d.%m.%Y'))

    # Rename columns
    sales_data.rename({'date_block_num': 'month_id', 'item_cnt_day': 'item_quantity'}, axis=1, inplace=True)

    # Handle duplicate rows
    # Doing nothing for now

    # Handle outliers in the items (price, item_cnt)
    # Deleting for now  (Not sure what to make the cutoffs)
    sales_data.drop(sales_data[sales_data['item_quantity'] > 1000].index, axis=0, inplace=True)
    sales_data.drop(sales_data[sales_data['item_price'] > 100000].index, axis=0, inplace=True)

    # Handle returned items (negative item_cnt)
    # # Delete them
    # sales_data.drop(sales_data[sales_data['item_quantity'] < 0].index, axis=0, inplace=True)
    # Set item_quantity to 0
    sales_data.loc[sales_data['item_quantity'] < 0, 'item_quantity'] = 0

    # Handle items with negative price
    # Deleting for now
    sales_data.drop(sales_data[sales_data['item_price'] < 0].index, axis=0, inplace=True)

    # Handle duplicate shops and shop names
    # Shops 0 and 57, 1 and 58, 10 and 11 are the same shops but different time period. Will use 57, 58, and 10 as the
    #   labels because they are in the test set, but shops 0, 1, and 11 are not
    # Shop 40 seems to be an "antenna" of shop 39 so combine their sales together labeled as shop 39
    # shops.drop(0, axis=0, inplace=True)
    # shops.drop(1, axis=0, inplace=True)
    # shops.drop(11, axis=0, inplace=True)
    # shops.drop(40, axis=0, inplace=True)

    sales_data.loc[sales_data['shop_id'] == 0, 'shop_id'] = 57
    sales_data.loc[sales_data['shop_id'] == 1, 'shop_id'] = 58
    sales_data.loc[sales_data['shop_id'] == 11, 'shop_id'] = 10
    sales_data.loc[sales_data['shop_id'] == 40, 'shop_id'] = 39

    # Edit the shop name
    # shops['shop_name'] = shops['shop_name'].apply(lambda x: x.lower()).str.\
    #     replace('[^\w\s]','').str.replace('\d+', '').str.strip()

    # # ?????????????????????
    # # Remove outlier shops
    # # ?????????????????????
    # # There are two shops only open in October months (9 and 20) so remove them
    # shops.drop(9, axis=0, inplace=True)
    # shops.drop(20, axis=0, inplace=True)
    # sales_data.drop(sales_data.loc[sales_data['shop_id'] == 9].index, axis=0, inplace=True)
    # sales_data.drop(sales_data.loc[sales_data['shop_id'] == 20].index, axis=0, inplace=True)
    #
    # # Remove shop 33 as it is only open for a short time in the middle of the training period
    # shops.drop(33, axis=0, inplace=True)
    # sales_data.drop(sales_data.loc[sales_data['shop_id'] == 33].index, axis=0, inplace=True)
    #
    # # ???????????????????????????????????????????????????????????????????????????????????????????
    # # Remove the two entries for shop 34 on month 18. Only two items were sold, each only 1 time
    # # ???????????????????????????????????????????????????????????????????????????????????????????
    # sales_data.drop(sales_data.loc[(sales_data['shop_id'] == 34) &
    #                                (sales_data['month_id'] == 18)].index, axis=0, inplace=True)

    # Rename item_categories
    # item_categories.loc[0, 'item_category_name'] = 'Аксессуары - PC (Гарнитуры/Наушники)'
    # item_categories.loc[8, 'item_category_name'] = 'Билеты - Билеты (Цифра)'
    # item_categories.loc[9, 'item_category_name'] = 'Доставка товара - Доставка товара'
    # item_categories.loc[26, 'item_category_name'] = 'Игры - Android (Цифра)'
    # item_categories.loc[27, 'item_category_name'] = 'Игры - MAC (Цифра)'
    # item_categories.loc[28, 'item_category_name'] = 'Игры - PC (Дополнительные издания)'
    # item_categories.loc[29, 'item_category_name'] = 'Игры - PC (Коллекционные издания)'
    # item_categories.loc[30, 'item_category_name'] = 'Игры - PC (Стандартные издания)'
    # item_categories.loc[31, 'item_category_name'] = 'Игры - PC (Цифра)'
    # item_categories.loc[32, 'item_category_name'] = 'Карты оплаты - Кино, Музыка, Игры'
    # item_categories.loc[
    #     79, 'item_category_name'] = 'Прием денежных средств для 1С-Онлайн - Прием денежных средств для 1С-Онлайн'
    # item_categories.loc[80, 'item_category_name'] = 'Билеты - Билеты'
    # item_categories.loc[81, 'item_category_name'] = 'Misc - Чистые носители (шпиль)'
    # item_categories.loc[82, 'item_category_name'] = 'Misc - Чистые носители (штучные)'
    # item_categories.loc[83, 'item_category_name'] = 'Misc - Элементы питания'

    # # ???????????????????????????
    # # Drop irrelevant categories
    # # ???????????????????????????
    # sales_data.drop(sales_data.loc[(sales_data['item_id'].map(items['item_category_id']) == 8)].index.values, axis=0,
    #                 inplace=True)
    # sales_data.drop(sales_data.loc[(sales_data['item_id'].map(items['item_category_id']) == 80)].index.values, axis=0,
    #                 inplace=True)
    # sales_data.drop(sales_data.loc[(sales_data['item_id'].map(items['item_category_id']) == 81)].index.values, axis=0,
    #                 inplace=True)
    # sales_data.drop(sales_data.loc[(sales_data['item_id'].map(items['item_category_id']) == 82)].index.values, axis=0,
    #                 inplace=True)
    # item_categories.drop(8, axis=0, inplace=True)
    # item_categories.drop(80, axis=0, inplace=True)
    # item_categories.drop(81, axis=0, inplace=True)
    # item_categories.drop(82, axis=0, inplace=True)
    # items.drop(items.loc[items['item_category_id'] == 8].index.values, axis=0, inplace=True)
    # items.drop(items.loc[items['item_category_id'] == 80].index.values, axis=0, inplace=True)
    # items.drop(items.loc[items['item_category_id'] == 81].index.values, axis=0, inplace=True)
    # items.drop(items.loc[items['item_category_id'] == 82].index.values, axis=0, inplace=True)

    # Add the month_id to the test set
    test_data['month_id'] = 34

    # Downcast dtypes
    downcast_dtypes(sales_data)
    sales_data['month_id'] = sales_data['month_id'].astype(np.int8)
    sales_data['shop_id'] = sales_data['shop_id'].astype(np.int8)

    downcast_dtypes(shops)
    shops['shop_id'] = shops['shop_id'].astype(np.int8)

    downcast_dtypes(items)
    items['item_category_id'] = items['item_category_id'].astype(np.int8)

    downcast_dtypes(item_categories)
    item_categories['item_category_id'] = item_categories['item_category_id'].astype(np.int8)

    test_data['month_id'] = test_data['month_id'].astype(np.int8)
    test_data['shop_id'] = test_data['shop_id'].astype(np.int8)
    test_data['item_id'] = test_data['item_id'].astype(np.int16)

    # Export data
    # output_dirname = ''
    # output_foldername = ''
#     output_dirname = '/kaggle/output'
#     output_foldername = '/kaggle/working/'
    sales_data.to_pickle('../data/cleaned/sales_data.pkl')
    shops.to_pickle('../data/cleaned/shops.pkl')
    items.to_pickle('../data/cleaned/items.pkl')
    item_categories.to_pickle('../data/cleaned/item_categories.pkl')
    test_data.to_pickle('../data/cleaned/test.pkl')

    del sales_data, shops, item_categories, items, test_data


def name_correction(x):
    x = x.lower()
    x = x.partition('[')[0]
    x = x.partition('(')[0]
    x = re.sub('[^A-Za-z0-9А-Яа-я]+', ' ', x)
    x = x.replace('  ', ' ')
    x = x.strip()
    return x


def get_basic_features():
    # Load d
    sales_data = pd.read_pickle('../data/cleaned/sales_data.pkl')
    shops = pd.read_pickle('../data/cleaned/shops.pkl')
    items = pd.read_pickle('../data/cleaned/items.pkl')
    item_categories = pd.read_pickle('../data/cleaned/item_categories.pkl')
    test_data = pd.read_pickle('../data/cleaned/test.pkl')

    ############
    # Shop City
    # Shop Type
    ############
    # Shop city
    shops.loc[shops['shop_name'] == 'Сергиев Посад ТЦ "7Я"', "shop_name"] = 'СергиевПосад ТЦ "7Я"'
    shops['shop_city'] = shops['shop_name'].str.split(' ').map(lambda x: x[0])
    shops.loc[shops['shop_city'] == "!Якутск", "shop_city"] = "Якутск"
    shops['city_code'] = LabelEncoder().fit_transform(shops['shop_city'])
    shops['city_code'] = shops['city_code'].astype(np.int8)
    # Shop type
    shops['shop_type'] = shops['shop_name'].str.split(' ').map(lambda x: x[1])
    types = []
    for cat in shops['shop_type'].unique():
        if len(shops[shops['shop_type'] == cat]) > 4:
            types.append(cat)
    shops['shop_type'] = shops['shop_type'].apply(lambda x: x if (x in types) else 'etc')
    shops['shop_type_code'] = LabelEncoder().fit_transform(shops['shop_type'])
    shops['shop_type_code'] = shops['shop_type_code'].astype(np.int8)

    # print('-------------------------------------')
    # for cat in shops['shop_type'].unique():
    #     print(cat, len(shops[shops['shop_type'] == cat]))

    # Only use the shop_id, city_code and shop_type_code
    shops = shops[['shop_id', 'shop_type_code', 'city_code']]

    ###################
    # Category type
    # Category subtype
    ###################
    # Category type
    item_categories['category_type'] = item_categories['item_category_name'].apply(lambda x: x.split(' ')[0]).astype(str)
    # item_categories.loc[(item_categories['category_type'] == 'Игровые') |
    #                     (item_categories['category_type'] == 'Аксессуары'), 'category_type'] = 'Игры'
    # print('\n-------------------------------------')
    types = []
    for cat in item_categories['category_type'].unique():
        # print(cat, len(item_categories[item_categories['category_type'] == cat]))
        if len(item_categories[item_categories['category_type'] == cat]) > 4:
            types.append(cat)
    item_categories['category_type'] = item_categories['category_type'].apply(lambda x: x if (x in types) else 'etc')

    item_categories['category_type_code'] = LabelEncoder().fit_transform(item_categories['category_type'])
    item_categories['category_type_code'] = item_categories['category_type_code'].astype(np.int8)

    # print('\n-------------------------------------')
    # for cat in item_categories['category_type'].unique():
    #     print(cat, len(item_categories[item_categories['category_type'] == cat]))

    # Category subtype
    item_categories['split'] = item_categories['item_category_name'].apply(lambda x: x.split('-'))
    item_categories['category_subtype'] = item_categories['split'].apply(lambda x: x[1].strip() if len(x) > 1 else x[0].strip())
    item_categories['category_subtype_code'] = LabelEncoder().fit_transform(item_categories['category_subtype'])
    item_categories['category_subtype_code'] = item_categories['category_subtype_code'].astype(np.int8)
    item_categories = item_categories[['item_category_id', 'category_subtype_code', 'category_type_code']]

    # print('\n-------------------------------------')
    # print(item_categories.head())

    #############
    # Item name2
    # Item name3
    #############
    items['name1'], items['name2'] = items['item_name'].str.split('[', 1).str
    items['name1'], items['name3'] = items['item_name'].str.split('(', 1).str
    items['name2'] = items['name2'].str.replace('[^A-Za-z0-9А-Яа-я]+', ' ').str.lower()
    items['name3'] = items['name3'].str.replace('[^A-Za-z0-9А-Яа-я]+', ' ').str.lower()
    items = items.fillna('0')

    # Name correction
    items['item_name'] = items['item_name'].apply(lambda x: name_correction(x))
    items['name2'] = items['name2'].apply(lambda x: x[:-1] if x != '0' else '0')

    # Get the item type
    items["type"] = items['name2'].apply(lambda x: x[0:8] if x.split(" ")[0] == "xbox" else x.split(" ")[0])
    items.loc[(items['type'] == "x360") | (items['type'] == "xbox360") | (items['type'] == "xbox 360"), "type"] = "xbox 360"
    items.loc[items['type'] == "", "type"] = "mac"
    items.type = items['type'].apply(lambda x: x.replace(" ", ""))
    items.loc[(items['type'] == 'pc') | (items['type'] == 'pс') | (items['type'] == "pc"), "type"] = "pc"
    items.loc[items['type'] == 'рs3', "type"] = "ps3"

    group_sum = items.groupby(['type']).agg({'item_id': 'count'}).reset_index()
    drop_cols = []
    for cat in group_sum['type'].unique():
        if group_sum.loc[(group_sum['type'] == cat), 'item_id'].values[0] < 40:
            drop_cols.append(cat)

    # print('\n-------------------------------------')
    # print(items.head())

    items['name2'] = items['name2'].apply(lambda x: 'etc' if (x in drop_cols) else x)
    items.drop(['type'], axis=1, inplace=True)
    items['name2'] = LabelEncoder().fit_transform(items['name2'])
    items['name3'] = LabelEncoder().fit_transform(items['name3'])
    items['name2'] = items['name2'].astype(np.int8)
    items['name3'] = items['name3'].astype(np.int16)
    items.drop(['item_name', 'name1'], axis=1, inplace=True)

    # print('\n-------------------------------------')
    # print(items.head())

    #######################################################################
    # Create a DataFrame with every possible shop/item pair for each month
    #######################################################################
    # Create DataFrame
    training = []
    cols = ['month_id', 'shop_id', 'item_id']
    for i in range(34):
        sales = sales_data[sales_data['month_id'] == i]
        training.append(
            np.array(list(product([i], sales['shop_id'].unique(), sales['item_id'].unique())), dtype='int16'))

    training = pd.DataFrame(np.vstack(training), columns=cols)
    training['month_id'] = training['month_id'].astype(np.int8)
    training['shop_id'] = training['shop_id'].astype(np.int8)
    training['item_id'] = training['item_id'].astype(np.int16)
    training.sort_values(by=cols, inplace=True)

    # Add revenue to the sales data
    sales_data['revenue'] = sales_data['item_quantity'] * sales_data['item_price']

    # Get the monthly sales
    monthly_sales = sales_data.groupby(cols).agg({'item_quantity': ['sum']})
    monthly_sales.columns = ['item_quantity']
    monthly_sales.reset_index(inplace=True)
    monthly_sales['month_id'] = monthly_sales['month_id'].astype(np.int8)
    monthly_sales['shop_id'] = monthly_sales['shop_id'].astype(np.int8)
    monthly_sales['item_id'] = monthly_sales['item_id'].astype(np.int16)

    # Merge with the large DataFrame
    training = pd.merge(training, monthly_sales, on=cols, how='left')

    # Fill missing data
    training['item_quantity'].fillna(0, inplace=True)

    #Add the test set
    training = pd.concat([training, test_data.drop(['ID'], axis=1)], ignore_index=True, sort=False, keys=cols)

    ####################################################
    # There are 3 types of items in the test set
    #   - Three groups:
    #       0 Completely new items
    #       1 Items never sold in this shop but not new
    #       2 Items sold in this shop before
    ####################################################
    # First month sold (month_id and month)
    training['month_id_item_release'] = training['item_id'].map(training[['month_id', 'item_id']].
                                                                groupby('item_id').min()['month_id'])

    # Number of months since the item has been released
    training['months_since_item_release'] = training['month_id'] - training['month_id_item_release']

    # Whether the item is new or not (Use for determining newness of item in test set)
    training['item_new'] = (training['months_since_item_release'] == 0)

    # Month that the item was first sold in the shop
    training = training.join(monthly_sales[['shop_id', 'month_id', 'item_id']].groupby(['shop_id', 'item_id']).min().
                             rename({'month_id': 'month_id_item_released_in_shop'}, axis=1), on=['shop_id', 'item_id'])

    # Number of months since the item has been released in this shop
    training['months_since_item_released_in_shop'] = (training['month_id'] - training['month_id_item_released_in_shop'])

    # Whether the item has been sold in this shop before (Used for determining newness of item in test set)
    training['item_never_sold_in_shop_before'] = ~(training['months_since_item_released_in_shop'] > 0)

    # Set the month of release and number of months since the item has been sold to -1
    #   if the item has never been sold in the shop
    training.loc[training['item_never_sold_in_shop_before'], 'months_since_item_released_in_shop'] = -1
    training.loc[training['item_never_sold_in_shop_before'], 'month_id_item_released_in_shop'] = -1

    # Downcast the new data
    training['months_since_item_released_in_shop'] = training['months_since_item_released_in_shop'].astype(np.int8)
    training['month_id_item_released_in_shop'] = training['month_id_item_released_in_shop'].astype(np.int8)

    # Get the seniority of each item
    training['item_seniority'] = (2 - training['item_new'].astype(int) -
                                  training['item_never_sold_in_shop_before'].astype(int)).astype(np.int8)

    # Drop unneeded columns
    training.drop(['month_id_item_release', 'months_since_item_release', 'item_new', 'month_id_item_released_in_shop',
                   'months_since_item_released_in_shop', 'item_never_sold_in_shop_before'], axis=1, inplace=True)

    ####################################
    # Merge all information to training
    ####################################
    # Fill missing data
    training.fillna(0, inplace=True)
    training = pd.merge(training, shops, on=['shop_id'], how='left')
    training = pd.merge(training, items, on=['item_id'], how='left')
    training = pd.merge(training, item_categories, on=['item_category_id'], how='left')

    ##############
    # Export data
    ##############
    sales_data.to_pickle('../data/cleaned/sales_data.pkl')
    item_categories.to_pickle('../data/cleaned/item_categories.pkl')
    shops.to_pickle('../data/cleaned/shops.pkl')
    training.to_pickle('../data/cleaned/full_data.pkl')


def lag_feature(df, lags, cols):
    for col in cols:
        print('Lagging column: ', col)
        temp = df[['month_id', 'shop_id', 'item_id', col]]
        for lag in lags:
            shifted = temp.copy()
            shifted.columns = ['month_id', 'shop_id', 'item_id', col + '_lag_' + str(lag)]
            shifted['month_id'] = shifted['month_id'] + lag
            df = pd.merge(df, shifted, on=['month_id', 'shop_id', 'item_id'], how='left')
    return df


def fill_na(df):
    for col in df.columns:
        if('_lag_' in col) & (df[col].isnull().any()):
            if 'item_quantity' in col:
                df[col].fillna(0, inplace=True)
            elif 'revenue' in col:
                df[col].fillna(0, inplace=True)
    return df


def clip_target(df):
    for col in df.columns:
        if 'item_quantity' in col:
            df[col] = df[col].clip(0, 20)
            df[col] = df[col].astype(np.float16)
    return df


def get_encoding_features():
    ##############
    # Import data
    ##############
    sales_data = pd.read_pickle('../data/cleaned/sales_data.pkl')
    training = pd.read_pickle('../data/cleaned/full_data.pkl')

    # Lag the item_quantity values
    training = lag_feature(training, [1, 2, 3], ['item_quantity'])

    # Get average item_quantity for each month
    group = training.groupby(['month_id']).agg({'item_quantity': ['mean']})
    group.columns = ['monthly_avg_item_quantity']
    group.reset_index(inplace=True)
    training = pd.merge(training, group, on=['month_id'], how='left')
    training['monthly_avg_item_quantity'] = training['monthly_avg_item_quantity'].astype(np.float16)
    # Lag the item_quantity
    training = lag_feature(training, [1], ['monthly_avg_item_quantity'])
    training.drop(['monthly_avg_item_quantity'], axis=1, inplace=True)

    # Get average item monthly item_quantity
    group = training.groupby(['month_id', 'item_id']).agg({'item_quantity': ['mean']})
    group.columns = ['item_monthly_avg_item_quantity']
    group.reset_index(inplace=True)
    training = pd.merge(training, group, on=['month_id', 'item_id'], how='left')
    training['item_monthly_avg_item_quantity'] = training['item_monthly_avg_item_quantity'].astype(np.float16)
    # Lag the item_quantity
    training = lag_feature(training, [1, 2, 3], ['item_monthly_avg_item_quantity'])
    training.drop(['item_monthly_avg_item_quantity'], axis=1, inplace=True)

    # Get average shop monthly item_quantity (lag 3)
    group = training.groupby(['month_id', 'shop_id']).agg({'item_quantity': ['mean']})
    group.columns = ['shop_monthly_avg_item_quantity']
    group.reset_index(inplace=True)
    training = pd.merge(training, group, on=['month_id', 'shop_id'], how='left')
    training['shop_monthly_avg_item_quantity'] = training['shop_monthly_avg_item_quantity'].astype(np.float16)
    # Lag the item_quantity
    training = lag_feature(training, [1, 2, 3], ['shop_monthly_avg_item_quantity'])
    training.drop(['shop_monthly_avg_item_quantity'], axis=1, inplace=True)

    # Get average item/shop monthly item_quantity (lag 3)
    group = training.groupby(['month_id', 'shop_id', 'item_id']).agg({'item_quantity': ['mean']})
    group.columns = ['shop_item_monthly_avg_item_quantity']
    group.reset_index(inplace=True)
    training = pd.merge(training, group, on=['month_id', 'shop_id', 'item_id'], how='left')
    training['shop_item_monthly_avg_item_quantity'] = training['shop_item_monthly_avg_item_quantity'].astype(np.float16)
    # Lag the item_quantity
    training = lag_feature(training, [1, 2, 3], ['shop_item_monthly_avg_item_quantity'])
    training.drop(['shop_item_monthly_avg_item_quantity'], axis=1, inplace=True)

    # Get average shop/category_subtype monthly item_quantity (lag 1)
    group = training.groupby(['month_id', 'shop_id', 'category_subtype_code']).agg({'item_quantity': ['mean']})
    group.columns = ['shop_subtype_monthly_avg_item_quantity']
    group.reset_index(inplace=True)
    training = pd.merge(training, group, on=['month_id', 'shop_id', 'category_subtype_code'], how='left')
    training['shop_subtype_monthly_avg_item_quantity'] = training['shop_subtype_monthly_avg_item_quantity'].astype(np.float16)
    # Lag the item_quantity
    training = lag_feature(training, [1], ['shop_subtype_monthly_avg_item_quantity'])
    training.drop(['shop_subtype_monthly_avg_item_quantity'], axis=1, inplace=True)

    # Get average shop city monthly item_quantity (lag 1)
    group = training.groupby(['month_id', 'city_code']).agg({'item_quantity': ['mean']})
    group.columns = ['city_monthly_avg_item_quantity']
    group.reset_index(inplace=True)
    training = pd.merge(training, group, on=['month_id', 'city_code'], how='left')
    training['city_monthly_avg_item_quantity'] = training['city_monthly_avg_item_quantity'].astype(np.float16)
    # Lag the item_quantity
    training = lag_feature(training, [1], ['city_monthly_avg_item_quantity'])
    training.drop(['city_monthly_avg_item_quantity'], axis=1, inplace=True)

    # Get average item/city monthly item_quantity (lag 1)
    group = training.groupby(['month_id', 'item_id', 'city_code']).agg({'item_quantity': ['mean']})
    group.columns = ['item_city_monthly_avg_item_quantity']
    group.reset_index(inplace=True)
    training = pd.merge(training, group, on=['month_id', 'item_id', 'city_code'], how='left')
    training['item_city_monthly_avg_item_quantity'] = training['item_city_monthly_avg_item_quantity'].astype(np.float16)
    # Lag the item_quantity
    training = lag_feature(training, [1], ['item_city_monthly_avg_item_quantity'])
    training.drop(['item_city_monthly_avg_item_quantity'], axis=1, inplace=True)

    print('Finished item_quantity encodings/lags\n')

    # Get average item price
    group = sales_data.groupby(['item_id']).agg({'item_price': ['mean']})
    group.columns = ['item_avg_item_price']
    group.reset_index(inplace=True)
    training = pd.merge(training, group, on=['item_id'], how='left')
    training['item_avg_item_price'] = training['item_avg_item_price'].astype(np.float16)

    # Get average item monthly item price
    group = sales_data.groupby(['month_id', 'item_id']).agg({'item_price': ['mean']})
    group.columns = ['item_monthly_avg_item_price']
    group.reset_index(inplace=True)
    training = pd.merge(training, group, on=['month_id', 'item_id'], how='left')
    training['item_monthly_avg_item_price'] = training['item_monthly_avg_item_price'].astype(np.float16)

    # Lag these price features
    lags = [1, 2, 3]
    training = lag_feature(training, lags, ['item_monthly_avg_item_price'])
    for lag in lags:
        training['delta_price_lag_' + str(lag)] = (training['item_monthly_avg_item_price_lag_' + str(lag)] -
                                                   training['item_avg_item_price']) / training['item_avg_item_price']

    del group
    gc.collect()

    print('Finished item_price encodings/lags\n')

    def select_trends(row):
        for lag in lags:
            if row['delta_price_lag_' + str(lag)]:
                return row['delta_price_lag_' + str(lag)]
        return 0

    training['delta_price_lag'] = training.apply(select_trends, axis=1)
    training['delta_price_lag'] = training['delta_price_lag'].astype(np.float16)
    training['delta_price_lag'].fillna(0, inplace=True)

    print('Finished getting delta_price_lag\n')

    features_to_drop = ['item_avg_item_price', 'item_monthly_avg_item_price']
    for lag in lags:
        features_to_drop.append('item_monthly_avg_item_price_lag_' + str(lag))
        features_to_drop.append('delta_price_lag_' + str(lag))
    training.drop(features_to_drop, axis=1, inplace=True)

    # Get total monthly revenue for each shop
    group = sales_data.groupby(['month_id', 'shop_id']).agg({'revenue': ['sum']})
    group.columns = ['shop_monthly_revenue']
    group.reset_index(inplace=True)
    training = pd.merge(training, group, on=['month_id', 'shop_id'], how='left')
    training['shop_monthly_revenue'] = training['shop_monthly_revenue'].astype(np.float32)

    # Get the shops average revenue
    group = group.groupby(['shop_id']).agg({'shop_monthly_revenue': ['mean']})
    group.columns = ['shop_avg_revenue']
    group.reset_index(inplace=True)
    training = pd.merge(training, group, on=['shop_id'], how='left')
    training['shop_avg_revenue'] = training['shop_avg_revenue'].astype(np.float32)

    del group
    gc.collect()

    training['delta_revenue'] = (training['shop_monthly_revenue'] - training['shop_avg_revenue']) / training['shop_avg_revenue']
    training['delta_revenue'] = training['delta_revenue'].astype(np.float32)

    # Lag the delta_revenue
    training = lag_feature(training, [1], ['delta_revenue'])
    training.drop(['shop_monthly_revenue', 'shop_avg_revenue', 'delta_revenue'], axis=1, inplace=True)

    print('Finished revenue encodings/lags\n')
    # print(training.head().T)

    # Create month and days in months data
    training['month'] = training['month_id'] % 12
    days = pd.Series([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
    training['days'] = training['month'].map(days).astype(np.int8)

    # Get the first month an item was sold in the shop
    training['item_shop_first_sale'] = training['month_id'] - training.groupby(['item_id', 'shop_id'])['month_id'].\
        transform('min')
    training['item_first_sale'] = training['month_id'] - training.groupby('item_id')['month_id'].transform('min')

    # Drop the first 3 months
    training = training[training['month_id'] > 2]
    training = fill_na(training)
    training = clip_target(training)

    # print('--------------------------------')
    # print(training.info(null_counts=True))

    training.to_pickle('../data/fully_processed/full_data.pkl')


#####################################################
# Creates the model and makes predictions
# Only 1 iteration with hyperparameters set manually
#####################################################
def model_prediction(seniority):
    print('----- Model Prediction for seniority', seniority, '-----')
    print('----- Loading Data -----')

    X_train = pd.read_pickle('../data/fully_processed/X_train.pkl')
    Y_train = pd.read_pickle('../data/fully_processed/Y_train.pkl')
    X_val = pd.read_pickle('../data/fully_processed/x_val_' + str(seniority) + '.pkl')
    Y_val = pd.read_pickle('../data/fully_processed/y_val_' + str(seniority) + '.pkl')

    max_depths = [15, 13, 9]
    min_child_weights = [6.5, 2.0, 9.0]
    colsample_bytrees = [0.3, 0.7, 0.3]
    subsamples = [0.9, 0.85, 0.92]
    learning_rates = [0.25, 0.15, 0.05]

    print('----- Training model -----\n')

    # Create the model
    model = XGBRegressor(
        max_depth=max_depths[seniority],
        min_child_weight=min_child_weights[seniority],
        colsample_bytree=colsample_bytrees[seniority],
        subsample=subsamples[seniority],
        eta=learning_rates[seniority],
        n_estimators=1000,
        tree_method='gpu_hist',     # Allows for gpu use, just comment it out if this is giving you issues and only the cpu will be used
        seed=42
    )
    model.fit(
        X_train,
        Y_train,
        eval_metric="rmse",
        eval_set=[(X_train, Y_train), (X_val, Y_val)],
        verbose=True,
        early_stopping_rounds=20
    )

    del X_train, Y_train, X_val, Y_val
    gc.collect()
    print('----- Trained model. Loading test data -----\n')

    test_df = pd.read_pickle('../data/fully_processed/test.pkl')
    X_test = test_df[test_df['item_seniority'] == seniority].drop(['item_quantity', 'item_seniority'], axis=1)

    print('----- Making predictions -----\n')

    # Make predictions
    Y_test = model.predict(X_test, model.best_ntree_limit).clip(0, 20)

    # Add the predictions to the test set that they belong to for combination later
    X_test['item_quantity'] = Y_test

    print('----- Saving results from model', seniority, '-----\n')

    X_test.to_pickle('../data/fully_processed/x_test_' + str(seniority) + '.pkl')

    print('----- Finished predictions -----\n')


#################################################################################################################################################
# Class to find the best hyperparameters
#   - From: https://towardsdatascience.com/an-example-of-hyperparameter-optimization-on-xgboost-lightgbm-and-catboost-using-hyperopt-12bc41a271e
#################################################################################################################################################
class HPOpt(object):
    def __init__(self, x_train, x_test, y_train, y_test):
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test

    def process(self, fn_name, space, trials, algo, max_evals):
        fn = getattr(self, fn_name)
        try:
            result = fmin(fn=fn, space=space, algo=algo, max_evals=max_evals, trials=trials)
        except Exception as e:
            return {'status': STATUS_FAIL,
                    'exception': str(e)}
        return result, trials

    def xgb_reg(self, para):
        reg = XGBRegressor(**para['reg_params'])
        return self.train_reg(reg, para)

    def train_reg(self, reg, para):
        reg.fit(self.x_train, self.y_train,
                eval_set=[(self.x_train, self.y_train), (self.x_test, self.y_test)],
                **para['fit_params'])
        pred = reg.predict(self.x_test)
        loss = para['loss_func'](self.y_test, pred)
        return {'loss': loss, 'status': STATUS_OK}


# Returns the k trials with the best rmse
# Selection sort but ends once the first k trials have been sorted
def get_best(trials, k):
    # If there are k or fewer elements return the entire list
    if k > len(trials):
        return trials
    for i in range(k):
        best = trials[i]
        best_score = best['result']['loss']
        best_pos = i
        # Find the best score between elements i - end
        for j in range(i+1, len(trials)):
            trial = trials[j]
            if trial['result']['loss'] < best_score:
                best = trial
                best_score = trial['result']['loss']
                best_pos = j
        # Move the best element to the ith position (i will be incremented so this element wont be moved)
        temp = trials[i]
        trials[i] = best
        trials[best_pos] = temp
    # Return the first k elements
    return trials[:k]


def split_data():
    training = pd.read_pickle('../data/fully_processed/full_data.pkl')

    X_train = training[training['month_id'] < 33].drop(['item_quantity', 'item_seniority'], axis=1)
    Y_train = training[training['month_id'] < 33]['item_quantity']
    val = training[training['month_id'] == 33]
    test = training[training['month_id'] == 34]

    # Create validation sets for each seniority type
    X_val_0 = val[val['item_seniority'] == 0].drop(['item_quantity', 'item_seniority'], axis=1)
    X_val_1 = val[val['item_seniority'] == 1].drop(['item_quantity', 'item_seniority'], axis=1)
    X_val_2 = val[val['item_seniority'] == 2].drop(['item_quantity', 'item_seniority'], axis=1)
    Y_val_0 = val[val['item_seniority'] == 0]['item_quantity']
    Y_val_1 = val[val['item_seniority'] == 1]['item_quantity']
    Y_val_2 = val[val['item_seniority'] == 2]['item_quantity']

    # Export the data
    X_train.to_pickle('../data/fully_processed/X_train.pkl')
    Y_train.to_pickle('../data/fully_processed/Y_train.pkl')
    val.to_pickle('../data/fully_processed/val.pkl')
    X_val_0.to_pickle('../data/fully_processed/x_val_0.pkl')
    X_val_1.to_pickle('../data/fully_processed/x_val_1.pkl')
    X_val_2.to_pickle('../data/fully_processed/x_val_2.pkl')
    Y_val_0.to_pickle('../data/fully_processed/y_val_0.pkl')
    Y_val_1.to_pickle('../data/fully_processed/y_val_1.pkl')
    Y_val_2.to_pickle('../data/fully_processed/y_val_2.pkl')
    test.to_pickle('../data/fully_processed/test.pkl')

    del test, X_train, val, Y_train, training, X_val_0, X_val_1, X_val_2, Y_val_0, Y_val_1, Y_val_2
    gc.collect()


# ###################################################
# # Creates the model and makes predictions
# # Uses the HPOpt class to tune the hyperparameters
# # Runs 100 iterations (This takes a long time)
# ###################################################
# def model_prediction(seniority):
#     print('----- Model Prediction for seniority', seniority, '-----')
#     print('----- Loading Data -----')
#
#     X_train = pd.read_pickle('../processed_data/xgb_sm/split_data/X_train.pkl')
#     Y_train = pd.read_pickle('../processed_data/xgb_sm/split_data/Y_train.pkl')
#     X_val = pd.read_pickle('../processed_data/xgb_sm/split_data/x_val_' + str(seniority) + '.pkl')
#     Y_val = pd.read_pickle('../processed_data/xgb_sm/split_data/y_val_' + str(seniority) + '.pkl')
#
#     # print(X_train.columns)
#
#     ###################################################
#     # TODO: If seniority 0/1, do we want all features?
#     #       - some lags will always be 0
#     ###################################################
#
#     print('----- Tuning hyperparameters -----\n')
#X
#     learning_rate_range = np.arange(0.05, 0.31, 0.05)
#     max_depth_range = np.arange(5, 16, 1, dtype=int)
#     min_child_weight_range = np.arange(0.5, 10, 0.5)
#     colsample_bytree_range = np.arange(0.3, 0.8, 0.1)
#     # subsample_range = np.arange(0.5, 1.0, 0.1)
#
#     # XGB Parameters
#     xgb_reg_params = {
#         'learning_rate': hp.choice('learning_rate', learning_rate_range),
#         'max_depth': hp.choice('max_depth', max_depth_range),
#         'min_child_weight:': hp.choice('min_child_weight', min_child_weight_range),
#         'colsample_bytree': hp.choice('colsample_bytree', colsample_bytree_range),
#         'subsample': hp.uniform('subsample', 0.8, 1),
#         'tree_method': 'gpu_hist',
#         'n_estimators': 1000,
#         'seed': 42
#     }
#     # print(xgb_reg_params)
#     xgb_fit_params = {
#         'eval_metric': 'rmse',
#         'early_stopping_rounds': 20,
#         'verbose': False
#     }
#     xgb_para = dict()
#     xgb_para['reg_params'] = xgb_reg_params
#     xgb_para['fit_params'] = xgb_fit_params
#     xgb_para['loss_func'] = lambda y, pred: np.sqrt(mean_squared_error(y, pred))
#
#     obj = HPOpt(X_train, X_val, Y_train, Y_val)
#     trials = Trials()
#     xgb_opt = obj.process(fn_name='xgb_reg', space=xgb_para, trials=trials, algo=tpe.suggest, max_evals=100)
#
#     # Get the best 10 trials
#     trials = trials.trials
#     best_trials = get_best(trials, 10)
#
#     del trials
#     gc.collect()
#
#     test_df = pd.read_pickle('../processed_data/xgb_sm/split_data/test.pkl')
#     X_test = test_df[test_df['item_seniority'] == seniority].drop(['item_quantity', 'item_seniority'], axis=1)
#
#     # Each of the best trials will train and make predictions.
#     # The average of all predictions will be used for submission
#     predictions = []
#     count = 1
#     for trial in best_trials:
#         print('----- Training using params ', count, ' -----')
#         # xgb_opt has the best index of the best parameters so we need to get the values from our ranges
#         learning_rate = learning_rate_range[trial['misc']['vals']['learning_rate']][0]
#         colsample_bytree = colsample_bytree_range[trial['misc']['vals']['colsample_bytree']][0]
#         max_depth = max_depth_range[trial['misc']['vals']['max_depth']][0]
#         min_child_weight = min_child_weight_range[trial['misc']['vals']['min_child_weight']][0]
#         # subsample = subsample_range[trial['misc']['vals']['subsample']][0]
#         subsample = trial['misc']['vals']['subsample'][0]
#
#         print('----- Params -----')
#         print('Learning rate: ', learning_rate)
#         print('colsample_bytree: ', colsample_bytree)
#         print('max_depth: ', max_depth)
#         print('min_child_weight: ', min_child_weight)
#         print('Subsample: ', subsample)
#         print('------------------')
#
#         # Create the model with the best parameters
#         model = XGBRegressor(
#             max_depth=max_depth,
#             min_child_weight=min_child_weight,
#             colsample_bytree=colsample_bytree,
#             subsample=subsample,
#             eta=learning_rate,
#             n_estimators=1000,
#             tree_method='gpu_hist',
#             seed=42
#         )
#         model.fit(
#             X_train,
#             Y_train,
#             eval_metric="rmse",
#             eval_set=[(X_train, Y_train), (X_val, Y_val)],
#             verbose=False,
#             early_stopping_rounds=20
#         )
#         print('----- Making predictions -----\n')
#
#         # Make predictions
#         Y_test = model.predict(X_test, model.best_ntree_limit)      # .clip(0, 20)
#         predictions.append(Y_test)
#         count += 1
#
#     # Get the average of all of the predictions
#     Y_test_final = predictions[0]
#     for sub in predictions[1:]:
#         Y_test_final = Y_test_final + sub
#     Y_test_final = Y_test_final / len(predictions)
#     Y_test_final = Y_test_final.clip(0, 20)
#
#     # Add the predictions to the test set that they belong to for combination later
#     X_test['item_quantity'] = Y_test_final
#
#     print('----- Saving results from model', seniority, '-----\n')
#
#     X_test.to_pickle('../processed_data/xgb_sm/final/x_test_' + str(seniority) + '.pkl')
#
#     # print(X_test.info(null_counts=True))
#
#     del test_df, X_test, X_val, X_train, Y_val, Y_train, learning_rate_range, max_depth_range, min_child_weight_range,\
#         colsample_bytree_range, xgb_reg_params, xgb_fit_params, xgb_para, xgb_opt, best_trials, predictions


def combine_predictions():
    # Import the data
    test = pd.read_pickle('../data/fully_processed/test.pkl')
    test_df = pd.read_pickle('../data/fully_processed/test.pkl')
    # Y_test = test_df['item_quantity']
    test_df = test_df[['shop_id', 'item_id']].reset_index().rename({'index': 'ID'}, axis=1)
    test_0 = pd.read_pickle('../data/fully_processed/.pkl')
    test_1 = pd.read_pickle('../data/fully_processed/x_test_1.pkl')
    test_2 = pd.read_pickle('../data/fully_processed/x_test_2.pkl')

    combined_test = pd.concat([test_0, test_1], axis=0)
    combined_test = pd.concat([combined_test, test_2], axis=0)

    # Merge the test
    predicted_test = pd.merge(test_df, combined_test, on=['shop_id', 'item_id'], how='left')
    predicted_test = predicted_test[['shop_id', 'item_id', 'item_quantity']]

    # Create the submission
    submission = pd.merge(test, predicted_test, on=['shop_id', 'item_id'], how='left')
    submission.drop(['shop_id', 'item_id', 'month_id'], axis=1, inplace=True)
    submission.rename({'item_quantity': 'item_cnt_month'}, axis=1, inplace=True)
    submission.to_csv('xgb_submission.csv', index=False)

    print(submission.info(null_counts=True))

    # # Check the RMSE
    # loss = np.sqrt(mean_squared_error(Y_test, submission['item_quantity']))
    # print('Loss = ', loss)
    

def one_hot_encode(X, column_names):
    for column_name in column_names:
        print('One hot encoding ', column_name)
        gc.collect()
        categorized = pd.get_dummies(X[column_name], prefix=[column_name], drop_first=False).astype(np.int8)
        X_categorized = pd.concat([X,categorized], axis=1)
#         X_categorized.drop([column_name], inplace=True)

    return X_categorized

def rmse(y_true, y_pred):
    return backend.sqrt(backend.mean(backend.square(y_pred - y_true), axis=-1))

def train_neural_net(data, coef, reg_type, num_epoc, es):
    Xtrain = data[0]
    Ytrain = data[1]
    Xval = data[2]
    Yval = data[3]

    kernel_coef = coef[0]
    bias_coef = coef[1]
    activity_coef = coef[2]
    if reg_type == 'l1':
        kern_reg = regularizers.l1(kernel_coef)
        bias_reg = regularizers.l1(bias_coef)
        activity_reg = regularizers.l1(activity_coef)
    elif reg_type == 'l2':
        kern_reg = regularizers.l2(kernel_coef)
        bias_reg = regularizers.l2(bias_coef)
        activity_reg = regularizers.l2(activity_coef)
    elif reg_type == 'l1l2':
        kern_reg = regularizers.l1_l2(kernel_coef)
        bias_reg = regularizers.l1_l2(bias_coef)
        activity_reg = regularizers.l1_l2(activity_coef)

    # Activations: linear, exponential, hard_sigmoid, sigmoid, tanh, relu, softsign, softplus, softmax, elu
    model = Sequential()
    model.add(Dense(1024, input_dim=32, use_bias=True, kernel_regularizer=kern_reg, bias_initializer='zeros', kernel_initializer='glorot_uniform', activation='elu',
        bias_regularizer=bias_reg, activity_regularizer=activity_reg))
    #model.add(Dropout(0.25))
    #model.add(Dense(128, activation='tanh', use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros'))
    #    kernel_regularizer=regularizers.l2(kernel_coef), bias_regularizer=None, 
    #      activity_regularizer=regularizers.l2(activity_coef), kernel_constraint=None, bias_constraint=None))
    #model.add(Dropout(0.25))
    #model.add(Dense(64, activation='relu', use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros'))
    #     kernel_regularizer=regularizers.l2(kernel_coef), bias_regularizer=None, 
    #     activity_regularizer=regularizers.l2(activity_coef), kernel_constraint=None, bias_constraint=None))
    # model.add(Dropout(0.25))
    # model.add(Dense(256, activation='relu', use_bias=True, kernel_initializer='glorot_uniform', 
    #     bias_initializer='zeros', kernel_regularizer=regularizers.l2(kernel_coef), bias_regularizer=None, 
    #     activity_regularizer=regularizers.l2(activity_coef), kernel_constraint=None, bias_constraint=None))
    # model.add(Dropout(0.25))
    # model.add(Dense(128, activation='relu', use_bias=True, kernel_initializer='glorot_uniform', 
    #     bias_initializer='zeros', kernel_regularizer=regularizers.l2(kernel_coef), bias_regularizer=None, 
    #     activity_regularizer=regularizers.l2(activity_coef), kernel_constraint=None, bias_constraint=None))
    # model.add(Dropout(0.25))
    # model.add(Dense(64, activation='relu', use_bias=True, kernel_initializer='glorot_uniform', 
    #     bias_initializer='zeros', kernel_regularizer=regularizers.l2(kernel_coef), bias_regularizer=None, 
    #     activity_regularizer=regularizers.l2(activity_coef), kernel_constraint=None, bias_constraint=None))
    # model.add(Dropout(0.25))
    # model.add(Dense(32, activation='relu', use_bias=True, kernel_initializer='glorot_uniform', 
    #     bias_initializer='zeros', kernel_regularizer=regularizers.l2(kernel_coef), bias_regularizer=None, 
    #     activity_regularizer=regularizers.l2(activity_coef), kernel_constraint=None, bias_constraint=None))
    # model.add(Dropout(0.25))
    model.add(Dense(1, activation='softplus'))
    model.compile(loss='mse', optimizer='adam', metrics=[rmse])
    
    history = model.fit(Xtrain, Ytrain, validation_data = (Xval, Yval), batch_size=2000, epochs=num_epoc, callbacks=[es])
    # Save model
    #model.save('my_model.h5')
    return model


def main():
    pre_processed = True
    reload_model = False

    if not pre_processed:
        gc.collect()
        # Clean the data
        print('Cleaning Data')
        start = time.time()
        clean_data()
        gc.collect()
        finish = time.time()
        print('Cleaning data took: ', str(finish - start))
        # input('Continue?')

        # Feature engineering: Basic features
        print('Constructing basic features')
        start = time.time()
        get_basic_features()
        gc.collect()
        finish = time.time()
        print('Basic features took: ', str(finish - start))
        # input('Continue?')

        # Feature engineering: Lags and encodings
        print('Constructing lags and encodings')
        start = time.time()
        get_encoding_features()
        gc.collect()
        finish = time.time()
        print('Encoding and lag features took: ', str(finish - start))
        # input('Continue?')

        split_data()
    
    min_max_scaler = preprocessing.MinMaxScaler()
    X_train = pd.read_pickle('../data/fully_processed/X_train.pkl')
    #X_train = (X_train.values).astype(dtype=np.float32)
    # Scaled
    X_train = min_max_scaler.fit_transform((X_train.values).astype(dtype=np.float32))
    Y_train = pd.read_pickle('../data/fully_processed/Y_train.pkl')
    Y_train = (Y_train.values).astype(dtype=np.float32)
    val = pd.read_pickle('../data/fully_processed/val.pkl')
    X_val = val.drop(['item_quantity', 'item_seniority'], axis=1)
    #X_val = (X_val.values).astype(dtype=np.float32)
    # Scaled
    X_val = min_max_scaler.fit_transform((X_val.values).astype(dtype=np.float32))
    Y_val = val['item_quantity']
    Y_val = (Y_val.values).astype(dtype=np.float32)
    test_df = pd.read_pickle('../data/fully_processed/test.pkl')
    X_test = test_df.drop(['item_quantity', 'item_seniority'], axis=1)
    #X_test = (X_test.values).astype(dtype=np.float32)
    # Scaled
    X_test = min_max_scaler.fit_transform((X_test.values).astype(dtype=np.float32))
    
#     names_to_categorize = ['month_id', 'shop_id', 'shop_type_code', 'city_code', 'item_category_id', 'name2', 'name3',
#                           'category_subtype_code', 'category_type_code'] # Took out 'item_id' due. tomemory limitations
    
#     X_train = one_hot_encode(X_train, names_to_categorize)

    print(X_train.shape)
    print(Y_train.shape)
    
    print('Training neural net')
    start = time.time()
#     split_data()

    if not(reload_model):
        # Find optimal parameters
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=30)
        num_epoc = 100
        coef = [0, 0, 0]
        reg_type = 'l2'
        model = train_neural_net([X_train, Y_train, X_val, Y_val], coef, reg_type, num_epoc, es)
        loss = model.history.history['loss']
        val_loss = model.history.history['val_loss']
        rmse_s = model.history.history['rmse'] # NECCESSARY TO NOT CONFUSE WITH rmse function
        val_rmse = model.history.history['val_rmse']
        # ----------------------------------- Find Optimal Hyper-parameters ------------------------------
    #     best_loss = math.inf
    #     best_val_loss = math.inf
    #     best_rmse = math.inf
    #     best_val_rmse = math.inf
    #     #kernel_coef = np.linspace(0,1,7)
    #     #bias_coef = np.linspace(0,1,7)
    #     #activity_coef = np.linspace(0,1,7)
    #     kernel_coef = [0, 0.5, 1]
    #     bias_coef = [0, 0.5, 1]
    #     activity_coef = [0, 0.5, 1]
    #     regularizer_type = ['l1','l2','l1l2']

    #     best_kernel_coef = None
    #     best_bias_coef = None
    #     best_activity_coef = None
    #     best_reg_type = None

    #     for k in kernel_coef:
            
    #         for b in bias_coef:
                
    #             for a in activity_coef:
                    
    #                 for r in regularizer_type:
    #                     print('From: ' + str(len(kernel_coef)) + ' Kernel Coefs, we are at k # ' + str(kernel_coef.index(k)) + '\n')
    #                     print('From: ' + str(len(bias_coef)) + ' Bias Coefs, we are at b # ' + str(bias_coef.index(b)) + '\n')
    #                     print('From: ' + str(len(activity_coef)) + ' Activity Coefs, we are at a # ' + str(activity_coef.index(a)) + '\n')
    #                     print('From: ' + str(len(regularizer_type)) + ' Regularizer types, we are at reg type # ' + str(regularizer_type.index(r)) + '\n')
    #                     coef = [k, b, a]
    #                     reg_type = r
    #                     model = train_neural_net([X_train, Y_train, X_val, Y_val], coef, reg_type, num_epoc, es)
    #                     loss = model.history.history['loss']
    #                     val_loss = model.history.history['val_loss']
    #                     rmse_s = model.history.history['rmse'] # NECCESSARY TO NOT CONFUSE WITH rmse function
    #                     val_rmse = model.history.history['val_rmse']
    #                     # Choosing val_loss here but rmse or val_rmse could be chosen as well
    #                     if val_loss[-1] < best_val_loss:
    #                         # This is a better model. Save its parameters and model
    #                         best_rmse = rmse_s[-1]
    #                         best_val_loss = val_loss[-1]
    #                         best_loss = loss[-1]
    #                         best_val_rmse = val_rmse[-1]
    #                         best_kernel_coef = k
    #                         best_bias_coef = b
    #                         best_activity_coef = a
    #                         best_reg_type = r
    #                         model.save('best_model.h5')
    #                     print('Current best values: \"[Val_loss, RMSE, Val_RMSE, Loss, Kernel Coef, Bias Coef, Activity Coef, Reg_type]\"\n' + str([best_val_loss, best_rmse, best_val_rmse, best_loss, best_kernel_coef, best_bias_coef, best_activity_coef, best_reg_type]) + '\n')
    # #-----------------------------------------------------------------------------------------------
    else:
        model = load_model('my_model.h5')
    finish = time.time()
    print('Training neural net took: ', str(finish - start))
    #print("Finished Hyper-Parameter Space scan\n")
    #print("Best Root Mean Squared Error:" + str(best_rmse) + "\nBest Val_loss" + str(best_val_loss) + "@ the following hyper-parameter values:\n")
    #print("Best Kernel Coef:" + str(best_kernel_coef) + "\nBest Bias Coef: " + str(best_bias_coef) + "\nBest Activity Coef" + str(best_activity_coef) + "\nBest Regularizer type:" + str(best_reg_type) +"")
    
    # Plot
    plt.plot(loss)
    plt.plot(val_loss)
    plt.title('Training Statistics')
    plt.ylabel('Loss, Validation_Loss')
    plt.xlabel('Epoch')
    plt.legend(['Training Loss', 'Validation Loss'], loc='upper right')
    plt.show()

    # Compare with our best submission
    best_sub=pd.read_csv("best_xgb_submission.csv")
    submission_df = pd.DataFrame(data=model.predict(X_test), index=range(214200), columns=['item_cnt_month'])
    submission_df.index.name = 'ID'
    pdb.set_trace()
    
    print("Best xgb boost submission predictions\n")
    print(best_sub.describe())
    print("Our model predictions\n")
    print(submission_df.describe())
    submission_df.to_csv('keras_nn_submission.csv')
    
    
    # Testing Accuracy
    # best_sub = best_sub.to_numpy(dtype=np.float32)
    # ypred = ypred.to_numpy(dtype=np.float32)
    # pred = list()
    # xgb_gtruth = list()
    # for i in range(len(ypred)):
    #     pred.append(np.argmax(ypred[i].round(10)))
    #     xgb_gtruth.append(np.argmax(best_sub[i].round(10)))
    # a = accuracy_score(best_sub, pred)
    # print('Accuracy on the test set after training + validation is:', a*100)
#     # Train and predict
#     print('Training models')
#     start = time.time()
#     # model_prediction()
#     print('----- Splitting the data -----\n')
#     split_data()
#     print('----- Data split -----\n')
#     for i in range(3):
#         model_prediction(i)
#         gc.collect()
#     finish = time.time()
#     print('Training and predicting took: ', str(finish - start))
#     # input('Continue?')

#     # Combine predictions
#     print('Combining predictions')
#     start = time.time()
#     combine_predictions()
#     finish = time.time()
#     print('Combining predictions took: ', str(finish - start))
#     # input('Continue?')

#     # Compare outputs
#     print('Comparing Submissions')
#     this_output = pd.read_csv('xgb_submission.csv')
#     dirname = '/kaggle/input'
#     foldername = '/xgb-best-submission/'
#     xgb3_best_output = pd.read_csv(dirname + foldername + 'xgb_submission_best.csv')

#     print('----- This Output -----')
#     print(this_output.describe())

#     print('----- XGBoost_Best Output -----')
#     print(xgb3_best_output.describe())
#     gc.collect()


if __name__ == '__main__':
    main()