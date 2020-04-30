#https://www.kaggle.com/lonewolf45/coursera-final-project
from itertools import product, combinations
from sklearn.preprocessing import LabelEncoder

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 100)

import gc
import time
import pickle
import re


def downcast_dtypes(df):
    float_cols = [c for c in df if df[c].dtype == "float64"]
    int_cols = [c for c in df if df[c].dtype in ["int64", "int32"]]
    df[float_cols] = df[float_cols].astype(np.float32)
    df[int_cols] = df[int_cols].astype(np.int16)
    return df


def clean_data():
    # Load the data
    sales_data = pd.read_csv('../input/sales_train.csv')
    test_data = pd.read_csv('../input/test.csv')
    items = pd.read_csv('../input/items.csv')
    item_categories = pd.read_csv('../input/item_categories.csv')
    shops = pd.read_csv('../input/shops.csv')

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
    sales_data.loc[sales_data['item_quantity'] < 1, 'item_quantity'] = 0

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
    sales_data.to_pickle('../processed_data/xgb3/cleaned/sales_data.pkl')
    shops.to_pickle('../processed_data/xgb3/cleaned/shops.pkl')
    items.to_pickle('../processed_data/xgb3/cleaned/items.pkl')
    item_categories.to_pickle('../processed_data/xgb3/cleaned/item_categories.pkl')
    test_data.to_pickle('../processed_data/xgb3/cleaned/test.pkl')

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
    # Load data
    sales_data = pd.read_pickle('../processed_data/xgb3/cleaned/sales_data.pkl')
    shops = pd.read_pickle('../processed_data/xgb3/cleaned/shops.pkl')
    items = pd.read_pickle('../processed_data/xgb3/cleaned/items.pkl')
    item_categories = pd.read_pickle('../processed_data/xgb3/cleaned/item_categories.pkl')
    test_data = pd.read_pickle('../processed_data/xgb3/cleaned/test.pkl')

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

    # Clip the target (0-20) and adjust the type
    training['item_quantity'] = training['item_quantity'].clip(0, 20)
    training['item_quantity'] = training['item_quantity'].astype(np.float16)

    ####################################
    # Merge all information to training
    ####################################
    training = pd.concat([training, test_data.drop(['ID'], axis=1)], ignore_index=True, sort=False, keys=cols)

    # Fill missing data
    training.fillna(0, inplace=True)
    training = pd.merge(training, shops, on=['shop_id'], how='left')
    training = pd.merge(training, items, on=['item_id'], how='left')
    training = pd.merge(training, item_categories, on=['item_category_id'], how='left')

    ##############
    # Export data
    ##############
    sales_data.to_pickle('../processed_data/xgb3/basic_features/sales_data.pkl')
    item_categories.to_pickle('../processed_data/xgb3/basic_features/item_categories.pkl')
    shops.to_pickle('../processed_data/xgb3/basic_features/shops.pkl')
    training.to_pickle('../processed_data/xgb3/basic_features/full_data.pkl')


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


def get_encoding_features():
    ##############
    # Import data
    ##############
    sales_data = pd.read_pickle('../processed_data/xgb3/basic_features/sales_data.pkl')
    training = pd.read_pickle('../processed_data/xgb3/basic_features/full_data.pkl')

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

    print('Finished revenue encodings/lags\nTraining head:')
    print(training.head().T)

    # Create month and days in months data
    training['month'] = training['month_id'] % 12
    days = pd.Series([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
    training['days'] = training['month'].map(days).astype(np.int8)

    # Get the first month an item was sold in the shop
    training['item_shop_first_sale'] = training['month_id'] - training.groupby(['item_id', 'shop_id'])['month_id'].\
        transform('min')
    training['item_first_sale'] = training['month_id'] - training.groupby('item_id')['month_id'].transform('min')

    # Drop the first 3 months
    training = training[training['month_id'] > 3]
    training = fill_na(training)

    print('--------------------------------')
    print(training.info(null_counts=True))

    training.to_pickle('../processed_data/xgb3/fully_processed/full_data.pkl')


def split_data():
    training = pd.read_pickle('../processed_data/xgb3/fully_processed/full_data.pkl')

    X_train = training[training['month_id'] < 33].drop(['item_quantity'], axis=1)
    Y_train = training[training['month_id'] < 33]['item_quantity']
    X_val = training[training['month_id'] == 33].drop(['item_quantity'], axis=1)
    Y_val = training[training['month_id'] == 33]['item_quantity']
    X_test = training[training['month_id'] == 34].drop(['item_quantity'], axis=1)

    X_train.to_pickle('../processed_data/xgb3/split_data/X_train.pkl')
    Y_train.to_pickle('../processed_data/xgb3/split_data/Y_train.pkl')
    X_val.to_pickle('../processed_data/xgb3/split_data/X_val.pkl')
    Y_val.to_pickle('../processed_data/xgb3/split_data/Y_val.pkl')
    X_test.to_pickle('../processed_data/xgb3/split_data/X_test.pkl')

    del X_test, X_train, X_val, Y_train, Y_val, training
    gc.collect()


def model_prediction():
    print('----- Splitting the data -----\n')
    split_data()
    print('----- Data split. Loading train and val sets -----\n')

    # Separate the data
    X_train = pd.read_pickle('../processed_data/xgb3/split_data/X_train.pkl')
    Y_train = pd.read_pickle('../processed_data/xgb3/split_data/Y_train.pkl')
    X_val = pd.read_pickle('../processed_data/xgb3/split_data/X_val.pkl')
    Y_val = pd.read_pickle('../processed_data/xgb3/split_data/Y_val.pkl')

    print('----- Training model -----\n')

    # Create the model
    model = XGBRegressor(
        max_depth=10,
        n_estimators=1000,
        min_child_weight=0.5,
        colsample_bytree=0.8,
        subsample=0.8,
        eta=0.1,
        # tree_method='gpu_hist',
        seed=42
    )
    model.fit(
        X_train,
        Y_train,
        eval_metric="rmse",
        eval_set=[(X_train, Y_train), (X_val, Y_val)],
        verbose=True,
        early_stopping_rounds=10
    )

    del X_train, Y_train, X_val, Y_val
    gc.collect()
    print('----- Trained model. Loading test data -----\n')

    test = pd.read_pickle('../processed_data/xgb3/cleaned/test.pkl')
    X_test = pd.read_pickle('../processed_data/xgb3/split_data/X_test.pkl')

    print('----- Making predictions -----\n')

    # Make predictions
    Y_test = model.predict(X_test, model.best_ntree_limit).clip(0, 20)
    submission = pd.DataFrame({'ID': test.index, 'item_cnt_month': Y_test})
    submission.to_csv('../output/xgboost/xgb3/xgb_submission.csv', index=False)

    print('----- Finished predictions -----')


def main():
    gc.collect()
    # # Clean the data
    # print('Cleaning Data')
    # start = time.time()
    # clean_data()
    # gc.collect()
    # finish = time.time()
    # print('Cleaning data took: ', str(finish - start))
    # input('Continue?')
    #
    # # Feature engineering: Basic features
    # print('Constructing basic features')
    # start = time.time()
    # get_basic_features()
    # gc.collect()
    # finish = time.time()
    # print('Basic features took: ', str(finish - start))
    # input('Continue?')
    #
    # # Feature engineering: Lags and encodings
    # print('Constructing lags and encodings')
    # start = time.time()
    # get_encoding_features()
    # gc.collect()
    # finish = time.time()
    # print('Encoding and lag features took: ', str(finish - start))
    # input('Continue?')

    # Train and predict
    print('Training model')
    start = time.time()
    model_prediction()
    gc.collect()
    finish = time.time()
    print('Training and predicting took: ', str(finish - start))
    input('Continue?')

    # Compare outputs
    xgb3_output = pd.read_csv('../output/xgboost/xgb3/xgb_submission.csv')
    xgb2_output = pd.read_csv('../output/xgboost/xgb2_submission.csv')
    xgb1_output = pd.read_csv('../output/xgboost/xgb_submission.csv')
    xgbTSNB_output = pd.read_csv('../output/xgboost/xgb_submission_TSNB.csv')
    # xgbBest_output = pd.read_csv('../output/xgboost/xgb_prediction.csv')
    xgbBest_output = pd.read_csv('../output/xgboost/xgbCFP_submission.csv')

    print('----- XGBoost_3rd Output -----')
    print(xgb3_output.describe())

    # print('----- XGBoost_2nd Output -----')
    # print(xgb2_output.describe())
    #
    # print('\n----- XGBoost_1st Output -----')
    # print(xgb1_output.describe())

    print('\n----- XGBoost_TSNB Output -----')
    print(xgbTSNB_output.describe())

    print('\n----- XGBoost_Best Output -----')
    print(xgbBest_output.describe())
    gc.collect()


if __name__ == '__main__':
    main()
