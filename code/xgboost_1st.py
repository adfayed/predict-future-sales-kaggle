import numpy as np
import pandas as pd
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 100)

from itertools import product
from sklearn.preprocessing import LabelEncoder

from xgboost import XGBRegressor
from xgboost import plot_importance

import time
import sys
import gc
import pickle


# Downcast the data to be able to save on space
def downcast_dtypes(df):
    float_cols = [c for c in df if df[c].dtype == "float64"]
    int_cols = [c for c in df if df[c].dtype in ['int64', 'int32']]
    df[float_cols] = df[float_cols].astype(np.float32)
    df[int_cols] = df[int_cols].astype(np.int16)
    return df


def clean_data():
    # Open files
    train_df = pd.read_csv('../data/sales_train.csv')
    test_df = pd.read_csv('../data/test.csv')
    items = pd.read_csv('../data/items.csv')
    item_categories = pd.read_csv('../data/item_categories.csv')
    shops = pd.read_csv('../data/shops.csv')

    # Rename features
    train_df.rename({'date_block_num': 'month_id', 'item_cnt_day': 'item_quantity'}, axis=1, inplace=True)

    # Convert date feature to datetime type
    train_df['date'] = pd.DataFrame(pd.to_datetime(train_df['date'], format='%d.%m.%Y'))
    train_df['day_id'] = (train_df['date'] - train_df['date'].min()).dt.days

    # ????????????????????????????????????????????????????????????????????????????????
    # Discard the returned items. We only care about the sales so they are irrelevant
    # ????????????????????????????????????????????????????????????????????????????????
    # print('percentage of realisations that represent returned articles : ' + str(
    #     round((train_df['item_quantity'] < 0).sum() / (train_df['item_quantity'] >= 0).sum() * 100, 2)) + ' %')
    train_df.drop(train_df[train_df['item_quantity'] < 0].index, axis=0, inplace=True)

    ########################
    # Processing the Shops #
    ########################
    # Remove the duplicated shops
    # Shops 0 and 57, 1 and 58, 10 and 11 are the same shops but different time period. Will use 57, 58, and 10 as the
    #   labels because they are in the test set, but shops 0, 1, and 11 are not
    # Shop 40 seems to be an "antenna" of shop 39 so combine their sales together as shop 39 since it is in the test set
    shops.drop(0, axis=0, inplace=True)
    shops.drop(1, axis=0, inplace=True)
    shops.drop(11, axis=0, inplace=True)
    shops.drop(40, axis=0, inplace=True)

    train_df.loc[train_df['shop_id'] == 0, 'shop_id'] = 57
    train_df.loc[train_df['shop_id'] == 1, 'shop_id'] = 58
    train_df.loc[train_df['shop_id'] == 11, 'shop_id'] = 10
    train_df.loc[train_df['shop_id'] == 40, 'shop_id'] = 39

    # Group all sales together by all columns except 'item_quantities'
    train_df = train_df.groupby(list(train_df.columns.drop('item_quantity')), as_index=False).sum()
    # train_df.info(null_counts=True)

    # Feature Engineering: Cities
    shops['city'] = shops['shop_name'].str.extract('(\S+)\s', expand=False)
    # print('number of cities in the dataset : ' + str(shops.city.nunique()))
    # print('number of null cities : ' + str(shops.city.isnull().sum()))
    # print(shops.city.unique())

    # Label encoding of the city names
    shops['city_id'] = pd.factorize(shops['city'])[0]

    # ?????????????????????
    # Remove outlier shops
    # ?????????????????????
    # There are two shops only open in October months (9 and 20) so remove them
    shops.drop(9, axis=0, inplace=True)
    shops.drop(20, axis=0, inplace=True)
    train_df.drop(train_df.loc[train_df['shop_id'] == 9].index, axis=0, inplace=True)
    train_df.drop(train_df.loc[train_df['shop_id'] == 20].index, axis=0, inplace=True)

    # Remove shop 33 as it is only open for a short time in the middle of the training period
    shops.drop(33, axis=0, inplace=True)
    train_df.drop(train_df.loc[train_df['shop_id'] == 33].index, axis=0, inplace=True)

    # ???????????????????????????????????????????????????????????????????????????????????????????
    # Remove the two entries for shop 34 on month 18. Only two items were sold, each only 1 time
    # ???????????????????????????????????????????????????????????????????????????????????????????
    train_df.drop(train_df.loc[(train_df['shop_id'] == 34) & (train_df['month_id'] == 18)].index, axis=0, inplace=True)

    ###############################
    # Process the Item_Categories #
    ###############################
    # Rename categories
    item_categories.loc[0, 'item_category_name'] = 'Аксессуары - PC (Гарнитуры/Наушники)'
    item_categories.loc[8, 'item_category_name'] = 'Билеты - Билеты (Цифра)'
    item_categories.loc[9, 'item_category_name'] = 'Доставка товара - Доставка товара'
    item_categories.loc[26, 'item_category_name'] = 'Игры - Android (Цифра)'
    item_categories.loc[27, 'item_category_name'] = 'Игры - MAC (Цифра)'
    item_categories.loc[28, 'item_category_name'] = 'Игры - PC (Дополнительные издания)'
    item_categories.loc[29, 'item_category_name'] = 'Игры - PC (Коллекционные издания)'
    item_categories.loc[30, 'item_category_name'] = 'Игры - PC (Стандартные издания)'
    item_categories.loc[31, 'item_category_name'] = 'Игры - PC (Цифра)'
    item_categories.loc[32, 'item_category_name'] = 'Карты оплаты - Кино, Музыка, Игры'
    item_categories.loc[
        79, 'item_category_name'] = 'Прием денежных средств для 1С-Онлайн - Прием денежных средств для 1С-Онлайн'
    item_categories.loc[80, 'item_category_name'] = 'Билеты - Билеты'
    item_categories.loc[81, 'item_category_name'] = 'Misc - Чистые носители (шпиль)'
    item_categories.loc[82, 'item_category_name'] = 'Misc - Чистые носители (штучные)'
    item_categories.loc[83, 'item_category_name'] = 'Misc - Элементы питания'

    # Create item_supercategory_name, item_category_console, and item_category_is_digital features from item_category
    item_categories['item_supercategory_name'] = item_categories['item_category_name']. \
        str.extract('([\S\s]+)\s\-', expand=False)
    item_categories['item_category_is_digital'] = (item_categories['item_category_name'].str.find('(Цифра)') >= 0)
    consoles = ['PS2', 'PS3', 'PS4', 'PSP', 'PSVita', 'XBOX 360', 'XBOX ONE', 'PC', 'MAC', 'Android']
    item_categories['item_category_console'] = ''
    for console in consoles:
        item_categories['item_category_console'] += item_categories['item_category_name']. \
            str.extract('(' + console + ')', expand=False).fillna('')
    item_categories.loc[item_categories['item_category_console'] == '', 'item_category_console'] = 'None'

    train_df['item_category_id'] = train_df['item_id'].map(items['item_category_id'])

    # ???????????????????????????
    # Drop irrelevant categories
    # ???????????????????????????
    train_df.drop(train_df.loc[(train_df['item_id'].map(items['item_category_id']) == 8)].index.values, axis=0,
                  inplace=True)
    train_df.drop(train_df.loc[(train_df['item_id'].map(items['item_category_id']) == 80)].index.values, axis=0,
                  inplace=True)
    train_df.drop(train_df.loc[(train_df['item_id'].map(items['item_category_id']) == 81)].index.values, axis=0,
                  inplace=True)
    train_df.drop(train_df.loc[(train_df['item_id'].map(items['item_category_id']) == 82)].index.values, axis=0,
                  inplace=True)
    item_categories.drop(8, axis=0, inplace=True)
    item_categories.drop(80, axis=0, inplace=True)
    item_categories.drop(81, axis=0, inplace=True)
    item_categories.drop(82, axis=0, inplace=True)
    items.drop(items.loc[items['item_category_id'] == 8].index.values, axis=0, inplace=True)
    items.drop(items.loc[items['item_category_id'] == 80].index.values, axis=0, inplace=True)
    items.drop(items.loc[items['item_category_id'] == 81].index.values, axis=0, inplace=True)
    items.drop(items.loc[items['item_category_id'] == 82].index.values, axis=0, inplace=True)

    # label encoding of item_category additional features
    item_categories['item_supercategory_id'] = item_categories['item_supercategory_name'].map(
        {'Игровые консоли': 0, 'Игры': 1, 'Аксессуары': 2, 'Доставка товара': 3,
         'Прием денежных средств для 1С-Онлайн': 4, 'Карты оплаты': 5, 'Кино': 6, 'Книги': 7, 'Музыка': 8, 'Подарки': 9,
         'Программы': 10, 'Misc': 11})
    item_categories['item_category_console_id'] = item_categories['item_category_console'].map(
        {console: i for i, console in enumerate(consoles + ['None'])})

    # reorder columns
    original_cols = ['item_category_name', 'item_supercategory_name', 'item_category_console',
                     'item_category_is_digital']
    label_cols = ['item_category_id', 'item_supercategory_id', 'item_category_console_id']
    item_categories = item_categories[original_cols + label_cols]

    # Join columns to training set
    for col in original_cols:
        train_df[col] = train_df['item_category_id'].map(item_categories[col])

    del original_cols, label_cols

    ######################
    # Process the Prices #
    ######################
    # ??????????????????????????????
    # Remove outliers in price data
    # ??????????????????????????????
    max_price = train_df['item_price'].max()
    most_expensive_item = train_df.loc[train_df['item_price'] == max_price, 'item_id'].values[0]
    # print('index of most expensive item : ' + str(most_expensive_item))
    # print('price of most expensive item : ' + str(max_price))
    # print('number of times where the most expensive item appears in training set : ' + str(
    #     train_df.loc[train_df['item_id'] == most_expensive_item].count()[0]))
    # print('most expensive item appears in test set : ' + str(most_expensive_item in test_df['item_id'].values))
    # # train_df.drop(train_df['item_price'].idxmax(), axis=0, inplace=True)
    del max_price, most_expensive_item

    # ??????????????????????????????????????????????????????????????????????????????
    # Drop the item with negative price (there is only one, probably missing value)
    # OR: Set to the median price
    # ??????????????????????????????????????????????????????????????????????????????
    # print('number of items with negative prices : ' + str((train_df['item_price'] < 0).sum()))
    train_df.drop(train_df['item_price'].idxmin(), axis=0, inplace=True)
    # print('number of items with negative prices : ' + str((train_df['item_price'] < 0).sum()))
    gc.collect()

    # print(train_df.head())
    train_df = train_df.iloc[:, 0:7]
    # print(train_df.head())

    # downcast dtypes for all dataframes
    downcast_dtypes(train_df)
    train_df['month_id'] = train_df['month_id'].astype(np.int8)
    train_df['shop_id'] = train_df['shop_id'].astype(np.int8)

    downcast_dtypes(shops)
    shops['shop_id'] = shops['shop_id'].astype(np.int8)

    downcast_dtypes(items)
    items['item_category_id'] = items['item_category_id'].astype(np.int8)

    downcast_dtypes(item_categories)
    item_categories['item_category_id'] = item_categories['item_category_id'].astype(np.int8)

    # Update the test set
    test_df['month_id'] = 34
    test_df['month_id'] = test_df['month_id'].astype(np.int8)
    test_df['shop_id'] = test_df['shop_id'].astype(np.int8)
    test_df['item_id'] = test_df['item_id'].astype(np.int16)

    train_df.to_pickle('../data/cleaned/train.pkl')
    shops.to_pickle('../data/cleaned/shops.pkl')
    items.to_pickle('../data/cleaned/items.pkl')
    item_categories.to_pickle('../data/cleaned/item_categories.pkl')
    test_df.to_pickle('../data/cleaned/test.pkl')


def lag_feature(df, lags, col):
    temp = df[['month_id', 'shop_id', 'item_id', col]]
    for i in lags:
        shifted = temp.copy()
        shifted.columns = ['month_id', 'shop_id', 'item_id', col + '_lag_' + str(i)]
        shifted['month_id'] += i
        df = pd.merge(df, shifted, on=['month_id', 'shop_id', 'item_id'], how='left')
    return df


def select_trend(row):
    lags = [1, 2, 3, 4, 5, 6]
    for i in lags:
        if row['delta_price_lag_' + str(i)]:
            return row['delta_price_lag_' + str(i)]
    return 0


def fill_na(df):
    for col in df.columns:
        if('_lag_' in col) & (df[col].isnull().any()):
            if 'item_cnt' in col:
                df[col].fillna(0, inplace=True)
            elif 'item_quantity' in col:
                df[col].fillna(0, inplace=True)
    return df


def feature_engineering():
    # Import the data
    train_df = pd.read_pickle('../data/cleaned/train.pkl')
    shops = pd.read_pickle('../data/cleaned/shops.pkl')
    items = pd.read_pickle('../data/cleaned/items.pkl')
    item_categories = pd.read_pickle('../data/cleaned/item_categories.pkl')
    test_df = pd.read_pickle('../data/cleaned/test.pkl')

    # Drop the day identifier in the training set
    train_df.drop('day_id', axis=1, inplace=True)

    # Aggregate the sales data by month, shop, and item
    col_agg = ['month_id', 'shop_id', 'item_id']
    train_agg = train_df.groupby(col_agg).agg('sum').drop(['item_price'], axis=1).reset_index()
    train_agg['month_id'] = train_agg['month_id'].astype(np.int8)
    train_agg['shop_id'] = train_agg['shop_id'].astype(np.int8)
    train_agg['item_id'] = train_agg['item_id'].astype(np.int16)

    # ??????????????????????????????????????????????
    # Clip the item_quantity to be between 0 and 20
    # Do we want to do this now? Use 20?
    # ??????????????????????????????????????????????
    train_agg['item_quantity'].clip(0, 20, inplace=True)
    train_agg['item_quantity'] = train_agg['item_quantity'].astype(np.float16)

    # Calculate monthly sales for each shop and item in a given month
    train_X = []
    for i in range(34):
        sales = train_df[train_df['month_id'] == i]
        train_X.append(np.array(list(product([i], sales['shop_id'].unique(), sales['item_id'].unique())), dtype='int16'))

    # ??????????????????????????????????????
    # Do we want to just use these columns ?
    # ??????????????????????????????????????
    test_df = test_df[col_agg]

    train_X = pd.DataFrame(np.vstack(train_X), columns=col_agg)
    train_X = pd.concat([train_X, test_df], ignore_index=True)
    train_X['month_id'] = train_X['month_id'].astype(np.int8)
    train_X['shop_id'] = train_X['shop_id'].astype(np.int8)
    train_X['item_id'] = train_X['item_id'].astype(np.int16)
    train_X.sort_values(by=col_agg, inplace=True)

    # Add aggregated sales by (month, shop, item) to the full DataFrame
    train_X = train_X.join(train_agg.set_index(col_agg), on=col_agg)

    # Fill missing values for the item_quantity
    train_X['item_quantity'].fillna(0, inplace=True)

    # Calculate a revenue column
    train_df['revenue'] = train_df['item_price'] * train_df['item_quantity']

    # Add shops, items, item_cats to train_X
    train_X = pd.merge(train_X, shops, on=['shop_id'], how='left')
    train_X = pd.merge(train_X, items, on=['item_id'], how='left')
    train_X = pd.merge(train_X, item_categories, on=['item_category_id'], how='left')
    train_X['city_id'] = train_X['city_id'].astype(np.int8)
    train_X['item_supercategory_id'] = train_X['item_supercategory_id'].astype(np.int8)
    train_X['item_category_console_id'] = train_X['item_category_console_id'].astype(np.int8)
    train_X.drop(['shop_name', 'city', 'item_name', 'item_category_name', 'item_supercategory_name',
                  'item_category_console'], axis=1, inplace=True)

    # Target Lags
    train_X = lag_feature(train_X, [1, 2, 3, 6, 12], 'item_quantity')

    # Mean encoding
    # By date
    group = train_X.groupby(['month_id']).agg({'item_quantity': ['mean']})
    group.columns = ['date_avg_item_cnt']
    group.reset_index(inplace=True)

    train_X = pd.merge(train_X, group, on=['month_id'], how='left')
    train_X['date_avg_item_cnt'] = train_X['date_avg_item_cnt'].astype(np.float16)
    train_X = lag_feature(train_X, [1], 'date_avg_item_cnt')
    train_X.drop(['date_avg_item_cnt'], axis=1, inplace=True)

    # By date and item
    group = train_X.groupby(['month_id', 'item_id']).agg({'item_quantity': ['mean']})
    group.columns = ['date_item_avg_item_cnt']
    group.reset_index(inplace=True)

    train_X = pd.merge(train_X, group, on=['month_id', 'item_id'], how='left')
    train_X['date_item_avg_item_cnt'] = train_X['date_item_avg_item_cnt'].astype(np.float16)
    train_X = lag_feature(train_X, [1, 2, 3, 6, 12], 'date_item_avg_item_cnt')
    train_X.drop(['date_item_avg_item_cnt'], axis=1, inplace=True)

    # By date and shop
    group = train_X.groupby(['month_id', 'shop_id']).agg({'item_quantity': ['mean']})
    group.columns = ['date_shop_avg_item_cnt']
    group.reset_index(inplace=True)

    train_X = pd.merge(train_X, group, on=['month_id', 'shop_id'], how='left')
    train_X['date_shop_avg_item_cnt'] = train_X['date_shop_avg_item_cnt'].astype(np.float16)
    train_X = lag_feature(train_X, [1, 2, 3, 6, 12], 'date_shop_avg_item_cnt')
    train_X.drop(['date_shop_avg_item_cnt'], axis=1, inplace=True)

    # By date and item_category
    group = train_X.groupby(['month_id', 'item_category_id']).agg({'item_quantity': ['mean']})
    group.columns = ['date_category_avg_item_cnt']
    group.reset_index(inplace=True)

    train_X = pd.merge(train_X, group, on=['month_id', 'item_category_id'], how='left')
    train_X['date_category_avg_item_cnt'] = train_X['date_category_avg_item_cnt'].astype(np.float16)
    train_X = lag_feature(train_X, [1], 'date_category_avg_item_cnt')
    train_X.drop(['date_category_avg_item_cnt'], axis=1, inplace=True)

    # By date, shop, and item_category
    group = train_X.groupby(['month_id', 'shop_id', 'item_category_id']).agg({'item_quantity': ['mean']})
    group.columns = ['date_shop_category_avg_item_cnt']
    group.reset_index(inplace=True)

    train_X = pd.merge(train_X, group, on=['month_id', 'shop_id', 'item_category_id'], how='left')
    train_X['date_shop_category_avg_item_cnt'] = train_X['date_shop_category_avg_item_cnt'].astype(np.float16)
    train_X = lag_feature(train_X, [1], 'date_shop_category_avg_item_cnt')
    train_X.drop(['date_shop_category_avg_item_cnt'], axis=1, inplace=True)

    # By date, shop, and item_category_console
    group = train_X.groupby(['month_id', 'shop_id', 'item_category_console_id']).agg({'item_quantity': ['mean']})
    group.columns = ['date_shop_console_avg_item_cnt']
    group.reset_index(inplace=True)

    train_X = pd.merge(train_X, group, on=['month_id', 'shop_id', 'item_category_console_id'], how='left')
    train_X['date_shop_console_avg_item_cnt'] = train_X['date_shop_console_avg_item_cnt'].astype(np.float16)
    train_X = lag_feature(train_X, [1], 'date_shop_console_avg_item_cnt')
    train_X.drop(['date_shop_console_avg_item_cnt'], axis=1, inplace=True)

    # By date, shop, supercategory
    group = train_X.groupby(['month_id', 'shop_id', 'item_supercategory_id']).agg({'item_quantity': ['mean']})
    group.columns = ['date_shop_super_avg_item_cnt']
    group.reset_index(inplace=True)

    train_X = pd.merge(train_X, group, on=['month_id', 'shop_id', 'item_supercategory_id'], how='left')
    train_X['date_shop_super_avg_item_cnt'] = train_X['date_shop_super_avg_item_cnt'].astype(np.float16)
    train_X = lag_feature(train_X, [1], 'date_shop_super_avg_item_cnt')
    train_X.drop(['date_shop_super_avg_item_cnt'], axis=1, inplace=True)

    # By date and city
    group = train_X.groupby(['month_id', 'city_id']).agg({'item_quantity': ['mean']})
    group.columns = ['date_city_avg_item_cnt']
    group.reset_index(inplace=True)

    train_X = pd.merge(train_X, group, on=['month_id', 'city_id'], how='left')
    train_X['date_city_avg_item_cnt'] = train_X['date_city_avg_item_cnt'].astype(np.float16)
    train_X = lag_feature(train_X, [1], 'date_city_avg_item_cnt')
    train_X.drop(['date_city_avg_item_cnt'], axis=1, inplace=True)

    # By date, item, and city
    group = train_X.groupby(['month_id', 'item_id', 'city_id']).agg({'item_quantity': ['mean']})
    group.columns = ['date_item_city_avg_item_cnt']
    group.reset_index(inplace=True)

    train_X = pd.merge(train_X, group, on=['month_id', 'item_id', 'city_id'], how='left')
    train_X['date_item_city_avg_item_cnt'] = train_X['date_item_city_avg_item_cnt'].astype(np.float16)
    train_X = lag_feature(train_X, [1], 'date_item_city_avg_item_cnt')
    train_X.drop(['date_item_city_avg_item_cnt'], axis=1, inplace=True)

    # By date and item_category_console
    group = train_X.groupby(['month_id', 'item_category_console_id']).agg({'item_quantity': ['mean']})
    group.columns = ['date_console_avg_item_cnt']
    group.reset_index(inplace=True)

    train_X = pd.merge(train_X, group, on=['month_id', 'item_category_console_id'], how='left')
    train_X['date_console_avg_item_cnt'] = train_X['date_console_avg_item_cnt'].astype(np.float16)
    train_X = lag_feature(train_X, [1], 'date_console_avg_item_cnt')
    train_X.drop(['date_console_avg_item_cnt'], axis=1, inplace=True)

    # By date and supercategpry
    group = train_X.groupby(['month_id', 'item_supercategory_id']).agg({'item_quantity': ['mean']})
    group.columns = ['date_super_avg_item_cnt']
    group.reset_index(inplace=True)

    train_X = pd.merge(train_X, group, on=['month_id', 'item_supercategory_id'], how='left')
    train_X['date_super_avg_item_cnt'] = train_X['date_super_avg_item_cnt'].astype(np.float16)
    train_X = lag_feature(train_X, [1], 'date_super_avg_item_cnt')
    train_X.drop(['date_super_avg_item_cnt'], axis=1, inplace=True)

    # print(train_X.info(null_counts=True))

    # Price trend for the last 6 months
    # Average price for each item
    group = train_df.groupby(['item_id']).agg({'item_price': ['mean']})
    group.columns = ['item_avg_item_price']
    group.reset_index(inplace=True)
    train_X = pd.merge(train_X, group, on=['item_id'], how='left')
    train_X['item_avg_item_price'] = train_X['item_avg_item_price'].astype(np.float16)
    
    # Average price for each item during each month
    group = train_df.groupby(['month_id', 'item_id']).agg({'item_price': ['mean']})
    group.columns = ['date_item_avg_item_price']
    group.reset_index(inplace=True)
    train_X = pd.merge(train_X, group, on=['month_id', 'item_id'], how='left')
    train_X['date_item_avg_item_price'] = train_X['date_item_avg_item_price'].astype(np.float16)
    
    # Lags for the average price for each item in each month
    lags = [1, 2, 3, 4, 5, 6]
    train_X = lag_feature(train_X, lags, 'date_item_avg_item_price')
    
    print(1)
    
    for i in lags:
        train_X['delta_price_lag_' + str(i)] = (train_X['date_item_avg_item_price_lag_' + str(i)] -
                                                train_X['item_avg_item_price']) / train_X['item_avg_item_price']
    
    print(2)
    
    train_X['delta_price_lag'] = train_X.apply(select_trend, axis=1)
    train_X['delta_price_lag'] = train_X['delta_price_lag'].astype(np.float16)
    train_X['delta_price_lag'].fillna(0, inplace=True)
    
    print(3)
    
    features_to_drop = ['item_avg_item_price', 'date_item_avg_item_price']
    for i in lags:
        features_to_drop += ['date_item_avg_item_price_lag_' + str(i)]
        features_to_drop += ['delta_price_lag_' + str(i)]
    train_X.drop(features_to_drop, axis=1, inplace=True)
    
    print(4)

    # Shop revenue trend for the last month
    group = train_df.groupby(['month_id', 'shop_id']).agg({'revenue': ['sum']})
    group.columns = ['date_shop_revenue']
    group.reset_index(inplace=True)
    train_X = pd.merge(train_X, group, on=['month_id', 'shop_id'], how='left')
    train_X['date_shop_revenue'] = train_X['date_shop_revenue'].astype(np.float32)

    group = group.groupby(['shop_id']).agg({'date_shop_revenue': ['mean']})
    group.columns = ['shop_avg_revenue']
    group.reset_index(inplace=True)
    train_X = pd.merge(train_X, group, on=['shop_id'], how='left')
    train_X['shop_avg_revenue'] = train_X['shop_avg_revenue'].astype(np.float32)

    train_X['delta_revenue'] = (train_X['date_shop_revenue'] - train_X['shop_avg_revenue']) / train_X['shop_avg_revenue']
    train_X['delta_revenue'] = train_X['delta_revenue'].astype(np.float16)

    train_X = lag_feature(train_X, [1], 'delta_revenue')

    train_X.drop(['date_shop_revenue', 'shop_avg_revenue', 'delta_revenue'], axis=1, inplace=True)

    # print(train_X.info(null_counts=True))

    # Month based features
    train_X['month'] = train_X['month_id'] % 12
    days = pd.Series([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
    train_X['num_days'] = train_X['month'].map(days).astype(np.int8)

    # Last sales for each item and shop pair
    temp_list = []
    for mid in train_X['month_id'].unique():
        temp = train_agg.loc[train_agg['month_id'] < mid, ['month_id', 'shop_id', 'item_id']]. \
            groupby(['shop_id', 'item_id']).last().rename({'month_id': 'item_month_id_of_last_sale_in_shop'}, axis=1). \
            astype(np.int16)
        temp['month_id'] = mid
        temp.reset_index(inplace=True)
        temp_list.append(temp)

    temp = pd.concat(temp_list)
    del temp_list
    train_X = train_X.join(temp.set_index(['month_id', 'shop_id', 'item_id']), on=['month_id', 'shop_id', 'item_id'])
    del temp

    # Downcast dtype
    train_X['item_month_id_of_last_sale_in_shop'] = train_X['item_month_id_of_last_sale_in_shop'].astype(np.float16)

    # Last sale for each item individually
    temp_list = []
    for mid in train_X['month_id'].unique():
        temp = train_agg.loc[train_agg['month_id'] < mid, ['month_id', 'item_id']].groupby('item_id').last().rename(
            {'month_id': 'item_month_id_of_last_sale'}, axis=1)
        temp['month_id'] = mid
        temp.reset_index(inplace=True)
        temp_list.append(temp)

    temp = pd.concat(temp_list)
    del temp_list
    train_X = train_X.join(temp.set_index(['month_id', 'item_id']), on=['month_id', 'item_id'])
    del temp

    # Downcast dtype
    train_X['item_month_id_of_last_sale'] = train_X['item_month_id_of_last_sale'].astype(np.float16)

    # First sales of items in the shop and individually
    train_X['item_shop_first_sale'] = train_X['month_id'] - train_X.groupby(['item_id', 'shop_id'])['month_id'].\
        transform('min')
    train_X['item_first_sale'] = train_X['month_id'] - train_X.groupby('item_id')['month_id'].transform('min')

    # Remove the first 12 months of data and fill NaN with 0
    train_X = train_X[train_X['month_id'] > 11]
    train_X = fill_na(train_X)

    # Export train_X
    train_X.to_pickle('../data/processed/train_X_limited.pkl')


def predict_xgboost():
    data = pd.read_pickle('../data/processed/train_X_limited.pkl')
    test = pd.read_pickle('../data/cleaned/test.pkl')
    # print(data.info(null_counts=True))
    # Use specific columns (dropped the ones dropped in the notebook or with NaN values)
    data = data[[
        'month_id',
        'shop_id',
        'item_id',
        'item_quantity',
        'city_id',
        'item_category_id',
        'item_category_is_digital',
        'item_supercategory_id',
        'item_category_console_id',
        'item_quantity_lag_1',
        'item_quantity_lag_2',
        'item_quantity_lag_3',
        'item_quantity_lag_6',
        'item_quantity_lag_12',
        'date_avg_item_cnt_lag_1',
        'date_item_avg_item_cnt_lag_1',
        'date_item_avg_item_cnt_lag_2',
        'date_item_avg_item_cnt_lag_3',
        'date_item_avg_item_cnt_lag_6',
        'date_item_avg_item_cnt_lag_12',
        'date_shop_avg_item_cnt_lag_1',
        'date_shop_avg_item_cnt_lag_2',
        'date_shop_avg_item_cnt_lag_3',
        'date_shop_avg_item_cnt_lag_6',
        'date_shop_avg_item_cnt_lag_12',
        'date_category_avg_item_cnt_lag_1',
        'date_shop_category_avg_item_cnt_lag_1',
        # 'date_shop_console_avg_item_cnt_lag_1',
        # 'date_shop_super_avg_item_cnt_lag_1',
        'date_city_avg_item_cnt_lag_1',
        'date_item_city_avg_item_cnt_lag_1',
        # 'date_console_avg_item_cnt_lag_1',
        # 'date_super_avg_item_cnt_lag_1',
        # 'delta_revenue_lag_1',
        'month',
        'num_days',
        # 'item_month_id_of_last_sale_in_shop',
        # 'item_month_id_of_last_sale',
        'item_shop_first_sale',
        'item_first_sale'
    ]]

    # Separate the data
    X_train = data[data['month_id'] < 33].drop(['item_quantity'], axis=1)
    Y_train = data[data['month_id'] < 33]['item_quantity']
    X_val = data[data['month_id'] == 33].drop(['item_quantity'], axis=1)
    Y_val = data[data['month_id'] == 33]['item_quantity']
    X_test = data[data['month_id'] == 34].drop(['item_quantity'], axis=1)

    model = XGBRegressor(
        max_depth=10,
        n_estimators=1000,
        min_child_weight=0.5,
        colsample_bytree=0.8,
        subsample=0.8,
        eta=0.1,
        tree_method='gpu_hist',
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

    Y_pred = model.predict(X_val).clip(0, 20)
    Y_test = model.predict(X_test).clip(0, 20)

    submission = pd.DataFrame({'ID': test.index, 'item_cnt_month': Y_test})
    submission.to_csv('./xgb_submission.csv', index=False)

    # print(submission.info(null_counts=True))
    # print(submission.describe())

    # Save predictions for an ensamble
    # pickle.dump(Y_pred, open('../output/xgboost/xgb_train.pickle', 'wb'))
    # pickle.dump(Y_test, open('../output/xgboost/xgb_test.pickle', 'wb'))


def main():
    # Clean all data
    clean_data()
    gc.collect()

    # Feature Engineering
    feature_engineering()
    gc.collect()

    # xgboost
    predict_xgboost()
    gc.collect()


if __name__ == '__main__':
    main()
