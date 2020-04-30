# Basic packages
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import random as rd # generating random numbers
from datetime import datetime # manipulating date formats
from scipy.optimize import minimize
#Normalizing the data
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
# Keras
from keras import regularizers
from keras.layers import Input, Dense
from keras.models import Model, Sequential
import pdb # python debugger

# Viz
import matplotlib.pyplot as plt # basic plotting
#import seaborn as sns # for prettier plots
# Google Translate package
#from mtranslate import translate


#-----------------------------------FUNCTIONS-------------------------------------------
def downcast_dtypes(df):
    float_cols = [c for c in df if df[c].dtype == "float64"]
    int_cols = [c for c in df if df[c].dtype in ["int64", "int32"]]
    df[float_cols] = df[float_cols].astype(np.float32)
    df[int_cols] = df[int_cols].astype(np.int16)
    return df

def data_processing():
    # Import all available csvs
    sales=pd.read_csv("../data/sales_train.csv")
    sales_Coll = pd.read_csv("../data/sales_train_en_complete.csv")
    item_cat=pd.read_csv("../data/item_cat_en.csv")
    item=pd.read_csv("../data/item_en.csv")
    output_sample=pd.read_csv("../data/sample_submission.csv")
    shops=pd.read_csv("../data/shops_en.csv")
    final_test=pd.read_csv("../data/test.csv")

    sales_Coll = downcast_dtypes(sales_Coll)
    sales_Coll = sales_Coll[sales_Coll.item_price > 0] #Testing
    #sales_Coll_grouped=sales_Coll.groupby(["date_block_num","shop_id","item_id"])[
    #    "item_price","item_cnt_day"].agg({"item_price":"mean","item_cnt_day":"sum"})

    # Take input for dataset 
    print("------------------------------------------------------------------------")
    print("Remember we have data of months 0 (Jan 2013) --> 33 (Oct 2015).\n")
    train_val_mnth_intrvl = 33#int(input("Number of months to use as training (80%) and validation (20%). (Starting from 0, E.g. 11 would be month 0 --> 11 which is Jan 2013 --> Dec 2013 as training): "))
    test_mnth = 33#int(input("\nMonth to predict. (E.g. 14 would be March of 2014): "))

    #Seperate and group dataset
    print("Generating subset files from complete dataset in current directory.")
    sales_Coll[sales_Coll["date_block_num"]<=train_val_mnth_intrvl].groupby(["date_block_num","item_category_id","shop_id","item_id"])[
        "item_price","item_cnt_day"].agg({"item_price":"mean","item_cnt_day":"sum"}).rename(columns={"item_cnt_day": "item_cnt_month"}).to_csv("./lin_reg_train_val.csv")

    sales_Coll[sales_Coll["date_block_num"]==test_mnth].groupby(["date_block_num","item_category_id","shop_id","item_id"])[
        "item_price","item_cnt_day"].agg({"item_price":"mean","item_cnt_day":"sum"}).rename(columns={"item_cnt_day": "item_cnt_month"}).to_csv("./lin_reg_test.csv")

    sales_Coll.groupby(["date_block_num","item_category_id","shop_id","item_id"])[
        "item_price","item_cnt_day"].agg({"item_price":"mean","item_cnt_day":"sum"}).rename(
            columns={"item_cnt_day": "item_cnt_month"}).groupby(["shop_id","item_id"])[
        "item_price","item_cnt_month"].agg({"item_price":"mean","item_cnt_month":"mean"}).rename(
            columns={"item_price": "item_price_mean"}).to_csv("./lin_reg_final_test.csv")

    train_val_df=downcast_dtypes(pd.read_csv("./lin_reg_train_val.csv"))
    test_df=downcast_dtypes(pd.read_csv("./lin_reg_test.csv"))
    item_mean_df = downcast_dtypes(pd.read_csv("./lin_reg_final_test.csv"))
    #train_val_df = train_val_df.query('item_cnt_month >= 0 and item_cnt_month <= 20 and item_price < 100000') #Testing
    # = train_val_df.reset_index().drop(columns='index') #Testing

    #Merge item mean prices to our final test dataframe
    final_test_df = pd.merge(final_test, item_mean_df, how='left', on=['shop_id','item_id'])
    #Merge item category id in to final test dataframe
    final_test_df = pd.merge(final_test_df, item, how='left', on=['item_id']).drop(columns=['item_name'])

    #Calculate item category mean price
    item_cat_mean_price_df=sales_Coll.groupby(["item_category_id","item_id"])["item_price"].agg({
        "item_price":"mean"}).rename(columns={"item_price": "item_cat_price_mean"}).groupby(['item_category_id'])['item_cat_price_mean'].agg('mean')
    
    #Merge item category mean price in to final test dataframe
    final_test_df = pd.merge(final_test_df, item_cat_mean_price_df, how='left', on=['item_category_id'])

    #Copy in the total mean over all months for items that we do not have a price for (never seen before in training)
    final_test_df.item_price_mean = np.where(final_test_df.item_price_mean.isnull(), final_test_df.item_cat_price_mean, final_test_df.item_price_mean)

    #Drop unneccesary columns for our linear regressor's features
    final_test_df = final_test_df.drop(columns=['ID','item_cnt_month','item_cat_price_mean'])
    #Swap columns to the order of features train data is using
    cols = final_test_df.columns.tolist()
    cols = cols[-1:] + cols[:-1]
    final_test_df = final_test_df[cols]

    #Insert Nov (e.g 2013) month column
    final_test_df.insert(0, "date_block_num", "34")

    #Convert categories into a non-ordered set
    num_categories = int(item_cat.describe().iloc[7].values[0]+1) # Testing
    #zeroes = np.zeroes(num_categories) # Testing
    #zeroes_temp = zeroes
    #int(final_test_df.iloc[1].item_category_id) # Testing
    #final_test_df['item_category_id'] = final_test_df['item_category_id'].astype(object) # Testing
    
    #zeroes_temp = zeroes_temp[int(final_test_df.iloc[1].item_category_id)] # Testing
    #final_test_df['item_category_id'] = final_test_df['item_category_id'].apply(lambda x: np.put(np.zeros(84),int(x),int(x))) # Testing
    #['item_category_id'] = final_test_df['item_category_id'].apply(lambda x: int(x)) # Testing

    #train_val_df[train_val_df['date_block_num']==0] #will give all sales for January
    #train_val_df.loc[train_val_df['item_id']==57]
    #train_val_df.iloc[[1888]] # will return one row dataframe of 1888th row


    # final_test_df['item_category_id']=final_test_df['item_category_id'].apply(lambda x: np.zeros(x) if x != 0 else np.zeros(1)) # Testing
    # final_test_df['item_category_id']=final_test_df['item_category_id'].apply(lambda x: np.append(x,np.zeros(num_categories-len(x)))) # Testing

    # train_val_df['item_category_id']=train_val_df['item_category_id'].apply(lambda x: np.zeros(x) if x != 0 else np.zeros(1)) # Testing
    # train_val_df['item_category_id']=train_val_df['item_category_id'].apply(lambda x: np.append(x,np.zeros(num_categories-len(x)))) # Testing

    # test_df['item_category_id']=test_df['item_category_id'].apply(lambda x: np.zeros(x) if x != 0 else np.zeros(1)) # Testing
    # test_df['item_category_id']=test_df['item_category_id'].apply(lambda x: np.append(x,np.zeros(num_categories-len(x)))) # Testing

    #Randomly remove 20% of data to create validation dataset (and rest is training)
    rd.seed(0)
    train_val_size = train_val_df.shape[0]
    validation_size = int(0.2*train_val_size)
    #Generate random ints from 0 index to len train_val_df 
    val_rows = rd.sample(range(train_val_size), validation_size)
    #Create empty df to store new validation set
    val_df = pd.DataFrame()
    #Fill val_df with the random rows and drop them from train_val_df saving the result in new train_df
    val_df = val_df.append(train_val_df.iloc[[*val_rows]])
    train_df = train_val_df.drop([*val_rows])
    #Delete (unwanted memory usage)
    del train_val_df
    Xtrain = train_df.loc[:,['date_block_num','item_category_id','shop_id','item_id','item_price']].to_numpy(dtype='float16')
    #Xtrain = np.hstack(np.hstack(Xtrain)).reshape((Xtrain.shape[0],Xtrain.shape[1]+num_categories-1)).astype(dtype='float16').astype(dtype='int16')
    #sc = StandardScaler()
    #Xtrain = sc.fit_transform(Xtrain)
    ytrain = train_df.loc[:,['item_cnt_month']].to_numpy(dtype='float16')

    Xval = val_df.loc[:,['date_block_num','item_category_id','shop_id','item_id','item_price']].to_numpy(dtype='float16')
    #Xval = np.hstack(np.hstack(Xval)).reshape((Xval.shape[0],Xval.shape[1]+num_categories-1)).astype(dtype='float16').astype(dtype='int16')
    yval = val_df.loc[:,['item_cnt_month']].to_numpy(dtype='float16')

    Xtest = test_df.loc[:,['date_block_num','item_category_id','shop_id','item_id','item_price']].to_numpy(dtype='float16')
    #Xtest = np.hstack(np.hstack(Xtest)).reshape((Xtest.shape[0],Xtest.shape[1]+num_categories-1)).astype(dtype='float16').astype(dtype='int16')
    ytest = test_df.loc[:,['item_cnt_month']].to_numpy(dtype='float16')

    Xfinaltest = final_test_df.to_numpy(dtype='float16')
    #Xfinaltest = np.hstack(np.hstack(Xfinaltest)).reshape((Xfinaltest.shape[0],Xfinaltest.shape[1]+num_categories-1)).astype(dtype='float16').astype(dtype='int16')

    return Xtrain, ytrain, Xval, yval, Xtest, ytest, Xfinaltest

#-----------------------------------Main-------------------------------------------
def main():
    #pdb.set_trace()
    Xtrain, ytrain, Xval, yval, Xtest, ytest, Xfinaltest = data_processing()

    bias_coef = 0.1
    kernel_coef = 0.1
    activity_coef = 0.1

    # Activations: linear, exponential, hard_sigmoid, sigmoid, tanh, relu, softsign, softplus, softmax, elu
    model = Sequential()
    model.add(Dense(20, input_dim=5, activation='relu', use_bias=True, kernel_regularizer=regularizers.l2(kernel_coef),
                activity_regularizer=regularizers.l1(activity_coef)))
    model.add(Dense(10, activation='relu', use_bias=True, kernel_regularizer=regularizers.l2(kernel_coef),
                activity_regularizer=regularizers.l1(activity_coef)))
    model.add(Dense(10, activation='relu', use_bias=True, kernel_regularizer=regularizers.l2(kernel_coef),
                activity_regularizer=regularizers.l1(activity_coef)))
    model.add(Dense(2, activation='tanh', use_bias=True, kernel_initializer='glorot_uniform', 
        bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, 
        activity_regularizer=None, kernel_constraint=None, bias_constraint=None))
    model.add(Dense(1))


    model.compile(loss='mean_squared_error', optimizer='adadelta', metrics=['accuracy'])


    # history = model.fit(Xtrain, ytrain, epochs=1, batch_size=64)


    # ypred = model.predict(Xtest)


    # #Converting predictions to label
    # pred = list()
    # for i in range(len(ypred)):
    #     pred.append(np.argmax(ypred[i]))
    # a = accuracy_score(pred,ytest)
    # print('Accuracy before training with validation is:', a*100)

    #model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
    history = model.fit(Xtrain, ytrain, validation_data = (Xval,yval), epochs=2, batch_size=64)

    pdb.set_trace()
    import matplotlib.pyplot as plt
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()


    ypred = model.predict(Xtest)
    pred = list()
    for i in range(len(ypred)):
        pred.append(np.argmax(ypred[i]))
    a = accuracy_score(pred,ytest)
    print('Accuracy after training with validation is:', a*100)

    yResult = model.predict(Xfinaltest)
    submission_df=pd.DataFrame(data=yResult,index=range(214200),columns=['item_cnt_month'])
    submission_df.index.name = 'ID'
    submission_df.to_csv('keras_nn_submission.csv')
    print(yResult)
if __name__ == "__main__":
    main()