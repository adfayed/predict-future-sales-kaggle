# Basic packages
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import random as rd # generating random numbers
from datetime import datetime # manipulating date formats
from scipy.optimize import minimize
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

def linear_regression_noreg(X, y):
    """
    Compute the weight parameter given X and y.
    Inputs:
    - X: A numpy array of shape (num_samples, D) containing feature.
    - y: A numpy array of shape (num_samples, ) containing label
    Returns:
    - w: a numpy array of shape (D, )
    """
    #####################################################
    #                YOUR CODE HERE                     #
    #####################################################
    w = np.linalg.inv(X.T.dot(X)).dot((X.T)).dot(y)
    return w

# Objective function
def objective(b, X, y, lambd):
	obj_val = (np.linalg.norm(X.dot(b) - y))**2 + lambd*(np.linalg.norm(b)**2)
	return obj_val

def createGenerator(my_list):
    for train_sample in my_list:
        yield train_sample

def regularized_linear_regression(w0, X, y, lambd):
    """
        Compute the weight parameter given X, y and lambda.
        Inputs:
        - X: A numpy array of shape (num_samples, D) containing feature.
        - y: A numpy array of shape (num_samples, ) containing label
        - lambd: a float number containing regularization strength
        Returns:
        - w: a numpy array of shape (D, )
        """   
    iterator = lambda x: next(x)
    solution = minimize(objective, w0, method='nelder-mead', options={'xatol': 1e-8, 'disp': True, 'maxfev': 999999999}, args= (iterator(X), iterator(y), lambd))
    w = solution.x

    #####################################################
    #                YOUR CODE HERE                     #
    #####################################################
    return w

def tune_lambda(w0, Xtrain, ytrain, Xval, yval, lambds):
    """
        Find the best lambda value.
        Inputs:
        - Xtrain: A numpy array of shape (num_training_samples, D) containing training feature.
        - ytrain: A numpy array of shape (num_training_samples, ) containing training label
        - Xval: A numpy array of shape (num_val_samples, D) containing validation feature.
        - yval: A numpy array of shape (num_val_samples, ) containing validation label
        - lambds: a list of lambdas
        Returns:
        - bestlambda: the best lambda you find in lambds
        """
    mse_min = float('inf')
    iterator = lambda x: next(x)
    for lambd in lambds:
        mygenerator_Xtrain = createGenerator(Xtrain)
        mygenerator_ytrain = createGenerator(ytrain)
        mygenerator_Xval = createGenerator(Xval)
        mygenerator_yval = createGenerator(yval)

        w = regularized_linear_regression(w0, mygenerator_Xtrain, mygenerator_ytrain, lambd)
        mse = test_error(w, mygenerator_Xval, mygenerator_yval)
        if mse < mse_min:
            mse_min = mse
            bestlambda = lambd
    #####################################################
    #                YOUR CODE HERE                     #
    #####################################################
    return bestlambda

def test_error(w, X, y):
    """
        Compute the mean squre error on test set given X, y, and model parameter w.
        Inputs:
        - X: A numpy array of shape (num_samples, D) containing test feature.
        - y: A numpy array of shape (num_samples, ) containing test label
        - w: a numpy array of shape (D, )
        Returns:
        - err: the mean square error
        """
    rss = 0
    count = 0
    for i in X:
        count = count + 1
        tempy = next(y)
        rss = (tempy - i.dot(w)).T.dot(tempy - i.dot(w)) + rss

    err = rss/count
    return err

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

    #sales_Coll_grouped=sales_Coll.groupby(["date_block_num","shop_id","item_id"])[
    #    "item_price","item_cnt_day"].agg({"item_price":"mean","item_cnt_day":"sum"})

    # Take input for dataset size
    print("Remember we have data of months 0 (Jan 2013) --> 33 (Oct 2015).")
    train_val_mnth_intrvl = int(input("Number of months to use as training (80%) and validation (20%). (Starting from 0, E.g. 11 would be month 0 --> 11 which is Jan 2013 --> Dec 2013 as training): "))
    test_mnth = int(input("Month to predict. (E.g. 14 would be March of 2014): "))

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

    #train_val_df[train_val_df['date_block_num']==0] #will give all sales for January
    #train_val_df.loc[train_val_df['item_id']==57]
    #train_val_df.iloc[[1888]] # will return one row dataframe of 1888th row


    #Randomly remove 20% of data to create validation dataset (and rest is training)
    rd.seed(0)
    train_val_size = train_val_df.shape[0]
    validation_size = int(0.2*train_val_size)
    #Generate random ints from 0 index to len train_val_df index
    val_rows = rd.sample(range(train_val_size), validation_size)
    #Create empty df to store new validation set
    val_df = pd.DataFrame()
    #Fill val_df with the random rows and drop them from train_val_df saving the result in new train_df
    val_df = val_df.append(train_val_df.iloc[[*val_rows]])
    train_df = train_val_df.drop([*val_rows])
    #Delete (unwanted memory usage)
    del train_val_df
    Xtrain = train_df.loc[:,['date_block_num','item_category_id','shop_id','item_id','item_price']].to_numpy(dtype='int16')
    ytrain = train_df.loc[:,['item_cnt_month']].to_numpy(dtype='int16')

    Xval = val_df.loc[:,['date_block_num','item_category_id','shop_id','item_id','item_price']].to_numpy(dtype='int16')
    yval = val_df.loc[:,['item_cnt_month']].to_numpy(dtype='int16')

    Xtest = test_df.loc[:,['date_block_num','item_category_id','shop_id','item_id','item_price']].to_numpy(dtype='int16')
    ytest = test_df.loc[:,['item_cnt_month']].to_numpy(dtype='int16')

    Xfinaltest = final_test_df.to_numpy(dtype='float16')

    return Xtrain, ytrain, Xval, yval, Xtest, ytest, Xfinaltest

#-----------------------------------Main-------------------------------------------
    
#Pass its features into objective function
#Tune lambda

#Set date_block_num as the index
#train_df.set_index("date_block_num",inplace=True)
#val_df.set_index("date_block_num",inplace=True)
#test_df.set_index("date_block_num",inplace=True)
#train_df.loc[0] #will give all sales for January
#train_df.loc[[0, 5]] #will give all sales for January and June
#train_df.loc[[0, 5], ['item_price','item_id']] #will return only item_price and item_id dataframe (remove one bracket '[]'' for series) for sales in January and June
def main():
    #Xtrain, ytrain, Xval, yval, Xtest, ytest = data_processing()
    Xtrain, ytrain, Xval, yval, Xtest, ytest, Xfinaltest = data_processing()

    mygenerator_Xtrain = createGenerator(Xtrain)
    mygenerator_ytrain = createGenerator(ytrain)
    mygenerator_Xval = createGenerator(Xval)
    mygenerator_yval = createGenerator(yval)
    mygenerator_Xtest = createGenerator(Xtest)
    mygenerator_ytest = createGenerator(ytest)
    iterator = lambda x: next(x)
    # =========================Q 1.1 linear_regression=================================
    w = linear_regression_noreg(Xtrain, ytrain)
    print("======== Question 1.1 Linear Regression ========")
    print("dimensionality of the model parameter is ", len(w), ".", sep="")
    print("model parameter is ", np.array_str(w))

    # =========================Q 1.2 regularized linear_regression=====================
    lambd = 5.0
    wl = regularized_linear_regression(w, mygenerator_Xtrain, mygenerator_ytrain, lambd)
    print("\n")
    print("======== Question 1.2 Regularized Linear Regression ========")
    print("dimensionality of the model parameter is ", len(wl), sep="")
    print("lambda = ", lambd, ", model parameter is ", np.array_str(wl), sep="")

    mygenerator_Xtrain2 = createGenerator(Xtrain)
    mygenerator_ytrain2 = createGenerator(ytrain)

    # =========================Q 1.3 tuning lambda======================
    lambds = [0, 10 ** -4, 10 ** -3, 10 ** -2, 10 ** -1, 1, 10, 10 ** 2]
    bestlambd = tune_lambda(w, Xtrain, ytrain, Xval, yval, lambds)
    print("\n")
    print("======== Question 1.3 tuning lambdas ========")
    print("tuning lambda, the best lambda =  ", bestlambd, sep="")

    mygenerator_Xtrain3 = createGenerator(Xtrain)
    mygenerator_ytrain3 = createGenerator(ytrain)

    # =========================Q 1.4 report mse on test ======================
    wbest = regularized_linear_regression(w, mygenerator_Xtrain3, mygenerator_ytrain3, bestlambd)
    mse = test_error(wbest, mygenerator_Xtest, mygenerator_ytest)
    print("\n")
    print("======== Question 1.4 report MSE ========")
    print("MSE on test is %.3f" % mse)
    

    
    yResult = Xfinaltest.dot(wbest)
    submission_df=pd.DataFrame(data=yResult,index=range(214200),columns=['item_cnt_month'])
    submission_df.index.name = 'ID'
    submission_df.to_csv('lin_reg_submission.csv')

if __name__ == "__main__":
    main()