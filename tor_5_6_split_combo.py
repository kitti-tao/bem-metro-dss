# import all necessary packages
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.ensemble import GradientBoostingRegressor
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression, Ridge, HuberRegressor
from google.cloud import bigquery
from datetime import datetime
from datetime import timezone
import pytz

'''
    This is the Machine Learning (ML) for predicting ridership. There are 4
    models: Ordinary Least Squared (OLS), Gradient Boosting Regressor (GBR),
    Ridge Regression (Rid), and Huber Regression (Hub). There are also two
    based line algorithms which are Weighted Moving Average (WMA) with
    window size of 3 and 6 (months).
    
    This code is trigger by Cloud Scheduler which can be set as daily, monthly,
    quarterly, etc. This can be set through xxxxxxxxxxx.
    
    Code explaination:
        1. When Cloud Scheduler is activated, function 'tor52()', main function
           is called.
        2. Import all necessary packages such as 'pandas', 'numpy', etc.
        3. Call 'bq_to_dataframe()' function to query all necessary data from 
           BigQuery for prediction and convert to dataframe.
        4. Parsing input dataframe and window size to 'wma()' for WMA 
           calculation.
        5. Call 'traintest()' function to split train and test dataframe.
        6. Call 'gettest()' function to get dataframe for predicting future
        7. Call 'modelpredict()' function for ML prediction.
        8. Call 'result_to_bq()' function for saving output to BigQuery
'''

def bq_to_dataframe(dataset_table):
    '''
    A fucntion used to query dataframe from bigquery.

    Returns
    -------
    None.

    '''
    
    print('Query data from Bigquery...')
    client = bigquery.Client(location="US")
    query = "SELECT * FROM {}".format(dataset_table)
   
    # query = """

    # -- Query without stations --

    # SELECT *
    #   FROM
    #     --`bem-metro-dss.5000_SP_Descriptive.v560_predict_revenue`

    # """
    
    query_job = client.query(query)  # API request - starts the query
    dataframe = query_job.to_dataframe() # Convert to dataframe

    return dataframe, client

def result_to_bq(X, client, dataset_id, dataset_table, append=True):
    '''
    Upload predicted result to GCP.

    Parameters
    ----------
    X : dataframe
        Predicted dataframe for GCP.
    client : client
        Big query client.
    dataset_id : str, optional
        Dataset ID. The default is 'datamart_for_report'.
    dataset_table : str, optional
        Dataset Table. The default is "tor_5_2".

    Returns
    -------
    None.

    '''
    
    dataset = client.create_dataset(dataset_id, exists_ok=True)  # API request
    table_ref = dataset.table(dataset_table)
    
    if append:
        job_config = bigquery.LoadJobConfig(write_disposition="WRITE_APPEND")
    else:
        job_config = bigquery.LoadJobConfig(write_disposition="WRITE_TRUNCATE")
        
    job = client.load_table_from_dataframe(X, table_ref, location="US", \
                                               job_config=job_config)
    job.result()  # Waits for table load to complete.
    
def group_by(dat, stationin, stationout, passenger):
    '''
    A fucntion used to group same combination, e.g. 'IBL->IBL, CSC, Adult'.

    Parameters
    ----------
    df : dataframe
        Input dataframe from bigquery
    crossline : str
        FactTrip Direction, e.g. IBL->IBL
    media : str
        Media Type, CSC or CST
    concession : str
        Concession Type, e.g. Adult

    Returns
    -------
    rdat : dataframe
        Return dataframe group by same combination

    '''
    
    # Select relevant dataframe for each model
    dat = dat[(dat['StationKeyIn']==stationin) & \
              (dat['StationKeyOut']==stationout) & \
              (dat['ConcessionTypeName']==passenger)]
    datcol = list(dat.columns)
    del datcol[-1:]
    
    # Target variable
    tarcol = list(dat.columns[-1:])
    
    # Aggregate ridership based on crossline, media type, and conession
    res = dat.groupby(datcol).sum()
    idat = np.vstack(res.index.values)
    rdat = pd.DataFrame(data=idat)
    rdat.columns = datcol
    # rdat['Year'] = rdat['Date'].astype(str).str[0:4]
    for col in range(0, len(tarcol)):
        rdat.loc[:, tarcol[col]] = res.loc[:, tarcol[col]].values
    
    return rdat
    
def clip_outlier(cl_df, lower=0.01, upper=0.99):
    '''
    A function used to clip outlier according to lower and upper bound.
    
    Parameters
    ----------
    cl_df : dataframe
        Input dataframe for ootlier clipping.
    lower : float, optional
        This set the lower bound. The default is 0.01.
    upper : float, optional
        This set the upper bound. The default is 0.99.

    Returns
    -------
    data_clip : dataframe
        Return clipped dataframe

    '''

    
    lower_bound = cl_df.quantile(lower) # Set the lower bound
    upper_bound = cl_df.quantile(upper) # Set the upper bound
    
    # Clip outlier
    data_clip = cl_df.clip(lower_bound, upper_bound, axis=1)
    
    return data_clip

def dummies(X, X_col):
    '''
    A function used to convert categorical data into one-hot encoding.

    Parameters
    ----------
    df : dataframe
        Input dataframe for one-hot conversion.

    Returns
    -------
    X_dum : dataframe
        Return dataframe with one-hot conversion.

    '''
    
#     X = df.copy()
    
    X_dum = pd.get_dummies(X, columns=['Month', 'Day', 'DayType'])
    X_dum = X_dum.set_index(X_col, append=True) # Include row index
    
    cols = X_dum.columns
    
    X_dum = pd.DataFrame(preprocessing.normalize(X_dum, norm='l2'), \
                         columns=cols)
    
    return X_dum
    
def traintest(dat, trainperiod=2020):
    '''
    A function used to split train and test dataframe.

    Parameters
    ----------
    df : dataframe
        Input dataframe for train-test split.
    trainperiod : int, optional
        Year of training period. The default is 2020.

    Returns
    -------
    Xtr_dum : dataframe
        Return independent variable dataframe for training.
    Xts_dum : dataframe
        Return independent variable dataframe for testing.
    y_tr : dataframe
        Return dependent variables dataframe for training.
    y_ts : dataframe
        Return dependent variables dataframe for testing.

    '''
    
    # copy input dataframe
    
    dat['yy_mm'] = dat['Date'].astype(str).str[0:7]
    dat['TrainPeriod'] = trainperiod
    period = trainperiod

    # Get dummies
    drop_col = ['Ridership']
    X = dat.drop(columns=drop_col, axis=1)
    X_col = list(['Date', 'Year', 'yy_mm', 'TrainPeriod', 
                  'StationKeyIn', 'StationKeyOut', 'ConcessionTypeName'])
    X_dum = dummies(X, X_col)
    X_dum = pd.concat([X_dum, X[X_col]], axis=1)
    
    Xtr_dum = X_dum[(X_dum.Year <= period)]

    Xtr_dum = Xtr_dum.set_index(X_col, append=True) # Include row index
    
    # Get test independent variables
    Xts_dum = X_dum.copy()
    Xts_dum = Xts_dum.set_index(X_col, append=True)
    
    # Get test dependent variables
    y_ts = dat[X_col].copy()
    y_ts['Ridership'] = dat.Ridership.copy()
    
    # Get train dependent variables
    y_tr = y_ts[(y_ts.Year <= period)]
    y_tr = y_tr.set_index(X_col, append=True)
    
    # Remove outlier
    y_tr = clip_outlier(y_tr).astype(int)
    
    y_ts = y_ts.set_index(X_col, append=True)
    
    return Xtr_dum, Xts_dum, y_tr, y_ts
    
def modelpredict(Xtr_gbr, Xts_gbr, y_train, y_test):
    '''
    A function used to generate machine learning models which are:
        1. Linear Regression (LNR)
        2. Ordinary Least Square (OLS)
        3. Ridge Regressor (Rid)
        4. Huber Regressor (Hub)
        5. Gradient Boosting Regressor

    Parameters
    ----------
    X_train : dataframe
        Independent dataframe for training.
    X_test : dataframe
        Dependent dataframe for testing.
    y_train : dataframe
        Dependent dataframe for training.
    y_test : dataframe
        Dependent dataframe for testing.

    Returns
    -------
    dat_out : dataframe
        Return predicted dataframe for every models.

    '''
    
    # Get train and test independent variable for ML    
    
    # GBR Model: Train and predict
#     Xtr_gbr = X_train.copy()
#     Xts_gbr = X_test.copy()
    mod_gbr = GradientBoostingRegressor(loss="ls", n_estimators=100)
    
    # Log Y
    y_train = np.log(y_train)
    
    mod_gbr.fit(Xtr_gbr, y_train.values.ravel())
    yht_gbr = mod_gbr.predict(Xts_gbr)
    
    # Exp Y
    yht_gbr = np.exp(yht_gbr)
    
    dat_out = pd.DataFrame(index=Xts_gbr.index)
    dat_out['Actual'] = y_test.copy()
    dat_out['Predict'] = yht_gbr.copy()
    dat_out = dat_out.fillna(0).round(0).astype(int)
    
    return dat_out

def gettest(gtdf, stationin, stationout, passenger, year):
    '''
    A function used to generate independent test dataframe for prediction.

    Parameters
    ----------
    df : dataframe
        Independent input dataframe for prediction.
    year : int
        Training period.
    crossline : str
        FactTrip Direction, e.g. IBL->IBL
    media : str
        Media Type, CSC or CST
    concession : str
        Concession Type, e.g. Adult

    Returns
    -------
    X_test : dataframe
        Return independent dataframe with combination.

    '''
    
    # Copy input dataframe
    rows = len(gtdf) # Get number of rows
    
    # Construct dataframe for testing variable
    gtdf['yy_mm'] = pd.DataFrame(gtdf.Date.astype(str).str[0:7])
    gtdf['TrainPeriod'] = np.repeat(year, rows)
    gtdf['StationKeyIn'] = np.repeat(stationin, rows)
    gtdf['StationKeyOut'] = np.repeat(stationout, rows)
    gtdf['ConcessionTypeName'] = np.repeat(passenger, rows)

    col_idx = ['Date', 'Year', 'yy_mm', 'TrainPeriod', 'StationKeyIn', 
               'StationKeyOut', 'ConcessionTypeName']
    X_test = dummies(gtdf, col_idx) # Convert to one-hot

    X_test = pd.concat([X_test, gtdf[col_idx].reset_index()], axis=1)
    X_test = X_test.set_index(col_idx, append=True)
    X_test = X_test.drop('index', axis=1)

    return X_test

def main(query, first_run):

    '''
    Main function for TOR 5.2 activated by Cloud Scheduler
    
    Returns
    -------
    output : dataframe
        Return output dataframe with features, predicted values for each
        models and upload to table in GCP
    
    '''
    
    print('Start.....')
    
    # Record start time
    with open('filename.txt', 'a+') as f:
        tz = pytz.timezone('Asia/Bangkok')
        now = datetime.now(tz=tz)        
        print("Start Time =", now, file=f)
    
    
    print('Read file from bucket...')
    cols = ['Date', 'Year', 'Month', 'Day', 'DayType', 'LongPH']
    tsdf = pd.read_csv('gs://data_sci_bucket/year2020.csv', usecols=cols)
    
    print('Download dataframe from bigquey...')
    df, client = bq_to_dataframe(query)
    df = df.fillna(0) # Eliminate null value
    
    stationin = sorted(df.StationKeyIn.unique())
    stationout = sorted(df.StationKeyOut.unique())
    passenger = sorted(df.ConcessionTypeName.unique())
    combo = sorted(df.Combo.unique())
    
    df = df.drop('Combo', axis=1)
    # year = sorted(df.Year.unique())
    # print('Type: {}'.format(type(year[0])))
    year = [2019, 2020]
    
    output = pd.DataFrame() # Create an output dataframe
    
    # Start ML prediction model
    for y in year:
        count = 1
        for ps in passenger:
            for si in stationin:
                for so in stationout:
                    concatCombo = str(y) + '-' + str(si) + '-' + str(so) + '-' + ps
                    
                    # If the combination doesn't exist, skip to next loop
                    if concatCombo not in combo:
    #                     print('Combination does not exist!!')
                        continue
    
                    try:
    
                        # Group dataframe based on crossline, transaction type
                        dat = group_by(df, si, so, ps)
                        
                        # Get train and test dataframe
                        X_train, X_test, y_train, y_test = traintest(dat, y)
                        
                        # Get dataframe for predicting future 
                        tsdf = tsdf[(tsdf.Date > max(dat.Date.astype(str)))]
                        Xts = gettest(tsdf, si, so, ps, y)
                        X_test = X_test.append(Xts)
                        X_test = X_test.fillna(0)
                        
                        train_col = X_train.columns
                        test_col = X_test.columns
                        if len(train_col) < len(test_col):
                            diff_cols = test_col.difference(train_col)
                            X_train[diff_cols] = pd.DataFrame(columns=diff_cols)
                            X_train = X_train.fillna(0)
                            X_train = X_train[X_test.columns]
                        elif len(train_col) > len(test_col):
                            diff_cols = train_col.differnce(test_col)
                            X_test[diff_cols] = pd.DataFrame(columns=diff_cols)
                            X_test = X_test.fillna(0)
                        
                        # Predict result
                        df_out = modelpredict(X_train, X_test, y_train, y_test)
                                                                            
                        output = output.append(df_out)
                        
                        print('Model {} Success ++'.format(concatCombo))
                        
                    except:
                        print('Model {} Fail --'.format(concatCombo))
                        # input()
                        pass
                    
            # Write to csv file and append
            output = output.reset_index().drop('level_0', axis=1)
            output['TrainDateTime'] = datetime.now()
            
            # Create table name
            table_name = 'report_5_6_' + str(y)
            
            # with open(table_name, 'a') as f:
            # Write output to BigQuery
            if first_run and count == 1:
                result_to_bq(output, client, dataset_id='datamart_for_report', \
                             dataset_table=table_name, append=False)
            else:
                result_to_bq(output, client, dataset_id='datamart_for_report', \
                             dataset_table=table_name, append=True)
            # Increase counter to append the table
            count += 1
            
            # Clear output dataframe
            output = pd.DataFrame() # Clear output dataframe
            
            with open('filename.txt', 'a+') as f:
                now = datetime.now(tz=tz)
                print('End Time for Model ' + concatCombo, now, file=f)

    # Record End time
    with open('filename.txt', 'a+') as f:
        now = datetime.now(tz=tz)
        print("End Time =", now, file=f)
    
    print("Finished....")


# Query dataframe from table
dataset = "bem-metro-dss.datamart_for_report"
table = "report_5_6_0_predict_ridership"
dataset_table = '`' + dataset + '.' + table + '`'

# This is the main function. If the first_run=True, then overwrite the table
main(dataset_table, first_run=True)