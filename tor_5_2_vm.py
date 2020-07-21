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
    
#     query = """

#     -- Query without stations --

#     SELECT *
#       FROM
#         `bem-metro-dss.5000_SP_Descriptive.v520_predict_ridership`

#     """
    
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
    dataset_id : str
        Dataset ID.
    dataset_table : str
        Dataset Table.
    append : bool
        True = Append, False = Overwrite

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
    
def group_by(dat, crossline, media, passenger):
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
    passenger : str
        Passenger Type, e.g. Adult

    Returns
    -------
    rdat : dataframe
        Return dataframe group by same combination

    '''
    
    dat = dat[(dat['CrossLine']==crossline) & \
              (dat['MediaType']==media) & \
              (dat['PassengerType']==passenger)]

    datcol = list(dat.columns)
    datcol.remove('Ridership')
    
    # Aggregate ridership based on crossline, media type, and conession
    res = dat.groupby(datcol).sum()
    idat = np.vstack(res.index.values)
    rdat = pd.DataFrame(data=idat)
    rdat.columns = datcol
    rdat['Ridership'] = res['Ridership'].values
    
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

def traintest(dat, trainperiod):
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
    X_col = list(['yy_mm','TrainPeriod','Date','Year','MediaType',\
                    'PassengerType','CrossLine'])

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
    
def modelpredict(X_train, X_test, y_train, y_test):
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
    # OLS Model
    Xtr_ols = sm.add_constant(X_train)
    Xts_ols = sm.add_constant(X_test)
    
    # Train and Test dataframe for models
    Xtr_gbr = Xtr_rig = Xtr_hub = X_train.copy()
    Xts_gbr = Xts_rig = Xts_hub = X_test.copy()
    
    # OLS Model: Train and predict
    mod_ols = sm.OLS(y_train, Xtr_ols).fit()
    yht_ols = mod_ols.predict(Xts_ols)
    
    # GBR Model: Train and predict
    mod_gbr = GradientBoostingRegressor(loss="ls", n_estimators=200)
    mod_gbr.fit(Xtr_gbr, y_train.values.ravel())
    yht_gbr = mod_gbr.predict(Xts_gbr)
    
    # Ridge Model: Train and predict
    mod_rig = Ridge(alpha=.5).fit(Xtr_rig, y_train)
    yht_rig = mod_rig.predict(Xts_rig)
    
    # Huber Model: Train and predict
    mod_hub = HuberRegressor(max_iter=3000)
    mod_hub.fit(Xtr_hub, y_train.values.ravel())
    yht_hub = mod_hub.predict(Xts_hub)

    # Ridership extraction for each feature
    cof_ols = mod_ols.params
    rid_ols = Xts_ols.multiply(cof_ols)
    rid_ols = rid_ols.rename(columns={'const': 'Base'})
    ols_gbr = yht_ols/yht_gbr
    fea_gbr = rid_ols.div(ols_gbr, axis=0)   
    bas_val = rid_ols['Base'].copy()    
    va_diff = fea_gbr.Base - bas_val
    tmp_fea = fea_gbr.drop('Base', axis=1)
    fea_rto = abs(tmp_fea).div((abs(tmp_fea).sum(axis=1)), axis=0)
    sp_diff = fea_rto.mul(va_diff, axis=0)
    dat_out = sp_diff.add(tmp_fea)

    # Construct output dataframe
    dat_out['Base'] = bas_val
    dat_out['Actual'] = y_test.copy()
    dat_out['OLS'] = yht_ols.copy()
    dat_out['Rig'] = yht_rig.copy()
    dat_out['Hub'] = yht_hub.copy()
    dat_out['GBR'] = yht_gbr.copy()
    dat_out = dat_out.fillna(0).round(0).astype(int)
    
    return dat_out

def wma(df, window=3):
    '''
    A function used to calculate weighted moving average.

    Parameters
    ----------
    df : dataframe
        Input dataframe for wma calculation.
    window : int, optional
        Window size. The default is 3.

    Returns
    -------
    out_wma : dataframe
        Return calculated wma values.

    '''
    
    w_index = np.append(np.arange(1,window+1), [0]) # Set window size
    dat_mth = monthgroup(df) # Group dataframe by month
    dat_idx = ssidx(dat_mth) # Get seasonal index
    dat_rid = dat_idx.Ridership.copy()
    
    out_wma = dat_rid.rolling(window+1).\
        apply(lambda rider: np.dot(rider, w_index)/w_index.sum(), raw=True)
    out_wma.index = dat_mth.index

    dat_idx = dat_idx.rename(columns={'ssidx': 'WMA'})
    out_wma = pd.DataFrame(out_wma).rename(columns={'Ridership': 'WMA'})
    out_wma = (out_wma.WMA * dat_idx.WMA).fillna(0).round(0).astype(int)
    
    return out_wma

def monthgroup(dat):
    '''
    A function used to group dataframe by month.

    Parameters
    ----------
    df : dataframe
        Input dataframe for grouping.

    Returns
    -------
    datgrp : dataframe
        Return grouped dataframe.

    '''
    
    dat['yy_mm'] = dat['Date'].astype(str).str[0:7]
    datcol = list(['yy_mm','Month','Year','MediaType','CrossLine'])
    datgrp = dat.groupby(datcol).sum() # Aggregate ridership by month
    
    return datgrp

def ssidx(dat):
    '''
    A function used to calculate seasonal index.

    Parameters
    ----------
    df : dataframe
        Input dataframe for seasonal index calculation.

    Returns
    -------
    dat_meg : dataframe
        Return seasonal index dataframe.

    '''
    
    yearsum = dat.mean(axis=0, level='Year')
    yearsum = yearsum.rename(columns={'Ridership': 'Average'})
    
    dat_meg = dat.reset_index().merge(yearsum, on='Year')
    dat_meg['ratio'] = dat_meg.Ridership/dat_meg.Average
    dat_meg['ssidx'] = 1 # Set 1st year seasonal index to 1
    
    for i in range(0, len(dat_meg) - 12):
        dat_meg.loc[i+12, 'ssidx'] = (dat_meg.loc[i, 'ssidx'] + \
                                      dat_meg.loc[i, 'ratio'])/2
            
    dat_col = list(['yy_mm','Month','Year','MediaType','CrossLine'])
    dat_meg = dat_meg.set_index(dat_col)
    
    return dat_meg

def gettest(gtdf, year, crossline, media, passenger):
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
    passenger : str
        Passenger Type, e.g. Adult

    Returns
    -------
    X_test : dataframe
        Return independent dataframe with combination.

    '''

    rows = len(gtdf) # Get number of rows
    
    # Construct dataframe for testing variable
    gtdf['yy_mm'] = pd.DataFrame(gtdf.Date.astype(str).str[0:7])
    gtdf['TrainPeriod'] = np.repeat(year, rows)
    gtdf['MediaType'] = np.repeat(media, rows)
    gtdf['PassengerType'] = np.repeat(passenger, rows)
    gtdf['CrossLine'] = np.repeat(crossline, rows)

    col_idx = ['Date', 'Year', 'yy_mm', 'TrainPeriod', 'MediaType',
               'PassengerType','CrossLine']
    X_test = dummies(gtdf, col_idx) # Convert to one-hot
    X_test = pd.concat([X_test, gtdf[col_idx].reset_index()], axis=1)
    X_test = X_test.set_index(col_idx, append=True)
    X_test = X_test.drop('index', axis=1)

    return X_test

def main(query, first_run):
    #     '''
    #     Main function for TOR 5.2 activated by Cloud Scheduler

    #     Returns
    #     -------
    #     output : dataframe
    #         Return output dataframe with features, predicted values for each
    #         models and upload to table in GCP

    #     '''

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

    crossline = sorted(df.CrossLine.unique()) # Get all available direction
    media = sorted(df.MediaType.unique()) # Get all available media type
    passenger = sorted(df.PassengerType.unique()) # Get all available concession
    combo = sorted(df.Combo.unique())
    year = sorted(df.Year.unique())
#     year = [2015]

    df = df.drop('Combo', axis=1)

    wma_mt = [3, 6] # Set WMA's window size

    output = pd.DataFrame() # Creat an output dataframe

    # Start ML prediction model
    for y in year:
        for c in crossline:
            for m in media:                
                for p in passenger:
                    concatCombo = str(y) + ', ' + c + ', ' + m + ', ' + p
                    # If the combination doesn't exist, skip to next loop
                    if concatCombo not in combo:
    #                     print('Model {} does not exist!!'.format(concatCombo))
                        continue

                    try:

                        dat = group_by(df, c, m, p)

                        # Calculate WMA3 & WMA6
                        dfwma = pd.DataFrame() # Create a dataframe for WMA
                        for w in wma_mt:
                            col_wma = 'WMA' + str(w)
                            dfwma[col_wma] = wma(dat,w)
                        
                        # Get train and test dataframe
                        X_train, X_test, y_train, y_test = traintest(dat, y)

                        # Get dataframe for predicting future 
                        tsdf = tsdf[(tsdf.Date > max(dat.Date.astype(str)))]

                        Xts = gettest(tsdf, y, c, m, p)
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

                        # Put WMA into dataframe
                        df_meg = df_out.reset_index().merge(dfwma, on='yy_mm', \
                                                            how='left')

                        # Eliminate null value, delect unused column, and
                        # append WMA to output dataframe
                        output = output.append(df_meg).fillna(0)
                        output = output.drop('level_0', axis=1)
                        output[['WMA3','WMA6']] = output[['WMA3','WMA6']].\
                            astype(int)

                        print('Model {} Success ++'.format(concatCombo))

                    except:
                        print('Model {} Fail --'.format(concatCombo))
    #                     input()
                        pass

    # Reset output frame index
    output = output.reset_index(drop=True).fillna(0)
    output['TrainDateTime'] = datetime.now()
#     output = output.drop('Year', axis=1)

    print("Complete Prediction process...")

    # Write output to BigQuery
    print('Write to BigQuey')
    # Create table name
    table_name = 'report_5_2_vm_test'
    
    if first_run:
        result_to_bq(output, client, dataset_id='datamart_for_report', \
                    dataset_table=table_name, append=False)
    else:
        result_to_bq(output, client, dataset_id='datamart_for_report', \
                    dataset_table=table_name, append=True)

    # Record End time
    with open('filename.txt', 'a+') as f:
        now = datetime.now(tz=tz)
        print("End Time =", now, file=f)

    print("Finished....")

# Query dataframe from table
dataset = "bem-metro-dss.datamart_for_report"
table = "report_5_2_0_predict_ridership"
dataset_table = '`' + dataset + '.' + table + '`'

# This is the main function. If the first_run=True, then overwrite the table
main(dataset_table, first_run=True)