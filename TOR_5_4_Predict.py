# TOR 5.4 Predict การเปิดส่วนต่อขยาย (รายสถานี)

# import all necessary packages
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model import LinearRegression, Ridge, HuberRegressor
from sklearn.ensemble import GradientBoostingRegressor

from sklearn import preprocessing
from google.cloud import bigquery

# Load data from Big Query
client = bigquery.Client(location="US")
query = """

-- Ridership, TripLength, Revenue, Crossline + ConcessionTypeName

SELECT
  ft.businessday,
  ft.businessdatekey,
  date.EnglishMonthName,
  date.calendaryear,
  date.EnglishDayNameOfWeek,
  date.daytype,
  case when ft.ServiceProviderKeyIn in (2,20) and ft.ServiceProviderKeyOut in (2,20) then 'Blue Line'
  when ft.ServiceProviderKeyIn = 10 and ft.ServiceProviderKeyOut=10 then 'Purple Line'
  else 'Cross Line' end as crossline, 
--   concat(od.linecodein,'-',od.linecodeout) as Direction,
  tkt.ConcessionTypeName,
  ext.nbStation_all,
  ext.InterchangeType_BTS,
  ext.InterchangeType_ARL,
  ext.InterchangeType_MRT,
  ext.Event,
  ext.Event_name,
  COUNT(*) AS Ridership,
  AVG(od.triplength) TripLength,
  sum(dfa.Fare) as ActualRevenue
FROM
  bem-metro-dss.Data_Warehouse.factTrip ft
LEFT JOIN
  bem-metro-dss.Data_Warehouse.dimDate date
ON
  ft.businessdatekey = date.datekey
LEFT JOIN
  bem-metro-dss.Data_Warehouse.dimOD od
ON
  ft.ODKey = od.id
LEFT JOIN bem-metro-dss.Data_Warehouse.dimFareApportion dfa
on ft.farekey = dfa.fareid
LEFT JOIN
  bem-metro-dss.DataScience.Extension_Factor_2 ext
ON
  ft.businessday = ext.businessday
LEFT JOIN
  bem-metro-dss.Data_Warehouse.dimTicketType tkt
ON
  ft.TicketTypeKey = tkt.TicketTypeKey
  
WHERE
  date.daytype != 'OF'
  AND ft.businessday >= '2015-01-01'
  AND ft.stationkeyin != 0
  AND od.triplength != 0
  AND (ft.ServiceProviderKeyIn is not null or ft.ServiceProviderKeyOut is not null)
GROUP BY
  ft.businessday,
  ft.businessdatekey,
  date.EnglishMonthName,
  date.calendaryear,
  date.EnglishDayNameOfWeek,
  date.daytype,
  tkt.ConcessionTypeName,
  crossline,
  ext.nbStation_all,
  ext.InterchangeType_BTS,
  ext.InterchangeType_ARL,
  ext.InterchangeType_MRT,
  ext.Event,
  ext.Event_name
--   Direction
ORDER BY

  businessday,crossline, ConcessionTypeName

  
    """
query_job = client.query(query)  # API request - starts the query
df_rd = query_job.to_dataframe() # Convert to dataframe

# Select feature
select_cols = ['businessday', 'EnglishMonthName', 'EnglishDayNameOfWeek', 
               'daytype','ConcessionTypeName', 'nbStation_all', 'InterchangeType_BTS', 'InterchangeType_ARL', 'InterchangeType_MRT', 
               'Event_name','crossline', 'ActualRevenue', 'Ridership']

raw_data = df_rd[select_cols]


col_list = ['InterchangeType_BTS', 'InterchangeType_ARL', 'InterchangeType_MRT', 'nbStation_all','Ridership']
for col in col_list:
    raw_data[col] = raw_data[col].astype(int)

cat_cols = ['EnglishMonthName', 'EnglishDayNameOfWeek', 'daytype']
# One-hot encoder
raw_data = pd.get_dummies(raw_data, columns=cat_cols)

lower=0.20
upper=0.99



def clip_outlier(cl_df, lower=lower, upper=upper):
    
    lower_bound = cl_df.quantile(lower) # Set the lower bound
    upper_bound = cl_df.quantile(upper) # Set the upper bound
    
    # Clip outlier
    data_clip = cl_df.clip(lower_bound, upper_bound, axis=1)
    
    return data_clip

data = raw_data.copy()
# data[['Ridership']] = clip_outlier(raw_data[['Ridership']], lower=lower, upper=upper)

print('Start prediction Ridership')


########## Central Function ##########
def create_table_train_test (df, i_ev, target_predict):
    
    train_list = []
    for train_period in df.Event_name.unique()[:i_ev]:  
        train_p = df.loc[df['Event_name'] == train_period]
        train_list.append(train_p)
    train_p_1 = pd.concat(train_list, axis=0,ignore_index=True)
    train_p_1_feat = train_p_1.drop([target_predict,'Event_name','crossline','ConcessionTypeName', 'ActualRevenue'], axis=1)
    train_p_1_feat = train_p_1_feat.set_index('businessday')
    train_feat = train_p_1_feat
    train_lab = train_p_1[[target_predict]]

#     print('Select',df.Event_name.unique()[:i_ev],'for Train data')

    test_list = []
    for test_period in df.Event_name.unique()[:i_ev+1]:
        test_p = df.loc[df['Event_name'] == test_period]
        test_list.append(test_p)
    test_p_1 = pd.concat(test_list, axis=0,ignore_index=True)
    test_feat = test_p_1.drop([target_predict,'Event_name','crossline','ConcessionTypeName', 'ActualRevenue'], axis=1)
    test_feat = test_feat.set_index('businessday')
    test_lab = test_p_1[[target_predict]]
  
#     print('Select',df.Event_name.unique()[:i_ev+1],'for Test data')

    # for filter Data
    num_con_1 = i_ev-1
    # for replace table 2
    num_con_2 = num_con_1 + 1
    condition = df['Event_name'].unique()

#     print('Train period',df.Event_name.unique()[0], 'to', condition[num_con_1], 'completed')
#     print('Predict', condition[num_con_2], 'completed')
#     print('From all condition', condition)

    # จำนวนสถานี
    n_st = test_p_1['nbStation_all'].loc[test_p_1['Event_name']== condition[num_con_1]].unique()
    # จำนวน Interchange
    n_inter_mrt = test_p_1['InterchangeType_MRT'].loc[test_p_1['Event_name']== condition[num_con_1]].unique()
    n_inter_bts = test_p_1['InterchangeType_BTS'].loc[test_p_1['Event_name']== condition[num_con_1]].unique()
    n_inter_arl = test_p_1['InterchangeType_ARL'].loc[test_p_1['Event_name']== condition[num_con_1]].unique()

#     print('Factor_nbStation_all in Table 2 =',n_st, 'station', 'Finished')
#     print('Factor_nbInterChangeMRT in Table 2 =',n_inter_mrt, 'station')
#     print('Factor_nbInterChangeBTS in Table 2 =',n_inter_bts, 'station')
#     print('Factor_nbInterChangeARL in Table 2 =',n_inter_arl, 'station')

    # Test table 2 feature
    test_feat_compare =  test_p_1.copy()
    test_feat_compare.loc[(test_feat_compare['Event_name'] == condition[num_con_2]), 'nbStation_all'] = n_st.max()
    test_feat_compare.loc[(test_feat_compare['Event_name'] == condition[num_con_2]), 'InterchangeType_MRT'] = n_inter_mrt.max()
    test_feat_compare.loc[(test_feat_compare['Event_name'] == condition[num_con_2]), 'InterchangeType_BTS'] = n_inter_bts.max()
    test_feat_compare.loc[(test_feat_compare['Event_name'] == condition[num_con_2]), 'InterchangeType_ARL'] = n_inter_arl.max()
    test_feat_compare.loc[(test_feat_compare['Event_name'] == condition[num_con_2]), target_predict] = 0
    # Test table 2 label
    test_lab_feat = test_feat_compare[[target_predict]]
    y_test_tab_2 = test_lab_feat
    test_feat_pred = test_feat_compare.drop([target_predict,'Event_name','crossline','ConcessionTypeName', 'ActualRevenue'], axis=1, inplace=True)
    test_feat_pred = test_feat_compare.set_index('businessday')
    x_test_tab_2 = test_feat_pred
    
#     print(train_feat.isna().sum(),test_feat.isna().sum(),x_test_tab_2.isna().sum())

    return train_feat, train_lab, test_feat, test_lab, x_test_tab_2, y_test_tab_2

########## Central Function ##########
def fit_and_predict (train_feature, train_label, test_feat_1, test_feat_2):
    
    # Call Model
    mod_lr = LinearRegression(normalize = True)
    mod_gbr = GradientBoostingRegressor(loss="ls",random_state=0)
    
    # Train model
    mod_lr.fit(train_feature, train_label)
    mod_gbr.fit(train_feature, train_label)

    y_lr_tab1 = mod_lr.predict(test_feat_1)
    y_gbr_tab1 = mod_gbr.predict(test_feat_1)
    
    # Predict table 2
    y_lr_tab2 = mod_lr.predict(test_feat_2)
    y_gbr_tab2 = mod_gbr.predict(test_feat_2)
#     print(test_feat_2.columns)
#     print(mod_lr.coef_)
    

    return y_lr_tab1, y_gbr_tab1, y_lr_tab2, y_gbr_tab2

# def fit_and_predict (train_feature, train_label, test_feat_1, test_feat_2):
    
#     # Call Model
#     mod_lr = LinearRegression(normalize = True)
#     mod_gbr = GradientBoostingRegressor(loss="ls",random_state=0)
    
#     # Log Y
#     train_label = np.log(train_label)
    
#     # Train model
#     mod_lr.fit(train_feature, train_label.values.ravel())
#     mod_gbr.fit(train_feature, train_label.values.ravel())
  
# #     # Train model
# #     mod_lr.fit(train_feature, train_label)
# #     mod_gbr.fit(train_feature, train_label)
    
#     y_lr_tab1 = mod_lr.predict(test_feat_1)
#     y_gbr_tab1 = mod_gbr.predict(test_feat_1)
    
#     # Exp Y
#     y_lr_tab1 = np.exp(y_lr_tab1)
#     y_gbr_tab1 = np.exp(y_gbr_tab1) 
    
#     # Predict table 2
#     y_lr_tab2 = mod_lr.predict(test_feat_2)
#     y_gbr_tab2 = mod_gbr.predict(test_feat_2)
    
#     # Exp Y
#     y_lr_tab2 = np.exp(y_lr_tab2)
#     y_gbr_tab2 = np.exp(y_gbr_tab2)  
    
# #     print(test_feat_2.columns)
# #     print(mod_lr.coef_)
    
#     return y_lr_tab1, y_gbr_tab1, y_lr_tab2, y_gbr_tab2

########## Individuul Function ##########
# Predict table result 1 function
def table_result_1 (y_lr, y_gbr, x_test_1, y_test_1, dat, train_period, new_section_condition):
    # reset index test data
    x_test_1 = x_test_1.reset_index()
    # Create empty dataframe
    frame = pd.DataFrame()
    frame['businessday'] = x_test_1.businessday
    frame['Ridership'] = y_test_1.reset_index(drop=True)
    frame['LR_Ridership'] = y_lr
    frame['GBR_Ridership'] = y_gbr
    # map date by merge original table
    frame = pd.merge(frame, dat[['businessday','ConcessionTypeName','Event_name','crossline','nbStation_all', 'ActualRevenue']], how="left", on=['businessday'])
    # create filter columns
    frame['TrainPeriod'] = train_period
    # creat filter by SectionType
    
    frame.loc[(frame['Event_name'] == new_section_condition), 'SectionType'] = 'New Section'
    frame['SectionType'] = frame['SectionType'].fillna('Existing Section')
    frame['RidershipType'] = 'Base'

    return frame

########## Individuul Function ##########
# Predict table result 2 function
def table_result_2 (y_lr_2, y_gbr_2, x_test_2, y_test_2, dat, train_period, new_section_condition):
    # reset index test data
    x_test_2 = x_test_2.reset_index()
    # Create empty dataframe
    frame = pd.DataFrame()
    frame['businessday'] = x_test_2.businessday
    frame['Ridership'] = y_test_2.reset_index(drop=True)
    frame['LR_Ridership'] = y_lr_2
    frame['GBR_Ridership'] = y_gbr_2
    # map date by merge original table
    frame = pd.merge(frame, dat[['businessday','ConcessionTypeName','Event_name','crossline','nbStation_all', 'ActualRevenue']], how="left", on=['businessday'])
    # create filter columns
    frame['TrainPeriod'] = train_period
    # creat filter by SectionType
    
    frame.loc[(frame['Event_name'] == new_section_condition), 'SectionType'] = 'New Section'
    frame['SectionType'] = frame['SectionType'].fillna('Existing Section')
    frame['RidershipType'] = 'Base'

    return frame

########## Individuul Function ##########
# Diff Table
def result_table (df_1, df_2):
    
    # Copy result datafram 1
    res_table = df_1.copy()
    # prediction result datafram 1 - prediction result datafram 2
    res_table['LR_Ridership'].loc[df_1['SectionType']=='New Section'] = df_1['LR_Ridership'].loc[df_1['SectionType']=='New Section'] - df_2['LR_Ridership'].loc[df_2['SectionType']=='New Section']
    res_table['GBR_Ridership'].loc[df_1['SectionType']=='New Section'] = df_1['GBR_Ridership'].loc[df_1['SectionType']=='New Section'] - df_2['GBR_Ridership'].loc[df_2['SectionType']=='New Section']
    df_2['LR_Ridership_Extension'] = 0
    df_2['LR_Ridership_Extension'].loc[(df_2['SectionType']=='New Section') | (df_2['RidershipType']=='Base')] = df_1['LR_Ridership'].loc[df_1['SectionType']=='New Section'] - df_2['LR_Ridership'].loc[df_2['SectionType']=='New Section']

    # Filter only Extention for
    res_table['RidershipType'].loc[res_table['SectionType']=='New Section'] = 'Extension'
    final_table = pd.concat([df_2, res_table.loc[res_table['SectionType']=='New Section']], axis=0,ignore_index=True)
    final_table['LR_Ridership_Extension'] = final_table['LR_Ridership_Extension'].fillna(0) 

    return final_table


data = data
target_predict = 'Ridership'

result_list = []
for c_line in data.crossline.unique():
    for concession in data.ConcessionTypeName.unique():

        combi_df = data.loc[(data['crossline'] == c_line) & (data['ConcessionTypeName'] == concession)]

        for i_ev in range(4, len(combi_df['Event_name'].unique())):

            x_train, y_train, x_test_1, y_test_1, x_test_2, y_test_2  = create_table_train_test (combi_df, i_ev, target_predict)
            y_lr, y_gbr, y_lr_2, y_gbr_2 = fit_and_predict(x_train, y_train, x_test_1, x_test_2)

            combi_event = combi_df.set_index('businessday')
            run_event = pd.merge(x_test_1, combi_event[['Event_name']], how="left", on=['businessday'])

            train_period = run_event.Event_name.unique()[-2]
            new_section_condition = run_event.Event_name.unique()[-1]

            res_1 = table_result_1 (y_lr, y_gbr, x_test_1, y_test_1,combi_df, train_period = train_period, new_section_condition = new_section_condition)
            res_2 = table_result_2 (y_lr_2, y_gbr_2, x_test_2, y_test_2,combi_df, train_period = train_period, new_section_condition = new_section_condition)
            frame = result_table (res_1,res_2)

            print('Ridership Combination',c_line,'-',concession,'-','Event_sequence =',i_ev ,'-','Train',train_period ,'-','Predict',new_section_condition, '=',len(frame),'rows')

            result_list.append(frame)


ridership_frame = pd.concat(result_list, axis=0,ignore_index=True)
print('Ridership prediction finished')
# ridership_frame


# TripLength Function

# ************************************************************************************************************* ##

query_job = client.query(query)  # API request - starts the query
df_tl = query_job.to_dataframe() # Convert to dataframe


# Select feature
select_cols = ['businessday', 'EnglishMonthName', 'EnglishDayNameOfWeek', 
               'daytype','ConcessionTypeName', 'nbStation_all', 'InterchangeType_BTS', 'InterchangeType_ARL', 'InterchangeType_MRT', 
               'Event_name','crossline', 'ActualRevenue', 'TripLength']

data = df_tl[select_cols]

col_list = ['InterchangeType_BTS', 'InterchangeType_ARL', 'InterchangeType_MRT', 'nbStation_all']
for col in col_list:
    data[col] = data[col].astype(int)    
# reduce decimals
data['TripLength'] = data['TripLength'].apply(lambda x: np.round(x, decimals=4))

# All dat after clean
# Get categorical columns
cat_cols = ['EnglishMonthName', 'EnglishDayNameOfWeek', 'daytype']
# One-hot encoder
data = pd.get_dummies(data, columns=cat_cols)
print('Start prediction TripLength')

## ************************************************************************************************************* ##

########## Individuul Function ##########
# Predict table result 1 function
def table_result_1 (y_lr, y_gbr, x_test_1, y_test_1, dat, train_period, new_section_condition):
    # reset index test data
    x_test_1 = x_test_1.reset_index()
    # Create empty dataframe
    frame = pd.DataFrame()
    frame['businessday'] = x_test_1.businessday
    frame['AVG_TripLength'] = y_test_1.reset_index(drop=True)
    frame['AVG_LR_TripLength'] = y_lr
    frame['AVG_GBR_TripLength'] = y_gbr
    # map date by merge original table
    frame = pd.merge(frame, dat[['businessday','ConcessionTypeName','Event_name','crossline','nbStation_all', 'ActualRevenue']], how="left", on=['businessday'])
    # create filter columns
    frame['TrainPeriod'] = train_period
    # creat filter by SectionType
    
    frame.loc[(frame['Event_name'] == new_section_condition), 'SectionType'] = 'New Section'
    frame['SectionType'] = frame['SectionType'].fillna('Existing Section')
    frame['RidershipType'] = 'Base'

    return frame

########## Individuul Function ##########
# Predict table result 2 function
def table_result_2 (y_lr_2, y_gbr_2, x_test_2, y_test_2, dat, train_period, new_section_condition):
    # reset index test data
    x_test_2 = x_test_2.reset_index()
    # Create empty dataframe
    frame = pd.DataFrame()
    frame['businessday'] = x_test_2.businessday
    frame['AVG_TripLength'] = y_test_2.reset_index(drop=True)
    frame['AVG_LR_TripLength'] = y_lr_2
    frame['AVG_GBR_TripLength'] = y_gbr_2
    # map date by merge original table
    frame = pd.merge(frame, dat[['businessday','ConcessionTypeName','Event_name','crossline','nbStation_all', 'ActualRevenue']], how="left", on=['businessday'])
    # create filter columns
    frame['TrainPeriod'] = train_period
    # creat filter by SectionType
    
    frame.loc[(frame['Event_name'] == new_section_condition), 'SectionType'] = 'New Section'
    frame['SectionType'] = frame['SectionType'].fillna('Existing Section')
    frame['RidershipType'] = 'Base'

    return frame

########## Individuul Function ##########
# Diff Table
def result_table (df_1, df_2):
    
    # Copy result datafram 1
    res_table = df_1.copy()
    # prediction result datafram 1 - prediction result datafram 2
    res_table['AVG_LR_TripLength'].loc[df_1['SectionType']=='New Section'] = df_1['AVG_LR_TripLength'].loc[df_1['SectionType']=='New Section'] - df_2['AVG_LR_TripLength'].loc[df_2['SectionType']=='New Section']
    res_table['AVG_GBR_TripLength'].loc[df_1['SectionType']=='New Section'] = df_1['AVG_GBR_TripLength'].loc[df_1['SectionType']=='New Section'] - df_2['AVG_GBR_TripLength'].loc[df_2['SectionType']=='New Section']
    df_2['AVG_LR_TripLength_Extension'] = 0
    df_2['AVG_LR_TripLength_Extension'].loc[(df_2['SectionType']=='New Section') | (df_2['RidershipType']=='Base')] = df_1['AVG_LR_TripLength'].loc[df_1['SectionType']=='New Section'] - df_2['AVG_LR_TripLength'].loc[df_2['SectionType']=='New Section']

    # Filter only Extention for
    res_table['RidershipType'].loc[res_table['SectionType']=='New Section'] = 'Extension'
    final_table = pd.concat([df_2, res_table.loc[res_table['SectionType']=='New Section']], axis=0,ignore_index=True)
    final_table['AVG_LR_TripLength_Extension'] = final_table['AVG_LR_TripLength_Extension'].fillna(0) 

    return final_table


data = data
target_predict = 'TripLength'

result_list = []
for c_line in data.crossline.unique():
    for concession in data.ConcessionTypeName.unique():

        combi_df = data.loc[(data['crossline'] == c_line) & (data['ConcessionTypeName'] == concession)]

        for i_ev in range(4, len(combi_df['Event_name'].unique())):

            x_train, y_train, x_test_1, y_test_1, x_test_2, y_test_2  = create_table_train_test (combi_df, i_ev, target_predict)
            y_lr, y_gbr, y_lr_2, y_gbr_2 = fit_and_predict(x_train, y_train, x_test_1, x_test_2)

            combi_event = combi_df.set_index('businessday')
            run_event = pd.merge(x_test_1, combi_event[['Event_name']], how="left", on=['businessday'])

            train_period = run_event.Event_name.unique()[-2]
            new_section_condition = run_event.Event_name.unique()[-1]

            res_1 = table_result_1 (y_lr, y_gbr, x_test_1, y_test_1,combi_df, train_period = train_period, new_section_condition = new_section_condition)
            res_2 = table_result_2 (y_lr_2, y_gbr_2, x_test_2, y_test_2,combi_df, train_period = train_period, new_section_condition = new_section_condition)
            frame = result_table (res_1,res_2)

            print('Triplength Combination',c_line,'-',concession,'-','Event_sequence =',i_ev ,'-','Train',train_period ,'-','Predict',new_section_condition, '=',len(frame),'rows')

            result_list.append(frame)


triplength_frame = pd.concat(result_list, axis=0,ignore_index=True)
print('Triplength prediction finished')


print('Compare dimension Ridership Table shape =', ridership_frame.shape,"and" ,'Triplength Table shape =', triplength_frame.shape)

# Join ตาราง Ridership กับ Triplength เข้าด้วยกัน

def concat_result (rid_df, tl_df):
    
    # ปัดเศษค่า ขึ้น/ลง ค่า Triplength เพื่อนำไป map กับ Fare
    tl_df['TripLength'] = tl_df['AVG_TripLength'].round().abs()
    tl_df['LR_TripLength'] = tl_df['AVG_LR_TripLength'].round().abs()
    tl_df['GBR_TripLength'] = tl_df['AVG_GBR_TripLength'].round().abs()
    
    select_col_tl = ['AVG_TripLength','AVG_LR_TripLength','AVG_LR_TripLength_Extension',
                     'AVG_GBR_TripLength','TripLength', 'LR_TripLength', 'GBR_TripLength']
    
    result = pd.concat([rid_df, tl_df[select_col_tl]], axis=1)
    print('Prediction finished')
    
    return result

res_table = concat_result (ridership_frame, triplength_frame)
# res_table

# Save result to big query

def result_to_bq(X, client, dataset_id, dataset_table):
    '''
    Upload predicted result to GCP.

    Parameters
    ----------
    X : dataframe
        Predicted dataframe for GCP.
    client : client
        Big query client.
    dataset_id : str, optional
        Dataset ID. The default is 'XXX'.
    dataset_table : str, optional
        Dataset Table. The default is "".

    Returns
    -------
    None.

    '''
    
    dataset = client.create_dataset(dataset_id, exists_ok=True)  # API request
    table_ref = dataset.table(dataset_table)
    
    job_config = bigquery.LoadJobConfig(write_disposition="WRITE_TRUNCATE")
    job = client.load_table_from_dataframe(X, table_ref, location="US", \
                                           job_config=job_config)
    job.result()  # Waits for table load to complete.
    

result_to_bq(res_table, client, dataset_id = "datamart_for_report", dataset_table = "report_5_4")

print('Save table to Big Query finished')    