import pandas as pd
import numpy as np
from google.cloud import bigquery
from constraint import *
import re


shortest_path_query = """
    SELECT
  dod.StationKeyIn,
  dod.StationKeyOut,
  dstin.StationCode as StationCodeIn,
  dstout.StationCode as StationCodeOut,
  dstin.SortOrder as SortOrderIn,
  dstout.SortOrder as SortOrderOut,
  dod.IBL+dod.BLE1+dod.BLE2+dod.BLE3 AS BL,
  dod.IBL,
  dod.BLE1+dod.BLE2+dod.BLE3 AS BLE,
  dod.PPL,
  dod.ServiceProviderNameIn,
  dod.ServiceProviderNameOut,
  dod.LineCodeIn,
  dod.LineCodeOut,
  dodkm.BLKM,
  dodkm.PPLKM,
  ConcessionTypeName as PassengerType
FROM
  `bem-metro-dss.Data_Warehouse.dimOD` dod
LEFT JOIN
  `bem-metro-dss.Data_Warehouse.dimODKM` dodkm
ON
  dod.stationkeyin = dodkm.stationkeyin
  AND dod.stationkeyout = dodkm.stationkeyout
LEFT JOIN
  `bem-metro-dss.Data_Warehouse.dimStation` dstin
ON
  dod.stationkeyin = dstin.stationkey
LEFT JOIN
  `bem-metro-dss.Data_Warehouse.dimStation` dstout
ON
  dod.stationkeyout = dstout.stationkey
cross join 
(select ConcessionTypeName from bem-metro-dss.Data_Warehouse.dimTicketType where TicketName = 'MRT Card' group by ConcessionTypeName)
  """

fare_config_query = """
    SELECT * 
    FROM `bem-metro-dss.datamart_for_report.58_km_config`
  """

# upload result dataframe to BigQuery
def result_to_bq(X, client, dataset_id='datamart_for_report', dataset_table="optimization_5_8_test", write_disposition="WRITE_APPEND"):
    '''
    Dump result to Bigquery
    '''
    print("Upload prediction result to Bigquery...")
    
    # The project defaults to the Client's project if not specified.
    dataset = client.create_dataset(dataset_id, exists_ok=True)  # API request
    table_ref = dataset.table(dataset_table)
    
    job_config = bigquery.LoadJobConfig(write_disposition=write_disposition,)

    job = client.load_table_from_dataframe(X, table_ref, location="US", job_config=job_config)

    job.result()  # Waits for table load to complete.
    print("Loaded dataframe to {}".format(table_ref.path))

# get query dataframe from BigQuery
def bq_to_dataframe(query):
    
    print('Query data from Bigquery...')
    client = bigquery.Client(location="US")
    query_job = client.query(query)  # API request - starts the query
    dataframe = query_job.to_dataframe()

    return client, dataframe

# get entry fee weights for each revenue sharing case
def get_feature(shortest_path, weight):

  ibl = np.where(shortest_path['ServiceProviderNameIn']=='IBL', 1, 0)
  ble = np.where(shortest_path['ServiceProviderNameIn']=='BLE', 1, 0)
  ppl = np.where(shortest_path['ServiceProviderNameIn']=='PPL', 1, 0)

  # only entry station
  entry = shortest_path.copy()
  entry['Mode'] = ['Entry'] * len(entry)
  entry['EntryWeightIBL'] = ibl
  entry['EntryWeightBLE'] = ble
  entry['EntryWeightPPL'] = ppl

  # share to all service providers
  shared = shortest_path.copy()
  shared['Mode'] = ['Shared'] * len(entry)
  shared['EntryWeightIBL'] = ((ibl + shared['IBL']) > 0)
  shared['EntryWeightBLE'] = ((ble + shared['BLE']) > 0)
  shared['EntryWeightPPL'] = ((ppl + shared['PPL']) > 0)

  # weighted
  weighted = shortest_path.copy()
  weighted['Mode'] = ['Weighted'] * len(entry)
  weighted['EntryWeightIBL'] = ((ibl + shared['IBL']) > 0) * weight['IBL']
  weighted['EntryWeightBLE'] = ((ble + shared['BLE']) > 0) * weight['BLE']
  weighted['EntryWeightPPL'] = ((ppl + shared['PPL']) > 0) * weight['PPL']
  
  # normalize
  res = pd.concat([entry, shared, weighted], axis=0)
  res['sum'] = res['EntryWeightIBL'] + res['EntryWeightBLE'] + res['EntryWeightPPL']
  res['EntryIBL'] = res['EntryWeightIBL'].divide(res['sum'], fill_value=0)
  res['EntryBLE'] = res['EntryWeightBLE'].divide(res['sum'], fill_value=0)
  res['EntryPPL'] = res['EntryWeightPPL'].divide(res['sum'], fill_value=0)
    
#   # normalize path hop
#   res['hop_sum'] = res.IBL + res.BLE + res.PPL
#   res['SharedHopIBL'] = res['IBL'].divide(res['hop_sum'], fill_value=0)
#   res['SharedHopBLE'] = res['BLE'].divide(res['hop_sum'], fill_value=0)
#   res['SharedHopPPL'] = res['PPL'].divide(res['hop_sum'], fill_value=0)

#   # normalize path km
#   res['km_sum'] = res['BLKM_SUM'] + res['PPLKM_SUM']
#   res['SharedKmIBL'] = res['BLKM_SUM'].divide(res['km_sum'], fill_value=0)
# #   res['SharedKmBLE'] = res['BLE'].divide(res['km_sum'], fill_value=0)
#   res['SharedKmPPL'] = res['PPLKM_SUM'].divide(res['km_sum'], fill_value=0)
    
  return res.drop(['EntryWeightIBL','EntryWeightBLE','EntryWeightPPL','sum'], axis=1)

# main function
# def hop_optimization(event_data, event_context):
print('start')

# get dataframe from BigQuery
client, shortest_path_df = bq_to_dataframe(query=shortest_path_query)
# _, ridership_df = bq_to_dataframe(query=ridership_query)
_, fare_config_df = bq_to_dataframe(query=fare_config_query)

path_df = shortest_path_df.copy()

# path_df['ArrBLKM'] = path_df['BLKM'].apply(lambda x: np.fromstring(x, dtype=float, sep=','))
# path_df['ArrPPLKM'] = path_df['PPLKM'].apply(lambda x: np.fromstring(x, dtype=float, sep=','))

# path_df['BLKM_SUM'] = path_df['ArrBLKM'].apply(lambda x: np.sum(x))
# path_df['PPLKM_SUM'] = path_df['ArrPPLKM'].apply(lambda x: np.sum(x))

weight = {'IBL':2,'BLE':1,'PPL':1}
feature_df = get_feature(path_df, weight)
# feature_df = feature_df.drop(['ArrBLKM','ArrPPLKM','BLKM_SUM','PPLKM_SUM'], axis=1)

config = fare_config_df.set_index('Line').to_dict()


output = pd.DataFrame()

problem = Problem()

# refers to baht increasing per hop or km
# A refers to blue line
# B refers to Purple line
# C refers to New line
problem.addVariable('B', range(config['MinPrice']['BL'], config['MaxPrice']['BL']+1))#int(config['Step']['BL'] * 100)))
problem.addVariable('P', range(config['MinPrice']['PPL'], config['MaxPrice']['PPL']+1))#int(config['Step']['PPL'] * 100)))
#   problem.addVariable('O', range(ol_dict['MinPrice'], ol_dict['MaxPrice'] +1, ol_dict['Step']))


# constraint
# def fare_limit(a, b):
#     if np.abs(a-b)/100 <= 2:
#         return True

# problem.addConstraint(fare_limit, "BP")
#   problem.addConstraint(fare_limit, "PO")
#   problem.addConstraint(fare_limit, "BO")

solution_found = {}
solutions = problem.getSolutions()

def add_entry_fee(df):
    if df.LineCodeIn == 'BL':
        return df.EntryFeeBL
    elif df.LineCodeIn == 'PPL':
        return df.EntryFeePPL
    else:
        return df.EntryFeeBL

def cal_fare(df):
    
# entry at BL

#             min(min(df.entry_fee+np.round(max(df.BL-1, 0)*df.BL_PPH)+df.OffsetBL, df.MaxInlineBL) + \
#                        np.round(df.PPL*df.PPL_PPH) + (int(df.PPL>0) * df.OffsetPPL),\
#                        df.MaxFareBL)

    if df.ServiceProviderNameIn == 'IBL':
        
        if df.FareTypeBL == 'HOP':
            offset_IBL = np.sum(df.ArrOffsetBL[:df.IBL])
            IBL = min(np.round(max(df.IBL-1, 0)*df.BL_PPH)+offset_IBL, df.MaxInlineBL - df.EntryFeeBL)
            
            offset_BLE = np.sum(df.ArrOffsetBL[:df.BL])
            BLE = min(np.round(max(df.BL-1, 0)*df.BL_PPH)+offset_BLE-IBL, df.MaxInlineBL - df.EntryFeeBL - IBL)
            
#             fare = fare + IBL + BLE
        elif df.FareTypeBL == 'KM':
            IBL = min(np.sum(np.round(df.ArrBLKM[:df.IBL]*df.BL_PPH)), df.MaxInlineBL - df.EntryFeeBL)
            BLE = min(np.sum(np.round(df.ArrBLKM[df.IBL:df.BL]*df.BL_PPH)), df.MaxInlineBL - df.EntryFeeBL - IBL)
            
        
        if df.FareTypePPL == 'HOP':
            offset_PPL = np.sum(df.ArrOffsetPPL[:df.PPL])
            PPL = min(np.round(df.PPL*df.PPL_PPH) + (int(df.PPL>0) * offset_PPL), df.MaxFareBL - df.MaxInlineBL)
            
        elif df.FareTypePPL == 'KM':
            PPL = min(np.sum(np.round(df.ArrPPLKM[:df.PPL]*df.PPL_PPH)), df.MaxFareBL - df.MaxInlineBL) 
        
        fare = df.EntryFeeBL + PPL + IBL + BLE

        return pd.Series([fare,IBL,BLE,PPL])
    
    elif df.ServiceProviderNameIn == 'BLE':
        
        if df.FareTypeBL == 'HOP':
            offset_BLE = np.sum(df.ArrOffsetBL[:df.BLE])
            BLE = min(np.round(max(df.BLE-1, 0)*df.BL_PPH)+offset_BLE, df.MaxInlineBL - df.EntryFeeBL)
            
            offset_IBL = np.sum(df.ArrOffsetBL[:df.BL])
            IBL = min(np.round(max(df.BL-1, 0)*df.BL_PPH)+offset_IBL-BLE, df.MaxInlineBL - df.EntryFeeBL - BLE)
            
#             fare = fare + IBL + BLE
        elif df.FareTypeBL == 'KM':
            BLE = min(np.sum(np.round(df.ArrBLKM[:df.BLE]*df.BL_PPH)), df.MaxInlineBL - df.EntryFeeBL)
            IBL = min(np.sum(np.round(df.ArrBLKM[df.BLE:df.BL]*df.BL_PPH)), df.MaxInlineBL - df.EntryFeeBL - BLE)
        
        if df.FareTypePPL == 'HOP':
            offset_PPL = np.sum(df.ArrOffsetPPL[:df.PPL])
            PPL = min(np.round(df.PPL*df.PPL_PPH) + (int(df.PPL>0) * offset_PPL), df.MaxFareBL - df.MaxInlineBL)
            
        elif df.FareTypePPL == 'KM':
            PPL = min(np.sum(np.round(df.ArrPPLKM[:df.PPL]*df.PPL_PPH)), df.MaxFareBL - df.MaxInlineBL) 
         
        fare = df.EntryFeeBL + PPL + IBL + BLE

        return pd.Series([fare,IBL,BLE,PPL])
#     entry at PPL
    elif df.ServiceProviderNameIn == 'PPL':
        
#         min(min(df.entry_fee+np.round(max(df.PPL-1, 0)*df.PPL_PPH)+df.OffsetPPL, df.MaxInlinePPL) + \
#                        np.round(df.BL*df.BL_PPH) + (int(df.BL>0) * df.OffsetBL), \
#                        df.MaxFarePPL)

        
        if df.FareTypePPL == 'HOP':
            offset_PPL = np.sum(df.ArrOffsetPPL[:df.PPL])
            PPL = min(np.round(max(df.PPL-1, 0)*df.PPL_PPH)+offset_PPL, df.MaxInlinePPL - df.EntryFeePPL)
#             fare = fare + PPL
        elif df.FareTypePPL == 'KM':
            PPL = min(np.sum(np.round(df.ArrPPLKM[:df.PPL]*df.PPL_PPH)), df.MaxInlinePPL - df.EntryFeePPL) 
        
        
        if df.FareTypeBL == 'HOP':
            offset_BLE = np.sum(df.ArrOffsetBL[:df.BLE])
            BLE = min(np.round(df.BLE*df.BL_PPH) + (int(df.BLE>0) * offset_BLE), df.MaxFarePPL - df.MaxInlinePPL)
            
            offset_IBL = np.sum(df.ArrOffsetBL[:df.BL])
            IBL = min(np.round(df.BL*df.BL_PPH)+offset_IBL-BLE, df.MaxFarePPL - df.MaxInlinePPL - BLE)
            
#             fare = np.round(df.BL*df.BL_PPH) + (int(df.BL>0) * df.OffsetBL)
        elif df.FareTypeBL == 'KM':
            BLE = min(np.sum(np.round(df.ArrBLKM[:df.BLE]*df.BL_PPH)), df.MaxFarePPL - df.MaxInlinePPL)
            IBL = min(np.sum(np.round(df.ArrBLKM[df.BLE:df.BL]*df.BL_PPH)), df.MaxFarePPL - df.MaxInlinePPL - BLE)
        
        
        fare = df.EntryFeePPL + PPL + IBL + BLE
        return pd.Series([fare,IBL,BLE,PPL])
    # entry at new line
    else:
        return
    
def passenger(df):
    if df.PassengerType == 'Adult':
        return pd.Series([df.BL_PPH,df.PPL_PPH,df.EntryFeeBL,df.EntryFeePPL,df.MaxInlineBL,df.MaxInlinePPL,df.MaxFareBL,df.MaxFarePPL])
    elif df.PassengerType == 'Student':
        BL_PPH = df.BL_PPH - (0.1 * df.BL_PPH)
        PPL_PPH = df.PPL_PPH - (0.1 * df.PPL_PPH)
        EntryFeeBL = df.EntryFeeBL - np.round(0.1 * df.EntryFeeBL)
        EntryFeePPL = df.EntryFeePPL - np.round(0.1 * df.EntryFeePPL)
        MaxInlineBL = df.MaxInlineBL - np.round(0.1 * df.MaxInlineBL)
        MaxInlinePPL = df.MaxInlinePPL - np.round(0.1 * df.MaxInlinePPL)
        MaxFareBL = df.MaxFareBL - np.round(0.1 * df.MaxFareBL)
        MaxFarePPL = df.MaxFarePPL - np.round(0.1 * df.MaxFarePPL)
        return pd.Series([BL_PPH,PPL_PPH,EntryFeeBL,EntryFeePPL,MaxInlineBL,MaxInlinePPL,MaxFareBL,MaxFarePPL])
    else:
        BL_PPH = df.BL_PPH - (0.5* df.BL_PPH)
        PPL_PPH = df.PPL_PPH - (0.5 * df.PPL_PPH)
        EntryFeeBL = df.EntryFeeBL - np.round(0.5 * df.EntryFeeBL)
        EntryFeePPL = df.EntryFeePPL - np.round(0.5 * df.EntryFeePPL)
        MaxInlineBL = df.MaxInlineBL - np.round(0.5 * df.MaxInlineBL)
        MaxInlinePPL = df.MaxInlinePPL - np.round(0.5 * df.MaxInlinePPL)
        MaxFareBL = df.MaxFareBL - np.round(0.5 * df.MaxFareBL)
        MaxFarePPL = df.MaxFarePPL - np.round(0.5 * df.MaxFarePPL)
        return pd.Series([BL_PPH,PPL_PPH,EntryFeeBL,EntryFeePPL,MaxInlineBL,MaxInlinePPL,MaxFareBL,MaxFarePPL])


# def passenger_fare(df):
#     if df.PassengerType == 'ADULT':
#         return pd.Series([df.fare,df.SharedIBL,df.SharedBLE,df.SharedPPL])
#     elif df.PassengerType == 'STUDENT':
#         fare = df.fare - np.round(0.1 * df.fare)
#         SharedIBL = df.SharedIBL - np.round(0.1 * df.SharedIBL)
#         SharedBLE = df.SharedBLE - np.round(0.1 * df.SharedBLE)
#         SharedPPL = df.SharedPPL - np.round(0.1 * df.SharedPPL)
#         return pd.Series([fare,SharedIBL,SharedBLE,SharedPPL])
#     else:
#         fare = df.fare - np.round(0.5 * df.fare)
#         SharedIBL = df.SharedIBL - np.round(0.5 * df.SharedIBL)
#         SharedBLE = df.SharedBLE - np.round(0.5 * df.SharedBLE)
#         SharedPPL = df.SharedPPL - np.round(0.5 * df.SharedPPL)
#         return pd.Series([fare,SharedIBL,SharedBLE,SharedPPL])
    
# def entry_fee(df):
#     if df.PassengerType == 'ADULT':
#         return df.entry_fee
#     elif df.PassengerType == 'STUDENT':
#         return df.entry_fee - np.round(0.1 * df.entry_fee)
#     else:
#         return df.entry_fee - np.round(0.5 * df.entry_fee)
    
# optimize_table = ridership_df[['Year','Month']].copy().drop_duplicates()
# optimize_table['B'] = 10
# optimize_table['P'] = 10
# print(len(optimize_table))

fare_table = pd.DataFrame()

shortest_path = shortest_path_df.copy()

shortest_path['EntryFeeBL'] = config['EntryFee']['BL']
shortest_path['EntryFeePPL'] = config['EntryFee']['PPL']

shortest_path['MaxInlineBL'] = config['MaxInlineFare']['BL']
shortest_path['MaxInlinePPL'] = config['MaxInlineFare']['PPL']

shortest_path['MaxFareBL'] = config['MaxFare']['BL']
shortest_path['MaxFarePPL'] = config['MaxFare']['PPL']

# OffsetBL = re.sub(r'[‘’]','',config['Offset']['BL'])
# OffsetPPL = re.sub(r'[‘’]','',config['Offset']['PPL'])

shortest_path['OffsetBL'] = config['Offset']['BL']
shortest_path['OffsetPPL'] = config['Offset']['PPL']

shortest_path['ArrOffsetBL'] = shortest_path['OffsetBL'].apply(lambda x: np.fromstring(re.sub(r'[‘’]','',x), dtype=int, sep=','))
shortest_path['ArrOffsetPPL'] = shortest_path['OffsetPPL'].apply(lambda x: np.fromstring(re.sub(r'[‘’]','',x), dtype=int, sep=','))

shortest_path['FareTypeBL'] = 'KM'
shortest_path['FareTypePPL'] = 'KM'

shortest_path['ArrBLKM'] = shortest_path['BLKM'].apply(lambda x: np.fromstring(x, dtype=float, sep=','))
shortest_path['ArrPPLKM'] = shortest_path['PPLKM'].apply(lambda x: np.fromstring(x, dtype=float, sep=','))

shortest_path['FileName'] = config['Filename']['BL']

index = 1

for s in solutions:
  path = shortest_path.copy()
#   ridership = ridership_df.copy()

  path['BL_PPH'] = s['B']
  path['PPL_PPH'] = s['P']
  path['Solution'] = 'Solution' + str(index)
  index = index + 1

#   passenger_type = pd.DataFrame(ridership_df.PassengerType.unique(), columns=['PassengerType'])
#   path = path.assign(tmp=1).merge(passenger_type.assign(tmp=1)).drop('tmp', 1)
    
  path[['BL_PPH','PPL_PPH','EntryFeeBL','EntryFeePPL','MaxInlineBL','MaxInlinePPL','MaxFareBL','MaxFarePPL']] = path.apply(lambda x: passenger(x), axis=1)

  
  path[['NewFare','SharedIBL','SharedBLE','SharedPPL']] = path.apply(lambda x: cal_fare(x), axis=1)
  path['EntryFee'] = path.apply(lambda x: add_entry_fee(x), axis=1)
#   path['new_fare'][path['StationKeyIn']==path['StationKeyOut']] = \
#   path['entry_fee'][path['StationKeyIn']==path['StationKeyOut']]
#   passenger_type = pd.DataFrame(ridership_df.PassengerType.unique(), columns=['PassengerType'])
#   path = path.assign(tmp=1).merge(passenger_type.assign(tmp=1)).drop('tmp', 1)
                                                                              
#   path['entry_fee'] = path.apply(lambda x: entry_fee(x), axis=1)  
#   path[['new_fare','SharedIBL','SharedBLE','SharedPPL']] = path.apply(lambda x: passenger_fare(x), axis=1)
    
#   path['remaining'] = path['new_fare'] - path['entry_fee']
    
  fare_table = pd.concat([fare_table, path], axis=0)

#   ridership = ridership.merge(path, left_on=['StationKeyIn','StationKeyOut','PassengerType'], right_on=['StationKeyIn','StationKeyOut','PassengerType'])

#   ridership['revenue_new_fare'] = ridership['Ridership'] * ridership['new_fare']
#   ridership['revenue_actual_fare'] = ridership['Ridership'] * ridership['Fare'].astype(float)

#   ridership = ridership.groupby(['Year','Month','B','P']).agg({'revenue_new_fare':'sum','revenue_actual_fare':'sum'}).reset_index()
  
#   # optimization
#   ridership['diff'] = ridership['revenue_new_fare'] - ridership['revenue_actual_fare']
#   tmp = ridership.loc[ridership['diff'] > 0]
# #   print(len(tmp))
#   optimize_table = optimize_table.merge(tmp, on=['Year', 'Month'], how='left')
# #   print(len(optimize_table))
#   optimize_table['B'] = np.nanmin(optimize_table[['B_x', 'B_y']], axis=1)
#   optimize_table['P'] = np.nanmin(optimize_table[['P_x', 'P_y']], axis=1)
# #   print(len(optimize_table))
#   optimize_table = optimize_table.drop(['B_x','P_x','B_y','P_y','revenue_new_fare','revenue_actual_fare','diff'], axis=1)
#   print(s)
#   print(len(optimize_table))

    
fare_table = fare_table.drop(['ArrBLKM','ArrPPLKM','ArrOffsetBL','ArrOffsetPPL'], axis=1)
fare_table = fare_table.merge(feature_df, how='outer') #.reset_index().drop('index', axis=1)
# optimize_table = optimize_table.drop('index', axis=1)
    
fare_table['EntryIBL'] = fare_table['EntryIBL'].mul(fare_table['EntryFee'])
fare_table['EntryBLE'] = fare_table['EntryBLE'].mul(fare_table['EntryFee'])
fare_table['EntryPPL'] = fare_table['EntryPPL'].mul(fare_table['EntryFee'])

# fare_table['SharedHopIBL'] = fare_table['SharedHopIBL'].mul(fare_table['remaining'])
# fare_table['SharedHopBLE'] = fare_table['SharedHopBLE'].mul(fare_table['remaining'])
# fare_table['SharedHopPPL'] = fare_table['SharedHopPPL'].mul(fare_table['remaining'])

result_to_bq(fare_table, client, dataset_id='datamart_for_report', dataset_table="km_fare_table", write_disposition="WRITE_TRUNCATE")
# result_to_bq(optimize_table, client, dataset_id='datamart_for_report', dataset_table="hop_optimize_table", write_disposition="WRITE_TRUNCATE")
print('finish')
  

