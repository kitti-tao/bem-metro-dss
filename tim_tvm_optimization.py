import pandas as pd
import numpy as np
from google.cloud import bigquery
from constraint import *

usage_query = """
SELECT
  *
FROM
  `bem-metro-dss.datamart_for_report.report_5_7_0_equipment_optimization_table`
WHERE
  MachineType = 'TIM/TVM'
"""

# upload result dataframe to BigQuery
def result_to_bq(X, client, dataset_id='datamart_for_report', dataset_table="optimization_5_8_test"):
    '''
    Dump result to Bigquery
    '''
    print("Upload prediction result to Bigquery...")
    
    # The project defaults to the Client's project if not specified.
    dataset = client.create_dataset(dataset_id, exists_ok=True)  # API request
    table_ref = dataset.table(dataset_table)
    
    job_config = bigquery.LoadJobConfig(write_disposition="WRITE_TRUNCATE",)

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

def tim_tvm_optimization(event_data, event_context):
  print('start')

  # get dataframe from BigQuery
  client, usage = bq_to_dataframe(query=usage_query)
  print('Done loading Bigquery')

  # fill null value with zero
  usage = usage.fillna(0)

  # prepare data
  throughput_df = usage.groupby(['TransactionTypeName'])['Throughput'].max().reset_index()

  max_throughput = throughput_df.max().values[1]
  throughput_df['Throughput'] = max_throughput/throughput_df['Throughput']

  transaction_df = usage.merge(throughput_df, left_on='TransactionTypeName', right_on='TransactionTypeName', suffixes=('', '_Weighted'))
  transaction_df['Usage'] = transaction_df['Usage']*transaction_df['Throughput_Weighted']

  columns_name = transaction_df.columns.to_list()
  columns_name = [e for e in columns_name if e not in ('Usage','Throughput','Throughput_Weighted','TransactionTypeName')]
  transaction_df = transaction_df.groupby(columns_name)['Usage'].sum().reset_index()
  transaction_df.fillna(0)


  # define decision variable
  problem = Problem()
  problem.addVariable("a", range(1,10))

  # define constraint
  def tim_tvm_constraint(a):
      if a > 0:
        return True

  problem.addConstraint(tim_tvm_constraint)

  # # find objective function
  solutions = problem.getSolutions()
  transaction_df['optimized_quantity'] = len(transaction_df) * [1]
  transaction_df['next_year_optimized_quantity'] = len(transaction_df) * [1]

  for s in solutions:
      transaction_df['optimized_quantity'] = transaction_df.apply(lambda x: s['a'] if x.Percentage/100*15*max_throughput*s['a']-x.Usage >= 0 else x.optimized_quantity, axis=1)
      transaction_df['next_year_optimized_quantity'] = transaction_df.apply(lambda x: s['a'] if x.Percentage/100*15*max_throughput*s['a']-x.GrowthRate*x.Usage >= 0 else x.next_year_optimized_quantity, axis=1)

  # save result to BigQuery
  result_to_bq(transaction_df, client, dataset_id='datamart_for_report', dataset_table="report_5_7_0_tim_tvm_optimization_result_table")
  print('finish')
  
