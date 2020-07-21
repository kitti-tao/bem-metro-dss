import pandas as pd
import numpy as np
from google.cloud import bigquery
from constraint import *

density_query = """
  SELECT
    *
  FROM
    `bem-metro-dss.datamart_for_report.report_5_7_0_train_optimization_table`
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

def train_optimization(event_data, event_context):
  print('start')

  # get dataframe from BigQuery
  client, density = bq_to_dataframe(query=density_query)
  print('Done loading Bigquery')

  # fill null value with zero
  density = density.fillna(0)

  # define decision variable
  problem = Problem()
  problem.addVariable("a", range(1,10))

  # define constraint
  def train_constraint(a):
    if a > 0:
      return True

  problem.addConstraint(train_constraint)

  # find objective function
  solutions = problem.getSolutions()
  density['NextYearDensity'] = density['GrowthRate']*density['Density']
  density['optimized_quantity'] = len(density) * [1]
  density['next_year_optimized_quantity'] = len(density) * [1]

  for s in solutions:
    density['optimized_quantity'] = density.apply(lambda x: s['a'] if x.PersonCountMax*s['a']-x.Density >= 0 else x.optimized_quantity, axis=1)
    density['next_year_optimized_quantity'] = density.apply(lambda x: s['a'] if x.PersonCountMax*s['a']- x.NextYearDensity >= 0 else x.next_year_optimized_quantity, axis=1)
  
  # save result to BigQuery
  result_to_bq(density, client, dataset_id='datamart_for_report', dataset_table="report_5_7_0_train_optimization_result_table")
  print('finish')
  
