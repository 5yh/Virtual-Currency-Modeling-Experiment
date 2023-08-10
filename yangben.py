from ctgan import CTGAN
# from ctgan import read_csv
import numpy as np
import pandas as pd


real_data = pd.read_csv('/home/lr/zengliang_test/zengqiang_0.01.csv')
real_data = real_data.sample(frac = 0.01)

# Names of the columns that are discrete
discrete_columns = ['label','Total_transactions', 'Number_of_received_addresses', 'Number_of_sent_addresses', 'Call_TRC10', 'Min_value_received', 'Max_value_received', 'Avg_value_received', 'Total_ether_sent_for_accounts', 'Min_value_sent', 'Max_value_sent', 'Avg_value_sent', 'Total_ether_received_for_accounts', 'Min_value_sent_to_trc10', 'Max_value_sent_to_trc10', 'Avg_value_sent_to_trc10', 'Total_ether_sent_to_trc10', 'outDegree', 'inDegree', 'Avg_time_between_sent', 'Avg_time_between_received', 'Time_between_first_last', 'mean_Total_transactions', 'mean_Number_of_received_addresses', 'mean_Number_of_sent_addresses', 'mean_Call_TRC10', 'mean_Min_value_received', 'mean_Max_value_received', 'mean_Avg_value_received', 'mean_Total_ether_sent_for_accounts', 'mean_Min_value_sent', 'mean_Max_value_sent', 'mean_Avg_value_sent', 'mean_Total_ether_received_for_accounts', 'mean_Min_value_sent_to_trc10', 'mean_Max_value_sent_to_trc10', 'mean_Avg_value_sent_to_trc10', 'mean_Total_ether_sent_to_trc10', 'mean_outDegree', 'mean_inDegree', 'mean_Avg_time_between_sent', 'mean_Avg_time_between_received', 'mean_Time_between_first_last']

# discrete_columns = [
#     'workclass',
#     'education',
#     'marital-status',
#     'occupation',
#     'relationship',
#     'race',
#     'sex',
#     'native-country',
#     'income'
# ]

ctgan = CTGAN(epochs=1)
ctgan.fit(real_data, discrete_columns)

# Create synthetic data
synthetic_data = ctgan.sample(10)
print(synthetic_data)