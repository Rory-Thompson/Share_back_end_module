import pytest

import numpy as np
from Backend_shares_py.dataframe_manager import shares_analysis
from Backend_shares_py.tests.test_data import test_data as test_data
from Backend_shares_py.tests.test_data import RSI_test_data2 as RSI_test_data2
import copy
print(type(test_data))
import pandas as pd 
def test_tests():
    obj ="This is a test"
    assert obj == "This is a test"

def test_my_data():
    window = 14
    min_periods = 14
    test_res = pd.DataFrame(test_data, columns = test_data.keys())#by default the output will be captured by pytest
    print(test_res.keys())
    print(f"test_res.dtypes{test_res.dtypes}")
    Test_df_manager = shares_analysis(location_base = r"\\DiskStation\Data\trading\files", shares_df= test_res)
    Test_df_manager.calc_rsi(min_periods = min_periods)
    print(f'the length of day df is: {len(Test_df_manager.day_df)}')
    
    assert_string = f"the lengths of the dataframes are not the same. This should not be possible. df_manager: {len(Test_df_manager.day_df)}, " \
        f"test_res before df_manager creation: {len(test_res)}"#Note if u miss days in the test data this could also cause issues. (no weekends required) 
    name_RSI = f'RSI_window_{window}_periods_{min_periods}'
    name_gain = f'avg_gain_window_{window}_periods_{min_periods}'
    name_loss = f'avg_loss_window_{window}_periods_{min_periods}'
    assert len(Test_df_manager.day_df) == len(test_res), assert_string
    
    
    Test_df_manager.day_df['RSI_test_value'] = test_res['RSI_test_value']
    pd.set_option("display.max_columns",15)
    test_val_res = Test_df_manager.day_df[~Test_df_manager.day_df['RSI_test_value'].isna()]#remove na values as na != na
    test_val = test_val_res[name_RSI].round(1) == test_val_res['RSI_test_value'].round(1)
    print(Test_df_manager.day_df.dtypes)
    print(test_val_res[~test_val][[name_RSI,'RSI_test_value']].round(1))

    assert test_val.all(), 'the RSI does not match the test_RSI real values.'


    ###test the updating of RSI. 
    print("beggining second test")
    test_df = copy.deepcopy(Test_df_manager.day_df.iloc[1:,:])
    #parse back in to create a new object
    #remove the first value. and make sure values are all the same after recalculation.
    Test_df_manager = shares_analysis(location_base = r"\\DiskStation\Data\trading\files", shares_df= test_df)
    Test_df_manager.calc_rsi(min_periods = min_periods)
    test_val_res = Test_df_manager.day_df[~Test_df_manager.day_df['RSI_test_value'].isna()]#remove na values as na != na
    test_val = test_val_res[name_RSI].round(1) == test_val_res['RSI_test_value'].round(1)
    print(test_val_res[~test_val][[name_RSI,'RSI_test_value']].round(1))

    assert test_val.all(), 'the RSI does not match the test_RSI real values.'



def test_nan_values():
    window = 14
    min_periods = 14
    test_res = pd.DataFrame(test_data, columns = test_data.keys())#by default the output will be captured by pytest
    test_res.loc[15,"change"] = np.nan#change 15th value to be na. so now the 15th value should == the 14th value after calculation .
    print(test_res.keys())
    print(f"test_res.dtypes{test_res.dtypes}")
    Test_df_manager = shares_analysis(location_base = r"\\DiskStation\Data\trading\files", shares_df= test_res)
    Test_df_manager.calc_rsi(min_periods = min_periods)
    name_RSI = f'RSI_window_{window}_periods_{min_periods}'
    name_gain = f'avg_gain_window_{window}_periods_{min_periods}'
    name_loss = f'avg_loss_window_{window}_periods_{min_periods}'
    print(Test_df_manager.day_df[["change","aest_day_datetime",name_RSI,name_gain, name_loss]])
    assert Test_df_manager.day_df.iloc[15,:][name_RSI] ==Test_df_manager.day_df.iloc[14,:][name_RSI], "the na value at index 15 is not the same as the previous RSI value."




def test_full_update():
    min_periods = 14
    window = 14
    name_RSI = f'RSI_window_{window}_periods_{min_periods}'
    name_gain = f'avg_gain_window_{window}_periods_{min_periods}'
    name_loss = f'avg_loss_window_{window}_periods_{min_periods}'
    test_res = pd.DataFrame(test_data, columns = test_data.keys())
    Test_df_manager = shares_analysis(location_base = r"\\DiskStation\Data\trading\files", shares_df= test_res)
    Test_df_manager.calc_rsi(min_periods = min_periods)
    #add an extra day 
    new_test_day = {
        'last': [243.20],
        "day_date_key": ["2025-02-13"],
        "updated_at": ["2025-02-13T04:30:01.000000Z"],
        "change": [1.67],
        "code": ["CBA"],
        "sector": ["finacials"],
        "ytd_percent_change": [5],
        "month_percent_change": [3],
        'week_percent_change': [2],
        'title': ["Commonwealth Bank of Australia"],
        'RSI_test_value':[55.88]

    }
    new_test_df = pd.concat([Test_df_manager.day_df,pd.DataFrame(new_test_day)])#append new test data
    new_test_df = new_test_df.sort_values(by=['code', 'aest_day_datetime']).reset_index(drop=True)
    new_test_df_slice = new_test_df.iloc[-5:,:]
    Test_df_manager_2 = shares_analysis(location_base = r"\\DiskStation\Data\trading\files", shares_df= new_test_df_slice)

    Test_df_manager_2.calc_rsi(min_periods = min_periods)
    test_val_res = Test_df_manager_2.day_df[~Test_df_manager_2.day_df['RSI_test_value'].isna()]#remove na values as na != na
    test_val = test_val_res[name_RSI].round(1) == test_val_res['RSI_test_value'].round(1)
    print(f"results test_full_update")
    print(test_val_res[~test_val][[name_RSI,'RSI_test_value']].round(1))

    assert test_val.all(), 'the RSI does not match the test_RSI real values.'



def test_multiple_codes():
    min_periods = 14
    window = 14
    name_RSI = f'RSI_window_{window}_periods_{min_periods}'
    name_gain = f'avg_gain_window_{window}_periods_{min_periods}'
    name_loss = f'avg_loss_window_{window}_periods_{min_periods}'
    df_0 = pd.DataFrame(test_data, columns = test_data.keys())
    df_1 = pd.DataFrame(RSI_test_data2)
    df_test = pd.concat([df_0,df_1])
    Test_df_manager_2 = shares_analysis(location_base = r"\\DiskStation\Data\trading\files", shares_df= df_test)
    Test_df_manager_2.calc_rsi(min_periods = min_periods)
    test_val_res = Test_df_manager_2.day_df[~Test_df_manager_2.day_df['RSI_test_value'].isna()]#remove na values as na != na
    test_val = test_val_res[name_RSI].round(1) == test_val_res['RSI_test_value'].round(1)
    print(f"results test_full_update")
    print(test_val_res[test_val][[name_RSI,'RSI_test_value']].round(1))

    assert test_val.all(), 'the RSI does not match the test_RSI real values.'

    # now do the test. 



def test_multiple_codes_update():
    min_periods = 14
    window = 14
    name_RSI = f'RSI_window_{window}_periods_{min_periods}'
    name_gain = f'avg_gain_window_{window}_periods_{min_periods}'
    name_loss = f'avg_loss_window_{window}_periods_{min_periods}'
    df_0 = pd.DataFrame(test_data, columns = test_data.keys())
    df_0_first = df_0.iloc[0:18,:]#not inclusive of 15 
    df_1 = pd.DataFrame(RSI_test_data2)
    
    df_1_first = df_1.iloc[0:18,:]
    df_test = pd.concat([df_0_first,df_1_first], ignore_index=True)
    print("raw data frame after concat of 2 dataset")

    Test_df_manager_1 = shares_analysis(location_base = r"\\DiskStation\Data\trading\files", shares_df= df_test)

    print("raw data frame after day_df initialisation before RSI calculation")


    Test_df_manager_1.calc_rsi(min_periods = min_periods)
    
    #make sure initial values are correct.
    test_val_res = Test_df_manager_1.day_df[~Test_df_manager_1.day_df['RSI_test_value'].isna()]#remove na values as na != na (only remove NA in the test data)
    test_val = test_val_res[name_RSI].round(1) == test_val_res['RSI_test_value'].round(1)
    print(f"results test_full_update")
    print(test_val_res[~test_val][[name_RSI,'RSI_test_value','code']].round(1))
    
    assert test_val.all(), 'the RSI does not match the test_RSI real values. (no update)'


    #now take only the last row of each code and see if it picks ip up and continues updating from it. 
    df_0_second = df_0.iloc[18:,:]#not inclusive of 15 
    
    df_1_second = df_1.iloc[18:,:]
    df_test_2 = pd.concat([df_0_second,df_1_second],ignore_index = True)
    last_values = Test_df_manager_1.day_df.groupby('code').last().reset_index()
    final_test = pd.concat([last_values, df_test_2],ignore_index = True)
    Test_df_manager_2 = shares_analysis(location_base = r"\\DiskStation\Data\trading\files", shares_df= final_test)
    
    Test_df_manager_2.calc_rsi(min_periods = min_periods)
    test_val_res = Test_df_manager_2.day_df[~Test_df_manager_2.day_df['RSI_test_value'].isna() & ~Test_df_manager_2.day_df['RSI_test_value'].isna()]#remove na values as na != na
    test_val = test_val_res[name_RSI].round(1) == test_val_res['RSI_test_value'].round(1)
    
    print(f"results test_full_update")
    print(test_val_res[~test_val][[name_RSI,'RSI_test_value']].round(1))
    
    assert test_val.all(), 'the RSI does not match the test_RSI real values. after update'





test_my_data()