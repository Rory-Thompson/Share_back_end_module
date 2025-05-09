

import pandas as pd


import pandas as pd
from pytz import timezone

aest = timezone('Australia/Sydney')


from matplotlib.colors import to_rgb


class TradingModel:
    
    '''
    A model is created with particular configurations
    configuration format is important.
    
    example model config file json. Multiple different gradients and moving averages can be had. 
    model_config = [
    {"type": "moving_average",
        "day_long": 21,
        "day_small":9,
        "difference_threshold_max": 5,  # Difference % between high and low
        "difference_threshold_min": 5, #when the difference threshold falls between this range between the long day and short day it will be True
        "buy_status": True, #if the shorter average is above the longer average.
        'column':'last',
        "min_streak": 3,  #the minimum and maximum number of days It has been buy status
        "max_streak": 7  
    },
    {"type":"gradient_average",
        "columns" = ['last'],
        "num_days": [9, 20, 3, 4, 5, 6],  #different rolling averages that will be averaged
        "greater_than": 0.5,  #gradient average min value
        "less_than": 2.0  #gradient average max value
    }
]
    
    '''
    def __init__(self, name, config, shares_analysis):
        self.name = name
        self.config = config
        self.shares_analysis = shares_analysis  # Reference to the main class
        self.results = []  # Store evaluation results
        self.shares_analysis.models[self.name] = self
        
        #	10/30_model_%_difference example name 
        
        self.config_df = pd.DataFrame(self.config)
        #the create model must now be run manually. 
        
        
    
    def create_model(self):
        
        for model_val in self.config:
            if model_val['type'] =='moving_average':
                print(model_val['day_long'])
                print(model_val['day_small'])
                
                self.shares_analysis.create_smoothing_function_model(day_long = model_val['day_long'],
                                                                     day_small = model_val['day_small'])
            elif model_val['type'] =='gradient_average':
                self.shares_analysis.calc_gradient_average(num_days=model_val['num_days'],
                                                           columns=[model_val['column']])
            elif model_val['type'] =='RSI':
                
                self.shares_analysis.calc_rsi(window = model_val["window"], min_periods = model_val["min_periods"])
                
                
    def share_test_values_get(self,df_series):
        
        
        '''
        This should work for both series and individual shares.
        input: either, dataframe containing relevant columns or series where the index is the column names

        output: for dataframe: a pandas series containing a True or False value, True being they should be bought. It should match the index of the original df_series inputted.
        
        '''
        lst_cur_res = []
        for model_val in self.config:
            
            
            if model_val['type'] =='moving_average':
                title = f'{model_val["day_small"]}/{model_val["day_long"]}_'
                #title_long = f'rolling average {day_long}'
                #title_small = f'rolling average {day_small}'
                #title = f'{day_small}/{day_long}_model_buy_status'
                #f'{day_small}/{day_long}_model_%_difference'
                
                
                difference_threshold = (
                (abs(df_series[title+'model_%_difference'])>= model_val["difference_threshold_min"]) & 
                (abs(df_series[title+'model_%_difference'])<= model_val["difference_threshold_max"])
                )
                
                
                streak_length = (
                (df_series[title+'streak_length'] >= model_val["min_streak"]) & 
                (df_series[title+'streak_length']<= model_val["max_streak"])
                )
                #global test7
                #test7=pd.concat([difference_threshold,streak_length], axis=1)
                
                diff_thresh_min = (abs(df_series[title+'model_%_difference'])>= model_val["difference_threshold_min"]).sum()
                diff_thresh_max = (abs(df_series[title+'model_%_difference'])<= model_val["difference_threshold_max"]).sum()
                min_streak = (df_series[title+'streak_length'] >= model_val["min_streak"]).sum()
                max_streak = (df_series[title+'streak_length'] <= model_val["max_streak"]).sum()

                print(model_val['type'])
                print(f"diff_thresh_min len {diff_thresh_min}")
                print(f"diff_thresh_max len {diff_thresh_max}")
                print(f"min_streak len {min_streak}")
                print(f"max_streak len {max_streak}")

                res = streak_length & difference_threshold & df_series[title+'model_buy_status']
                print(f" total for {model_val['type']}: {res.sum()}")
                lst_cur_res.append(res)
                
                
                
            
            elif model_val['type'] =='gradient_average':
                columns = [model_val["column"]]
                title = "_".join(columns) + f"_num_days_{'_'.join(map(str, model_val['num_days']))}_average"
                
                res = (
                (df_series[title] >= model_val['greater_than']) & 
                (df_series[title] <= model_val['less_than'])
                )

                gradient_min = (df_series[title] >= model_val['greater_than']).sum()
                gradient_max = (df_series[title] <= model_val['less_than']).sum()
                print(model_val['type'])
                print(f"gradient_min len {gradient_min}")
                print(f"gradient_max len {gradient_max}")
                print(f" total for {model_val['type']}: {res.sum()}")
                lst_cur_res.append(res)
            elif model_val['type'] =='RSI':
                #True if it is less than the RSI min value.
                title = f'RSI_window_{model_val["window"]}_periods_{model_val["min_periods"]}'

                res = df_series[title]<= model_val["rsi_max"]

                lst_cur_res.append(res)
                RSI_max = (df_series[title]<= model_val["rsi_max"]).sum()

                print(model_val['type'])
                print(f"RSI_max len {RSI_max}")
                print(f" total for {model_val['type']}: {res.sum()}")
        
        print(len(lst_cur_res))
                    
                
        final_res = pd.concat(lst_cur_res, axis=1)
        print(len(final_res.columns))
        
        final_res_2 = (final_res.sum(axis=1) == len(final_res.columns))
        self.results = final_res_2    
        return final_res_2#either returns a boolean or a True false value.
        
            
                
            
        
        
    def shares_to_buy_now(self):
        
        if self.shares_analysis.model_res_df:
            pass
         
                
                
            
            
