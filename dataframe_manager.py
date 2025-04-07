import requests
import pandas as pd
import json
import logging
import os
from tqdm import tqdm
import pandas as pd
from pytz import timezone
import warnings
import numpy as np
import requests
aest = timezone('Australia/Sydney')
import traceback
import time
import yfinance as yf

logging.basicConfig(level=logging.INFO)  # Default logging level


class shares_analysis:
    
    '''
    Notes to improve. This class must update the self.shares_df and self.model_res_df, and constantly has complex logic 
    copied and pasted, some sort of auxilary method that updates a column with an input of a series. Would make it alot cleaner. \
    
    Save memory, need to cache the entire dataset and see how it runs. 
    issue is it will create self.shares_df from the cache_day_df location. Then a bunch of code is run to create the day df again.
    This means we are doubling the amount of memory required. Instead if the get_cache_df is true and successfully loads, we pass it directly to day_df and dont have a 
    self.shares_df.
    
    '''
    
    def __init__(self, location_base,shares_df=None,cache_df_location = None):

        '''
        Description of some of the arguments.
        location_base: used for the model_res_df as well as the yfinance data.
        shares_df: used to pass if you wish a pandas dataframe instead of using cached data
        
        '''
       
        self.day_df = None#initialization
        self.shares_df = shares_df#is None, not the best logic. will be set to parameter shares_df if shares_df is not none and get_cache_df is True
        aest = timezone('Australia/Sydney')
        self.df_is_updated = True

        #in all cases we want to read the yfinance results. It is broken constantly. with api client errors of too many requests. 
        logging.info("reading metrics yfinance results")
        self.yfinance_location = os.path.join(location_base, "rory_model_results","yfinance_results.csv") 
        self.share_metric_df= pd.read_csv(self.yfinance_location,index_col = "code")
        logging.info("metric df yfinance has been read")
       
        self.model_res_df = pd.DataFrame()
    
        
        #the only way for day_df to be not None is if cache_df was used and was successful.
        #this is a requirement for the proceeding lines. 
        logging.debug(f"{type(self.shares_df)} shares df type in initialization")
        self.all_codes = self.shares_df['code'].unique()


        #the below logic requires inplace operations. df_complete is pointing to the same reference objects as either self.shares_df or self.day_df.
        #if a copy is done it will no longer point to the same location, doubling memory usage.
        self.shares_df["updated_at"]= pd.to_datetime(self.shares_df["updated_at"])
        

        #the below lines are done if a cashed
        
        self.shares_df['aest_time'] = self.shares_df['updated_at'].dt.tz_convert(aest)
        self.shares_df["aest_day"] = self.shares_df["aest_time"].dt.strftime('%d/%m/%Y')
        self.shares_df.reset_index(inplace=True,drop = True)#if this is not done via inplace it will create a copy. also need to drop useless index column
        
        
        
        self.moving_average = {}
        #self.gen_share_lst()#Note this does not update anything if using cache df. although it probably should.
        self.models = {}
        self.averages_calculated = []
        type(self.day_df)
        self.day_df is None and self.create_day_df()#if self.day_df is None (implies it must be created from shares_df) Then it will do the function

        self.columns_to_drop = ["path","is_asr", "star_stock","status", "deleted_at","type"]
        self.day_df = self.day_df[list(set(list(self.day_df.columns)) - set(self.columns_to_drop))]

        #assume the day df is created we will update the date time values and sort it just in case.
        assert self.day_df["aest_day"].dtype == "object", f'Column is not of type string column is {self.day_df["aest_day"].dtype}'
        logging.debug(f"first value in day df type: {type(self.day_df.iloc[0,:]['aest_day'])}")
        self.day_df['aest_day_datetime'] = pd.to_datetime(self.day_df['aest_day'], format="%d/%m/%Y")
        #self.save_day_df_cache()df

        self.day_df.dropna(subset=['aest_day_datetime'], inplace=True)#there is not much point having data that has no day datetime, it causes issues down the line. 
        self.update_price_model_res_df()
        self.df_is_updated = False
        self.completed_tickers = []
        
    
    def get_all_raw_data(self):
        files = os.listdir(self.json_raw_location)
        
                
        df_data = pd.DataFrame()
        
        print(f"init function shares_analysis running, creating day_to_df from scratch")
        for file in tqdm(files):
            
            df_temp = self.get_data_to_df(file)
            df_data = pd.concat([df_data,df_temp])
            self.saved_files_this_run.append(file)
        self.shares_df = df_data



    def get_data_to_df(self,file):
        full_path = os.path.join(self.json_raw_location, file)
        df_data = pd.DataFrame()
        temp = pd.read_json(full_path)
        df_data = pd.concat([temp,df_data], join = "outer")
        lst = []
        def lambda_func(a):
            if a['company_sector'] != None:
                #print(a['company_sector']['gics_sector'])
                ind = a.index
                lst.append(a['company_sector']['gics_sector'].replace(' ','_'))
            else:
                lst.append("No_industry")
        df_data.apply(lambda_func, axis = 1)
        df_data['sector']= lst
        self.saved_files_this_run.append(file)
        return df_data

    def update_price_model_res_df(self):
        
        '''
        updates the 'last 
        
        '''
        
        idx = self.day_df.groupby('code')['aest_day_datetime'].idxmax()
        temp_df = self.day_df.loc[idx]
        #we now have the latest for day for each code in the day df. 
        # needs to handle if empty, currently doesnt. 
        temp_df = temp_df.set_index('code')
        columns = ['change', 'ytd_percent_change', 'month_percent_change','week_percent_change', 'company_sector', 'updated_at','last','aest_day']
        self.model_res_df[columns] = temp_df[columns]
        
        
        
            
    
    def create_day_df(self):
        idx = self.shares_df.groupby(["code","aest_day"])["aest_time"].idxmax()#should be the lowest idx. 
        temp_df = self.shares_df.loc[idx]
        temp_df = temp_df.sort_values("aest_time")
        full_dates = pd.date_range(start=temp_df["aest_time"].min(), end=temp_df["aest_time"].max(), freq='B',normalize = True)# we must normalize. date_range only works best with dates not datetimes.

        
        #FIXED LFG
        ##create multi index dataframe
        full_df = pd.DataFrame(
            [(date.strftime('%d/%m/%Y'), code) for date in full_dates for code in self.all_codes],
            columns=['aest_day', 'code']
        )
        merged_df = pd.merge(full_df, temp_df, on = ["aest_day", "code"], how="left")
        self.day_df = merged_df
        self.day_df['aest_day_datetime'] = pd.to_datetime(self.day_df['aest_day'], format="%d/%m/%Y")
        
        
        
        
    def calc_moving_average(self, num_days,min_periods,start_date =None, end_date= None, shares_codes=[]):
        if len(shares_codes) == 0:
            shares_codes = list(self.all_codes)
        
        '''
        input
        num_days = 50. means an average will be calculated using 50 days
        min_periods, the number of days requried for a valid entry (will exclude the missing days if above)
        min_periods is handy for when 1 day is missing. 
        share_codes = ["BHP"],
        start date. #not used
        end date. #not used. 
        
        a warning will be returned if the start  date and end date do not have enough days to be passed
        
        output: a dataframe, each row is an entry of a smoothed share proce for a particular code. 
        
        
        '''
        title = f'rolling average {num_days}'
        self.averages_calculated.append(num_days)
        self.day_df[title] = self.day_df.groupby('code')['last'].transform(lambda x: x.rolling(window=num_days, min_periods = min_periods).mean())
        logging.info("shares analysis, calc moving average done: "+title)
            
    def gen_share_lst(self,extra_metrics = [],codes_to_update =[]):
        """
        What does this do? 
        talks to the yfinance api. It will search for all codes in the metric
        """
        base_metrics = ["longName","industry","sector","marketCap","trailingPE",'forwardPE','trailingEps','forwardEps','returnOnAssets','earningsGrowth','revenueGrowth','totalRevenue',
         'returnOnAssets','returnOnEquity','profitMargins','dividendYield']
        metrics = list(set(extra_metrics) | set(base_metrics))
        
        self.df_lst = []
        #here we take all codes in day_df, then - codes already in the share metric df, and these codes as well as any new share codes passed into the function.
        list(self.share_metric_df.index)
        tickers = list(set(list(self.all_codes)) - set(list(self.share_metric_df.index)))

        tickers = list(set(tickers) | set(codes_to_update))

        logging.info(f"amount of codes to check len(tickers)")
        tickers = [code + '.AX' for code in tickers]
        for ticker_name in tickers:
            
            ticker_res = {}
            ticker_res["code"] = ticker_name[:-3]
            
            success = False
            retries = 0
            
            while not success and retries < 1:
                try:
                    ticker = yf.Ticker(ticker_name)
                    try:
                        hist = ticker.history(period="1d")
                        ticker_res["datetime"] = hist.index[-1] if not hist.empty else np.nan
                        
                    except Exception as e:
                        print(f"❌ History data failed for {ticker_name}: {e}")
                        ticker_res["datetime"] = np.nan
                        
                    ticker_info = ticker.info
                    
                    
                    for metric in metrics:
                        ticker_res[metric] = ticker_info.get(metric, np.nan)
                        
                    self.df_lst.append(ticker_res)
                    success = True
                    self.completed_tickers.append(ticker_name)
                    
                except Exception  as e:
                    print(f"❌ Full error for {ticker_name}: {e}")
                    print("Traceback details:")
                    traceback.print_exc()  # Prints the full traceback
                    #cannot find a way to know if it is a rate limit error. 
                      # Rate limit error
                    retries += 1
                    wait_time = 2 ** retries  # Exponential backoff
                    print(f"⚠️ Rate limit hit! Waiting {wait_time}s before retrying ({retries}/5)...")
                    time.sleep(wait_time)
                    #else:
                        #print(f"❌ Failed to fetch {ticker_name}: {e}")
                        #break  # Skip this ticker after other errors
        res = pd.DataFrame(self.df_lst)
        #res.set_index("code")
        
        #for some reason this code is cooked, needs to save as proper df before update can work
        self.gen_share_df()
        return res
    
    def gen_share_df(self):
        '''
        Auxillary method to update yfinance. It only updates data if it is new.
        This does not happen for nas. if the value is na or nan in new df from df_lst it will not update existing numbers. 
        This is called after gen_share_df is run .
        '''
        
        if len(self.df_lst)>0:
            res = pd.DataFrame(self.df_lst)

            res.set_index("code")
            #now self.df_lst is updated completelty, but share metric df must not be, it should only be updated,
            #test this out, will obviously take some time
            self.share_metric_df.update(res)
            
            #we assume that because the yfinance_results are loaded upon the initiation of shares_analysis.
            #it is the FULL dataset. Hence we can then update it, which should only updates changes and not delete data.

            self.share_metric_df.to_csv(self.yfinance_location)
            
                    
                    
                    
        
        
    def create_smoothing_function_model(self,day_long, day_small, min_periods_long= None, min_periods_small = None):
        assert day_long > day_small, "day long must be greater than day small"
        
        '''
        note that the aest_time and aest_datetime are a bit flawed, it will update whenever run, but models that are not 
        updated will technically have the wrong datetime, so you need to update every model at each time to make this 
        value accurate. or you have seperate date time values for each model which makes more sense in my opinion. 
        '''
        if min_periods_long == None:
            min_periods_long = int(day_long//1.4)
        if min_periods_small == None:

            min_periods_small = int(day_small//1.4)
        
        #whenever this is called we want to update the values. 
        self.calc_moving_average(num_days = day_long, min_periods = min_periods_long)
        
            
        self.calc_moving_average(num_days = day_small, min_periods = min_periods_small)
            
        #required averages have been calculated. 
        title_long = f'rolling average {day_long}'
        title_small = f'rolling average {day_small}'
        #if the shorter day period price is greater than the longer period price, it is a buy.
        self.day_df[f'{day_small}/{day_long}_model_buy_status'] = self.day_df[title_small]>self.day_df[title_long]
        self.day_df[f'{day_small}/{day_long}_model_difference'] = self.day_df[title_small]-self.day_df[title_long]
        self.day_df[f'{day_small}/{day_long}_model_%_difference'] = (self.day_df[f'{day_small}/{day_long}_model_difference']/(self.day_df['last']))*100
        
        
        ##these next lines are genius
        #ne returns the same dataframe of True False values.
        #shift, returns previous value, that means we need to sort by code and aest_day_datetime
        #this means it checks if the previous day is different from the sameday
        #cumsum (hehe) then adds 1 to a cumulative sum through the series whenever there was a change
        #we then create the buy streaks series that will show the size of each buy streak,
        #we can then join these buy streaks back into the dataset.
        title = f'{day_small}/{day_long}_model_buy_status'
        self.day_df = self.day_df.sort_values(by=['code', 'aest_day_datetime']).reset_index(drop=True)
        self.day_df[f'{day_small}/{day_long}_buy_streak'] = self.day_df.ne(self.day_df.shift())[title].cumsum()
        
        
        streak_length = self.day_df.groupby(['code', f'{day_small}/{day_long}_buy_streak']).size()
        multi_index = pd.MultiIndex.from_frame(self.day_df[['code', f'{day_small}/{day_long}_buy_streak']])#creates a multi index that can be used to map streak length back
        #this dataframe now is buy streaks 
        self.day_df[f'{day_small}/{day_long}_streak_length'] = multi_index.map(streak_length)

        #below is the cleanest way to update model_res_df
        extra_cols = set(["aest_day","aest_day_datetime"])
        extra_cols_to_update = list(set(list(self.model_res_df.columns)) & extra_cols)
        temp_df = self.day_df[["code",
                               f'{day_small}/{day_long}_model_buy_status',
                               f'{day_small}/{day_long}_model_difference',
                               f'{day_small}/{day_long}_model_%_difference', 
                               f'{day_small}/{day_long}_streak_length']+
                               list(extra_cols)]
            
        
        idx = temp_df.groupby('code')['aest_day_datetime'].idxmax()
        temp_df = temp_df.loc[idx]
        temp_df.set_index('code', inplace=True)
        logging.info(f"Duplicated index values temp_df: {pd.Series(temp_df.index)[pd.Series(temp_df.index).duplicated()]}")
        logging.info(f"Duplicated columns values temp_df: {pd.Series(temp_df.columns)[pd.Series(temp_df.columns).duplicated()]}")
        logging.info(f"Duplicated index values self.model_res_df: {pd.Series(self.model_res_df.index)[pd.Series(self.model_res_df.index).duplicated()]}")
        logging.info(f"Duplicated columns values self.model_res_df: {pd.Series(self.model_res_df.columns)[pd.Series(self.model_res_df.columns).duplicated()]}")
        self.model_res_df[list(temp_df.columns)] = temp_df#update all columns in temporary df
        #temp_df now contains the latest values for each share only, then can be merged
        self.day_df.drop(columns = [f'{day_small}/{day_long}_buy_streak',f'{day_small}/{day_long}_model_difference' ], inplace = True)#drop columns to save memory. 
            
    def calc_gradient(self, num_days=[5], columns=[]):
        """
        Calculate the gradient (rate of change) for each share price over a given number of days.

        Parameters:
        num_days (int): The number of days to calculate the change. Default is 5.
        column must be a valid column inside of the day_df dataframe. 
        Returns:
        Adds a new column 'gradient_<num_days>' to self.day_df with the calculated gradient.
        
        due to the nature of it being column for column , we can assume that if this column already exists it will update fine. 
        
        
        """
        for day in num_days:
            self.day_df = self.day_df.sort_values(by=['code', 'aest_day_datetime']).reset_index(drop=True)
            col_names = []
            for col in columns:
                logging.info(f"calculating gradient for {col}")
                gradient_col_name = f'gradient_{col}_{day}'
                col_names.append(gradient_col_name)
                # Ensure the data is sorted by 'code' and 'aest_day_datetime'


                # Calculate the gradient using the difference in 'last' over the difference in days
                self.day_df[gradient_col_name] = self.day_df.groupby('code')[col].transform(
                    lambda x: (x - x.shift(day)) / day
                )


            #now we need to update the self.model_res_df
            idx = self.day_df.groupby('code')['aest_day_datetime'].idxmax()
            temp_df = self.day_df.loc[idx]
            temp_df = temp_df[["code","aest_day"]+col_names]
            #we now have the latest prices stored in the temp df 
            temp_df = temp_df.set_index("code")
            self.model_res_df
            self.model_res_df.update(temp_df)
            logging.info(temp_df.columns)

            # Add new columns
            for col in temp_df.columns:

                if col not in self.model_res_df.columns:
                    self.model_res_df[col] = temp_df[col]
            
        

    
    
    
    def calc_gradient_average(self, num_days=[5], columns=[]):
        """
        Uses the calculate gradient method then averages over the different num day column pairs. 
        makes sense to only input 1 column at a time. 
        
    
        """
        self.calc_gradient(num_days = num_days, columns = columns)
        new_column_names = [f'gradient_{col}_{days}' for col in columns for days in num_days]
        
        name = "_".join(columns) + f"_num_days_{'_'.join(map(str, num_days))}_average"
        
        self.day_df[name] = self.day_df[new_column_names].mean(axis = 1)
        self.model_res_df[name] = self.model_res_df[new_column_names].mean(axis =1)
        
        
    def generate_results_to_day_df(self, model):
        model.create_model()#make sure columns are generated. 
        self.day_df[f'{model.name}_result'] = model.share_test_values_get(df_series = self.day_df)
        
        
                
            
    def calc_rsi(self, window = 14, min_periods = 13):
        name_RSI = f'RSI_window_{window}_periods_{min_periods}'
        self.day_df = self.day_df.sort_values(by=['code', 'aest_day_datetime']).reset_index(drop=True)#required for the smoothing logic for average gain and loss. 
        logging.info("beggining calc_rsi")
        name_RSI = f'RSI_window_{window}_periods_{min_periods}'
        name_gain = f'avg_gain_window_{window}_periods_{min_periods}'
        name_loss = f'avg_loss_window_{window}_periods_{min_periods}'
        #get avg_gain
        logging.info("starting calc gain")
        self.day_df['gain'] = self.day_df['change'].mask(self.day_df['change'] < 0, 0)#mask -s opposite to .where but handles na differently. 

        self.day_df['loss'] = self.day_df['change'].mask(self.day_df['change'] > 0, 0).abs()#.mask instead of,where to ensure na values remain na values in the loss column
        logging.info("starting calc loss")

        #issues we may have 
        
        if name_gain not in self.day_df.columns:
            self.day_df[name_gain] = np.nan
        if name_loss not in self.day_df.columns:
            self.day_df[name_loss] = np.nan
        #also this does not handle if days are missing. 
        
        #essentially the problem is it is dependency basec vectorized calculation. Meaning you cannot know ahead of time the valuesof the previously updated avg gain.
        #vector calculations is  actually just splitting up each inidividual calculation into a seperate thread kind of.
        #so becase the previous avg gain updates, the current avg gain, each calculation is dependent on the calculation before. So essentially we need to do a loop. 
        # it is best to do the cloop on an numpy array. 

        #initialise the columns if they do not exist. 
        
        ###make it work for multiple codes with a loop for each code and store the index. 
        logging.debug("the dataframe before RSI calculated")
        logging.debug(self.day_df[["code","updated_at","last"]])
        for code, group in self.day_df.groupby('code'):
            idx = group.index#store idx for reassigning back to the dataframe.
            
            logging.debug(f"the idx for this group {code} is: {idx}")

            avg_gain = group[name_gain].to_numpy()
            avg_loss = group[name_loss].to_numpy()#doesnt save the indexing obviously
            gain_temp = group["gain"].to_numpy()
            loss_temp = group["loss"].to_numpy()
            gain_temp = np.where(gain_temp == None, np.nan, gain_temp)
            loss_temp = np.where(loss_temp == None, np.nan, loss_temp)
            avg_gain = np.where(avg_gain == None, np.nan, avg_gain)
            avg_loss = np.where(avg_loss == None, np.nan, avg_loss)
            #above is required for some functionality. It should not be None. it should be nan for numeric claculations. I changed it to none which impacted this. not the best idea.
            logging.debug("avg gain")
            logging.debug(avg_gain)
            assert len(avg_gain) == len(avg_loss), f"The length must be the same of avg gain and avg loss. avg gain: {len(avg_gain)}, avg_loss {len(avg_loss)}"
            # if np.isnan(avg_gain[0]):
            #     number_nan_days = 1
            number_of_nan_days = 0
            RSI_initialised = False
            logging.debug(f"this is the value of gain_temp {gain_temp}")

            #this below if conditions could maybe be written better
            #essentially there is 3 main conditions we are checking. is the RSI_initialized is the previous avg gain na, is the current avg gain na,
            #has the number of na days exceeded 14.
            for i in range(1,len(avg_gain)):#why did I make it skip the first row? 
                logging.debug(i)          
                if (pd.isna(avg_gain[i-1])) and not RSI_initialised:

                    if not pd.isna(gain_temp[i]):
                        #if there is padding of extra days for a code it should be handled. 
                        number_of_nan_days +=1
                        logging.debug(f"number of days updated to {number_of_nan_days}")
                    if number_of_nan_days >= 14:
                        
                        #there has been 14 days in a row of na values in avg_gain
                        logging.debug(f"check to see if the current gain is na {gain_temp[i]}")
                        if not np.isnan(gain_temp[i]):
                            logging.debug("RSI initialised.")
                            logging.debug(f"{gain_temp[i-13:i+1]} evalulates to {gain_temp[i-13:i+1].mean()} type of gain_temp: {gain_temp.dtype}")
                            logging.debug(f"{loss_temp[i-13:i+1]} evalulates to {loss_temp[i-13:i+1].mean()} type of loss_temp: {loss_temp.dtype}")
                            # we can update and initialise the RSI
                           
                            avg_gain[i] = np.nanmean(gain_temp[i-13:i+1])#take last 14 values and initialise the RSI
                            #do nanmean just to ensure a bit more stability for the algorithm. might give slightly incorrect values but will be mostly fine. 
                            avg_loss[i] = np.nanmean(loss_temp[i-13:i+1])#use the initial value.
                            number_of_nan_days=0
                            last_valid_gain = avg_gain[i]
                            last_valid_loss = avg_loss[i] 
                            RSI_initialised = True
                        else:
                            logging.debug("the avg_gain was meant to be initialised but the gain_temp was na. ")
                            pass# we dont want to do anything and assume we are still missing values
                
                else:#this block only executes if RSI is initialised
                    logging.debug("smoothing calculation to be done. ")
                    RSI_initialised = True
                    logging.debug(gain_temp[i])
                    if pd.isna(gain_temp[i]):#condition to use last valid value. 
                        logging.debug("going down missing day path")
                        #for a na current gain (we are missing a day)
                        #The RSI has been initialised (there must be a none na value in this column lower than this idx)
                        #we hence want to use last valid value. 
                        logging.debug(f"last valid gain {last_valid_gain}")
                        avg_gain[i] = last_valid_gain
                        # Get indices of non-NaN values before i
                        avg_loss[i] = last_valid_loss
                        
                    else:#we know that gain is not na and avg_gain not na, and RSI has been initialised
                        logging.debug(f"avg_")
                        avg_gain[i] = (avg_gain[i-1]*13 + gain_temp[i])/14
                        avg_loss[i] = (avg_loss[i-1]*13 + loss_temp[i])/14#if previous is na it will be na.
                        logging.debug(f"avg_gain new = {avg_gain[i]}")
                        last_valid_gain = avg_gain[i]
                        last_valid_loss = avg_loss[i] 
            logging.debug(f"the average gain results are {avg_gain}")
            logging.debug(f"the average loss results are {avg_loss}")
            self.day_df.loc[idx, name_gain] = avg_gain
            self.day_df.loc[idx, name_loss] = avg_loss
        
        self.day_df[name_RSI] = 100 - (100/(1+(self.day_df[name_gain]/self.day_df[name_loss])))
        
        
        
        #set model_res_df
        
        
        idx = self.day_df.groupby('code')['aest_day_datetime'].idxmax()
        temp_df = self.day_df.loc[idx]
        
        #we now have the latest prices stored in the temp df 

        temp_df = temp_df.set_index("code")
        self.model_res_df[name_RSI] = temp_df[name_RSI]
        #self.day_df.drop(columns = ['gain', 'loss'], inplace = True)#remove columns to save memory. (must preserve )
        

    def test_model(self, model):
        '''
        input
        model. model is a function. input = share prices up to a given day. output= shares to buty
        
        output of this test. 
        using the original shares_df it will take different points in time, find what shares should be bought. 
        Then finally test the shares bought and see if they went up in price or not. to validate its efficacy. 
        
        '''
        pass
        
        
    