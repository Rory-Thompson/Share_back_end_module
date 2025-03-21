import requests
import pandas as pd
import json
import logging
from datetime import datetime
import os
from tqdm import tqdm
import pandas as pd
from pytz import timezone
import matplotlib.dates as mdates
import warnings
import numpy as np
import yfinance
import requests
aest = timezone('Australia/Sydney')
import traceback
import time
import seaborn as sns
import matplotlib.pyplot as plt
import io
import base64

from matplotlib.colors import to_rgb




class SharesPlotter:
    def __init__(self, shares_analysis_instance,plot_website =False):
        """Initialize with an instance of the shares_analysis class."""
        self.shares_analysis = shares_analysis_instance
        self.plot_website = plot_website
    
    @property
    def day_df(self):
        return self.shares_analysis.day_df
    
    @property
    def share_metric_df(self):
        return self.shares_analysis.share_metric_df
    @property
    def model_res_df(self):
        return self.shares_analysis.model_res_df

    def plot_rsi(self, codes, window=14, min_periods = 13,ax=None):
        """
        Helper method to plot RSI for the given ASX codes.
        Plots the RSI with overbought (70) and oversold (30) lines.
        """
        rsi_column = f'RSI_window_{window}_periods_{min_periods}'
        if (rsi_column not in list(self.day_df.columns)) or self.shares_analysis.df_is_updated:
            print("print calculate_rsi_required_from_plotting_module: "+rsi_column)
            self.shares_analysis.calc_rsi(window = window,min_periods = min_periods)

        plt.style.use("dark_background")  # Enable dark mode
        if ax is None:
            fig, ax = plt.subplots(figsize=(14, 4))

        for code in codes:
            temp_df = self.day_df[self.day_df['code'] == code]

            # RSI column name
            #RSI_window_14_periods_13
            #rsi_column = f'RSI_window_{window}_periods_{window}'
            
            # Plot RSI
            ax.plot(temp_df["aest_day_datetime"].dt.tz_localize(None), temp_df[rsi_column], label=f'RSI for {code}', linewidth=2)

        # Add horizontal lines for overbought (70) and oversold (30) levels
        ax.axhline(y=70, color='r', linestyle='--', label='Overbought (70)')
        ax.axhline(y=30, color='g', linestyle='--', label='Oversold (30)')
        ax.fill_between(temp_df["aest_day_datetime"].dt.tz_localize(None), 30, 70, color='blueviolet', alpha=0.15)

        # Format the x-axis for the RSI plot
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))  # Show dates in YYYY-MM-DD format
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())  # Show ticks for each month
        ax.tick_params(axis='x', rotation=45)  # Rotate date labels for better readability

        # Add gridlines, title, labels, and legend to the RSI plot
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.set_title('Relative Strength Index (RSI)', fontsize=16, fontweight='bold')
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('RSI', fontsize=12)
        ax.legend(fontsize=12)

        # Display the plot
        if ax is None:
            plt.tight_layout()
            plt.show()
            
    def plot_averages(self,codes, averages= [50], plot_rsi=False,min_periods = None, window = None):
        
        
        """
        codes = list. List of ASX codes. No error will be thrown for invalid codes.
        
        averages = list of averages. Calcualtion beforehand not required.
        
        output: graph with every combination of code and average.
        
        works best with only 1 code and multiple averages. 
        
        """
        plt.style.use("dark_background")  # Enable dark mode
        #print(f"checking averages: {pd.Series(list(self.day_df.columns)).str[-len(str(average)):].values}")#IDK WAT I WAS THINKIN WITH THIS LOGIC.
        print(f"{averages} averages to be plotting in SharesPlotter")
        print(f"{codes}, codes to be plotting in SharesPlotter")
        for average in averages:
            #print(pd.Series(list(self.day_df.columns)))
            title = f'rolling average {average}'
            if (title not in list(self.model_res_df.columns)) or self.shares_analysis.df_is_updated:

                #the average has not been calculated yet
                print(f'calculating missing average {average} updated variable: {self.shares_analysis.df_is_updated}')
                self.shares_analysis.calc_moving_average(num_days = average, min_periods = int(average//1.4))
                self.shares_analysis.averages_calculated.append(average)

                #note arandom min pperiods is used with some basic logic
                
                
        # Filter the data for 'BHP'
        fig, ax1 = plt.subplots(figsize=(18, 12),
                                nrows=(2 if plot_rsi else 1),
                                sharex=True,
                                gridspec_kw={'height_ratios': [2, 1] if plot_rsi else [1]})
        
        if not isinstance(ax1, np.ndarray):
            ax1 = [ax1]  
        for code in codes:
            temp_df = self.day_df[self.day_df['code'] == code]

            # Set up the plot
            
            # Plot the 'last' prices
            ax1[0].plot(temp_df["aest_day_datetime"].dt.tz_localize(None), temp_df["last"], label=f'Last Price for {code}', linewidth=2, marker='o')

        # Plot the rolling average
            for average in averages:
            
                title = f'rolling average {average}'
                
                ax1[0].plot(temp_df["aest_day_datetime"].dt.tz_localize(None), temp_df[title], label=f'{code} {title}', linewidth=2)
        
        # Format the x-axis to show dates nicely
        ax1[0].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax1[0].xaxis.set_major_locator(mdates.AutoDateLocator())

        # Titles and labelspl
        ax1[0].set_title('Stock Price & Moving Averages', fontsize=16, fontweight='bold')
        ax1[0].set_ylabel('Price', fontsize=12)#for better readability

        # Add gridlines for better readability
        ax1[0].grid(True, linestyle='--', alpha=0.5)
        ax1[0].legend(fontsize=12)

        # Add a legend
        if plot_rsi:
            self.plot_rsi(codes, ax=ax1[1],min_periods = min_periods, window = window)

        # Add a vertical line for the current date (optional)
        #         current_date = pd.Timestamp.now().date()
        #         if current_date in temp_df["aest_day"].values:
        #             plt.axvline(x=current_date, color='red', linestyle='--', label='Today')

        #         # Display the plot
        plt.tight_layout()  # Adjust layout to prevent overlap
        if self.plot_website:
            img_buffer = io.BytesIO()
            fig.savefig(img_buffer, format="png")  # Save to buffer
            img_buffer.seek(0)
            
            # Encode to base64 so it can be displayed in Dash
            encoded_image = base64.b64encode(img_buffer.getvalue()).decode("utf-8")
            plt.close(fig)  # Free memory
            return f"data:image/png;base64,{encoded_image}"
        
        plt.show()
        
       
    
    
    def plot_metric_comparison(self, code, metric_x, metric_y, size_metric="marketCap"):
        """
        Plots a scatter plot comparing two selected metrics for all stocks in the same sector as the given stock.
        
        Parameters:
        - code (str): The stock code to highlight.
        - metric_x (str): The metric for the x-axis.
        - metric_y (str): The metric for the y-axis.
        - size_metric (str): The metric determining dot size (default: market cap).
        """
        
        # Get the dataframe of fundamental metrics
        df = self.shares_analysis.share_metric_df.copy()

        # Ensure metrics exist
        

        sector = df.loc[code, "sector"]
        sector_df = df[df["sector"] == sector].dropna(subset=[metric_x, metric_y, size_metric])
        
        # Normalize sizes for better visibility
        min_size = 20
        max_size = 500
        sector_df["size"] = np.interp(sector_df[size_metric], (sector_df[size_metric].min(), sector_df[size_metric].max()), (min_size, max_size))

        # Create scatter plot
        fig = plt.figure(figsize=(12, 8))
        
        threshold = 5
        
        sector_df = sector_df[~sector_df[metric_x].isin([np.inf, -np.inf])]
        
        sector_df = sector_df[~sector_df[metric_y].isin([np.inf, -np.inf])]
        
        
        mean_val_metric_x = sector_df[metric_x].mean()
        std_val_metric_x = sector_df[metric_x].std()
        upper_bound_metric_x = mean_val_metric_x + (threshold * std_val_metric_x)
        print(std_val_metric_x)
        print(mean_val_metric_x)
        
        mean_val_metric_y = sector_df[metric_y].mean()
        std_val_metric_y = sector_df[metric_y].std()
        upper_bound_metric_y = mean_val_metric_y + (threshold * std_val_metric_y)
        
        print(upper_bound_metric_y)
        print(upper_bound_metric_x)
        
        
        sector_df = sector_df[(sector_df[metric_x] <= upper_bound_metric_x)&(sector_df[metric_y] <= upper_bound_metric_y)]
        
        if metric_x not in df.columns or metric_y not in df.columns or size_metric not in df.columns:
            raise ValueError(f"One or more selected metrics are missing from the dataset: {metric_x}, {metric_y}, {size_metric}")

        # Get the sector of the selected stock
        if code not in df.index:
            raise ValueError(f"Code {code} not found in dataset.")


        top_stocks = list(sector_df.nlargest(3, size_metric).index)#for legend.
        print(top_stocks)
        if code not in top_stocks:
            top_stocks.append(code)
        hue_order = list(top_stocks) + ['Other']
        global test9
        palette = sns.color_palette("Set2", len(hue_order))#set initial palette
        test9 = palette
        
        sector_df["highlight"] = sector_df.index.map(lambda x: x if x in top_stocks else "Other")
        global test8
        test8 = sector_df
        
        dictionary_color = dict(zip(hue_order, palette))
        dictionary_color[code] = to_rgb('#880808')
        dictionary_color["Other"] = to_rgb('#F4E2A8')



       
        #get hue order from the dictionary, ensure it is the right order.
        hue_order, palette = zip(*dictionary_color.items())
        test9 = palette
        #for this function set the palett. it should be in order with the hue_order. palett order = hue_order to get correct coloring
        sns.set_palette(palette)
        
        print(f"hue_order_{hue_order}")
        scatter = sns.scatterplot(
            data=test8, x=metric_x, y=metric_y, size="size", sizes=(min_size, max_size), alpha=0.6, edgecolor="black",
            hue="highlight",hue_order = hue_order, legend = "brief"
        )
        
        
        ###do legend stuff
        #need to make final updates to the legend, remove the sizes, other and highlight variables.
        scatter.get_legend()

        handles, labels = scatter.get_legend_handles_labels()

        dictionary_color_2 = dict(zip(labels, handles))
        
        #remove other and highlight. 
        dictionary_color_2.pop('Other', None)
        dictionary_color_2.pop('highlight', None)

        labels, handles = zip(*dictionary_color_2.items())

        if 'size' in labels:
            #this is a bit of a hack job but thats aite
            #assumes same ordering
            #assumes the size labels are after the 'size' value in the list. 
            size_index = labels.index('size')  # Get index of 'size'
            filtered_handles = handles[:size_index]  # Keep only items before 'size'
            filtered_labels = labels[:size_index]    # Keep only labels before 'size'
        else:
            filtered_handles, filtered_labels = handles, labels  # No change if 'size' is missing

        print(filtered_handles)
        print(filtered_labels)
        # Set new legend
        scatter.legend(filtered_handles, filtered_labels, title="Filtered Shares")
        #
        
        
        # Highlight the selected stock
      
        # Add median reference lines
        plt.axvline(sector_df[metric_x].median(), linestyle="--", color="gray", alpha=0.7, label=f"Median {metric_x}")
        plt.axhline(sector_df[metric_y].median(), linestyle="--", color="gray", alpha=0.7, label=f"Median {metric_y}")

        # Labels and title
        plt.xlabel(metric_x)
        plt.ylabel(metric_y)
        plt.title(f"{metric_x} vs {metric_y} for {sector} Stocks", fontsize=14, fontweight="bold")
        highlight_x = sector_df.loc[code, metric_x]
        highlight_y = sector_df.loc[code, metric_y]
        x_range = max(sector_df[metric_x]) - min(sector_df[metric_x])
        y_range = max(sector_df[metric_y]) - min(sector_df[metric_y])
        text_offset_x = x_range * 0.05  # Move 5% of the x-range to the right
        text_offset_y = y_range * 0.1   # Move 10% of the y-range up
        plt.annotate(
            f'selected_share: {code}',
            xy=(highlight_x, highlight_y),
            xytext=(highlight_x + text_offset_x, highlight_y + text_offset_y),
            arrowprops=dict(facecolor='red', arrowstyle='->')
        )


        plt.grid(True, linestyle="--", alpha=0.5)
        if self.plot_website:
            img_buffer = io.BytesIO()
            fig.savefig(img_buffer, format="png")  # Save to buffer
            img_buffer.seek(0)
            
            # Encode to base64 so it can be displayed in Dash
            encoded_image = base64.b64encode(img_buffer.getvalue()).decode("utf-8")
            plt.close(fig)  # Free memory
            return f"data:image/png;base64,{encoded_image}" 
        # Show plot
        plt.show()