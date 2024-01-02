import time 

from .utils import compute_parallel_ratio

from abc import ABC, abstractmethod
import numpy as np 
import pandas as pd
from scipy import stats

import plotly.graph_objects as go



class ChannelStrategyBase(ABC): 
    """
    A base Channel Strategy class 

    Attributes
    ----- 

    window_size (int): 
        integer specifying the amount of left and right neighbors of a candle 
        Therefore, a neighborhoud has 2 * window_size + 1 neighbors. 
    parallel_threshold (float): 
        threshold value for the parallelism of the High and Low lines
    r_sqrt_ratio (float): 
        threshold value for the high and low lines r2 squared score
    """
    def __init__(self, window_size, parallel_threshold=0.3, r_sqrt_ratio=0.85): 
        self.window_size = window_size 
        self.parallel_threshold = parallel_threshold
        self.r_sqrt_ratio = r_sqrt_ratio

    @abstractmethod
    def _detect_pivots(self, df, candle_id): 
        """
        Return for a given candle, its pivot type. 
        
        Arguments
        ----
        df : the dataframe 
        row_id : index 
        window_size : the window size
        """
        pass         
    @abstractmethod
    def _set_pivot_value_(self, df, candle_id): 
        """
        Gives a specific value with respect to the pivot type. 
        Note : this value is not used for computing but for plotting only !! 
        """
        pass

    @abstractmethod
    def _set_breakout_value(self, df, candle_id): 
        """
        Given a candle_id, assign a specific value with respect to its breakout type

        Argument
        ----
        df : dataframe 
        candle_id : index 
        """
        pass
    
    @abstractmethod
    def _detect_breakout(self, df, candle_id): 
        """
        For A given candle_id, return the channel breakout label 

        Arugment 
        ---- 
        df : dataframe 
        candle_id : index 
        """
        pass
        
    
    

class ChannelBreakoutStrategy(ChannelStrategyBase): 

    def __init__(self, window_size, parallel_threshold): 
        super().__init__(window_size, parallel_threshold) 


    def visualize_candle(self, df, candle_id, num_backcandles=None, channel_window_size=None,optimize=True):
        """
        Visualize the candle based on the processed dataframe. 
        This dataframe must contain the plot_pivot_val prior to that. 

        Argument
        ----- 

        df : dataframe 
        cnadle_id : int 
        num_backcandles : (int or None)
            Not needed if optimize is set to True
        channel_window_size : (int or None)
            Not needed if optimize is set to True
        optimize : if True, run best channel hyper parameters search, otherwise, used given backcandles and channel_window_size 
        """ 

        if "plot_pivot_val" not in df.columns: 
            raise ValueError("'plot_pivot_val' column missing. You must call get_all_pivots() first or process_dataframe()")

        if optimize and (num_backcandles or channel_window_size): 
            print("Warning : When optimize = True, num_backcandles and channel_window_size are useless. ")

        layout = go.Layout(
            autosize=False,
            width=2000,
            height=800,
        )

        fig = go.Figure(data=[go.Candlestick(x=df.index,
                open=df['Open'],
                high=df['High'],
                low=df['Low'],
                close=df['Close'])], layout=layout)

        fig.add_scatter(x=df.index, y=df["plot_pivot_val"], mode="markers", name="pivot")
        if "plot_breakout_val" in df.columns: 
            fig.add_scatter(x=df.index, y=df["plot_breakout_val"], mode="markers", name="breakpoint", marker_symbol="hexagram", marker=dict(size=10, color="white"))
        
        if optimize: 
            d_best = self._best_channel_search(df, candle_id, verbose=True)
            low_slope, low_intercept, _, high_slope, high_intercept, _ = d_best["parameters"] 
            num_backcandles = d_best["hyperparameters"]["num_backcandles"] 
            channel_window_size = d_best["hyperparameters"]["channel_window_size"]
        else: 
            low_slope, low_intercept, low_r, high_slope, high_intercept, high_r = self.detect_channels(df, candle_id, channel_window_size, num_backcandles, verbose=True)
            if low_slope:
                print("low r2 : ", low_r**2) 
                print("high r2 : ", high_r**2)
        
        if low_intercept is not None:
            range_backward = list(range(candle_id - num_backcandles - channel_window_size, candle_id+1))

            fig.add_scatter(
                x=range_backward,
                y=np.array(range_backward) * low_slope + low_intercept, 
                mode="lines", 
                name="lower"
            )
        if high_intercept is not None:
            fig.add_scatter(
                x=range_backward,
                y=np.array(range_backward) * high_slope + high_intercept, 
                mode="lines", 
                name="upper"
            )


        fig.show()


    def process_dataframe(self, df, channel_window_size, num_backcandles):
        """
        Pre-process a dataframe 

        Arugment 
        ---- 

        df : dataframe
        channel_window_size : the number of previous candles to ignore as support
        num_backcandles : the number of backcandles to fit in the support and resistance lines

        Return 
        ---- 
        process_df : preprocessed dataframe
        """ 
        pre_processed_df = self.get_all_pivots(df) 
        pre_processed_df = self.get_all_channels(pre_processed_df, channel_window_size, num_backcandles) 
        print(f"Breakout counts : {pre_processed_df['breakout_type'].value_counts()}")
        return pre_processed_df 
        

    def get_all_pivots(self, df): 
        """
        create two new columns to the df dataframe : 'pivot_type' and 'plot_pivot_val'
        'pivot_type' describes the type of pivot : 0 for no pivot, 1 for low and 2 for high pivot
        'plot_pivot_val' describes a value used for plotting, with respect to the pivot_type

        Note : this is not an inplace operation
        """
        _df = df.copy()
        _df["pivot_type"] = _df.apply(lambda x: self._detect_pivots(_df, x.name), axis=1)
        _df["plot_pivot_val"] = _df.apply(lambda x: self._set_pivot_value_(_df, x.name), axis=1)
        return _df 
    
    def get_all_channels(self, df, channel_window_size, num_backcandles): 
        """
        Create two new columns to the df dataframe : 'breakout_type' and 'plot_breakout_val'
        'breakout_type' describes the type of breakout : 0 for no breakout, 1 for low and 2 for high breakout
        'plot_breakout_val' describes a value used for plotting, with respect to the breakout_type. 
        

        Note : this is not an inplace operation
        """
        _df = df.copy()
        _df[["breakout_type", "num_backcandles"]] = _df.apply(lambda row: self._detect_breakout(_df, row.name, channel_window_size, num_backcandles, optimize=False), axis=1, result_type='expand') 
        _df["plot_breakout_val"] = _df.apply(lambda row: self._set_breakout_value(_df, row.name), axis=1) 
        return _df 
    

    def detect_channels(self, df, candle_id, channel_window_size, num_backcandles, verbose=False): 
        """
        For a given candle_id, identify its channel 

        Arugment 
        ---- 

        df : dataframe 
        candle_id : index 
        verbose : Turn on or off terminal print statement
        channel_window_size : the number of previous candles to ignore as support
        num_backcandles : the number of backcandles to fit in the support and resistance lines

        Return 
        ---- 

        (low slope, low intercept, low r value, high slope, high intercept, high r value)
        """
        prev_data = df.iloc[
            candle_id - channel_window_size - num_backcandles: candle_id - channel_window_size
        ]    

        highs_idx = prev_data[prev_data["pivot_type"] == 2].index 
        highs_value = prev_data[prev_data["pivot_type"] == 2]["High"].values
        lows_idx = prev_data[prev_data["pivot_type"] == 1].index 
        lows_value = prev_data[prev_data["pivot_type"] == 1]["Low"].values
        parallel_ratio = 0

        if verbose: 
            print("highs", highs_idx)
            print("low", lows_idx)

        if len(highs_idx) >= 2 and len(lows_idx) >= 2: 
            # highs_idx = highs_idx[-2:] 
            # highs_value = highs_value[-2:]
            # lows_idx = lows_idx[-2:]
            # lows_value = lows_value[-2:]

            low_slope, low_intercept, low_r, _, _ = stats.linregress(list(lows_idx), lows_value) 
            high_slope, high_intercept, high_r, _, _ = stats.linregress(list(highs_idx), highs_value) 
            # check for how parallel they are 

            parallel_ratio = compute_parallel_ratio(low_slope, high_slope)
            if verbose: 
                print(low_slope, high_slope)
                print("parallel ratio : ", parallel_ratio)
            if parallel_ratio <= self.parallel_threshold: 
                return (low_slope, low_intercept, low_r, high_slope, high_intercept, high_r)
            
        if verbose:
            print(f"No channel identified for candle id : {candle_id} \n (parallel ratio = {parallel_ratio}"
                  f"low supports = {len(lows_idx)},"  
                  f"high supports = {len(highs_idx)}" 
                  )
        return (None, None, None, None, None, None)
    

    def _detect_pivots(self, df, candle_id): 
        """
        Return for a given candle, its pivot type. 
        
        Arguments
        ----
        df : the dataframe 
        row_id : index 
        window_size : the window size

        Return 
        ----- 
        pivot_type (int) : 
            0 if not pivot 
            1 if low pivot 
            2 if high pivot 
            3 if both 
        """
        if candle_id + self.window_size > len(df) - 1 or candle_id - self.window_size < 0: 
            return 0 
        isHighPivot = True 
        isLowPivot = True 

        candle_row = df.iloc[candle_id]

        for i in range(candle_id - self.window_size, candle_id + self.window_size + 1):  
            neihbor_row = df.iloc[i]
            if neihbor_row.Low < candle_row.Low: 
                isLowPivot = False 
            if neihbor_row.High > candle_row.High: 
                isHighPivot = False 
        
        # both high and low pivot
        if isLowPivot and isHighPivot: 
            return 3 
        elif isLowPivot: 
            return 1 
        elif isHighPivot: 
            return 2 
        else: 
            return 0
        

    def _set_pivot_value_(self, df, candle_id): 
        """
        Gives a specific value with respect to the pivot type. 
        Note : this value is not used for computing but for plotting only !! 

        Argument
        ----
        df : dataframe 
        candle_id : index 

        Return: 
        ----- 
        pivot_value (float) : 
            depending if low pivot or high pivot, return the Low - 0.1 or High + 0.1 
        """
        candle_row = df.iloc[candle_id]

        if candle_row["pivot_type"] == 0: 
            return np.nan
        elif candle_row["pivot_type"] == 1: 
            return candle_row["Low"] - 0.1 
        elif candle_row["pivot_type"] == 2: 
            return candle_row["High"] + 0.1 
        else: 
            return (candle_row["Low"] + candle_row["High"]) / 2
        
    
    def _best_channel_search(self, df, candle_id, verbose=False):
        """
        Given a candle_id, finds the best matching channel by searching the optimal 
        pair of channel window size and number of backcandles, yielding the lowest 'low' and 'high' r squared score.

        Argument
        ----
        df : dataframe 
        candle_id : index 
        verbose : bool

        Return 
        ---- 
        dict : 
            'parameters' (low slope, low intercept, low r value, high slope, high intercept, high r value), 
            'hypterparameters' : 
                'num_backcandles' : int 
                'channel_window_size' : int 
        """

        if verbose:
            print("Running optimization ...")

        channel_window_size_ranges = range(1, 5) 
        num_backcandles_ranges = range(10, 40)

        # find best suited channel 
        best_low_r = 0
        best_high_r = 0
        best_low_slope = None 
        best_high_slope = None 
        best_low_intercept = None
        best_high_intercept = None 
        best_num_backcanles = None 
        best_channel_window_size = None

        start_time = time.time()
        for channel_window_size in channel_window_size_ranges: 
            for num_backcandles in num_backcandles_ranges: 

                low_slope, low_intercept, low_r, high_slope, high_intercept, high_r = self.detect_channels(df, candle_id, channel_window_size, num_backcandles)

                if low_slope is not None: 
                    if verbose: 
                        print("num_backcandles = ", num_backcandles) 
                        print("channel window size = ", channel_window_size) 
                        print("r2² low = ", low_r**2) 
                        print("r2² high = ", high_r**2)
                    if low_r**2 >= best_low_r**2 and high_r**2 >= best_high_r**2: 
                        best_num_backcanles = num_backcandles
                        best_channel_window_size = channel_window_size
                        best_low_r = low_r
                        best_high_r= high_r
                        best_low_slope = low_slope 
                        best_high_slope = high_slope 
                        best_low_intercept = low_intercept 
                        best_high_intercept = high_intercept 
        end_time = time.time()

        if verbose: 
            print(f"Searching for optimal channel in : {end_time - start_time} seconds")
            print("Optimal num_backcandles = ", best_num_backcanles) 
            print("Optimal channel window size = ", best_channel_window_size) 
            print("Optimization : DONE ", "\n")
        return {
            "parameters" : (best_low_slope, best_low_intercept, best_low_r, best_high_slope, best_high_intercept, best_high_r), 
            "hyperparameters" : {
                "num_backcandles" : best_num_backcanles, 
                "channel_window_size" : best_channel_window_size
            }
        }


        
    def _detect_breakout(self, df, candle_id, channel_window_size, num_backcandles, optimize=False): 
        """
        For A given candle_id, return the channel breakout label and as well as the 
        num_backcandles and channel_window_size given.

        Arugment 
        ---- 
        df : dataframe 
        candle_id : index 
        channel_window_size : the number of previous candles to ignore as support
        num_backcandles : the number of backcandles to fit in the support and resistance lines

        Return
        -----

        label (int) :  
                        0 if no breakout 
                        1 if down breakout 
                        2 if high brekaout 
        """

        if optimize: 
            d_best = self._best_channel_search(df, candle_id)
            low_slope, low_intercept, low_r, high_slope, high_intercept, high_r = d_best["parameters"]
            num_backcandles = d_best["hyperparameters"]["num_backcandles"] 
            channel_window_size = d_best["hyperparameters"]["channel_window_size"]
        else: 
            low_slope, low_intercept, low_r, high_slope, high_intercept, high_r = self.detect_channels(df, candle_id, channel_window_size, num_backcandles)
        # no channel
            
        if low_slope is None: 
            return 0, None
        
        current_candle = df.iloc[candle_id] 
        current_candle_open = current_candle["Open"] 
        current_candle_close = current_candle["Close"] 

        # no breakout
        if high_r**2 < 0.85 and low_r**2 < 0.85: 
            return 0, num_backcandles
        # low breakout 
        if current_candle_close <= low_slope * candle_id + low_intercept and current_candle_open <= low_slope * candle_id + low_intercept: 
            return 1, num_backcandles
        # high breakout 
        elif current_candle_close > high_slope * candle_id + high_intercept and current_candle_open > high_slope * candle_id + high_intercept: 
            return 2, num_backcandles
        else: 
            return 0, None

        
    def _set_breakout_value(self, df, candle_id): 
        """
        Given a candle_id, assign a specific value with respect to its breakout type

        Argument
        ----
        df : dataframe 
        candle_id : index 

        Return: 
        ----- 
        breakout_value (float or np.nan) : 
            value depending on breakout type
        """

        candle = df.iloc[candle_id]
        if candle["breakout_type"] == 1: 
            return candle["Low"] - 1e-4
        elif candle["breakout_type"] == 2: 
            return candle["High"] - 1e-4
        elif candle["breakout_type"] == 0: 
            return np.nan
        
