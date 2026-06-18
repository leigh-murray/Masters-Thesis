import sys
import numpy as np
import pandas as pd

from contract_dates_functions import *

def get_implied_cnh_rate(trading_date, tenor, forward_df, sofr_df, spot_df, 
                         domestic_day_count=360, foreign_day_count=360):
    """
    Calculate implied CNH rate using covered interest rate parity.
    Works for forward contract (price date, tenor) pair and for option contract (price date, tenor) pair
    
    Parameters:
    -----------
    trading_date : pd.Timestamp
        Trade date
    tenor : str
        Tenor string (e.g., '1M', '3M', '1Y')
    forward_df : pd.DataFrame
        Forward rates indexed by date, columns are days to delivery
    sofr_df : pd.DataFrame
        SOFR rates with multi-index (Pricing Date, Expiry Date)
    spot_df : pd.DataFrame
        Spot rates with 'Spot' column
    domestic_day_count : int
        Day count convention for CNH (default 360)
    foreign_day_count : int
        Day count convention for SOFR (default 360)
    
    Returns:
    --------
    float : Implied CNH rate (as decimal, not percentage)
    """
    
    # Get contract dates
    settlement_date, expiry_date, delivery_date = get_contract_dates(trading_date, tenor)
    
    # Calculate days
    trade_to_expiry_days = (expiry_date - trading_date).days
    settlement_to_delivery_days = (delivery_date - settlement_date).days
    trade_to_delivery_days = (delivery_date - trading_date).days
    
    # Get market data
    forward = forward_df.loc[trading_date][trade_to_delivery_days-1]
    foreign_interest_rate = sofr_df.loc[(trading_date, expiry_date), 'SOFR']
    spot = spot_df.loc[trading_date, 'Spot']
    
    # Calculate time discounts
    time_discount_foreign = settlement_to_delivery_days / foreign_day_count
    time_discount_domestic = settlement_to_delivery_days / domestic_day_count
    
    # Calculate implied CNH rate using covered interest rate parity
    implied_domestic_interest_rate = (forward / spot * (1 + foreign_interest_rate*(time_discount_foreign)) - 1) /(time_discount_domestic) 
    
    
    return implied_domestic_interest_rate

def get_implied_cnh_rate_no_tenor(trading_date, settlement_date, expiry_date, delivery_date, forward_df, sofr_df, spot_df, 
                         domestic_day_count=360, foreign_day_count=360):
    """
    Calculate implied CNH rate using covered interest rate parity.
    Works for forward contract (price date, tenor) pair and for option contract (price date, tenor) pair
    
    Parameters:
    -----------
    trading_date : pd.Timestamp
        Trade date
    tenor : str
        Tenor string (e.g., '1M', '3M', '1Y')
    forward_df : pd.DataFrame
        Forward rates indexed by date, columns are days to delivery
    sofr_df : pd.DataFrame
        SOFR rates with multi-index (Pricing Date, Expiry Date)
    spot_df : pd.DataFrame
        Spot rates with 'Spot' column
    domestic_day_count : int
        Day count convention for CNH (default 360)
    foreign_day_count : int
        Day count convention for SOFR (default 360)
    
    Returns:
    --------
    float : Implied CNH rate (as decimal, not percentage)
    """
    
    # Calculate days
    trade_to_expiry_days = (expiry_date - trading_date).days
    settlement_to_delivery_days = (delivery_date - settlement_date).days
    trade_to_delivery_days = (delivery_date - trading_date).days
    
    # Get market data
    forward = forward_df.loc[trading_date][trade_to_delivery_days-1]
    foreign_interest_rate = sofr_df.loc[(trading_date, expiry_date), 'SOFR']
    spot = spot_df.loc[trading_date, 'Spot']
    
    # Calculate time discounts
    time_discount_foreign = settlement_to_delivery_days / foreign_day_count
    time_discount_domestic = settlement_to_delivery_days / domestic_day_count
    
    # Calculate implied CNH rate using covered interest rate parity
    implied_domestic_interest_rate = (
        (forward / spot * (1 + foreign_interest_rate * time_discount_foreign) - 1) 
        / time_discount_domestic
    )
    
    return implied_domestic_interest_rate

