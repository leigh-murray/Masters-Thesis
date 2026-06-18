import numpy as np
from datetime import timedelta
import pandas as pd

# ============================================================================
# LOAD HOLIDAY CALENDARS AND EXPIRY DATES
# ============================================================================
# Convert to sets of pd.Timestamp for O(1) lookup
FED_HOLIDAYS = set(
    pd.to_datetime(
        pd.read_csv('/').iloc[:, 0], 
        dayfirst=True
    )
)

CNH_HOLIDAYS = set(
    pd.to_datetime(
        pd.read_csv('/').iloc[:, 0], 
        dayfirst=True
    )
)

# Get all column names except index
tenor_cols = ['1W','2W','3W','1M','2M','3M','4M','5M','6M','7M','8M','9M','10M','11M','1Y'] + \
             [str(i) for i in range(1, 96)]

EXPIRY_DF = pd.read_csv(
    '/',
    parse_dates=['Trading date'] + tenor_cols,  # Parse ALL date columns
    dayfirst=True,
    index_col='Trading date'
)


# ============================================================================
# [DATE, TENOR] -> EXPIRY DATE FUNCTION
# ============================================================================

def get_expiry_date(date: pd.Timestamp, tenor, expiry_df=EXPIRY_DF) -> pd.Timestamp:
    """
    Lookup expiry date from pre-calculated CSV.
    
    Parameters:
    date: pd.Timestamp - trading date
    tenor: str or int - e.g., '1W', '3M', '1Y', '90D', '90', or 90
    
    Returns:
    pd.Timestamp - expiry date from CSV lookup
    """
    
    # Validate input type
    if not isinstance(date, pd.Timestamp):
        raise TypeError(f"date must be pd.Timestamp, got {type(date)}")
    
    # Handle tenor with 'D' suffix (e.g., '90D' -> '90')
    if isinstance(tenor, str) and tenor.endswith('D'):
        tenor = tenor[:-1]  # Strip the 'D' suffix
    
    # Convert integer tenor to string if needed
    tenor_col = str(tenor)
    
    # Check if date exists in index
    if date not in expiry_df.index:
        raise ValueError(f"date {date.strftime('%d/%m/%Y')} not found in CSV")
    
    # Check if tenor exists in columns
    if tenor_col not in expiry_df.columns:
        raise ValueError(f"Tenor {tenor} not found in CSV columns")
    
    # Lookup expiry date
    expiry_date = expiry_df.loc[date, tenor_col]

    return expiry_date

# ============================================================================
# T+2 SETTLEMENT FUNCTION
# ============================================================================
def get_T2_lag_date(date: pd.Timestamp,
                    fed_holidays=FED_HOLIDAYS,
                    cnh_holidays=CNH_HOLIDAYS) -> pd.Timestamp:
    """
    Calculate T+2 settlement date from initial date. Initial date must be a weekday.
    
    Parameters:
    date: pd.Timestamp - input date
    
    Returns:
    pd.Timestamp - T+2 settlement date
    """
    
    # Validate input type
    if not isinstance(date, pd.Timestamp):
        raise TypeError(f"date must be pd.Timestamp, got {type(date)}")
    
    # Check if weekend
    if date.weekday() >= 5:
        raise ValueError(f"Input date {date.strftime('%d/%m/%Y')} is a weekend")
    
    business_days_counted = 0
    current_date = date
    
    # Count forward T+2 business days
    while business_days_counted < 2:
        current_date = current_date + pd.Timedelta(days=1)
        
        # Don't count: weekends, CNH holidays, joint FED+CNH holidays
        # Do count: FED-only holidays, regular business days
        if current_date in fed_holidays and current_date not in cnh_holidays:
            business_days_counted += 1
        elif current_date.weekday() < 5 and \
             current_date not in fed_holidays and \
             current_date not in cnh_holidays:
            business_days_counted += 1
    
    # Adjust if lands on non-business day in any calendar
    while current_date.weekday() >= 5 or \
          current_date in fed_holidays or \
          current_date in cnh_holidays:
        current_date = current_date + pd.Timedelta(days=1)
    
    return current_date

# ============================================================================
# CONTRACT DATES FUNCTION
# ============================================================================
    
def get_contract_dates(trading_date: pd.Timestamp,
                       tenor) -> tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]:
    """
    Calculate settlement, expiry, and delivery dates for a USDCNH option contract.
    
    Parameters:
    trading_date: pd.Timestamp - trading date
    tenor: str or int - e.g., '1W', '3M', '1Y', '90D', '90', or 90
    
    Returns:
    tuple: (settlement_date, expiry_date, delivery_date) as pd.Timestamp
    """
    
    # Validate input type
    if not isinstance(trading_date, pd.Timestamp):
        raise TypeError(f"trading_date must be pd.Timestamp, got {type(trading_date)}")
    
    # Calculate settlement date
    settlement_date = get_T2_lag_date(trading_date)
    
    # Get expiry date using lookup function 
    expiry_date = get_expiry_date(trading_date, tenor)
    
    # Calculate delivery date
    delivery_date = get_T2_lag_date(expiry_date)
    
    return settlement_date, expiry_date, delivery_date