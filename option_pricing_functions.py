# Import standard libraries
import pandas as pd
import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm


from volatility_functions import *

from contract_dates_functions import *

def price_fx_option(spot: float,
                    trade_to_expiry_days: int,
                    settlement_to_delivery_days: int,
                    strike: float,
                    foreign_interest_rate: float,
                    forward: float,
                    volatility: float,
                    option_type: str,
                    notional_ccy: str,
                    premium_ccy: str,
                    foreign_day_count = 360,
                    domestic_day_count = 360,
                    notional = 1
  ) -> list:
    """
    Prices a European FX option using the Garman-Kohlhagen formula (modified Black-Scholes model) according to Reiswich and Wystup (2010) extended to different discounting periods for delivery and expiry.


    Args:

    Returns: list [premium, notional_percentage_price, unadjusted_spot_delta, unadjusted_forward_delta, premium_adjusted_spot_delta, premium_adjusted_forward_delta, d1, d2]
    FOR-DOM
        
    """

    # PRICING FUNCTIONS
    
    omega = 1 if option_type[0].lower() == "c" else -1 # call/put variable

    time_discount_foreign = settlement_to_delivery_days / foreign_day_count 
    time_discount_domestic = settlement_to_delivery_days / domestic_day_count
    time_vol = trade_to_expiry_days / 365 

    # Domestic interest rate by covered interest rate parity
    implied_domestic_interest_rate = (forward / spot * (1 + foreign_interest_rate*(time_discount_foreign)) - 1) /(time_discount_domestic) 

    d1 = (np.log(forward / strike) + ( 0.5 * volatility ** 2) * time_vol) / (volatility * np.sqrt(time_vol))
    d2 = d1 - volatility * np.sqrt(time_vol)

    domestic_discount_factor = (1 / (1 + implied_domestic_interest_rate * time_discount_domestic)) # simple compounding for now

    # STANDARD CASE: 1 unit of foreign in exchange for K units of domestic
    # foreign/domestic price
    value_fordom = omega * domestic_discount_factor * (forward * norm.cdf(omega * d1) - strike * norm.cdf(omega * d2)) 
  
    # Notional in foreign currency
    if notional_ccy[0].lower() == "f":
      premium_domestic = notional * value_fordom # in domestic currency
      notional_percentage_price = value_fordom / spot # foreign percentage
      premium_foreign = notional_percentage_price * notional # in foreign currency
    
    # CHANGE OF NUMERAIRE: 1 unit of domestic in exchange for 1/K units of foreign
    # domestic/foreign price
    value_domfor = (value_fordom / spot) / strike

    # Notional in domestic currency
    if notional_ccy[0].lower() == "d":
      premium_foreign = notional * value_domfor # in foreign currency
      premium_domestic = premium_foreign * spot # in domestic currency
      notional_percentage_price = value_domfor / strike # domestic percentage

    # Setting premium to correct currency
    if premium_ccy[0].lower() == "f":
      premium = premium_foreign
    else:
      premium = premium_domestic

    # DELTA FUNCTIONS
    unadjusted_spot_delta = omega * (1 / (1 + foreign_interest_rate * time_discount_foreign))*(norm.cdf(omega*d1))
    unadjusted_forward_delta = omega * norm.cdf(omega*d1)

    # adjust for premium if premium in foreign currency
    premium_adjusted_spot_delta = omega * (1 / (1 + foreign_interest_rate * time_discount_foreign)) * (strike / forward) * (norm.cdf(omega*d2))
    premium_adjusted_forward_delta = omega * (strike / forward) * norm.cdf(omega * d2)

    # fix for put deltas greater than -1
    if premium_adjusted_spot_delta < -1:
        premium_adjusted_spot_delta = unadjusted_spot_delta
    if premium_adjusted_forward_delta < -1:
        premium_adjusted_forward_delta = unadjusted_forward_delta
    
    return [premium, notional_percentage_price, unadjusted_spot_delta, unadjusted_forward_delta, premium_adjusted_spot_delta, premium_adjusted_forward_delta, d1, d2]

def delta_to_strike(delta: float,
                    is_premium_adjusted: bool,
                    is_spot_delta: bool,
                    spot: float,
                    trade_to_expiry_days: int,
                    settlement_to_delivery_days: int,
                    foreign_interest_rate: float,
                    forward: float,
                    volatility: float,
                    option_type: str,
                    notional_ccy: str,
                    premium_ccy: str,
                    foreign_day_count = 360,
                    domestic_day_count = 360,
                    notional = 1,
                    tolerance: float = 1e-6,
                    max_iterations: int = 1000) -> float:
    """
    Calculates strike from target delta, using method outlined in 'A Guide to FX Option Quoting Conventions' by Reiswich and Wystup.

    Assumes delta in form 0.25 for 25% delta.

    ######################
    CURRENTLY ONLY WORKING FOR A CALL OPTION
    ######################

    A FOR-DOM call = FOR call = DOM put and a FOR-DOM put = FOR put = DOM call.

    For USDCNH, a USDCNH call = USD call / CNH put
    a USDCNH put = USD put / CNH call. 

    Note: premium-adjusted call spot delta is non-monotonic in strike, premium-adjusted put spot delta is monotonic strike. 

    Premium adjusted call spot delta: search only done for K greater than the K at maximum delta (to the right of maximum delta). This is appropriate only for calls that are OTM. So this function should only be used when the target delta is less than 50. For my project this is suitable. 
    """
    omega = 1 if option_type[0].lower() == "c" else -1 # call/put variable

    time_discount_foreign = settlement_to_delivery_days / foreign_day_count 
    time_discount_domestic = settlement_to_delivery_days / domestic_day_count
    time_vol = trade_to_expiry_days / 365 

    # non premium-adjusted deltas have a closed form solution
    if not is_premium_adjusted:
      
      if is_spot_delta:
        strike = forward * np.exp(-omega * norm.ppf(omega * (1 + foreign_interest_rate * time_discount_foreign)* delta)*volatility*np.sqrt(time_vol)+0.5*volatility**2*time_vol)
      
      else:
        strike = forward * np.exp(-omega * norm.ppf(omega * delta)*volatility*np.sqrt(time_vol)+0.5*volatility**2*time_vol)

    # premium-adjusted deltas
    else:
        # Define objective function for Brent's method (used for both put and call)
        def delta_objective(K_trial):
            result = price_fx_option(spot, trade_to_expiry_days, settlement_to_delivery_days, 
                                    K_trial, foreign_interest_rate, forward, volatility, 
                                    option_type, notional_ccy, premium_ccy, 
                                    foreign_day_count, domestic_day_count, notional)
            [premium, notional_percentage_price, unadjusted_spot_delta, unadjusted_forward_delta, 
            premium_adjusted_spot_delta, premium_adjusted_forward_delta, d1, d2] = result
            
            # Select appropriate delta based on type
            current_delta = premium_adjusted_spot_delta if is_spot_delta else premium_adjusted_forward_delta
            return current_delta - delta
        
        # premium-adjusted put
        if omega == -1:
            # Simple bounds for put (monotonic delta)
            reference_price = spot if is_spot_delta else forward
            Kmin = reference_price / 100
            Kmax = reference_price * 100
        
        # premium-adjusted call
        else:
            # Calculate Kmax: strike from non-premium-adjusted delta (closed form)
            if is_spot_delta:
                Kmax = forward * np.exp(-omega * norm.ppf(omega * (1 + foreign_interest_rate * time_discount_foreign) * delta) * volatility * np.sqrt(time_vol) + 0.5 * volatility**2 * time_vol)
            else:
                Kmax = forward * np.exp(-omega * norm.ppf(omega * delta) * volatility * np.sqrt(time_vol) + 0.5 * volatility**2 * time_vol)
            
            # Calculate Kmin: strike at maximum delta
            # Solve: σ√t × N(d2) = n(d2) for d2
            def delta_max_equation(d2_val):
                return volatility * np.sqrt(time_vol) * norm.cdf(d2_val) - norm.pdf(d2_val)
            
            # Find d2 at maximum (typically around 0)
            d2_at_max = brentq(delta_max_equation, -5, 5)
            
            # Back out Kmin from d2
            Kmin = forward * np.exp(-d2_at_max * volatility * np.sqrt(time_vol) + 0.5 * volatility**2 * time_vol)
            
            if Kmin >= Kmax:
                raise ValueError(
                    f"Invalid bounds: Kmin={Kmin:.4f} >= Kmax={Kmax:.4f}\n"
                    f"Inputs: spot={spot:.6f}, forward={forward:.6f}, vol={volatility:.6f}\n"
                    f"Days: expiry={trade_to_expiry_days}, settlement={settlement_to_delivery_days}\n"
                    f"Delta={delta:.4f}, sofr={foreign_interest_rate:.6f}\n"
                    f"d2_at_max={d2_at_max:.6f}, time_vol={time_vol:.6f}")
        
        # Use Brent's method for both put and call
        try:
            strike = brentq(delta_objective, Kmin, Kmax, xtol=tolerance, maxiter=max_iterations)
        except ValueError:
            # If root not bracketed, return midpoint
            strike = (Kmin + Kmax) / 2
    return strike

def premium_to_strike_put_nested(premium: float,
                                  spot: float,
                                  trade_to_expiry_days: int,
                                  settlement_to_delivery_days: int,
                                  foreign_interest_rate: float,
                                  forward: float,
                                  vol_bsplines_df,
                                  vol_date_tenor_df,
                                  trading_date,
                                  notional_ccy: str = 'foreign',
                                  premium_ccy: str = 'foreign',
                                  foreign_day_count: int = 360,
                                  domestic_day_count: int = 360,
                                  notional: float = 1,
                                  tolerance: float = 0.000000000001) -> tuple:
    """
    Finds put strike from premium using nested iteration:
    - Outer loop: guess strike K
    - Inner loop: compute delta → get vol from surface → price option
    
    Returns: strike
    """
    
    option_type = 'P'
    
    def strike_objective(K_trial):
        # Price option with trial strike
        # Need volatility from surface, but need delta first
        
        # Initial volatility guess (use ATM vol)
        vol_guess = get_volatility_from_bsplines(trading_date, trade_to_expiry_days,
                                                  25, option_type, vol_bsplines_df, 
                                                  vol_date_tenor_df) / 100
        
        # Inner iteration: converge on consistent vol-delta pair
        for _ in range(10):  # Max inner iterations
            # Price with current vol
            result = price_fx_option(spot, trade_to_expiry_days, settlement_to_delivery_days,
                                    K_trial, foreign_interest_rate, forward, vol_guess,
                                    option_type, notional_ccy, premium_ccy,
                                    foreign_day_count, domestic_day_count, notional)
            
            # Extract premium-adjusted spot delta
            delta = result[4]  # premium_adjusted_spot_delta
            
            # Get vol from surface using this delta
            vol_new = get_volatility_from_bsplines(trading_date, trade_to_expiry_days,
                                                    abs(delta), option_type,
                                                    vol_bsplines_df, vol_date_tenor_df)
            
            # Check convergence
            if abs(vol_new - vol_guess) < 1e-6:
                break
            vol_guess = vol_new
        
        # Return premium error
        return result[0] - premium
    
    # Outer loop: search for strike
    K_min = forward * 0.5  # Put strikes below forward
    K_max = forward * 1.5
    
    strike = brentq(strike_objective, K_min, K_max, xtol=tolerance)
    
    # Final calculation with converged strike
    vol_final = get_volatility_from_bsplines(trading_date, trade_to_expiry_days,
                                              0.25, option_type, vol_bsplines_df,
                                              vol_date_tenor_df)
    
    for _ in range(10):
        result = price_fx_option(spot, trade_to_expiry_days, settlement_to_delivery_days,
                                strike, foreign_interest_rate, forward, vol_final,
                                option_type, notional_ccy, premium_ccy,
                                foreign_day_count, domestic_day_count, notional)
        delta_final = result[4]
        vol_new = get_volatility_from_bsplines(trading_date, trade_to_expiry_days,
                                                abs(delta_final), option_type,
                                                vol_bsplines_df, vol_date_tenor_df)
        if abs(vol_new - vol_final) < 1e-6:
            break
        vol_final = vol_new
    
    return strike, delta_final, vol_final

def enter_risk_reversal(trading_date,
                        tenor: str,
                        call_delta: float,
                        spot_df,
                        forward_df,
                        sofr_df,
                        vol_bsplines_df,
                        vol_date_tenor_df,
                        notional_ccy: str = 'foreign',
                        premium_ccy: str = 'foreign',
                        is_spot_delta: bool = True,
                        is_premium_adjusted: bool = True) -> dict:
    """
    Prices a risk reversal (long call, short put with equal premiums).
    
    Args:
        trading_date: Trade date (pd.Timestamp)
        tenor: Option tenor (e.g., '3M')
        call_delta: Target call delta (e.g., 0.25 for 25%)
        spot_df, forward_df, sofr_df: Market data dataframes
        vol_bsplines_df, vol_date_tenor_df: Volatility surface dataframes
        
    Returns:
        dict with keys: call_strike, call_premium, call_delta, 
                       put_strike, put_premium, put_delta
    """
    
    # Get contract dates
    settlement_date, expiry_date, delivery_date = get_contract_dates(trading_date, tenor)
    trade_to_expiry_days = (expiry_date - trading_date).days
    settlement_to_delivery_days = (delivery_date - settlement_date).days
    trade_to_delivery_days = (delivery_date - trading_date).days
    
    # Get market data with error checking
    try:
        spot = spot_df.loc[trading_date, 'Spot']
        forward = forward_df.loc[trading_date][trade_to_delivery_days - 1]
        foreign_interest_rate = sofr_df.loc[(trading_date, expiry_date), 'SOFR']
    except KeyError:
        raise ValueError(f"Missing market data for date {trading_date}")
    
    # Validate market data
    if not all(isinstance(x, (int, float)) and not np.isnan(x) 
               for x in [spot, forward, foreign_interest_rate]):
        raise ValueError("Market data contains invalid values (NaN or non-numeric)")
    
    # Step 1: Calculate call strike from delta
    call_vol_interp = get_volatility_from_bsplines(trading_date, trade_to_expiry_days, 
                                                    call_delta, 'C', vol_bsplines_df, 
                                                    vol_date_tenor_df)
    
    if call_vol_interp > 1:
      raise ValueError(f"Volatility {call_vol_interp:.4f} > 1")
    
    call_strike = delta_to_strike(call_delta, is_premium_adjusted, is_spot_delta,
                                  spot, trade_to_expiry_days, settlement_to_delivery_days,
                                  foreign_interest_rate, forward, call_vol_interp, 'C',
                                  notional_ccy, premium_ccy)
    
    # Step 2: Price call option
    results_call = price_fx_option(spot, trade_to_expiry_days, settlement_to_delivery_days,
                                   call_strike, foreign_interest_rate, forward, call_vol_interp,
                                   'C', notional_ccy, premium_ccy)
    call_premium = results_call[0] # premium
    call_delta_actual = results_call[4]  # premium_adjusted_spot_delta
    
    # Step 3: Find put strike matching call premium
    put_strike, put_delta, put_vol = premium_to_strike_put_nested(
        call_premium, spot, trade_to_expiry_days, settlement_to_delivery_days,
        foreign_interest_rate, forward, vol_bsplines_df, vol_date_tenor_df, trading_date,
        notional_ccy, premium_ccy)
    
    # Step 2: Price put option
    results_put = price_fx_option(spot, trade_to_expiry_days, settlement_to_delivery_days,
                                   put_strike, foreign_interest_rate, forward, put_vol,
                                   'P', notional_ccy, premium_ccy)
    put_premium = results_put[0] # premium
    put_delta_actual = results_put[4]  # premium_adjusted_spot_delta
    
    return {
        'call_strike': call_strike,
        'call_premium': call_premium,
        'call_delta': call_delta_actual,
        'put_strike': put_strike,
        'put_premium': call_premium,  # Equal by construction
        'put_delta': put_delta_actual
    }

def enter_risk_reversal_no_tenor(trading_date,
                                    settlement_date, 
                                    expiry_date, 
                                    delivery_date,
                                    call_delta: float,
                                    spot_df,
                                    forward_df,
                                    sofr_df,
                                    vol_bsplines_df,
                                    vol_date_tenor_df,
                                    notional_ccy: str = 'foreign',
                                    premium_ccy: str = 'foreign',
                                    is_spot_delta: bool = True,
                                    is_premium_adjusted: bool = True) -> dict:
    """
    Prices a risk reversal (long call, short put with equal premiums).
    
    Args:
        trading_date: Trade date (pd.Timestamp)
        tenor: Option tenor (e.g., '3M')
        call_delta: Target call delta (e.g., 0.25 for 25%)
        spot_df, forward_df, sofr_df: Market data dataframes
        vol_bsplines_df, vol_date_tenor_df: Volatility surface dataframes
        
    Returns:
        dict with keys: call_strike, call_premium, call_delta, 
                       put_strike, put_premium, put_delta
    """
    
    # Get contract dates
    trade_to_expiry_days = (expiry_date - trading_date).days
    settlement_to_delivery_days = (delivery_date - settlement_date).days
    trade_to_delivery_days = (delivery_date - trading_date).days
    
    # Get market data with error checking
    try:
        spot = spot_df.loc[trading_date, 'Spot']
        forward = forward_df.loc[trading_date][trade_to_delivery_days - 1]
        foreign_interest_rate = sofr_df.loc[(trading_date, expiry_date), 'SOFR']
    except KeyError:
        raise ValueError(f"Missing market data for date {trading_date}")
    
    # Validate market data
    if not all(isinstance(x, (int, float)) and not np.isnan(x) 
               for x in [spot, forward, foreign_interest_rate]):
        raise ValueError("Market data contains invalid values (NaN or non-numeric)")
    
    # Step 1: Calculate call strike from delta
    call_vol_interp = get_volatility_from_bsplines(trading_date, trade_to_expiry_days, 
                                                    call_delta, 'C', vol_bsplines_df, 
                                                    vol_date_tenor_df)
    
    if call_vol_interp > 1:
      raise ValueError(f"Volatility {call_vol_interp:.4f} > 1")
    
    call_strike = delta_to_strike(call_delta, is_premium_adjusted, is_spot_delta,
                                  spot, trade_to_expiry_days, settlement_to_delivery_days,
                                  foreign_interest_rate, forward, call_vol_interp, 'C',
                                  notional_ccy, premium_ccy)
    
    # Step 2: Price call option
    results_call = price_fx_option(spot, trade_to_expiry_days, settlement_to_delivery_days,
                                   call_strike, foreign_interest_rate, forward, call_vol_interp,
                                   'C', notional_ccy, premium_ccy)
    call_premium = results_call[0] # premium
    call_delta_actual = results_call[4]  # premium_adjusted_spot_delta
    
    # Step 3: Find put strike matching call premium
    put_strike, put_delta, put_vol = premium_to_strike_put_nested(
        call_premium, spot, trade_to_expiry_days, settlement_to_delivery_days,
        foreign_interest_rate, forward, vol_bsplines_df, vol_date_tenor_df, trading_date,
        notional_ccy, premium_ccy)
    
    # Step 2: Price put option
    results_put = price_fx_option(spot, trade_to_expiry_days, settlement_to_delivery_days,
                                   put_strike, foreign_interest_rate, forward, put_vol,
                                   'P', notional_ccy, premium_ccy)
    put_premium = results_put[0] # premium
    put_delta_actual = results_put[4]  # premium_adjusted_spot_delta
    
    return {
        'call_strike': call_strike,
        'call_premium': call_premium,
        'call_delta': call_delta_actual,
        'put_strike': put_strike,
        'put_premium': call_premium,  # Equal by construction
        'put_delta': put_delta_actual
    }

def strike_to_volatility(spot: float,
                        trading_date,
                        settlement_date,
                        expiry_date,
                        delivery_date,
                        strike: float,
                        foreign_interest_rate: float,
                        forward: float,
                        option_type: str,
                        vol_bsplines_df,
                        vol_date_tenor_df,
                        notional_ccy: str = 'foreign',
                        premium_ccy: str = 'foreign',
                        tolerance: float = 1e-5,
                        max_iterations: int = 1000) -> float:
    """
    Finds volatility for a given strike using iterative delta-vol matching.
    
    Algorithm:
    1. Start with ATM vol (delta=0.5)
    2. Price option → get premium-adjusted spot delta
    3. Get vol from surface for that delta
    4. Repeat until convergence

    References: FX option pricing conventions page 10/12, Wystup FX_Options_and_Structured_Products_----_(1.4_TECHNICAL_ISSUES_FOR_VANILLA_OPTIONS)) page 24/40

    continuous vol delta mapping -> iterative process  -> strike (make sure it considers the type of delta) there is a proof of convergence in the textbook
    
    Returns: volatility (as decimal, e.g., 0.0364 for 3.64%)
    """
    
    trade_to_expiry_days = (expiry_date - trading_date).days
    settlement_to_delivery_days = (delivery_date - settlement_date).days
    
    # Step 1: Initial vol guess (ATM = 50 delta)
    vol_0 = get_volatility_from_bsplines(trading_date, trade_to_expiry_days,
                                         0.5, option_type, 
                                         vol_bsplines_df, vol_date_tenor_df)
    
    # Iterate to convergence
    for _ in range(max_iterations):
        # Step 2: Price option with current vol to get delta
        result = price_fx_option(spot, trade_to_expiry_days, settlement_to_delivery_days,
                                strike, foreign_interest_rate, forward, vol_0,
                                option_type, notional_ccy, premium_ccy)
        
        delta = result[4]  # premium_adjusted_spot_delta
        
        # Step 3: Get vol from surface for this delta
        vol_new = get_volatility_from_bsplines(trading_date, trade_to_expiry_days,
                                                abs(delta), option_type,
                                                vol_bsplines_df, vol_date_tenor_df)
        
        # Step 4: Check convergence
        if abs(vol_new - vol_0) < tolerance:
            return vol_new
        
        # Update for next iteration
        vol_0 = vol_new
    
    # Return last value if no convergence
    print("reached max iterations")
    return vol_new