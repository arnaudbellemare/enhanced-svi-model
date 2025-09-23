"""
Advanced Probability Calculator for SVI Model

This module provides sophisticated probability calculation methods
for analyzing market expectations through implied probabilities.
"""

import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.integrate import quad
from scipy.optimize import minimize_scalar
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class ProbabilityCalculator:
    """
    Advanced probability calculator for SVI implied probability analysis.
    """
    
    def __init__(self, risk_free_rate: float = 0.05, dividend_yield: float = 0.0):
        """
        Initialize probability calculator.
        
        Args:
            risk_free_rate: Risk-free interest rate
            dividend_yield: Dividend yield
        """
        self.risk_free_rate = risk_free_rate
        self.dividend_yield = dividend_yield
    
    def calculate_implied_probability_density(self, strikes: np.ndarray, 
                                           call_prices: np.ndarray, 
                                           put_prices: np.ndarray,
                                           spot_price: float,
                                           time_to_expiry: float) -> Dict:
        """
        Calculate implied probability density from option prices.
        
        Args:
            strikes: Array of strike prices
            call_prices: Array of call option prices
            put_prices: Array of put option prices
            spot_price: Current spot price
            time_to_expiry: Time to expiry in years
            
        Returns:
            Dictionary containing probability density and related metrics
        """
        # Calculate implied volatilities
        call_ivs = self._calculate_implied_volatilities(
            call_prices, spot_price, strikes, time_to_expiry, 'call'
        )
        put_ivs = self._calculate_implied_volatilities(
            put_prices, spot_price, strikes, time_to_expiry, 'put'
        )
        
        # Calculate probability density using Breeden-Litzenberger formula
        density = self._breeden_litzenberger_density(
            strikes, call_prices, put_prices, spot_price, time_to_expiry
        )
        
        # Calculate cumulative probabilities
        call_probs = self._calculate_call_probabilities(
            strikes, spot_price, time_to_expiry, call_ivs
        )
        put_probs = self._calculate_put_probabilities(
            strikes, spot_price, time_to_expiry, put_ivs
        )
        
        return {
            'strikes': strikes,
            'density': density,
            'call_probabilities': call_probs,
            'put_probabilities': put_probs,
            'call_ivs': call_ivs,
            'put_ivs': put_ivs,
            'spot_price': spot_price,
            'time_to_expiry': time_to_expiry
        }
    
    def _calculate_implied_volatilities(self, prices: np.ndarray, S: float, 
                                      K: np.ndarray, T: float, option_type: str) -> np.ndarray:
        """Calculate implied volatilities using Newton-Raphson method."""
        ivs = []
        
        for price, strike in zip(prices, K):
            if price <= 0:
                ivs.append(np.nan)
                continue
                
            try:
                iv = self._newton_raphson_iv(price, S, strike, T, option_type)
                ivs.append(iv)
            except:
                ivs.append(np.nan)
        
        return np.array(ivs)
    
    def _newton_raphson_iv(self, price: float, S: float, K: float, 
                          T: float, option_type: str, max_iter: int = 100) -> float:
        """Newton-Raphson method for implied volatility calculation."""
        if T <= 0:
            return np.nan
        
        # Initial guess
        sigma = 0.2
        
        for _ in range(max_iter):
            try:
                d1 = (np.log(S/K) + (self.risk_free_rate - self.dividend_yield + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
                d2 = d1 - sigma*np.sqrt(T)
                
                if option_type == 'call':
                    bs_price = S*np.exp(-self.dividend_yield*T)*norm.cdf(d1) - K*np.exp(-self.risk_free_rate*T)*norm.cdf(d2)
                    vega = S*np.exp(-self.dividend_yield*T)*np.sqrt(T)*norm.pdf(d1)
                else:
                    bs_price = K*np.exp(-self.risk_free_rate*T)*norm.cdf(-d2) - S*np.exp(-self.dividend_yield*T)*norm.cdf(-d1)
                    vega = S*np.exp(-self.dividend_yield*T)*np.sqrt(T)*norm.pdf(d1)
                
                if vega == 0:
                    break
                
                diff = bs_price - price
                if abs(diff) < 1e-6:
                    break
                
                sigma = sigma - diff / vega
                sigma = max(0.001, min(5.0, sigma))
                
            except:
                return np.nan
        
        return sigma
    
    def _breeden_litzenberger_density(self, strikes: np.ndarray, call_prices: np.ndarray,
                                    put_prices: np.ndarray, spot_price: float, 
                                    time_to_expiry: float) -> np.ndarray:
        """
        Calculate risk-neutral density using Breeden-Litzenberger formula.
        
        The density is given by: q(K) = e^(rT) * d²C/dK²
        """
        # Calculate second derivative of call prices with respect to strike
        density = np.zeros_like(strikes)
        
        for i in range(1, len(strikes) - 1):
            # Second derivative approximation
            d2C_dK2 = (call_prices[i+1] - 2*call_prices[i] + call_prices[i-1]) / (strikes[i+1] - strikes[i-1])**2
            density[i] = np.exp(self.risk_free_rate * time_to_expiry) * d2C_dK2
        
        # Handle boundaries
        density[0] = density[1]
        density[-1] = density[-2]
        
        # Ensure non-negative density
        density = np.maximum(density, 0)
        
        return density
    
    def _calculate_call_probabilities(self, strikes: np.ndarray, spot_price: float,
                                    time_to_expiry: float, ivs: np.ndarray) -> np.ndarray:
        """Calculate call option probabilities."""
        probs = []
        
        for strike, iv in zip(strikes, ivs):
            if np.isnan(iv):
                probs.append(np.nan)
                continue
                
            d2 = (np.log(spot_price/strike) + (self.risk_free_rate - self.dividend_yield - 0.5*iv**2)*time_to_expiry) / (iv*np.sqrt(time_to_expiry))
            prob = norm.cdf(d2)
            probs.append(prob)
        
        return np.array(probs)
    
    def _calculate_put_probabilities(self, strikes: np.ndarray, spot_price: float,
                                   time_to_expiry: float, ivs: np.ndarray) -> np.ndarray:
        """Calculate put option probabilities."""
        probs = []
        
        for strike, iv in zip(strikes, ivs):
            if np.isnan(iv):
                probs.append(np.nan)
                continue
                
            d2 = (np.log(spot_price/strike) + (self.risk_free_rate - self.dividend_yield - 0.5*iv**2)*time_to_expiry) / (iv*np.sqrt(time_to_expiry))
            prob = norm.cdf(-d2)
            probs.append(prob)
        
        return np.array(probs)
    
    def calculate_moment_risk_metrics(self, strikes: np.ndarray, density: np.ndarray,
                                    spot_price: float) -> Dict:
        """
        Calculate moment-based risk metrics from probability density.
        
        Args:
            strikes: Array of strike prices
            density: Probability density function
            spot_price: Current spot price
            
        Returns:
            Dictionary of risk metrics
        """
        # Normalize density
        density = density / np.trapz(density, strikes)
        
        # Calculate moments
        mean = np.trapz(strikes * density, strikes)
        variance = np.trapz((strikes - mean)**2 * density, strikes)
        std_dev = np.sqrt(variance)
        
        # Skewness
        skewness = np.trapz((strikes - mean)**3 * density, strikes) / (std_dev**3)
        
        # Kurtosis
        kurtosis = np.trapz((strikes - mean)**4 * density, strikes) / (std_dev**4)
        
        # Tail risk metrics
        tail_risk = self._calculate_tail_risk_metrics(strikes, density, spot_price)
        
        # Value at Risk (VaR) and Expected Shortfall (ES)
        var_metrics = self._calculate_var_metrics(strikes, density)
        
        return {
            'mean': mean,
            'variance': variance,
            'std_dev': std_dev,
            'skewness': skewness,
            'kurtosis': kurtosis,
            'tail_risk': tail_risk,
            'var_metrics': var_metrics
        }
    
    def _calculate_tail_risk_metrics(self, strikes: np.ndarray, density: np.ndarray,
                                   spot_price: float) -> Dict:
        """Calculate tail risk metrics."""
        # Left tail (downside risk)
        left_tail_mask = strikes < spot_price * 0.9
        left_tail_prob = np.trapz(density[left_tail_mask], strikes[left_tail_mask])
        
        # Right tail (upside risk)
        right_tail_mask = strikes > spot_price * 1.1
        right_tail_prob = np.trapz(density[right_tail_mask], strikes[right_tail_mask])
        
        # Extreme tail risk
        extreme_left_mask = strikes < spot_price * 0.8
        extreme_left_prob = np.trapz(density[extreme_left_mask], strikes[extreme_left_mask])
        
        extreme_right_mask = strikes > spot_price * 1.2
        extreme_right_prob = np.trapz(density[extreme_right_mask], strikes[extreme_right_mask])
        
        return {
            'left_tail_probability': left_tail_prob,
            'right_tail_probability': right_tail_prob,
            'extreme_left_probability': extreme_left_prob,
            'extreme_right_probability': extreme_right_prob,
            'tail_ratio': right_tail_prob / left_tail_prob if left_tail_prob > 0 else np.inf
        }
    
    def _calculate_var_metrics(self, strikes: np.ndarray, density: np.ndarray) -> Dict:
        """Calculate Value at Risk and Expected Shortfall metrics."""
        # Sort strikes and density
        sorted_indices = np.argsort(strikes)
        sorted_strikes = strikes[sorted_indices]
        sorted_density = density[sorted_indices]
        
        # Calculate cumulative distribution
        cumulative = np.cumsum(sorted_density)
        cumulative = cumulative / cumulative[-1]  # Normalize
        
        # VaR at different confidence levels
        var_levels = [0.01, 0.05, 0.1, 0.9, 0.95, 0.99]
        var_values = {}
        
        for level in var_levels:
            idx = np.searchsorted(cumulative, level)
            if idx < len(sorted_strikes):
                var_values[f'var_{int(level*100)}'] = sorted_strikes[idx]
            else:
                var_values[f'var_{int(level*100)}'] = sorted_strikes[-1]
        
        # Expected Shortfall (Conditional VaR)
        es_values = {}
        for level in [0.01, 0.05, 0.1]:
            idx = np.searchsorted(cumulative, level)
            if idx > 0:
                tail_density = sorted_density[:idx]
                tail_strikes = sorted_strikes[:idx]
                if np.sum(tail_density) > 0:
                    es_values[f'es_{int(level*100)}'] = np.sum(tail_strikes * tail_density) / np.sum(tail_density)
                else:
                    es_values[f'es_{int(level*100)}'] = sorted_strikes[0]
            else:
                es_values[f'es_{int(level*100)}'] = sorted_strikes[0]
        
        return {**var_values, **es_values}
    
    def calculate_butterfly_spread_probability(self, strikes: np.ndarray, 
                                             call_prices: np.ndarray,
                                             put_prices: np.ndarray,
                                             spot_price: float,
                                             time_to_expiry: float) -> Dict:
        """
        Calculate butterfly spread probabilities for different strike combinations.
        
        Args:
            strikes: Array of strike prices
            call_prices: Array of call option prices
            put_prices: Array of put option prices
            spot_price: Current spot price
            time_to_expiry: Time to expiry in years
            
        Returns:
            Dictionary with butterfly spread probabilities
        """
        butterfly_probs = {}
        
        # Find ATM strike
        atm_idx = np.argmin(np.abs(strikes - spot_price))
        atm_strike = strikes[atm_idx]
        
        # Calculate butterfly spreads for different wings
        wing_sizes = [0.05, 0.1, 0.15, 0.2]  # 5%, 10%, 15%, 20% wings
        
        for wing_size in wing_sizes:
            lower_strike = atm_strike * (1 - wing_size)
            upper_strike = atm_strike * (1 + wing_size)
            
            # Find closest strikes
            lower_idx = np.argmin(np.abs(strikes - lower_strike))
            upper_idx = np.argmin(np.abs(strikes - upper_strike))
            
            if lower_idx < atm_idx < upper_idx:
                # Butterfly spread probability
                butterfly_prob = self._calculate_butterfly_probability(
                    strikes[lower_idx], strikes[atm_idx], strikes[upper_idx],
                    call_prices[lower_idx], call_prices[atm_idx], call_prices[upper_idx],
                    spot_price, time_to_expiry
                )
                
                butterfly_probs[f'wing_{int(wing_size*100)}pct'] = {
                    'lower_strike': strikes[lower_idx],
                    'atm_strike': strikes[atm_idx],
                    'upper_strike': strikes[upper_idx],
                    'probability': butterfly_prob
                }
        
        return butterfly_probs
    
    def _calculate_butterfly_probability(self, K1: float, K2: float, K3: float,
                                       C1: float, C2: float, C3: float,
                                       S: float, T: float) -> float:
        """Calculate butterfly spread probability."""
        # Butterfly spread: long K1, short 2*K2, long K3
        # Probability of expiring between K1 and K3
        try:
            # Calculate implied volatilities
            iv1 = self._newton_raphson_iv(C1, S, K1, T, 'call')
            iv2 = self._newton_raphson_iv(C2, S, K2, T, 'call')
            iv3 = self._newton_raphson_iv(C3, S, K3, T, 'call')
            
            if any(np.isnan([iv1, iv2, iv3])):
                return np.nan
            
            # Calculate probabilities
            d1_1 = (np.log(S/K1) + (self.risk_free_rate - self.dividend_yield + 0.5*iv1**2)*T) / (iv1*np.sqrt(T))
            d1_3 = (np.log(S/K3) + (self.risk_free_rate - self.dividend_yield + 0.5*iv3**2)*T) / (iv3*np.sqrt(T))
            
            prob_between = norm.cdf(d1_3) - norm.cdf(d1_1)
            return max(0, min(1, prob_between))
            
        except:
            return np.nan
    
    def calculate_volatility_smile_metrics(self, strikes: np.ndarray, ivs: np.ndarray,
                                         spot_price: float) -> Dict:
        """
        Calculate volatility smile metrics.
        
        Args:
            strikes: Array of strike prices
            ivs: Array of implied volatilities
            spot_price: Current spot price
            
        Returns:
            Dictionary of smile metrics
        """
        # Remove NaN values
        valid_mask = ~np.isnan(ivs)
        valid_strikes = strikes[valid_mask]
        valid_ivs = ivs[valid_mask]
        
        if len(valid_ivs) < 3:
            return {}
        
        # Calculate moneyness
        moneyness = valid_strikes / spot_price
        
        # Fit quadratic curve to volatility smile
        coeffs = np.polyfit(moneyness, valid_ivs, 2)
        
        # Calculate smile metrics
        smile_metrics = {
            'quadratic_coeffs': coeffs.tolist(),
            'smile_curvature': coeffs[0],  # Second derivative
            'smile_slope': coeffs[1],      # First derivative at ATM
            'atm_volatility': coeffs[2],   # Intercept (ATM vol)
            'smile_range': np.max(valid_ivs) - np.min(valid_ivs),
            'smile_skew': self._calculate_volatility_skew(valid_strikes, valid_ivs, spot_price)
        }
        
        return smile_metrics
    
    def _calculate_volatility_skew(self, strikes: np.ndarray, ivs: np.ndarray, 
                                 spot_price: float) -> float:
        """Calculate volatility skew (slope of smile)."""
        # Find ATM strike
        atm_idx = np.argmin(np.abs(strikes - spot_price))
        
        if atm_idx == 0 or atm_idx == len(strikes) - 1:
            return 0
        
        # Calculate skew as difference between OTM put and OTM call vols
        put_strikes = strikes[strikes < spot_price]
        call_strikes = strikes[strikes > spot_price]
        
        if len(put_strikes) > 0 and len(call_strikes) > 0:
            put_ivs = ivs[strikes < spot_price]
            call_ivs = ivs[strikes > spot_price]
            
            # Average OTM put and call volatilities
            avg_put_vol = np.mean(put_ivs)
            avg_call_vol = np.mean(call_ivs)
            
            skew = avg_put_vol - avg_call_vol
        else:
            skew = 0
        
        return skew


# Example usage
if __name__ == "__main__":
    # Create sample data
    np.random.seed(42)
    spot_price = 100.0
    strikes = np.linspace(80, 120, 20)
    time_to_expiry = 0.25  # 3 months
    
    # Generate synthetic option prices
    call_prices = []
    put_prices = []
    
    for strike in strikes:
        # Simple Black-Scholes prices with smile
        vol = 0.2 + 0.1 * (strike - spot_price) / spot_price
        d1 = (np.log(spot_price/strike) + 0.05 * time_to_expiry + 0.5 * vol**2 * time_to_expiry) / (vol * np.sqrt(time_to_expiry))
        d2 = d1 - vol * np.sqrt(time_to_expiry)
        
        call_price = spot_price * norm.cdf(d1) - strike * np.exp(-0.05 * time_to_expiry) * norm.cdf(d2)
        put_price = strike * np.exp(-0.05 * time_to_expiry) * norm.cdf(-d2) - spot_price * norm.cdf(-d1)
        
        call_prices.append(max(call_price, 0.01))
        put_prices.append(max(put_price, 0.01))
    
    call_prices = np.array(call_prices)
    put_prices = np.array(put_prices)
    
    # Initialize calculator
    calc = ProbabilityCalculator()
    
    # Calculate probability density
    print("Calculating implied probability density...")
    prob_data = calc.calculate_implied_probability_density(
        strikes, call_prices, put_prices, spot_price, time_to_expiry
    )
    
    # Calculate risk metrics
    print("Calculating risk metrics...")
    risk_metrics = calc.calculate_moment_risk_metrics(
        prob_data['strikes'], prob_data['density'], spot_price
    )
    
    # Calculate butterfly probabilities
    print("Calculating butterfly spread probabilities...")
    butterfly_probs = calc.calculate_butterfly_spread_probability(
        strikes, call_prices, put_prices, spot_price, time_to_expiry
    )
    
    # Calculate volatility smile metrics
    print("Calculating volatility smile metrics...")
    smile_metrics = calc.calculate_volatility_smile_metrics(
        strikes, prob_data['call_ivs'], spot_price
    )
    
    # Display results
    print("\nRisk Metrics:")
    for key, value in risk_metrics.items():
        if isinstance(value, dict):
            print(f"  {key}:")
            for sub_key, sub_value in value.items():
                print(f"    {sub_key}: {sub_value:.4f}")
        else:
            print(f"  {key}: {value:.4f}")
    
    print("\nButterfly Spread Probabilities:")
    for key, value in butterfly_probs.items():
        print(f"  {key}: {value['probability']:.4f}")
    
    print("\nVolatility Smile Metrics:")
    for key, value in smile_metrics.items():
        print(f"  {key}: {value}")
    
    print("\nProbability calculation complete!")
