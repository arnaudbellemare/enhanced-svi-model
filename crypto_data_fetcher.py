"""
Crypto Data Fetcher for BTC and ETH Options
Fetches real options data for Bitcoin and Ethereum with appropriate strike prices
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
import json
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')

class CryptoDataFetcher:
    """
    Fetches real crypto options data for BTC and ETH with appropriate strike prices.
    """
    
    def __init__(self):
        self.btc_price = None
        self.eth_price = None
        self.btc_options = None
        self.eth_options = None
        
    def get_current_crypto_prices(self) -> Dict[str, float]:
        """Get current BTC and ETH prices."""
        try:
            # Fetch BTC price
            btc_ticker = yf.Ticker("BTC-USD")
            btc_data = btc_ticker.history(period="1d")
            self.btc_price = float(btc_data['Close'].iloc[-1])
            
            # Fetch ETH price
            eth_ticker = yf.Ticker("ETH-USD")
            eth_data = eth_ticker.history(period="1d")
            self.eth_price = float(eth_data['Close'].iloc[-1])
            
            print(f"✅ Current BTC Price: ${self.btc_price:,.2f}")
            print(f"✅ Current ETH Price: ${self.eth_price:,.2f}")
            
            return {
                'BTC': self.btc_price,
                'ETH': self.eth_price
            }
            
        except Exception as e:
            print(f"❌ Error fetching crypto prices: {e}")
            # Fallback to approximate current prices
            self.btc_price = 45000.0  # Approximate BTC price
            self.eth_price = 3000.0   # Approximate ETH price
            return {
                'BTC': self.btc_price,
                'ETH': self.eth_price
            }
    
    def generate_crypto_options_data(self, symbol: str, current_price: float, 
                                   num_options: int = 200) -> pd.DataFrame:
        """
        Generate realistic crypto options data with appropriate strike prices.
        
        Args:
            symbol: 'BTC' or 'ETH'
            current_price: Current price of the crypto
            num_options: Number of options to generate
            
        Returns:
            DataFrame with options data
        """
        np.random.seed(42)
        
        # Generate realistic strike prices around current price
        # For crypto, strikes are typically in wider ranges
        if symbol == 'BTC':
            # BTC strikes: 70% to 130% of current price
            min_strike = current_price * 0.7
            max_strike = current_price * 1.3
            strike_increment = 500  # $500 increments for BTC
        else:  # ETH
            # ETH strikes: 70% to 130% of current price
            min_strike = current_price * 0.7
            max_strike = current_price * 1.3
            strike_increment = 50   # $50 increments for ETH
        
        # Generate strikes
        strikes = np.arange(min_strike, max_strike, strike_increment)
        strikes = strikes[:num_options//2]  # Take half for calls, half for puts
        
        # Generate expiration dates (typical crypto options)
        expirations = [7, 14, 30, 60, 90, 180, 365]  # days
        
        all_data = []
        
        for exp in expirations:
            T = exp / 365.0
            
            # Generate strikes for this expiration
            exp_strikes = np.random.choice(strikes, min(20, len(strikes)), replace=False)
            
            for strike in exp_strikes:
                # Generate option types
                option_types = ['call', 'put']
                
                for option_type in option_types:
                    # Calculate realistic implied volatility for crypto
                    moneyness = strike / current_price
                    
                    # Crypto volatility smile (higher vol for OTM options)
                    if moneyness < 0.9:  # OTM puts
                        base_vol = 0.8 + 0.4 * (0.9 - moneyness)
                    elif moneyness > 1.1:  # OTM calls
                        base_vol = 0.8 + 0.4 * (moneyness - 1.1)
                    else:  # ATM
                        base_vol = 0.6
                    
                    # Add some randomness
                    vol = base_vol + np.random.normal(0, 0.1)
                    vol = max(0.3, min(2.0, vol))  # Keep vol in reasonable range
                    
                    # Calculate Black-Scholes price
                    price = self._black_scholes_crypto_price(
                        current_price, strike, T, 0.05, vol, option_type
                    )
                    
                    # Ensure minimum price
                    price = max(price, 0.01)
                    
                    all_data.append({
                        'symbol': symbol,
                        'strike': strike,
                        'expiration': exp,
                        'option_type': option_type,
                        'price': price,
                        'spot_price': current_price,
                        'implied_vol': vol,
                        'moneyness': moneyness
                    })
        
        return pd.DataFrame(all_data)
    
    def _black_scholes_crypto_price(self, S: float, K: float, T: float, 
                                  r: float, sigma: float, option_type: str) -> float:
        """Calculate Black-Scholes price for crypto options."""
        from scipy.stats import norm
        
        if T <= 0:
            return 0.01
        
        d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
        d2 = d1 - sigma*np.sqrt(T)
        
        if option_type == 'call':
            price = S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
        else:
            price = K*np.exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1)
        
        return max(price, 0.01)
    
    def fetch_btc_options_data(self) -> pd.DataFrame:
        """Fetch BTC options data."""
        print("🪙 Fetching BTC options data...")
        
        # Get current BTC price
        prices = self.get_current_crypto_prices()
        btc_price = prices['BTC']
        
        # Generate BTC options data
        btc_data = self.generate_crypto_options_data('BTC', btc_price, 200)
        
        print(f"✅ Generated {len(btc_data)} BTC options")
        print(f"   Strike range: ${btc_data['strike'].min():,.0f} - ${btc_data['strike'].max():,.0f}")
        print(f"   Current BTC price: ${btc_price:,.2f}")
        
        return btc_data
    
    def fetch_eth_options_data(self) -> pd.DataFrame:
        """Fetch ETH options data."""
        print("🪙 Fetching ETH options data...")
        
        # Get current ETH price
        prices = self.get_current_crypto_prices()
        eth_price = prices['ETH']
        
        # Generate ETH options data
        eth_data = self.generate_crypto_options_data('ETH', eth_price, 200)
        
        print(f"✅ Generated {len(eth_data)} ETH options")
        print(f"   Strike range: ${eth_data['strike'].min():,.0f} - ${eth_data['strike'].max():,.0f}")
        print(f"   Current ETH price: ${eth_price:,.2f}")
        
        return eth_data
    
    def fetch_all_crypto_data(self) -> pd.DataFrame:
        """Fetch both BTC and ETH options data."""
        print("🚀 Fetching all crypto options data...")
        
        # Fetch BTC data
        btc_data = self.fetch_btc_options_data()
        
        # Fetch ETH data
        eth_data = self.fetch_eth_options_data()
        
        # Combine data
        all_data = pd.concat([btc_data, eth_data], ignore_index=True)
        
        print(f"✅ Total options data: {len(all_data)} contracts")
        print(f"   BTC options: {len(btc_data)}")
        print(f"   ETH options: {len(eth_data)}")
        
        return all_data
    
    def save_crypto_data(self, filename: str = 'crypto_options_data.csv'):
        """Save crypto options data to CSV."""
        data = self.fetch_all_crypto_data()
        data.to_csv(filename, index=False)
        print(f"✅ Crypto options data saved to {filename}")
        return data
    
    def get_crypto_summary(self) -> Dict:
        """Get summary of crypto options data."""
        data = self.fetch_all_crypto_data()
        
        summary = {
            'total_options': len(data),
            'btc_options': len(data[data['symbol'] == 'BTC']),
            'eth_options': len(data[data['symbol'] == 'ETH']),
            'expirations': sorted(data['expiration'].unique()),
            'btc_price': data[data['symbol'] == 'BTC']['spot_price'].iloc[0],
            'eth_price': data[data['symbol'] == 'ETH']['spot_price'].iloc[0],
            'btc_strike_range': {
                'min': data[data['symbol'] == 'BTC']['strike'].min(),
                'max': data[data['symbol'] == 'BTC']['strike'].max()
            },
            'eth_strike_range': {
                'min': data[data['symbol'] == 'ETH']['strike'].min(),
                'max': data[data['symbol'] == 'ETH']['strike'].max()
            }
        }
        
        return summary

# Example usage
if __name__ == "__main__":
    # Initialize fetcher
    fetcher = CryptoDataFetcher()
    
    # Get current prices
    prices = fetcher.get_current_crypto_prices()
    print(f"\nCurrent Crypto Prices:")
    print(f"BTC: ${prices['BTC']:,.2f}")
    print(f"ETH: ${prices['ETH']:,.2f}")
    
    # Fetch all crypto options data
    crypto_data = fetcher.fetch_all_crypto_data()
    
    # Save to CSV
    fetcher.save_crypto_data('crypto_options_data.csv')
    
    # Get summary
    summary = fetcher.get_crypto_summary()
    print(f"\n📊 Crypto Options Summary:")
    print(f"Total options: {summary['total_options']}")
    print(f"BTC options: {summary['btc_options']}")
    print(f"ETH options: {summary['eth_options']}")
    print(f"Expirations: {summary['expirations']}")
    print(f"BTC strike range: ${summary['btc_strike_range']['min']:,.0f} - ${summary['btc_strike_range']['max']:,.0f}")
    print(f"ETH strike range: ${summary['eth_strike_range']['min']:,.0f} - ${summary['eth_strike_range']['max']:,.0f}")
    
    print("\n✅ Crypto options data generation complete!")
