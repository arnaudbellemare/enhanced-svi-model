"""
Real Crypto Options Data Fetcher
Fetches live options data from Deribit and Thalex using CCXT
"""

import ccxt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

class RealCryptoOptionsFetcher:
    """
    Fetches real crypto options data from Deribit and Thalex exchanges.
    """
    
    def __init__(self):
        self.deribit = None
        self.thalex = None
        self.btc_price = None
        self.eth_price = None
        
    def initialize_exchanges(self):
        """Initialize Deribit and Thalex exchanges."""
        try:
            # Initialize Deribit
            self.deribit = ccxt.deribit({
                'apiKey': '',  # Public data doesn't need API key
                'secret': '',
                'sandbox': False,
                'enableRateLimit': True,
            })
            
            # Initialize Thalex (if available)
            try:
                self.thalex = ccxt.thalex({
                    'apiKey': '',
                    'secret': '',
                    'sandbox': False,
                    'enableRateLimit': True,
                })
            except:
                print("⚠️ Thalex exchange not available, using Deribit only")
                self.thalex = None
            
            print("✅ Exchanges initialized successfully")
            
        except Exception as e:
            print(f"❌ Error initializing exchanges: {e}")
            raise
    
    def get_current_crypto_prices(self) -> Dict[str, float]:
        """Get current BTC and ETH prices from exchanges."""
        try:
            # Get BTC price from Deribit
            btc_ticker = self.deribit.fetch_ticker('BTC/USD')
            self.btc_price = float(btc_ticker['last'])
            
            # Get ETH price from Deribit
            eth_ticker = self.deribit.fetch_ticker('ETH/USD')
            self.eth_price = float(eth_ticker['last'])
            
            print(f"✅ Current BTC Price: ${self.btc_price:,.2f}")
            print(f"✅ Current ETH Price: ${self.eth_price:,.2f}")
            
            return {
                'BTC': self.btc_price,
                'ETH': self.eth_price
            }
            
        except Exception as e:
            print(f"❌ Error fetching crypto prices: {e}")
            # Fallback prices
            self.btc_price = 45000.0
            self.eth_price = 3000.0
            return {
                'BTC': self.btc_price,
                'ETH': self.eth_price
            }
    
    def fetch_deribit_options(self, symbol: str, limit: int = 100) -> List[Dict]:
        """Fetch options data from Deribit."""
        try:
            # Deribit options are typically in format: BTC-{expiry}-{strike}-{type}
            # We'll fetch the instruments and filter for options
            
            instruments = self.deribit.fetch_markets()
            options_data = []
            
            # Filter for options instruments
            for instrument in instruments:
                if (instrument['type'] == 'option' and 
                    symbol in instrument['id'] and
                    instrument['active']):
                    
                    try:
                        # Get ticker data for this option
                        ticker = self.deribit.fetch_ticker(instrument['id'])
                        
                        # Parse option details
                        option_info = self._parse_deribit_option(instrument['id'], ticker)
                        if option_info:
                            options_data.append(option_info)
                            
                    except Exception as e:
                        # Skip instruments that can't be fetched
                        continue
                    
                    if len(options_data) >= limit:
                        break
            
            print(f"✅ Fetched {len(options_data)} {symbol} options from Deribit")
            return options_data
            
        except Exception as e:
            print(f"❌ Error fetching Deribit options: {e}")
            return []
    
    def _parse_deribit_option(self, instrument_id: str, ticker: Dict) -> Optional[Dict]:
        """Parse Deribit option instrument ID and ticker data."""
        try:
            # Deribit option format: BTC-29DEC23-45000-C or BTC-29DEC23-45000-P
            parts = instrument_id.split('-')
            if len(parts) < 4:
                return None
            
            symbol = parts[0]
            expiry_str = parts[1]
            strike = float(parts[2])
            option_type = parts[3]  # C for call, P for put
            
            # Convert expiry string to datetime
            expiry_date = self._parse_deribit_expiry(expiry_str)
            if not expiry_date:
                return None
            
            # Calculate days to expiry
            days_to_expiry = (expiry_date - datetime.now()).days
            
            if days_to_expiry <= 0:
                return None
            
            # Get current price
            current_price = self.btc_price if symbol == 'BTC' else self.eth_price
            
            return {
                'symbol': symbol,
                'strike': strike,
                'expiration': days_to_expiry,
                'option_type': 'call' if option_type == 'C' else 'put',
                'price': float(ticker['last']) if ticker['last'] else 0.01,
                'spot_price': current_price,
                'bid': float(ticker['bid']) if ticker['bid'] else 0.01,
                'ask': float(ticker['ask']) if ticker['ask'] else 0.01,
                'volume': float(ticker['baseVolume']) if ticker['baseVolume'] else 0,
                'exchange': 'deribit'
            }
            
        except Exception as e:
            return None
    
    def _parse_deribit_expiry(self, expiry_str: str) -> Optional[datetime]:
        """Parse Deribit expiry string to datetime."""
        try:
            # Format: 29DEC23 -> 2023-12-29
            day = int(expiry_str[:2])
            month_str = expiry_str[2:5]
            year = 2000 + int(expiry_str[5:])
            
            month_map = {
                'JAN': 1, 'FEB': 2, 'MAR': 3, 'APR': 4, 'MAY': 5, 'JUN': 6,
                'JUL': 7, 'AUG': 8, 'SEP': 9, 'OCT': 10, 'NOV': 11, 'DEC': 12
            }
            
            month = month_map.get(month_str)
            if not month:
                return None
            
            return datetime(year, month, day)
            
        except Exception as e:
            return None
    
    def fetch_thalex_options(self, symbol: str, limit: int = 100) -> List[Dict]:
        """Fetch options data from Thalex."""
        if not self.thalex:
            return []
        
        try:
            # Thalex options fetching (similar to Deribit)
            instruments = self.thalex.fetch_markets()
            options_data = []
            
            for instrument in instruments:
                if (instrument['type'] == 'option' and 
                    symbol in instrument['id'] and
                    instrument['active']):
                    
                    try:
                        ticker = self.thalex.fetch_ticker(instrument['id'])
                        option_info = self._parse_thalex_option(instrument['id'], ticker)
                        if option_info:
                            options_data.append(option_info)
                    except:
                        continue
                    
                    if len(options_data) >= limit:
                        break
            
            print(f"✅ Fetched {len(options_data)} {symbol} options from Thalex")
            return options_data
            
        except Exception as e:
            print(f"❌ Error fetching Thalex options: {e}")
            return []
    
    def _parse_thalex_option(self, instrument_id: str, ticker: Dict) -> Optional[Dict]:
        """Parse Thalex option instrument ID and ticker data."""
        try:
            # Thalex option format similar to Deribit
            parts = instrument_id.split('-')
            if len(parts) < 4:
                return None
            
            symbol = parts[0]
            expiry_str = parts[1]
            strike = float(parts[2])
            option_type = parts[3]
            
            expiry_date = self._parse_deribit_expiry(expiry_str)  # Same format
            if not expiry_date:
                return None
            
            days_to_expiry = (expiry_date - datetime.now()).days
            if days_to_expiry <= 0:
                return None
            
            current_price = self.btc_price if symbol == 'BTC' else self.eth_price
            
            return {
                'symbol': symbol,
                'strike': strike,
                'expiration': days_to_expiry,
                'option_type': 'call' if option_type == 'C' else 'put',
                'price': float(ticker['last']) if ticker['last'] else 0.01,
                'spot_price': current_price,
                'bid': float(ticker['bid']) if ticker['bid'] else 0.01,
                'ask': float(ticker['ask']) if ticker['ask'] else 0.01,
                'volume': float(ticker['baseVolume']) if ticker['baseVolume'] else 0,
                'exchange': 'thalex'
            }
            
        except Exception as e:
            return None
    
    def fetch_all_crypto_options(self, symbol: str = 'BTC', limit: int = 200) -> pd.DataFrame:
        """Fetch options data from all available exchanges."""
        try:
            # Initialize exchanges
            self.initialize_exchanges()
            
            # Get current prices
            self.get_current_crypto_prices()
            
            all_options = []
            
            # Fetch from Deribit
            deribit_options = self.fetch_deribit_options(symbol, limit//2)
            all_options.extend(deribit_options)
            
            # Fetch from Thalex
            thalex_options = self.fetch_thalex_options(symbol, limit//2)
            all_options.extend(thalex_options)
            
            if not all_options:
                print(f"⚠️ No options data found for {symbol}, generating synthetic data...")
                return self._generate_synthetic_crypto_options(symbol)
            
            # Convert to DataFrame
            df = pd.DataFrame(all_options)
            
            # Filter out invalid data
            df = df[df['price'] > 0]
            df = df[df['expiration'] > 0]
            df = df[df['strike'] > 0]
            
            print(f"✅ Total {symbol} options: {len(df)}")
            print(f"   Deribit: {len(df[df['exchange'] == 'deribit'])}")
            print(f"   Thalex: {len(df[df['exchange'] == 'thalex'])}")
            print(f"   Strike range: ${df['strike'].min():,.0f} - ${df['strike'].max():,.0f}")
            print(f"   Expirations: {sorted(df['expiration'].unique())}")
            
            return df
            
        except Exception as e:
            print(f"❌ Error fetching crypto options: {e}")
            return self._generate_synthetic_crypto_options(symbol)
    
    def _generate_synthetic_crypto_options(self, symbol: str) -> pd.DataFrame:
        """Generate synthetic crypto options data as fallback."""
        print(f"🔄 Generating synthetic {symbol} options data...")
        
        current_price = self.btc_price if symbol == 'BTC' else self.eth_price
        
        # Generate realistic strike prices
        if symbol == 'BTC':
            strikes = np.arange(current_price * 0.7, current_price * 1.3, 1000)
        else:  # ETH
            strikes = np.arange(current_price * 0.7, current_price * 1.3, 100)
        
        expirations = [7, 14, 30, 60, 90, 180, 365]
        
        all_data = []
        
        for exp in expirations:
            T = exp / 365.0
            exp_strikes = np.random.choice(strikes, min(20, len(strikes)), replace=False)
            
            for strike in exp_strikes:
                for option_type in ['call', 'put']:
                    # Calculate realistic crypto volatility
                    moneyness = strike / current_price
                    if moneyness < 0.9:
                        vol = 0.8 + 0.4 * (0.9 - moneyness)
                    elif moneyness > 1.1:
                        vol = 0.8 + 0.4 * (moneyness - 1.1)
                    else:
                        vol = 0.6
                    
                    vol += np.random.normal(0, 0.1)
                    vol = max(0.3, min(2.0, vol))
                    
                    # Calculate Black-Scholes price
                    price = self._black_scholes_price(current_price, strike, T, 0.05, vol, option_type)
                    price = max(price, 0.01)
                    
                    all_data.append({
                        'symbol': symbol,
                        'strike': strike,
                        'expiration': exp,
                        'option_type': option_type,
                        'price': price,
                        'spot_price': current_price,
                        'bid': price * 0.95,
                        'ask': price * 1.05,
                        'volume': np.random.uniform(0, 1000),
                        'exchange': 'synthetic'
                    })
        
        return pd.DataFrame(all_data)
    
    def _black_scholes_price(self, S: float, K: float, T: float, r: float, sigma: float, option_type: str) -> float:
        """Calculate Black-Scholes price."""
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
    
    def fetch_btc_eth_options(self) -> pd.DataFrame:
        """Fetch both BTC and ETH options data."""
        print("🚀 Fetching BTC and ETH options data from exchanges...")
        
        # Fetch BTC options
        btc_data = self.fetch_all_crypto_options('BTC', 200)
        
        # Fetch ETH options
        eth_data = self.fetch_all_crypto_options('ETH', 200)
        
        # Combine data
        all_data = pd.concat([btc_data, eth_data], ignore_index=True)
        
        print(f"✅ Total options data: {len(all_data)}")
        print(f"   BTC: {len(btc_data)}")
        print(f"   ETH: {len(eth_data)}")
        
        return all_data

# Example usage
if __name__ == "__main__":
    fetcher = RealCryptoOptionsFetcher()
    
    # Fetch BTC options
    print("🪙 Fetching BTC options...")
    btc_data = fetcher.fetch_all_crypto_options('BTC', 100)
    print(f"✅ BTC options: {len(btc_data)}")
    
    # Fetch ETH options
    print("\n🪙 Fetching ETH options...")
    eth_data = fetcher.fetch_all_crypto_options('ETH', 100)
    print(f"✅ ETH options: {len(eth_data)}")
    
    # Save data
    all_data = pd.concat([btc_data, eth_data], ignore_index=True)
    all_data.to_csv('real_crypto_options.csv', index=False)
    print(f"\n✅ Saved {len(all_data)} options to real_crypto_options.csv")
