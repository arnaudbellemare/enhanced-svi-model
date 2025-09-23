#!/usr/bin/env python3
"""
Debug Ticker Data
Debug what's happening with Deribit ticker data
"""

import ccxt
import pandas as pd

def debug_deribit_ticker():
    """Debug Deribit ticker data."""
    print("🔍 Debugging Deribit Ticker Data")
    print("=" * 50)
    
    try:
        # Initialize Deribit
        deribit = ccxt.deribit({
            'apiKey': '',
            'secret': '',
            'sandbox': False,
            'enableRateLimit': True,
        })
        
        # Get markets
        markets = deribit.fetch_markets()
        btc_options = [m for m in markets if m['type'] == 'option' and 'BTC' in m['id']]
        
        print(f"Found {len(btc_options)} BTC options")
        
        # Test a few options
        for i, option in enumerate(btc_options[:5]):
            print(f"\n🔍 Testing option {i+1}: {option['id']}")
            
            try:
                ticker = deribit.fetch_ticker(option['id'])
                print(f"   Ticker data:")
                print(f"     Last: {ticker.get('last', 'None')}")
                print(f"     Bid: {ticker.get('bid', 'None')}")
                print(f"     Ask: {ticker.get('ask', 'None')}")
                print(f"     Volume: {ticker.get('baseVolume', 'None')}")
                print(f"     High: {ticker.get('high', 'None')}")
                print(f"     Low: {ticker.get('low', 'None')}")
                
                # Test our parsing logic
                last_price = None
                if ticker['last'] and ticker['last'] > 0:
                    last_price = float(ticker['last'])
                    print(f"     ✅ Using last price: {last_price}")
                elif ticker['bid'] and ticker['ask'] and ticker['bid'] > 0 and ticker['ask'] > 0:
                    last_price = (float(ticker['bid']) + float(ticker['ask'])) / 2
                    print(f"     ✅ Using mid-price: {last_price}")
                else:
                    print(f"     ❌ No valid price data")
                
                if last_price:
                    print(f"     ✅ Option would be included")
                else:
                    print(f"     ❌ Option would be skipped")
                    
            except Exception as e:
                print(f"   ❌ Error fetching ticker: {e}")
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    debug_deribit_ticker()
