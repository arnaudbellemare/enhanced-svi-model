#!/usr/bin/env python3
"""
Debug All Options
Debug all option formats and find valid ones
"""

import ccxt
from datetime import datetime

def debug_all_options():
    """Debug all option formats."""
    print("🔍 Debugging All Option Formats")
    print("=" * 50)
    
    try:
        # Initialize Deribit
        deribit = ccxt.deribit({
            'apiKey': '',
            'secret': '',
            'enableRateLimit': True,
        })
        
        # Get markets
        markets = deribit.fetch_markets()
        btc_options = [m for m in markets if m['type'] == 'option' and 'BTC' in m['id']]
        
        print(f"Found {len(btc_options)} BTC options")
        
        # Check different option formats
        print(f"\n📋 Sample option IDs:")
        for i, option in enumerate(btc_options[:20]):
            print(f"   {i+1}. {option['id']}")
        
        # Find options with valid formats
        valid_options = []
        for option in btc_options:
            parts = option['id'].split('-')
            if len(parts) >= 4:
                try:
                    symbol = parts[0]
                    expiry_str = parts[1]
                    strike = float(parts[2])
                    option_type = parts[3]
                    
                    # Try to parse expiry
                    expiry_date = parse_deribit_expiry(expiry_str)
                    if expiry_date:
                        days_to_expiry = (expiry_date - datetime.now()).days
                        if days_to_expiry > 0:
                            valid_options.append({
                                'id': option['id'],
                                'expiry': expiry_str,
                                'strike': strike,
                                'type': option_type,
                                'days': days_to_expiry
                            })
                except:
                    continue
        
        print(f"\n✅ Found {len(valid_options)} valid options with future expirations")
        
        # Show some examples
        for i, option in enumerate(valid_options[:10]):
            print(f"   {i+1}. {option['id']} - {option['days']} days to expiry")
        
        # Test ticker data for valid options
        if valid_options:
            print(f"\n🔍 Testing ticker data for valid options:")
            for i, option in enumerate(valid_options[:3]):
                print(f"\n   Testing {option['id']}:")
                try:
                    ticker = deribit.fetch_ticker(option['id'])
                    print(f"     Last: {ticker.get('last', 'None')}")
                    print(f"     Bid: {ticker.get('bid', 'None')}")
                    print(f"     Ask: {ticker.get('ask', 'None')}")
                    
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
                    print(f"     ❌ Error fetching ticker: {e}")
        
    except Exception as e:
        print(f"❌ Error: {e}")

def parse_deribit_expiry(expiry_str: str):
    """Parse Deribit expiry string to datetime."""
    try:
        # Handle different formats
        if len(expiry_str) == 7:  # 24SEP25
            day = int(expiry_str[:2])
            month_str = expiry_str[2:5]
            year = 2000 + int(expiry_str[5:])
        elif len(expiry_str) == 8:  # 24SEP25
            day = int(expiry_str[:2])
            month_str = expiry_str[2:5]
            year = 2000 + int(expiry_str[5:])
        else:
            return None
        
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

if __name__ == "__main__":
    debug_all_options()
