#!/usr/bin/env python3
"""
Debug Expirations
Find options with longer expirations
"""

import ccxt
from datetime import datetime

def debug_expirations():
    """Debug option expirations."""
    print("🔍 Debugging Option Expirations")
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
        
        # Check expirations
        expirations = set()
        for option in btc_options:
            parts = option['id'].split('-')
            if len(parts) >= 2:
                expiry_str = parts[1]
                expirations.add(expiry_str)
        
        print(f"\n📅 Found {len(expirations)} unique expirations:")
        for exp in sorted(expirations):
            print(f"   {exp}")
        
        # Test a few options with different expirations
        print(f"\n🔍 Testing options with different expirations:")
        for i, option in enumerate(btc_options[:20]):
            parts = option['id'].split('-')
            if len(parts) >= 4:
                expiry_str = parts[1]
                strike = float(parts[2])
                option_type = parts[3]
                
                # Parse expiry
                expiry_date = parse_deribit_expiry(expiry_str)
                if expiry_date:
                    days_to_expiry = (expiry_date - datetime.now()).days
                    if days_to_expiry > 0:
                        print(f"   ✅ {option['id']} - {days_to_expiry} days to expiry")
                    else:
                        print(f"   ❌ {option['id']} - {days_to_expiry} days (expired)")
        
    except Exception as e:
        print(f"❌ Error: {e}")

def parse_deribit_expiry(expiry_str: str):
    """Parse Deribit expiry string to datetime."""
    try:
        # Format: 24SEP25 -> 2025-09-24
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

if __name__ == "__main__":
    debug_expirations()
