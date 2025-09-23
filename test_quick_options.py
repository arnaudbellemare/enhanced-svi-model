#!/usr/bin/env python3
"""
Test Quick Options
Fetch a reasonable amount of real options quickly
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from real_crypto_options_fetcher import RealCryptoOptionsFetcher

def test_quick_options():
    """Test fetching a reasonable amount of options quickly."""
    print("⚡ Testing Quick Options Fetch")
    print("=" * 50)
    
    try:
        fetcher = RealCryptoOptionsFetcher()
        
        # Test BTC options with reasonable limit
        print("\n🪙 Fetching BTC options (limit 100)...")
        btc_data = fetcher.fetch_all_crypto_options('BTC', 100)
        
        print(f"\n✅ BTC Results:")
        print(f"   Total options: {len(btc_data)}")
        print(f"   Exchanges: {btc_data['exchange'].value_counts().to_dict()}")
        print(f"   Strike range: ${btc_data['strike'].min():,.0f} - ${btc_data['strike'].max():,.0f}")
        print(f"   Expiration range: {btc_data['expiration'].min()} - {btc_data['expiration'].max()} days")
        print(f"   Price range: ${btc_data['price'].min():.4f} - ${btc_data['price'].max():.4f}")
        
        # Show sample data
        print(f"\n📋 Sample BTC Options:")
        sample = btc_data.head(5)
        for _, row in sample.iterrows():
            print(f"   {row['symbol']} {row['strike']:,.0f} {row['option_type']} - {row['expiration']} days - ${row['price']:.4f} ({row['exchange']})")
        
        # Test ETH options with reasonable limit
        print(f"\n🪙 Fetching ETH options (limit 100)...")
        eth_data = fetcher.fetch_all_crypto_options('ETH', 100)
        
        print(f"\n✅ ETH Results:")
        print(f"   Total options: {len(eth_data)}")
        print(f"   Exchanges: {eth_data['exchange'].value_counts().to_dict()}")
        print(f"   Strike range: ${eth_data['strike'].min():,.0f} - ${eth_data['strike'].max():,.0f}")
        print(f"   Expiration range: {eth_data['expiration'].min()} - {eth_data['expiration'].max()} days")
        print(f"   Price range: ${eth_data['price'].min():.4f} - ${eth_data['price'].max():.4f}")
        
        # Show sample data
        print(f"\n📋 Sample ETH Options:")
        sample = eth_data.head(5)
        for _, row in sample.iterrows():
            print(f"   {row['symbol']} {row['strike']:,.0f} {row['option_type']} - {row['expiration']} days - ${row['price']:.4f} ({row['exchange']})")
        
        print(f"\n🎉 SUCCESS! Fetched {len(btc_data)} BTC + {len(eth_data)} ETH = {len(btc_data) + len(eth_data)} total options!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_quick_options()
