#!/usr/bin/env python3
"""
Test Real Deribit Data
Quick test to verify we're getting real options data from Deribit
"""

import ccxt
import pandas as pd
from datetime import datetime

def test_deribit_connection():
    """Test direct connection to Deribit."""
    print("🧪 Testing Direct Deribit Connection")
    print("=" * 50)
    
    try:
        # Initialize Deribit
        deribit = ccxt.deribit({
            'apiKey': '',
            'secret': '',
            'sandbox': False,
            'enableRateLimit': True,
        })
        
        print("✅ Deribit exchange initialized")
        
        # Test basic connection
        print("\n🔍 Testing basic connection...")
        markets = deribit.fetch_markets()
        print(f"✅ Fetched {len(markets)} markets from Deribit")
        
        # Filter for BTC options
        btc_options = [m for m in markets if m['type'] == 'option' and 'BTC' in m['id']]
        print(f"✅ Found {len(btc_options)} BTC option markets")
        
        # Show sample BTC options
        print("\n📋 Sample BTC Options:")
        for i, option in enumerate(btc_options[:10]):
            print(f"  {i+1}. {option['id']} - {option['type']} - Active: {option['active']}")
        
        # Test fetching ticker for one option
        if btc_options:
            test_option = btc_options[0]
            print(f"\n🔍 Testing ticker fetch for: {test_option['id']}")
            
            try:
                ticker = deribit.fetch_ticker(test_option['id'])
                print(f"✅ Ticker data fetched:")
                print(f"   Last: {ticker.get('last', 'N/A')}")
                print(f"   Bid: {ticker.get('bid', 'N/A')}")
                print(f"   Ask: {ticker.get('ask', 'N/A')}")
                print(f"   Volume: {ticker.get('baseVolume', 'N/A')}")
                
            except Exception as e:
                print(f"❌ Error fetching ticker: {e}")
        
        # Test ETH options
        eth_options = [m for m in markets if m['type'] == 'option' and 'ETH' in m['id']]
        print(f"\n✅ Found {len(eth_options)} ETH option markets")
        
        # Show sample ETH options
        print("\n📋 Sample ETH Options:")
        for i, option in enumerate(eth_options[:10]):
            print(f"  {i+1}. {option['id']} - {option['type']} - Active: {option['active']}")
        
        print("\n✅ Deribit connection test successful!")
        return True
        
    except Exception as e:
        print(f"❌ Error testing Deribit connection: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_real_options_fetcher():
    """Test our real options fetcher."""
    print("\n🧪 Testing Real Options Fetcher")
    print("=" * 50)
    
    try:
        from real_crypto_options_fetcher import RealCryptoOptionsFetcher
        
        fetcher = RealCryptoOptionsFetcher()
        
        # Test initialization
        print("🔌 Initializing exchanges...")
        fetcher.initialize_exchanges()
        
        # Test price fetching
        print("\n💰 Fetching current prices...")
        prices = fetcher.get_current_crypto_prices()
        print(f"✅ BTC: ${prices['BTC']:,.2f}")
        print(f"✅ ETH: ${prices['ETH']:,.2f}")
        
        # Test BTC options fetching
        print("\n🪙 Testing BTC options fetching...")
        btc_data = fetcher.fetch_all_crypto_options('BTC', 20)
        print(f"✅ BTC options: {len(btc_data)}")
        
        if len(btc_data) > 0:
            print(f"   Strike range: ${btc_data['strike'].min():,.0f} - ${btc_data['strike'].max():,.0f}")
            print(f"   Expirations: {sorted(btc_data['expiration'].unique())}")
            print(f"   Exchanges: {btc_data['exchange'].value_counts().to_dict()}")
            
            # Show sample data
            print("\n📋 Sample BTC Options Data:")
            sample = btc_data.head()
            print(sample[['symbol', 'strike', 'expiration', 'option_type', 'price', 'exchange', 'instrument_id']].to_string(index=False))
        
        print("\n✅ Real options fetcher test successful!")
        return True
        
    except Exception as e:
        print(f"❌ Error testing real options fetcher: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🚀 Testing Real Crypto Options Data")
    print("=" * 60)
    
    # Test direct Deribit connection
    deribit_success = test_deribit_connection()
    
    # Test our fetcher
    fetcher_success = test_real_options_fetcher()
    
    if deribit_success and fetcher_success:
        print("\n🎉 All tests passed! Real crypto options data is working!")
    else:
        print("\n❌ Some tests failed. Check the errors above.")
