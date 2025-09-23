#!/usr/bin/env python3
"""
Test Simple Fetch
Test the simple fetching logic
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from real_crypto_options_fetcher import RealCryptoOptionsFetcher

def test_simple_fetch():
    """Test simple fetching."""
    print("🧪 Testing Simple Fetch")
    print("=" * 50)
    
    try:
        fetcher = RealCryptoOptionsFetcher()
        
        # Initialize exchanges
        print("🔌 Initializing exchanges...")
        fetcher.initialize_exchanges()
        
        # Get current prices
        print("\n💰 Fetching current prices...")
        prices = fetcher.get_current_crypto_prices()
        print(f"✅ BTC: ${prices['BTC']:,.2f}")
        print(f"✅ ETH: ${prices['ETH']:,.2f}")
        
        # Test Deribit options fetching with small limit
        print("\n🪙 Testing Deribit options fetching (limit=5)...")
        deribit_options = fetcher.fetch_deribit_options('BTC', 5)
        print(f"✅ Deribit options: {len(deribit_options)}")
        
        if len(deribit_options) > 0:
            print("\n📋 Sample Deribit options:")
            for i, option in enumerate(deribit_options[:3]):
                print(f"   {i+1}. {option['symbol']} {option['strike']} {option['option_type']} - {option['expiration']} days - ${option['price']:.4f}")
        
        # Test full fetching
        print("\n🔄 Testing full fetching...")
        all_options = fetcher.fetch_all_crypto_options('BTC', 10)
        print(f"✅ Total options: {len(all_options)}")
        
        if len(all_options) > 0:
            print("\n📋 Sample options:")
            sample = all_options.head()
            print(sample[['symbol', 'strike', 'expiration', 'option_type', 'price', 'exchange']].to_string(index=False))
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_simple_fetch()
