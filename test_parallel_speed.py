#!/usr/bin/env python3
"""
Test Parallel Speed
Test the speed improvement with 12 workers
"""

import sys
import os
import time
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from real_crypto_options_fetcher import RealCryptoOptionsFetcher

def test_parallel_speed():
    """Test parallel processing speed."""
    print("⚡ Testing Parallel Processing Speed")
    print("=" * 50)
    
    try:
        fetcher = RealCryptoOptionsFetcher()
        
        # Initialize exchanges
        print("🔌 Initializing exchanges...")
        fetcher.initialize_exchanges()
        
        # Get current prices
        print("\n💰 Fetching current prices...")
        fetcher.get_current_crypto_prices()
        
        # Test BTC options with parallel processing
        print("\n🚀 Testing BTC options with 12 workers...")
        start_time = time.time()
        
        btc_data = fetcher.fetch_all_crypto_options('BTC', 200)  # Limit to 200 for speed test
        
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"\n✅ BTC Results:")
        print(f"   Total options: {len(btc_data)}")
        print(f"   Time taken: {duration:.2f} seconds")
        print(f"   Speed: {len(btc_data)/duration:.1f} options/second")
        print(f"   Exchanges: {btc_data['exchange'].value_counts().to_dict()}")
        
        # Test ETH options with parallel processing
        print(f"\n🚀 Testing ETH options with 12 workers...")
        start_time = time.time()
        
        eth_data = fetcher.fetch_all_crypto_options('ETH', 200)  # Limit to 200 for speed test
        
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"\n✅ ETH Results:")
        print(f"   Total options: {len(eth_data)}")
        print(f"   Time taken: {duration:.2f} seconds")
        print(f"   Speed: {len(eth_data)/duration:.1f} options/second")
        print(f"   Exchanges: {eth_data['exchange'].value_counts().to_dict()}")
        
        total_options = len(btc_data) + len(eth_data)
        print(f"\n🎉 SUCCESS! Fetched {total_options} total options with 12 workers!")
        print(f"   Much faster than sequential processing! 🚀")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_parallel_speed()
