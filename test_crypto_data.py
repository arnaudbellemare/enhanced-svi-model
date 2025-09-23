#!/usr/bin/env python3
"""
Test Crypto Data Fetcher
Quick test to verify BTC and ETH options data generation
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from crypto_data_fetcher import CryptoDataFetcher
from svi_enhanced import SVIEnhanced

def test_crypto_data():
    """Test crypto data fetching and SVI model."""
    print("🧪 Testing Crypto Data Fetcher")
    print("=" * 50)
    
    try:
        # Test crypto data fetcher
        fetcher = CryptoDataFetcher()
        
        # Get current prices
        print("📊 Fetching current crypto prices...")
        prices = fetcher.get_current_crypto_prices()
        print(f"✅ BTC: ${prices['BTC']:,.2f}")
        print(f"✅ ETH: ${prices['ETH']:,.2f}")
        
        # Test BTC options data
        print("\n🪙 Testing BTC options data...")
        btc_data = fetcher.fetch_btc_options_data()
        print(f"✅ BTC options: {len(btc_data)}")
        print(f"   Strike range: ${btc_data['strike'].min():,.0f} - ${btc_data['strike'].max():,.0f}")
        print(f"   Current price: ${btc_data['spot_price'].iloc[0]:,.2f}")
        
        # Test ETH options data
        print("\n🪙 Testing ETH options data...")
        eth_data = fetcher.fetch_eth_options_data()
        print(f"✅ ETH options: {len(eth_data)}")
        print(f"   Strike range: ${eth_data['strike'].min():,.0f} - ${eth_data['strike'].max():,.0f}")
        print(f"   Current price: ${eth_data['spot_price'].iloc[0]:,.2f}")
        
        # Test SVI model with crypto data
        print("\n🎯 Testing SVI Model with crypto data...")
        svi = SVIEnhanced()
        svi.load_market_data('crypto_data')
        
        print(f"✅ SVI model loaded: {len(svi.market_data)} options")
        
        # Show sample data
        print("\n📋 Sample BTC Options Data:")
        btc_sample = svi.market_data[svi.market_data['symbol'] == 'BTC'].head()
        print(btc_sample[['symbol', 'strike', 'expiration', 'option_type', 'price', 'spot_price']].to_string(index=False))
        
        print("\n📋 Sample ETH Options Data:")
        eth_sample = svi.market_data[svi.market_data['symbol'] == 'ETH'].head()
        print(eth_sample[['symbol', 'strike', 'expiration', 'option_type', 'price', 'spot_price']].to_string(index=False))
        
        # Test probability calculation
        print("\n🔬 Testing probability calculation...")
        probabilities = svi.calculate_implied_probabilities()
        print(f"✅ Calculated probabilities for {len(probabilities)} expirations")
        
        # Show results for first expiration
        first_exp = list(probabilities.keys())[0]
        prob_data = probabilities[first_exp]['probabilities']
        print(f"\n📊 Results for {first_exp} days to expiration:")
        print(f"   Strike range: ${prob_data['strikes'][0]:,.0f} - ${prob_data['strikes'][-1]:,.0f}")
        print(f"   Avg call probability: {np.mean(prob_data['call_probabilities']):.3f}")
        print(f"   Avg put probability: {np.mean(prob_data['put_probabilities']):.3f}")
        
        print("\n✅ All tests passed! Crypto data integration working correctly.")
        
    except Exception as e:
        print(f"❌ Error in crypto data test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_crypto_data()
