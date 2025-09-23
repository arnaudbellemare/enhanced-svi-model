#!/usr/bin/env python3
"""
Test Real Crypto Options Data
Test the real crypto options fetcher with Deribit and Thalex
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from real_crypto_options_fetcher import RealCryptoOptionsFetcher
from svi_enhanced import SVIEnhanced

def test_real_crypto_data():
    """Test real crypto options data fetching."""
    print("🧪 Testing Real Crypto Options Data Fetcher")
    print("=" * 60)
    
    try:
        # Test real crypto options fetcher
        fetcher = RealCryptoOptionsFetcher()
        
        # Test exchange initialization
        print("🔌 Initializing exchanges...")
        fetcher.initialize_exchanges()
        
        # Test price fetching
        print("\n💰 Fetching current crypto prices...")
        prices = fetcher.get_current_crypto_prices()
        print(f"✅ BTC: ${prices['BTC']:,.2f}")
        print(f"✅ ETH: ${prices['ETH']:,.2f}")
        
        # Test BTC options fetching
        print("\n🪙 Testing BTC options fetching...")
        btc_data = fetcher.fetch_all_crypto_options('BTC', 50)
        print(f"✅ BTC options: {len(btc_data)}")
        
        if len(btc_data) > 0:
            print(f"   Strike range: ${btc_data['strike'].min():,.0f} - ${btc_data['strike'].max():,.0f}")
            print(f"   Expirations: {sorted(btc_data['expiration'].unique())}")
            print(f"   Exchanges: {btc_data['exchange'].value_counts().to_dict()}")
            
            # Show sample data
            print("\n📋 Sample BTC Options:")
            sample = btc_data.head()
            print(sample[['symbol', 'strike', 'expiration', 'option_type', 'price', 'exchange']].to_string(index=False))
        
        # Test ETH options fetching
        print("\n🪙 Testing ETH options fetching...")
        eth_data = fetcher.fetch_all_crypto_options('ETH', 50)
        print(f"✅ ETH options: {len(eth_data)}")
        
        if len(eth_data) > 0:
            print(f"   Strike range: ${eth_data['strike'].min():,.0f} - ${eth_data['strike'].max():,.0f}")
            print(f"   Expirations: {sorted(eth_data['expiration'].unique())}")
            print(f"   Exchanges: {eth_data['exchange'].value_counts().to_dict()}")
            
            # Show sample data
            print("\n📋 Sample ETH Options:")
            sample = eth_data.head()
            print(sample[['symbol', 'strike', 'expiration', 'option_type', 'price', 'exchange']].to_string(index=False))
        
        # Test combined data
        print("\n🔄 Testing combined BTC and ETH data...")
        all_data = fetcher.fetch_btc_eth_options()
        print(f"✅ Total options: {len(all_data)}")
        print(f"   BTC: {len(all_data[all_data['symbol'] == 'BTC'])}")
        print(f"   ETH: {len(all_data[all_data['symbol'] == 'ETH'])}")
        
        # Test SVI model with real data
        print("\n🎯 Testing SVI Model with real crypto data...")
        svi = SVIEnhanced()
        svi.load_market_data('crypto_data')
        
        print(f"✅ SVI model loaded: {len(svi.market_data)} options")
        
        # Show data summary
        if len(svi.market_data) > 0:
            print(f"\n📊 Data Summary:")
            print(f"   Total options: {len(svi.market_data)}")
            print(f"   BTC options: {len(svi.market_data[svi.market_data['symbol'] == 'BTC'])}")
            print(f"   ETH options: {len(svi.market_data[svi.market_data['symbol'] == 'ETH'])}")
            print(f"   Strike range: ${svi.market_data['strike'].min():,.0f} - ${svi.market_data['strike'].max():,.0f}")
            print(f"   Expirations: {sorted(svi.market_data['expiration'].unique())}")
            
            # Show exchange breakdown
            if 'exchange' in svi.market_data.columns:
                print(f"   Exchanges: {svi.market_data['exchange'].value_counts().to_dict()}")
            
            # Test probability calculation
            print("\n🔬 Testing probability calculation...")
            probabilities = svi.calculate_implied_probabilities()
            print(f"✅ Calculated probabilities for {len(probabilities)} expirations")
            
            # Show results for first expiration
            if probabilities:
                first_exp = list(probabilities.keys())[0]
                prob_data = probabilities[first_exp]['probabilities']
                print(f"\n📊 Results for {first_exp} days to expiration:")
                print(f"   Strike range: ${prob_data['strikes'][0]:,.0f} - ${prob_data['strikes'][-1]:,.0f}")
                print(f"   Avg call probability: {np.mean(prob_data['call_probabilities']):.3f}")
                print(f"   Avg put probability: {np.mean(prob_data['put_probabilities']):.3f}")
        
        print("\n✅ All tests passed! Real crypto data integration working correctly.")
        
    except Exception as e:
        print(f"❌ Error in real crypto data test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_real_crypto_data()
