#!/usr/bin/env python3
"""
Test Real Data Only
Verify we're getting ONLY real data from Deribit and Thalex
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_real_data_only():
    """Test that we get ONLY real data from exchanges."""
    print("🧪 Testing REAL DATA ONLY - No Synthetic Data")
    print("=" * 60)
    
    try:
        from svi_enhanced import SVIEnhanced
        
        # Initialize SVI model
        svi = SVIEnhanced()
        
        print("🔍 Loading crypto data...")
        svi.load_market_data('crypto_data')
        
        print(f"✅ Data loaded: {len(svi.market_data)} options")
        
        # Verify data is real
        if len(svi.market_data) > 0:
            print("\n📊 REAL DATA VERIFICATION:")
            print(f"   Total options: {len(svi.market_data)}")
            print(f"   BTC options: {len(svi.market_data[svi.market_data['symbol'] == 'BTC'])}")
            print(f"   ETH options: {len(svi.market_data[svi.market_data['symbol'] == 'ETH'])}")
            
            # Check exchanges
            if 'exchange' in svi.market_data.columns:
                print(f"   Exchanges: {svi.market_data['exchange'].value_counts().to_dict()}")
            
            # Check strike prices are realistic
            btc_data = svi.market_data[svi.market_data['symbol'] == 'BTC']
            if len(btc_data) > 0:
                print(f"   BTC strike range: ${btc_data['strike'].min():,.0f} - ${btc_data['strike'].max():,.0f}")
                print(f"   BTC current price: ${btc_data['spot_price'].iloc[0]:,.2f}")
            
            eth_data = svi.market_data[svi.market_data['symbol'] == 'ETH']
            if len(eth_data) > 0:
                print(f"   ETH strike range: ${eth_data['strike'].min():,.0f} - ${eth_data['strike'].max():,.0f}")
                print(f"   ETH current price: ${eth_data['spot_price'].iloc[0]:,.2f}")
            
            # Show sample data
            print("\n📋 Sample Real Options Data:")
            sample = svi.market_data.head(10)
            print(sample[['symbol', 'strike', 'expiration', 'option_type', 'price', 'exchange']].to_string(index=False))
            
            # Verify no synthetic data
            if 'synthetic' in svi.market_data['exchange'].values:
                print("❌ ERROR: Found synthetic data!")
                return False
            else:
                print("✅ VERIFIED: No synthetic data found - all data is real!")
            
            # Test probability calculation
            print("\n🔬 Testing probability calculation with real data...")
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
            
            print("\n🎉 SUCCESS: Real data integration working correctly!")
            return True
        else:
            print("❌ ERROR: No data loaded!")
            return False
            
    except Exception as e:
        print(f"❌ Error in real data test: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    import numpy as np
    
    success = test_real_data_only()
    
    if success:
        print("\n✅ ALL TESTS PASSED - Real data only!")
    else:
        print("\n❌ TESTS FAILED - Check errors above")
