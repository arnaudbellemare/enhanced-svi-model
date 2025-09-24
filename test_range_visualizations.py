#!/usr/bin/env python3
"""
Test script to create range-specific probability density visualizations.
Creates visualizations for 1-30, 1-60, and 1-90 day ranges.
"""

from svi_enhanced import SVIEnhanced
from visualization import SVIVisualizer
import numpy as np

def test_range_visualizations():
    """Test the new range probability density functions."""
    print('📊 Testing Range-Specific Probability Density Visualizations')
    print('=' * 70)
    
    # Load real crypto data
    print('🔄 Loading real crypto data...')
    svi = SVIEnhanced()
    svi.load_market_data('crypto_data')
    svi.calculate_implied_probabilities()
    
    print('✅ SVI model loaded with real data')
    
    # Create visualizer
    visualizer = SVIVisualizer(svi)
    
    # Test 1-30 days range
    print('\n🔧 Testing 1-30 days range...')
    try:
        fig_30 = visualizer.plot_range_probability_density(1, 30, 'range_1_30_days.html')
        print('✅ 1-30 days range visualization created!')
        print('📁 Saved as range_1_30_days.html')
    except Exception as e:
        print(f'❌ Error creating 1-30 days range: {e}')
        import traceback
        traceback.print_exc()
    
    # Test 1-60 days range
    print('\n🔧 Testing 1-60 days range...')
    try:
        fig_60 = visualizer.plot_range_probability_density(1, 60, 'range_1_60_days.html')
        print('✅ 1-60 days range visualization created!')
        print('📁 Saved as range_1_60_days.html')
    except Exception as e:
        print(f'❌ Error creating 1-60 days range: {e}')
        import traceback
        traceback.print_exc()
    
    # Test 1-90 days range
    print('\n🔧 Testing 1-90 days range...')
    try:
        fig_90 = visualizer.plot_range_probability_density(1, 90, 'range_1_90_days.html')
        print('✅ 1-90 days range visualization created!')
        print('📁 Saved as range_1_90_days.html')
    except Exception as e:
        print(f'❌ Error creating 1-90 days range: {e}')
        import traceback
        traceback.print_exc()
    
    print('\n🎉 All range visualizations completed!')
    print('📊 Created visualizations for:')
    print('   • 1-30 days range')
    print('   • 1-60 days range') 
    print('   • 1-90 days range')
    print('🌐 Open the HTML files to see the interactive visualizations!')

if __name__ == "__main__":
    test_range_visualizations()
