#!/usr/bin/env python3
"""
Enhanced SVI Model Demo

This script demonstrates the complete functionality of the Enhanced SVI Model
with implied probability calculations for all expiration dates.
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from svi_enhanced import SVIEnhanced
from probability_calculator import ProbabilityCalculator
from visualization import SVIVisualizer

def print_banner():
    """Print welcome banner."""
    print("=" * 80)
    print("ENHANCED SVI MODEL WITH IMPLIED PROBABILITY ANALYSIS")
    print("=" * 80)
    print("Advanced Stochastic Volatility Inspired Model")
    print("with Comprehensive Implied Probability Calculations")
    print("for All Expiration Dates")
    print("=" * 80)
    print(f"Demo started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

def demonstrate_core_functionality():
    """Demonstrate core SVI functionality."""
    print("STEP 1: Initializing Enhanced SVI Model")
    print("-" * 50)
    
    # Initialize the enhanced SVI model
    svi = SVIEnhanced()
    print("✓ SVI model initialized")
    
    # Load market data (will generate synthetic data for demo)
    print("✓ Loading market data...")
    svi.load_market_data('synthetic_data')
    print(f"✓ Market data loaded: {len(svi.market_data)} options")
    
    # Calculate implied probabilities for all expirations
    print("✓ Calculating implied probabilities for all expirations...")
    probabilities = svi.calculate_implied_probabilities()
    print(f"✓ Calculated probabilities for {len(probabilities)} expirations")
    
    return svi

def demonstrate_probability_analysis(svi):
    """Demonstrate probability analysis capabilities."""
    print("\nSTEP 2: Implied Probability Analysis")
    print("-" * 50)
    
    # Initialize probability calculator
    calc = ProbabilityCalculator(risk_free_rate=0.05, dividend_yield=0.02)
    print("✓ Probability calculator initialized")
    
    # Analyze each expiration
    print("\nDetailed Analysis by Expiration:")
    print("Expiration | Strike Points | Avg Call Prob | Avg Put Prob | Risk Level")
    print("-" * 70)
    
    for exp in sorted(svi.implied_probabilities.keys()):
        prob_data = svi.implied_probabilities[exp]['probabilities']
        strikes = prob_data['strikes']
        call_probs = prob_data['call_probabilities']
        put_probs = prob_data['put_probabilities']
        
        avg_call_prob = np.mean(call_probs)
        avg_put_prob = np.mean(put_probs)
        risk_level = "High" if avg_call_prob > 0.6 else "Medium" if avg_call_prob > 0.4 else "Low"
        
        print(f"{exp:>10} | {len(strikes):>12} | {avg_call_prob:>12.3f} | {avg_put_prob:>11.3f} | {risk_level:>10}")
    
    return calc

def demonstrate_risk_metrics(svi):
    """Demonstrate risk metrics calculation."""
    print("\nSTEP 3: Risk Metrics Analysis")
    print("-" * 50)
    
    # Calculate risk metrics
    risk_metrics = svi.get_risk_metrics()
    print("✓ Risk metrics calculated")
    
    # Display risk summary
    print("\nRisk Metrics Summary:")
    print("Expiration | ATM Prob | Skew | Tail Risk | Up Move | Down Move")
    print("-" * 65)
    
    for exp in sorted(risk_metrics.keys()):
        metrics = risk_metrics[exp]
        print(f"{exp:>10} | {metrics['atm_probability']:>8.3f} | {metrics['volatility_skew']:>4.2f} | {metrics['tail_risk']:>9.3f} | {metrics['large_up_move_prob']:>7.3f} | {metrics['large_down_move_prob']:>9.3f}")
    
    # Calculate portfolio-level metrics
    print("\nPortfolio-Level Risk Analysis:")
    avg_atm_prob = np.mean([m['atm_probability'] for m in risk_metrics.values()])
    avg_skew = np.mean([m['volatility_skew'] for m in risk_metrics.values()])
    avg_tail_risk = np.mean([m['tail_risk'] for m in risk_metrics.values()])
    
    print(f"  Average ATM Probability: {avg_atm_prob:.3f}")
    print(f"  Average Volatility Skew: {avg_skew:.3f}")
    print(f"  Average Tail Risk: {avg_tail_risk:.3f}")
    
    # Risk trend analysis
    expirations = sorted(risk_metrics.keys())
    tail_risks = [risk_metrics[exp]['tail_risk'] for exp in expirations]
    if len(tail_risks) > 1:
        risk_trend = np.polyfit(expirations, tail_risks, 1)[0]
        trend_direction = "increasing" if risk_trend > 0 else "decreasing"
        print(f"  Risk Trend: {trend_direction} over time (slope: {risk_trend:.6f})")
    
    return risk_metrics

def demonstrate_visualization(svi):
    """Demonstrate visualization capabilities."""
    print("\nSTEP 4: Advanced Visualization")
    print("-" * 50)
    
    # Initialize visualizer
    visualizer = SVIVisualizer(svi)
    print("✓ Visualizer initialized")
    
    # Create visualizations
    print("✓ Creating 3D probability surface...")
    visualizer.plot_3d_probability_surface('demo_3d_surface.png')
    
    print("✓ Creating interactive heatmap...")
    visualizer.plot_interactive_probability_heatmap('demo_interactive_heatmap.html')
    
    print("✓ Creating risk metrics dashboard...")
    visualizer.plot_risk_metrics_dashboard('demo_risk_dashboard.png')
    
    print("✓ Creating SVI parameter analysis...")
    visualizer.plot_svi_parameter_analysis('demo_svi_parameters.png')
    
    print("✓ Creating comprehensive report...")
    visualizer.create_comprehensive_report('demo_comprehensive_report.html')
    
    print("\nVisualization files created:")
    print("  - demo_3d_surface.png")
    print("  - demo_interactive_heatmap.html")
    print("  - demo_risk_dashboard.png")
    print("  - demo_svi_parameters.png")
    print("  - demo_comprehensive_report.html")

def demonstrate_export_functionality(svi):
    """Demonstrate data export capabilities."""
    print("\nSTEP 5: Data Export and Analysis")
    print("-" * 50)
    
    # Export results
    print("✓ Exporting probability data...")
    svi.export_results('demo_implied_probabilities.csv')
    
    # Create detailed analysis export
    print("✓ Creating detailed analysis export...")
    
    export_data = []
    for exp, prob_info in svi.implied_probabilities.items():
        strikes = prob_info['probabilities']['strikes']
        call_probs = prob_info['probabilities']['call_probabilities']
        put_probs = prob_info['probabilities']['put_probabilities']
        density = prob_info['probabilities']['risk_neutral_density']
        
        for i, strike in enumerate(strikes):
            export_data.append({
                'expiration': exp,
                'strike': strike,
                'call_probability': call_probs[i],
                'put_probability': put_probs[i],
                'risk_neutral_density': density[i],
                'moneyness': strike / svi.market_data['spot_price'].iloc[0],
                'log_moneyness': np.log(strike / svi.market_data['spot_price'].iloc[0])
            })
    
    detailed_df = pd.DataFrame(export_data)
    detailed_df.to_csv('demo_detailed_analysis.csv', index=False)
    
    # Create summary statistics
    summary_stats = detailed_df.groupby('expiration').agg({
        'call_probability': ['mean', 'std', 'min', 'max'],
        'put_probability': ['mean', 'std', 'min', 'max'],
        'risk_neutral_density': ['mean', 'std', 'min', 'max']
    }).round(4)
    
    summary_stats.to_csv('demo_summary_statistics.csv')
    
    print("✓ Export files created:")
    print("  - demo_implied_probabilities.csv")
    print("  - demo_detailed_analysis.csv")
    print("  - demo_summary_statistics.csv")
    
    # Display sample statistics
    print(f"\nSample Statistics (first expiration):")
    first_exp = detailed_df['expiration'].iloc[0]
    first_exp_data = detailed_df[detailed_df['expiration'] == first_exp]
    
    print(f"  Expiration: {first_exp} days")
    print(f"  Strike Range: {first_exp_data['strike'].min():.2f} - {first_exp_data['strike'].max():.2f}")
    print(f"  Call Probability Range: {first_exp_data['call_probability'].min():.3f} - {first_exp_data['call_probability'].max():.3f}")
    print(f"  Put Probability Range: {first_exp_data['put_probability'].min():.3f} - {first_exp_data['put_probability'].max():.3f}")

def demonstrate_advanced_features(svi, calc):
    """Demonstrate advanced features."""
    print("\nSTEP 6: Advanced Features")
    print("-" * 50)
    
    # Get market data for detailed analysis
    market_data = svi.market_data
    
    # Analyze butterfly spread probabilities
    print("✓ Calculating butterfly spread probabilities...")
    
    for exp in sorted(svi.implied_probabilities.keys())[:3]:  # Analyze first 3 expirations
        exp_data = market_data[market_data['expiration'] == exp]
        call_data = exp_data[exp_data['option_type'] == 'call']
        put_data = exp_data[exp_data['option_type'] == 'put']
        
        if len(call_data) > 0 and len(put_data) > 0:
            butterfly_probs = calc.calculate_butterfly_spread_probability(
                call_data['strike'].values,
                call_data['price'].values,
                put_data['price'].values,
                call_data['spot_price'].iloc[0],
                exp / 365.0
            )
            
            print(f"\n  {exp} days to expiration - Butterfly Spread Probabilities:")
            for wing, data in butterfly_probs.items():
                print(f"    {wing}: {data['probability']:.3f}")
    
    # Calculate volatility smile metrics
    print("\n✓ Calculating volatility smile metrics...")
    
    for exp in sorted(svi.implied_probabilities.keys())[:2]:  # Analyze first 2 expirations
        prob_info = svi.implied_probabilities[exp]['probabilities']
        strikes = prob_info['strikes']
        
        # Generate synthetic implied volatilities for demonstration
        spot_price = svi.market_data['spot_price'].iloc[0]
        ivs = np.array([0.2 + 0.1 * (strike - spot_price) / spot_price for strike in strikes])
        
        smile_metrics = calc.calculate_volatility_smile_metrics(strikes, ivs, spot_price)
        
        if smile_metrics:
            print(f"\n  {exp} days to expiration - Volatility Smile Metrics:")
            print(f"    ATM Volatility: {smile_metrics['atm_volatility']:.3f}")
            print(f"    Smile Curvature: {smile_metrics['smile_curvature']:.3f}")
            print(f"    Smile Skew: {smile_metrics['smile_skew']:.3f}")
            print(f"    Smile Range: {smile_metrics['smile_range']:.3f}")

def print_summary():
    """Print demo summary."""
    print("\n" + "=" * 80)
    print("DEMO SUMMARY")
    print("=" * 80)
    print("✓ Enhanced SVI Model successfully demonstrated")
    print("✓ Implied probability calculations completed for all expirations")
    print("✓ Risk metrics calculated and analyzed")
    print("✓ Advanced visualizations created")
    print("✓ Data export functionality demonstrated")
    print("✓ Advanced features showcased")
    print("\nGenerated Files:")
    print("  📊 Visualizations:")
    print("    - demo_3d_surface.png")
    print("    - demo_interactive_heatmap.html")
    print("    - demo_risk_dashboard.png")
    print("    - demo_svi_parameters.png")
    print("    - demo_comprehensive_report.html")
    print("  📁 Data Exports:")
    print("    - demo_implied_probabilities.csv")
    print("    - demo_detailed_analysis.csv")
    print("    - demo_summary_statistics.csv")
    print("\nThe Enhanced SVI Model provides comprehensive implied probability")
    print("analysis across all expiration dates, enabling deeper market insights")
    print("through probability-based risk assessment and visualization.")
    print("=" * 80)

def main():
    """Main demo function."""
    try:
        print_banner()
        
        # Step 1: Core functionality
        svi = demonstrate_core_functionality()
        
        # Step 2: Probability analysis
        calc = demonstrate_probability_analysis(svi)
        
        # Step 3: Risk metrics
        risk_metrics = demonstrate_risk_metrics(svi)
        
        # Step 4: Visualization
        demonstrate_visualization(svi)
        
        # Step 5: Export functionality
        demonstrate_export_functionality(svi)
        
        # Step 6: Advanced features
        demonstrate_advanced_features(svi, calc)
        
        # Summary
        print_summary()
        
    except Exception as e:
        print(f"\n❌ Error during demo: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
