"""
Example Usage of Enhanced SVI Model with Implied Probability Analysis

This script demonstrates comprehensive usage of the enhanced SVI model
for analyzing implied probabilities across all expiration dates.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from svi_enhanced import SVIEnhanced
from probability_calculator import ProbabilityCalculator
from visualization import SVIVisualizer
import warnings
warnings.filterwarnings('ignore')

def create_sample_market_data():
    """Create comprehensive sample market data for demonstration."""
    np.random.seed(42)
    
    # Market parameters
    spot_price = 100.0
    risk_free_rate = 0.05
    dividend_yield = 0.02
    
    # Generate multiple expiration dates
    expirations = [7, 14, 30, 60, 90, 180, 365]
    n_options_per_exp = 50
    
    all_data = []
    
    for exp in expirations:
        T = exp / 365.0
        
        # Generate strikes around spot price with some skew
        strikes = np.linspace(spot_price * 0.7, spot_price * 1.3, n_options_per_exp)
        
        # Generate option types
        option_types = np.random.choice(['call', 'put'], n_options_per_exp)
        
        for i, (strike, option_type) in enumerate(zip(strikes, option_types)):
            # Create volatility smile
            moneyness = strike / spot_price
            base_vol = 0.2
            vol_smile = 0.1 * (moneyness - 1.0)**2
            vol = base_vol + vol_smile + np.random.normal(0, 0.02)
            vol = max(0.05, min(0.8, vol))  # Keep vol in reasonable range
            
            # Calculate Black-Scholes price
            d1 = (np.log(spot_price/strike) + (risk_free_rate - dividend_yield + 0.5*vol**2)*T) / (vol*np.sqrt(T))
            d2 = d1 - vol*np.sqrt(T)
            
            if option_type == 'call':
                price = spot_price*np.exp(-dividend_yield*T)*norm.cdf(d1) - strike*np.exp(-risk_free_rate*T)*norm.cdf(d2)
            else:
                price = strike*np.exp(-risk_free_rate*T)*norm.cdf(-d2) - spot_price*np.exp(-dividend_yield*T)*norm.cdf(-d1)
            
            price = max(price, 0.01)  # Ensure positive price
            
            all_data.append({
                'strike': strike,
                'expiration': exp,
                'option_type': option_type,
                'price': price,
                'spot_price': spot_price,
                'implied_vol': vol
            })
    
    return pd.DataFrame(all_data)

def demonstrate_basic_usage():
    """Demonstrate basic usage of the enhanced SVI model."""
    print("=" * 60)
    print("BASIC USAGE DEMONSTRATION")
    print("=" * 60)
    
    # Initialize the enhanced SVI model
    svi = SVIEnhanced()
    
    # Load market data
    print("Loading market data...")
    svi.load_market_data('crypto_data')  # This will generate synthetic data
    
    # Calculate implied probabilities for all expirations
    print("Calculating implied probabilities...")
    probabilities = svi.calculate_implied_probabilities()
    
    # Display results
    print(f"\nCalculated probabilities for {len(probabilities)} expirations:")
    for exp, data in probabilities.items():
        print(f"  {exp} days: {len(data['probabilities']['strikes'])} strike points")
    
    # Calculate and display risk metrics
    print("\nRisk Metrics:")
    risk_metrics = svi.get_risk_metrics()
    for exp, metrics in risk_metrics.items():
        print(f"  {exp} days:")
        print(f"    ATM Probability: {metrics['atm_probability']:.3f}")
        print(f"    Volatility Skew: {metrics['volatility_skew']:.3f}")
        print(f"    Tail Risk: {metrics['tail_risk']:.3f}")
    
    return svi

def demonstrate_advanced_analysis(svi):
    """Demonstrate advanced analysis capabilities."""
    print("\n" + "=" * 60)
    print("ADVANCED ANALYSIS DEMONSTRATION")
    print("=" * 60)
    
    # Initialize probability calculator
    calc = ProbabilityCalculator(risk_free_rate=0.05, dividend_yield=0.02)
    
    # Get market data
    market_data = svi.market_data
    
    # Analyze each expiration in detail
    for exp in sorted(svi.implied_probabilities.keys()):
        print(f"\nAnalyzing {exp} days to expiration...")
        
        # Filter data for this expiration
        exp_data = market_data[market_data['expiration'] == exp]
        call_data = exp_data[exp_data['option_type'] == 'call']
        put_data = exp_data[exp_data['option_type'] == 'put']
        
        if len(call_data) > 0 and len(put_data) > 0:
            # Calculate detailed probability metrics
            prob_data = calc.calculate_implied_probability_density(
                call_data['strike'].values,
                call_data['price'].values,
                put_data['price'].values,
                call_data['spot_price'].iloc[0],
                exp / 365.0
            )
            
            # Calculate risk metrics
            risk_metrics = calc.calculate_moment_risk_metrics(
                prob_data['strikes'],
                prob_data['density'],
                call_data['spot_price'].iloc[0]
            )
            
            print(f"  Mean: {risk_metrics['mean']:.2f}")
            print(f"  Std Dev: {risk_metrics['std_dev']:.2f}")
            print(f"  Skewness: {risk_metrics['skewness']:.3f}")
            print(f"  Kurtosis: {risk_metrics['kurtosis']:.3f}")
            
            # Calculate butterfly spread probabilities
            butterfly_probs = calc.calculate_butterfly_spread_probability(
                call_data['strike'].values,
                call_data['price'].values,
                put_data['price'].values,
                call_data['spot_price'].iloc[0],
                exp / 365.0
            )
            
            print("  Butterfly Spread Probabilities:")
            for wing, data in butterfly_probs.items():
                print(f"    {wing}: {data['probability']:.3f}")

def demonstrate_visualization(svi):
    """Demonstrate comprehensive visualization capabilities."""
    print("\n" + "=" * 60)
    print("VISUALIZATION DEMONSTRATION")
    print("=" * 60)
    
    # Initialize visualizer
    visualizer = SVIVisualizer(svi)
    
    print("Creating 3D probability surface...")
    visualizer.plot_3d_probability_surface('3d_probability_surface.png')
    
    print("Creating interactive probability heatmap...")
    visualizer.plot_interactive_probability_heatmap('interactive_heatmap.html')
    
    print("Creating probability evolution animation...")
    visualizer.plot_probability_evolution_animation('probability_evolution.png')
    
    print("Creating risk metrics dashboard...")
    visualizer.plot_risk_metrics_dashboard('risk_metrics_dashboard.png')
    
    print("Creating SVI parameter analysis...")
    visualizer.plot_svi_parameter_analysis('svi_parameters.png')
    
    print("Creating comprehensive report...")
    visualizer.create_comprehensive_report('comprehensive_report.html')

def demonstrate_risk_analysis(svi):
    """Demonstrate comprehensive risk analysis."""
    print("\n" + "=" * 60)
    print("RISK ANALYSIS DEMONSTRATION")
    print("=" * 60)
    
    # Get risk metrics
    risk_metrics = svi.get_risk_metrics()
    
    # Create risk summary
    risk_summary = pd.DataFrame([
        {
            'Expiration': exp,
            'ATM_Prob': metrics['atm_probability'],
            'Skew': metrics['volatility_skew'],
            'Tail_Risk': metrics['tail_risk'],
            'Up_Move_Prob': metrics['large_up_move_prob'],
            'Down_Move_Prob': metrics['large_down_move_prob']
        }
        for exp, metrics in risk_metrics.items()
    ])
    
    print("Risk Summary Table:")
    print(risk_summary.to_string(index=False, float_format='%.3f'))
    
    # Calculate portfolio risk metrics
    print("\nPortfolio Risk Analysis:")
    
    # Average risk across all expirations
    avg_atm_prob = np.mean([m['atm_probability'] for m in risk_metrics.values()])
    avg_skew = np.mean([m['volatility_skew'] for m in risk_metrics.values()])
    avg_tail_risk = np.mean([m['tail_risk'] for m in risk_metrics.values()])
    
    print(f"  Average ATM Probability: {avg_atm_prob:.3f}")
    print(f"  Average Volatility Skew: {avg_skew:.3f}")
    print(f"  Average Tail Risk: {avg_tail_risk:.3f}")
    
    # Risk concentration analysis
    short_term_risk = np.mean([m['tail_risk'] for exp, m in risk_metrics.items() if exp <= 30])
    long_term_risk = np.mean([m['tail_risk'] for exp, m in risk_metrics.items() if exp > 30])
    
    print(f"  Short-term Tail Risk (≤30 days): {short_term_risk:.3f}")
    print(f"  Long-term Tail Risk (>30 days): {long_term_risk:.3f}")
    
    # Risk trend analysis
    expirations = sorted(risk_metrics.keys())
    tail_risks = [risk_metrics[exp]['tail_risk'] for exp in expirations]
    
    if len(tail_risks) > 1:
        risk_trend = np.polyfit(expirations, tail_risks, 1)[0]
        print(f"  Risk Trend (slope): {risk_trend:.6f}")
        
        if risk_trend > 0:
            print("  → Risk increases with time to expiration")
        else:
            print("  → Risk decreases with time to expiration")

def demonstrate_export_functionality(svi):
    """Demonstrate data export capabilities."""
    print("\n" + "=" * 60)
    print("EXPORT FUNCTIONALITY DEMONSTRATION")
    print("=" * 60)
    
    # Export results to CSV
    print("Exporting results to CSV...")
    svi.export_results('implied_probabilities_export.csv')
    
    # Create detailed export with additional metrics
    print("Creating detailed export...")
    
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
    detailed_df.to_csv('detailed_probability_analysis.csv', index=False)
    print("Detailed analysis exported to 'detailed_probability_analysis.csv'")
    
    # Create summary statistics
    summary_stats = detailed_df.groupby('expiration').agg({
        'call_probability': ['mean', 'std', 'min', 'max'],
        'put_probability': ['mean', 'std', 'min', 'max'],
        'risk_neutral_density': ['mean', 'std', 'min', 'max']
    }).round(4)
    
    summary_stats.to_csv('probability_summary_statistics.csv')
    print("Summary statistics exported to 'probability_summary_statistics.csv'")

def main():
    """Main demonstration function."""
    print("Enhanced SVI Model with Implied Probability Analysis")
    print("Comprehensive Demonstration")
    print("=" * 60)
    
    try:
        # Basic usage demonstration
        svi = demonstrate_basic_usage()
        
        # Advanced analysis demonstration
        demonstrate_advanced_analysis(svi)
        
        # Visualization demonstration
        demonstrate_visualization(svi)
        
        # Risk analysis demonstration
        demonstrate_risk_analysis(svi)
        
        # Export functionality demonstration
        demonstrate_export_functionality(svi)
        
        print("\n" + "=" * 60)
        print("DEMONSTRATION COMPLETE")
        print("=" * 60)
        print("All analysis files have been generated:")
        print("  - 3d_probability_surface.png")
        print("  - interactive_heatmap.html")
        print("  - probability_evolution.png")
        print("  - risk_metrics_dashboard.png")
        print("  - svi_parameters.png")
        print("  - comprehensive_report.html")
        print("  - implied_probabilities_export.csv")
        print("  - detailed_probability_analysis.csv")
        print("  - probability_summary_statistics.csv")
        
    except Exception as e:
        print(f"Error during demonstration: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Import required modules
    from scipy.stats import norm
    
    # Run the demonstration
    main()
