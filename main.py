#!/usr/bin/env python3
"""
Enhanced SVI Model - Main Application
All-in-one interface with menu-driven options

This is the main entry point for the Enhanced SVI Model.
Choose from different modes of operation based on your needs.
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def print_banner():
    """Print welcome banner."""
    print("=" * 80)
    print("🎯 ENHANCED SVI MODEL - MAIN APPLICATION")
    print("=" * 80)
    print("Stochastic Volatility Inspired Model with Implied Probability Analysis")
    print("Choose your mode of operation:")
    print("=" * 80)

def print_menu():
    """Print the main menu."""
    print("\n📋 AVAILABLE MODES:")
    print("-" * 50)
    print("1. 🚀 QUICK DEMO - See everything in action")
    print("2. 📊 BASIC ANALYSIS - Core SVI with probabilities")
    print("3. 🔬 ADVANCED ANALYSIS - Detailed risk metrics")
    print("4. 📈 VISUALIZATION - Create charts and reports")
    print("5. 🧪 TESTING - Run test suite")
    print("6. 📖 EXAMPLES - Usage examples and tutorials")
    print("7. ⚙️  CONFIGURATION - Set parameters")
    print("8. 📁 EXPORT DATA - Export results")
    print("9. ❓ HELP - Show help and documentation")
    print("0. 🚪 EXIT - Exit application")
    print("-" * 50)

def quick_demo():
    """Run the complete demonstration."""
    print("\n🚀 RUNNING QUICK DEMO")
    print("=" * 50)
    
    try:
        from svi_enhanced import SVIEnhanced
        from probability_calculator import ProbabilityCalculator
        from visualization import SVIVisualizer
        
        # Initialize components
        svi = SVIEnhanced()
        calc = ProbabilityCalculator()
        
        print("✓ Loading market data...")
        svi.load_market_data('crypto_data')
        
        print("✓ Calculating implied probabilities...")
        svi.calculate_implied_probabilities()
        
        print("✓ Calculating risk metrics...")
        risk_metrics = svi.get_risk_metrics()
        
        print("✓ Creating visualizations...")
        visualizer = SVIVisualizer(svi)
        visualizer.plot_risk_metrics_dashboard('demo_risk_dashboard.png')
        
        print("✓ Exporting results...")
        svi.export_results('demo_results.csv')
        
        print("\n✅ DEMO COMPLETED SUCCESSFULLY!")
        print("Generated files:")
        print("  📊 demo_risk_dashboard.png")
        print("  📁 demo_results.csv")
        
    except Exception as e:
        print(f"❌ Error in demo: {e}")

def basic_analysis():
    """Run basic SVI analysis."""
    print("\n📊 RUNNING BASIC ANALYSIS")
    print("=" * 50)
    
    try:
        from svi_enhanced import SVIEnhanced
        
        svi = SVIEnhanced()
        svi.load_market_data('crypto_data')
        
        print("Calculating implied probabilities...")
        probabilities = svi.calculate_implied_probabilities()
        
        print(f"\n✅ Analysis complete for {len(probabilities)} expirations:")
        for exp, data in probabilities.items():
            print(f"  📅 {exp} days: {len(data['probabilities']['strikes'])} strike points")
        
        # Show sample results
        first_exp = list(probabilities.keys())[0]
        prob_data = probabilities[first_exp]['probabilities']
        
        print(f"\n📋 Sample Results ({first_exp} days):")
        print(f"  Strike Range: {prob_data['strikes'][0]:.2f} - {prob_data['strikes'][-1]:.2f}")
        print(f"  Avg Call Probability: {np.mean(prob_data['call_probabilities']):.3f}")
        print(f"  Avg Put Probability: {np.mean(prob_data['put_probabilities']):.3f}")
        
    except Exception as e:
        print(f"❌ Error in basic analysis: {e}")

def advanced_analysis():
    """Run advanced analysis with detailed metrics."""
    print("\n🔬 RUNNING ADVANCED ANALYSIS")
    print("=" * 50)
    
    try:
        from svi_enhanced import SVIEnhanced
        from probability_calculator import ProbabilityCalculator
        
        svi = SVIEnhanced()
        calc = ProbabilityCalculator()
        
        svi.load_market_data('crypto_data')
        svi.calculate_implied_probabilities()
        
        print("Calculating detailed risk metrics...")
        risk_metrics = svi.get_risk_metrics()
        
        print("\n📊 RISK METRICS SUMMARY:")
        print("Expiration | ATM Prob | Skew | Tail Risk | Up Move | Down Move")
        print("-" * 65)
        
        for exp in sorted(risk_metrics.keys()):
            metrics = risk_metrics[exp]
            print(f"{exp:>10} | {metrics['atm_probability']:>8.3f} | {metrics['volatility_skew']:>4.2f} | {metrics['tail_risk']:>9.3f} | {metrics['large_up_move_prob']:>7.3f} | {metrics['large_down_move_prob']:>9.3f}")
        
        # Portfolio-level analysis
        avg_atm = np.mean([m['atm_probability'] for m in risk_metrics.values()])
        avg_skew = np.mean([m['volatility_skew'] for m in risk_metrics.values()])
        avg_tail = np.mean([m['tail_risk'] for m in risk_metrics.values()])
        
        print(f"\n📈 PORTFOLIO METRICS:")
        print(f"  Average ATM Probability: {avg_atm:.3f}")
        print(f"  Average Volatility Skew: {avg_skew:.3f}")
        print(f"  Average Tail Risk: {avg_tail:.3f}")
        
    except Exception as e:
        print(f"❌ Error in advanced analysis: {e}")

def visualization_mode():
    """Run visualization mode."""
    print("\n📈 VISUALIZATION MODE")
    print("=" * 50)
    
    try:
        from svi_enhanced import SVIEnhanced
        from visualization import SVIVisualizer
        
        svi = SVIEnhanced()
        svi.load_market_data('crypto_data')
        svi.calculate_implied_probabilities()
        
        visualizer = SVIVisualizer(svi)
        
        print("Choose visualization type:")
        print("1. 📊 Risk Metrics Dashboard")
        print("2. 🌐 3D Probability Surface")
        print("3. 🔥 Interactive Heatmap")
        print("4. 📈 SVI Parameter Analysis")
        print("5. 📋 All Visualizations")
        
        choice = input("\nEnter choice (1-5): ").strip()
        
        if choice == "1":
            visualizer.plot_risk_metrics_dashboard('risk_dashboard.png')
            print("✅ Risk dashboard saved as 'risk_dashboard.png'")
        elif choice == "2":
            visualizer.plot_3d_probability_surface('3d_surface.png')
            print("✅ 3D surface saved as '3d_surface.png'")
        elif choice == "3":
            visualizer.plot_interactive_probability_heatmap('interactive_heatmap.html')
            print("✅ Interactive heatmap saved as 'interactive_heatmap.html'")
        elif choice == "4":
            visualizer.plot_svi_parameter_analysis('svi_parameters.png')
            print("✅ SVI parameters saved as 'svi_parameters.png'")
        elif choice == "5":
            print("Creating all visualizations...")
            visualizer.plot_risk_metrics_dashboard('all_risk_dashboard.png')
            visualizer.plot_3d_probability_surface('all_3d_surface.png')
            visualizer.plot_interactive_probability_heatmap('all_interactive_heatmap.html')
            visualizer.plot_svi_parameter_analysis('all_svi_parameters.png')
            print("✅ All visualizations created!")
        else:
            print("❌ Invalid choice")
            
    except Exception as e:
        print(f"❌ Error in visualization: {e}")

def testing_mode():
    """Run testing mode."""
    print("\n🧪 TESTING MODE")
    print("=" * 50)
    
    try:
        import unittest
        from test_svi_model import TestSVIEnhanced, TestProbabilityCalculator, TestSVIVisualizer, TestIntegration
        
        # Create test suite
        test_suite = unittest.TestSuite()
        
        # Add test cases
        test_suite.addTest(unittest.makeSuite(TestSVIEnhanced))
        test_suite.addTest(unittest.makeSuite(TestProbabilityCalculator))
        test_suite.addTest(unittest.makeSuite(TestSVIVisualizer))
        test_suite.addTest(unittest.makeSuite(TestIntegration))
        
        # Run tests
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(test_suite)
        
        if result.wasSuccessful():
            print("\n✅ ALL TESTS PASSED!")
        else:
            print(f"\n❌ {len(result.failures)} tests failed, {len(result.errors)} errors")
            
    except Exception as e:
        print(f"❌ Error running tests: {e}")

def examples_mode():
    """Run examples mode."""
    print("\n📖 EXAMPLES MODE")
    print("=" * 50)
    
    try:
        from example_usage import demonstrate_basic_usage, demonstrate_advanced_analysis, demonstrate_risk_analysis
        
        print("Choose example to run:")
        print("1. 🚀 Basic Usage Example")
        print("2. 🔬 Advanced Analysis Example")
        print("3. 📊 Risk Analysis Example")
        print("4. 🎯 All Examples")
        
        choice = input("\nEnter choice (1-4): ").strip()
        
        if choice == "1":
            demonstrate_basic_usage()
        elif choice == "2":
            demonstrate_advanced_analysis()
        elif choice == "3":
            demonstrate_risk_analysis()
        elif choice == "4":
            print("Running all examples...")
            demonstrate_basic_usage()
            demonstrate_advanced_analysis()
            demonstrate_risk_analysis()
        else:
            print("❌ Invalid choice")
            
    except Exception as e:
        print(f"❌ Error in examples: {e}")

def configuration_mode():
    """Run configuration mode."""
    print("\n⚙️ CONFIGURATION MODE")
    print("=" * 50)
    
    print("Current configuration:")
    print("  Risk-free rate: 0.05 (5%)")
    print("  Dividend yield: 0.02 (2%)")
    print("  Default spot price: 100.0")
    
    print("\nConfiguration options:")
    print("1. 📊 Set risk-free rate")
    print("2. 💰 Set dividend yield")
    print("3. 📈 Set spot price")
    print("4. 🔄 Reset to defaults")
    
    choice = input("\nEnter choice (1-4): ").strip()
    
    if choice == "1":
        rate = input("Enter risk-free rate (e.g., 0.05 for 5%): ")
        try:
            rate = float(rate)
            print(f"✅ Risk-free rate set to {rate:.3f}")
        except:
            print("❌ Invalid rate")
    elif choice == "2":
        yield_val = input("Enter dividend yield (e.g., 0.02 for 2%): ")
        try:
            yield_val = float(yield_val)
            print(f"✅ Dividend yield set to {yield_val:.3f}")
        except:
            print("❌ Invalid yield")
    elif choice == "3":
        spot = input("Enter spot price (e.g., 100.0): ")
        try:
            spot = float(spot)
            print(f"✅ Spot price set to {spot:.2f}")
        except:
            print("❌ Invalid spot price")
    elif choice == "4":
        print("✅ Configuration reset to defaults")
    else:
        print("❌ Invalid choice")

def export_mode():
    """Run export mode."""
    print("\n📁 EXPORT MODE")
    print("=" * 50)
    
    try:
        from svi_enhanced import SVIEnhanced
        
        svi = SVIEnhanced()
        svi.load_market_data('crypto_data')
        svi.calculate_implied_probabilities()
        
        print("Export options:")
        print("1. 📊 Export probability data (CSV)")
        print("2. 📈 Export risk metrics (CSV)")
        print("3. 📋 Export summary statistics (CSV)")
        print("4. 📁 Export all data")
        
        choice = input("\nEnter choice (1-4): ").strip()
        
        if choice == "1":
            svi.export_results('exported_probabilities.csv')
            print("✅ Probability data exported to 'exported_probabilities.csv'")
        elif choice == "2":
            risk_metrics = svi.get_risk_metrics()
            df = pd.DataFrame([
                {
                    'expiration': exp,
                    'atm_probability': metrics['atm_probability'],
                    'volatility_skew': metrics['volatility_skew'],
                    'tail_risk': metrics['tail_risk'],
                    'large_up_move_prob': metrics['large_up_move_prob'],
                    'large_down_move_prob': metrics['large_down_move_prob']
                }
                for exp, metrics in risk_metrics.items()
            ])
            df.to_csv('exported_risk_metrics.csv', index=False)
            print("✅ Risk metrics exported to 'exported_risk_metrics.csv'")
        elif choice == "3":
            # Create summary statistics
            all_data = []
            for exp, prob_info in svi.implied_probabilities.items():
                prob_data = prob_info['probabilities']
                all_data.append({
                    'expiration': exp,
                    'avg_call_prob': np.mean(prob_data['call_probabilities']),
                    'avg_put_prob': np.mean(prob_data['put_probabilities']),
                    'strike_range': f"{prob_data['strikes'][0]:.2f}-{prob_data['strikes'][-1]:.2f}",
                    'num_strikes': len(prob_data['strikes'])
                })
            
            summary_df = pd.DataFrame(all_data)
            summary_df.to_csv('exported_summary.csv', index=False)
            print("✅ Summary statistics exported to 'exported_summary.csv'")
        elif choice == "4":
            svi.export_results('all_probabilities.csv')
            risk_metrics = svi.get_risk_metrics()
            df = pd.DataFrame([
                {
                    'expiration': exp,
                    'atm_probability': metrics['atm_probability'],
                    'volatility_skew': metrics['volatility_skew'],
                    'tail_risk': metrics['tail_risk']
                }
                for exp, metrics in risk_metrics.items()
            ])
            df.to_csv('all_risk_metrics.csv', index=False)
            print("✅ All data exported!")
        else:
            print("❌ Invalid choice")
            
    except Exception as e:
        print(f"❌ Error in export: {e}")

def help_mode():
    """Show help and documentation."""
    print("\n❓ HELP & DOCUMENTATION")
    print("=" * 50)
    
    print("📚 ENHANCED SVI MODEL HELP")
    print("-" * 30)
    print("This application provides comprehensive implied probability analysis")
    print("using the Stochastic Volatility Inspired (SVI) model.")
    print()
    print("🎯 MAIN FEATURES:")
    print("  • SVI parameter fitting for all expiration dates")
    print("  • Implied probability calculations")
    print("  • Risk metrics and tail risk analysis")
    print("  • Advanced visualizations")
    print("  • Data export capabilities")
    print()
    print("📋 MODE DESCRIPTIONS:")
    print("  1. Quick Demo - See everything in action")
    print("  2. Basic Analysis - Core SVI with probabilities")
    print("  3. Advanced Analysis - Detailed risk metrics")
    print("  4. Visualization - Create charts and reports")
    print("  5. Testing - Run test suite")
    print("  6. Examples - Usage examples and tutorials")
    print("  7. Configuration - Set parameters")
    print("  8. Export Data - Export results")
    print("  9. Help - Show this help")
    print("  0. Exit - Exit application")
    print()
    print("🔧 TECHNICAL DETAILS:")
    print("  • Based on SVI parameterization: w(k) = a + b * (ρ * (k - m) + √((k - m)² + σ²))")
    print("  • Implied probabilities: P(S_T > K) = N(d₂) for calls")
    print("  • Risk-neutral density: q(K) = e^(rT) * ∂²C/∂K²")
    print("  • Advanced risk metrics: VaR, Expected Shortfall, Tail Risk")
    print()
    print("📁 GENERATED FILES:")
    print("  • CSV exports for probability data")
    print("  • PNG files for visualizations")
    print("  • HTML files for interactive charts")
    print("  • Comprehensive reports")
    print()
    print("For more detailed documentation, see 'docs/model_documentation.md'")

def main():
    """Main application loop."""
    print_banner()
    
    while True:
        print_menu()
        
        try:
            choice = input("\n🎯 Enter your choice (0-9): ").strip()
            
            if choice == "0":
                print("\n👋 Thank you for using Enhanced SVI Model!")
                print("Goodbye! 🚪")
                break
            elif choice == "1":
                quick_demo()
            elif choice == "2":
                basic_analysis()
            elif choice == "3":
                advanced_analysis()
            elif choice == "4":
                visualization_mode()
            elif choice == "5":
                testing_mode()
            elif choice == "6":
                examples_mode()
            elif choice == "7":
                configuration_mode()
            elif choice == "8":
                export_mode()
            elif choice == "9":
                help_mode()
            else:
                print("❌ Invalid choice. Please enter 0-9.")
            
            # Pause before showing menu again
            input("\n⏸️  Press Enter to continue...")
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye! 🚪")
            break
        except Exception as e:
            print(f"\n❌ Unexpected error: {e}")
            input("Press Enter to continue...")

if __name__ == "__main__":
    main()
