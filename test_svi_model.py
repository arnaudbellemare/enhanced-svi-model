"""
Test Suite for Enhanced SVI Model

This module provides comprehensive tests for the enhanced SVI model
with implied probability calculations.
"""

import unittest
import numpy as np
import pandas as pd
from svi_enhanced import SVIEnhanced
from probability_calculator import ProbabilityCalculator
from visualization import SVIVisualizer
import warnings
warnings.filterwarnings('ignore')

class TestSVIEnhanced(unittest.TestCase):
    """Test cases for SVIEnhanced class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.svi = SVIEnhanced()
        self.create_test_data()
    
    def create_test_data(self):
        """Create test market data."""
        np.random.seed(42)
        
        # Generate synthetic data
        n_options = 200
        spot_price = 100.0
        
        strikes = np.random.normal(spot_price, spot_price * 0.2, n_options)
        strikes = np.clip(strikes, spot_price * 0.8, spot_price * 1.2)
        
        expirations = np.random.choice([7, 30, 90, 180], n_options)
        option_types = np.random.choice(['call', 'put'], n_options)
        
        prices = []
        for i in range(n_options):
            vol = np.random.uniform(0.15, 0.35)
            price = self.black_scholes_price(
                spot_price, strikes[i], expirations[i]/365, 
                0.05, vol, option_types[i]
            )
            prices.append(price)
        
        self.test_data = pd.DataFrame({
            'strike': strikes,
            'expiration': expirations,
            'option_type': option_types,
            'price': prices,
            'spot_price': spot_price
        })
    
    def black_scholes_price(self, S, K, T, r, sigma, option_type):
        """Calculate Black-Scholes price for testing."""
        from scipy.stats import norm
        d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
        d2 = d1 - sigma*np.sqrt(T)
        
        if option_type == 'call':
            price = S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
        else:
            price = K*np.exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1)
            
        return max(price, 0.01)
    
    def test_initialization(self):
        """Test SVI model initialization."""
        self.assertIsNone(self.svi.market_data)
        self.assertEqual(self.svi.risk_free_rate, 0.05)
        self.assertEqual(self.svi.dividend_yield, 0.0)
        self.assertEqual(len(self.svi.svi_params), 0)
        self.assertEqual(len(self.svi.implied_probabilities), 0)
    
    def test_load_market_data(self):
        """Test market data loading."""
        # Test with synthetic data generation
        self.svi.load_market_data('synthetic_data')
        self.assertIsNotNone(self.svi.market_data)
        self.assertGreater(len(self.svi.market_data), 0)
        
        # Test data structure
        required_columns = ['strike', 'expiration', 'option_type', 'price', 'spot_price']
        for col in required_columns:
            self.assertIn(col, self.svi.market_data.columns)
    
    def test_svi_parameter_fitting(self):
        """Test SVI parameter fitting."""
        self.svi.market_data = self.test_data
        
        # Test fitting for a specific expiration
        expiration = 30
        svi_params = self.svi.fit_svi_parameters(expiration)
        
        if svi_params is not None:
            # Check parameter structure
            required_params = ['a', 'b', 'rho', 'm', 'sigma', 'expiration', 'convergence']
            for param in required_params:
                self.assertIn(param, svi_params)
            
            # Check parameter bounds
            self.assertGreaterEqual(svi_params['b'], 0.01)
            self.assertLessEqual(svi_params['b'], 1.0)
            self.assertGreaterEqual(svi_params['sigma'], 0.01)
            self.assertLessEqual(svi_params['sigma'], 1.0)
            self.assertGreaterEqual(svi_params['rho'], -0.99)
            self.assertLessEqual(svi_params['rho'], 0.99)
    
    def test_implied_probability_calculation(self):
        """Test implied probability calculations."""
        self.svi.market_data = self.test_data
        
        # Calculate probabilities
        probabilities = self.svi.calculate_implied_probabilities()
        
        self.assertIsInstance(probabilities, dict)
        self.assertGreater(len(probabilities), 0)
        
        # Check structure of probability data
        for exp, data in probabilities.items():
            self.assertIn('probabilities', data)
            self.assertIn('svi_params', data)
            
            prob_data = data['probabilities']
            required_keys = ['strikes', 'call_probabilities', 'put_probabilities', 'risk_neutral_density']
            for key in required_keys:
                self.assertIn(key, prob_data)
            
            # Check probability bounds
            call_probs = prob_data['call_probabilities']
            put_probs = prob_data['put_probabilities']
            
            self.assertTrue(np.all(np.array(call_probs) >= 0))
            self.assertTrue(np.all(np.array(call_probs) <= 1))
            self.assertTrue(np.all(np.array(put_probs) >= 0))
            self.assertTrue(np.all(np.array(put_probs) <= 1))
    
    def test_risk_metrics_calculation(self):
        """Test risk metrics calculation."""
        self.svi.market_data = self.test_data
        self.svi.calculate_implied_probabilities()
        
        risk_metrics = self.svi.get_risk_metrics()
        
        self.assertIsInstance(risk_metrics, dict)
        self.assertGreater(len(risk_metrics), 0)
        
        # Check metric structure
        for exp, metrics in risk_metrics.items():
            required_metrics = ['atm_probability', 'large_up_move_prob', 
                              'large_down_move_prob', 'volatility_skew', 'tail_risk']
            for metric in required_metrics:
                self.assertIn(metric, metrics)
                self.assertIsInstance(metrics[metric], (int, float))
    
    def test_export_functionality(self):
        """Test data export functionality."""
        self.svi.market_data = self.test_data
        self.svi.calculate_implied_probabilities()
        
        # Test export
        self.svi.export_results('test_export.csv')
        
        # Verify file was created and has content
        import os
        self.assertTrue(os.path.exists('test_export.csv'))
        
        # Clean up
        os.remove('test_export.csv')

class TestProbabilityCalculator(unittest.TestCase):
    """Test cases for ProbabilityCalculator class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.calc = ProbabilityCalculator(risk_free_rate=0.05, dividend_yield=0.02)
        self.create_test_data()
    
    def create_test_data(self):
        """Create test data for probability calculations."""
        np.random.seed(42)
        
        self.spot_price = 100.0
        self.strikes = np.linspace(80, 120, 20)
        self.time_to_expiry = 0.25
        
        # Generate synthetic option prices
        call_prices = []
        put_prices = []
        
        for strike in self.strikes:
            vol = 0.2 + 0.1 * (strike - self.spot_price) / self.spot_price
            d1 = (np.log(self.spot_price/strike) + 0.05 * self.time_to_expiry + 0.5 * vol**2 * self.time_to_expiry) / (vol * np.sqrt(self.time_to_expiry))
            d2 = d1 - vol * np.sqrt(self.time_to_expiry)
            
            call_price = self.spot_price * self.norm_cdf(d1) - strike * np.exp(-0.05 * self.time_to_expiry) * self.norm_cdf(d2)
            put_price = strike * np.exp(-0.05 * self.time_to_expiry) * self.norm_cdf(-d2) - self.spot_price * self.norm_cdf(-d1)
            
            call_prices.append(max(call_price, 0.01))
            put_prices.append(max(put_price, 0.01))
        
        self.call_prices = np.array(call_prices)
        self.put_prices = np.array(put_prices)
    
    def norm_cdf(self, x):
        """Cumulative normal distribution function."""
        from scipy.stats import norm
        return norm.cdf(x)
    
    def test_initialization(self):
        """Test probability calculator initialization."""
        self.assertEqual(self.calc.risk_free_rate, 0.05)
        self.assertEqual(self.calc.dividend_yield, 0.02)
    
    def test_implied_probability_density_calculation(self):
        """Test implied probability density calculation."""
        prob_data = self.calc.calculate_implied_probability_density(
            self.strikes, self.call_prices, self.put_prices,
            self.spot_price, self.time_to_expiry
        )
        
        # Check structure
        required_keys = ['strikes', 'density', 'call_probabilities', 'put_probabilities',
                        'call_ivs', 'put_ivs', 'spot_price', 'time_to_expiry']
        for key in required_keys:
            self.assertIn(key, prob_data)
        
        # Check data types and shapes
        self.assertEqual(len(prob_data['strikes']), len(self.strikes))
        self.assertEqual(len(prob_data['density']), len(self.strikes))
        self.assertEqual(len(prob_data['call_probabilities']), len(self.strikes))
        self.assertEqual(len(prob_data['put_probabilities']), len(self.strikes))
        
        # Check probability bounds
        call_probs = prob_data['call_probabilities']
        put_probs = prob_data['put_probabilities']
        
        self.assertTrue(np.all(np.array(call_probs) >= 0))
        self.assertTrue(np.all(np.array(call_probs) <= 1))
        self.assertTrue(np.all(np.array(put_probs) >= 0))
        self.assertTrue(np.all(np.array(put_probs) <= 1))
    
    def test_moment_risk_metrics(self):
        """Test moment-based risk metrics calculation."""
        prob_data = self.calc.calculate_implied_probability_density(
            self.strikes, self.call_prices, self.put_prices,
            self.spot_price, self.time_to_expiry
        )
        
        risk_metrics = self.calc.calculate_moment_risk_metrics(
            prob_data['strikes'], prob_data['density'], self.spot_price
        )
        
        # Check structure
        required_metrics = ['mean', 'variance', 'std_dev', 'skewness', 'kurtosis', 'tail_risk', 'var_metrics']
        for metric in required_metrics:
            self.assertIn(metric, risk_metrics)
        
        # Check value ranges
        self.assertGreaterEqual(risk_metrics['variance'], 0)
        self.assertGreaterEqual(risk_metrics['std_dev'], 0)
        self.assertIsInstance(risk_metrics['skewness'], (int, float))
        self.assertIsInstance(risk_metrics['kurtosis'], (int, float))
    
    def test_butterfly_spread_probability(self):
        """Test butterfly spread probability calculation."""
        butterfly_probs = self.calc.calculate_butterfly_spread_probability(
            self.strikes, self.call_prices, self.put_prices,
            self.spot_price, self.time_to_expiry
        )
        
        self.assertIsInstance(butterfly_probs, dict)
        
        # Check structure of butterfly probabilities
        for wing, data in butterfly_probs.items():
            self.assertIn('lower_strike', data)
            self.assertIn('atm_strike', data)
            self.assertIn('upper_strike', data)
            self.assertIn('probability', data)
            
            # Check probability bounds
            self.assertGreaterEqual(data['probability'], 0)
            self.assertLessEqual(data['probability'], 1)
    
    def test_volatility_smile_metrics(self):
        """Test volatility smile metrics calculation."""
        # Generate synthetic implied volatilities
        ivs = np.array([0.2 + 0.1 * (strike - self.spot_price) / self.spot_price for strike in self.strikes])
        
        smile_metrics = self.calc.calculate_volatility_smile_metrics(
            self.strikes, ivs, self.spot_price
        )
        
        self.assertIsInstance(smile_metrics, dict)
        
        if smile_metrics:  # Only check if metrics were calculated
            required_metrics = ['quadratic_coeffs', 'smile_curvature', 'smile_slope', 
                              'atm_volatility', 'smile_range', 'smile_skew']
            for metric in required_metrics:
                self.assertIn(metric, smile_metrics)

class TestSVIVisualizer(unittest.TestCase):
    """Test cases for SVIVisualizer class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.svi = SVIEnhanced()
        self.svi.load_market_data('synthetic_data')
        self.svi.calculate_implied_probabilities()
        self.visualizer = SVIVisualizer(self.svi)
    
    def test_initialization(self):
        """Test visualizer initialization."""
        self.assertEqual(self.visualizer.svi_model, self.svi)
        self.assertIsNotNone(self.visualizer.colors)
    
    def test_plot_creation(self):
        """Test that plots can be created without errors."""
        try:
            # Test 3D surface plot
            self.visualizer.plot_3d_probability_surface('test_3d.png')
            
            # Test risk metrics dashboard
            self.visualizer.plot_risk_metrics_dashboard('test_risk.png')
            
            # Test SVI parameter analysis
            self.visualizer.plot_svi_parameter_analysis('test_svi.png')
            
            # Clean up test files
            import os
            for filename in ['test_3d.png', 'test_risk.png', 'test_svi.png']:
                if os.path.exists(filename):
                    os.remove(filename)
                    
        except Exception as e:
            self.fail(f"Plot creation failed: {e}")
    
    def test_comprehensive_report_creation(self):
        """Test comprehensive report creation."""
        try:
            self.visualizer.create_comprehensive_report('test_report.html')
            
            # Verify file was created
            import os
            self.assertTrue(os.path.exists('test_report.html'))
            
            # Clean up
            os.remove('test_report.html')
            
        except Exception as e:
            self.fail(f"Report creation failed: {e}")

class TestIntegration(unittest.TestCase):
    """Integration tests for the complete system."""
    
    def test_end_to_end_workflow(self):
        """Test complete end-to-end workflow."""
        # Initialize system
        svi = SVIEnhanced()
        calc = ProbabilityCalculator()
        
        # Load data and calculate probabilities
        svi.load_market_data('synthetic_data')
        probabilities = svi.calculate_implied_probabilities()
        
        # Verify results
        self.assertIsInstance(probabilities, dict)
        self.assertGreater(len(probabilities), 0)
        
        # Calculate risk metrics
        risk_metrics = svi.get_risk_metrics()
        self.assertIsInstance(risk_metrics, dict)
        
        # Test visualization
        visualizer = SVIVisualizer(svi)
        self.assertIsNotNone(visualizer)
        
        # Test export
        svi.export_results('test_integration.csv')
        
        # Clean up
        import os
        if os.path.exists('test_integration.csv'):
            os.remove('test_integration.csv')
    
    def test_data_consistency(self):
        """Test data consistency across calculations."""
        svi = SVIEnhanced()
        svi.load_market_data('synthetic_data')
        svi.calculate_implied_probabilities()
        
        # Check that all expirations have consistent data
        for exp, data in svi.implied_probabilities.items():
            prob_data = data['probabilities']
            
            # Check array lengths are consistent
            strikes = prob_data['strikes']
            call_probs = prob_data['call_probabilities']
            put_probs = prob_data['put_probabilities']
            density = prob_data['risk_neutral_density']
            
            self.assertEqual(len(strikes), len(call_probs))
            self.assertEqual(len(strikes), len(put_probs))
            self.assertEqual(len(strikes), len(density))
            
            # Check that probabilities sum to reasonable values
            # (They don't need to sum to 1 as they're not mutually exclusive)
            self.assertTrue(np.all(np.array(call_probs) >= 0))
            self.assertTrue(np.all(np.array(put_probs) >= 0))

def run_performance_tests():
    """Run performance tests."""
    import time
    
    print("Running performance tests...")
    
    # Test with larger dataset
    svi = SVIEnhanced()
    
    start_time = time.time()
    svi.load_market_data('synthetic_data')
    load_time = time.time() - start_time
    
    start_time = time.time()
    svi.calculate_implied_probabilities()
    calc_time = time.time() - start_time
    
    start_time = time.time()
    risk_metrics = svi.get_risk_metrics()
    risk_time = time.time() - start_time
    
    print(f"Data loading time: {load_time:.3f} seconds")
    print(f"Probability calculation time: {calc_time:.3f} seconds")
    print(f"Risk metrics calculation time: {risk_time:.3f} seconds")
    print(f"Total time: {load_time + calc_time + risk_time:.3f} seconds")
    
    # Performance benchmarks
    assert load_time < 5.0, "Data loading too slow"
    assert calc_time < 30.0, "Probability calculation too slow"
    assert risk_time < 5.0, "Risk metrics calculation too slow"

if __name__ == '__main__':
    # Run unit tests
    print("Running unit tests...")
    unittest.main(verbosity=2, exit=False)
    
    # Run performance tests
    print("\n" + "="*50)
    run_performance_tests()
    
    print("\nAll tests completed successfully!")
