"""
Enhanced SVI Model with Implied Probability Analysis

This module extends the Stochastic Volatility Inspired (SVI) model to include
comprehensive implied probability calculations for all expiration dates.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class SVIEnhanced:
    """
    Enhanced SVI model with implied probability calculations for all expiration dates.
    """
    
    def __init__(self):
        self.market_data = None
        self.svi_params = {}
        self.implied_probabilities = {}
        self.volatility_surface = {}
        self.risk_free_rate = 0.05  # Default risk-free rate
        self.dividend_yield = 0.0   # Default dividend yield
        
    def load_market_data(self, data_source: str):
        """
        Load market data from various sources.
        
        Args:
            data_source: Path to CSV file or 'crypto_data' for real crypto options
        """
        try:
            if data_source.endswith('.csv'):
                self.market_data = pd.read_csv(data_source)
            elif data_source == 'crypto_data':
                # Load real crypto options data from Deribit and Thalex
                self._load_crypto_data()
            else:
                raise Exception(f"Unknown data source: {data_source}. Use 'crypto_data' for real crypto options or provide a CSV file path.")
                
            print(f"Market data loaded successfully. Shape: {self.market_data.shape}")
            
        except Exception as e:
            print(f"Error loading market data: {e}")
            raise Exception("Failed to load market data. No synthetic data will be generated.")
    
    
    def _load_crypto_data(self):
        """Load REAL crypto options data for BTC and ETH from Deribit and Thalex ONLY."""
        try:
            from real_crypto_options_fetcher import RealCryptoOptionsFetcher
            
            # Initialize real crypto options fetcher
            fetcher = RealCryptoOptionsFetcher()
            
            # Fetch real crypto options data from exchanges
            crypto_data = fetcher.fetch_btc_eth_options()
            
            # Store the data
            self.market_data = crypto_data
            
            print(f"✅ REAL crypto options data loaded:")
            print(f"   Total options: {len(crypto_data)}")
            print(f"   BTC options: {len(crypto_data[crypto_data['symbol'] == 'BTC'])}")
            print(f"   ETH options: {len(crypto_data[crypto_data['symbol'] == 'ETH'])}")
            
            if len(crypto_data) > 0:
                btc_data = crypto_data[crypto_data['symbol'] == 'BTC']
                eth_data = crypto_data[crypto_data['symbol'] == 'ETH']
                
                if len(btc_data) > 0:
                    print(f"   Current BTC price: ${btc_data['spot_price'].iloc[0]:,.2f}")
                    print(f"   BTC strike range: ${btc_data['strike'].min():,.0f} - ${btc_data['strike'].max():,.0f}")
                    print(f"   BTC exchanges: {btc_data['exchange'].value_counts().to_dict()}")
                
                if len(eth_data) > 0:
                    print(f"   Current ETH price: ${eth_data['spot_price'].iloc[0]:,.2f}")
                    print(f"   ETH strike range: ${eth_data['strike'].min():,.0f} - ${eth_data['strike'].max():,.0f}")
                    print(f"   ETH exchanges: {eth_data['exchange'].value_counts().to_dict()}")
            
        except Exception as e:
            print(f"❌ Error loading real crypto data: {e}")
            raise Exception("Failed to load real crypto options data. No synthetic data will be generated.")
    
    def _black_scholes_price(self, S, K, T, r, sigma, option_type):
        """Calculate Black-Scholes option price."""
        d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
        d2 = d1 - sigma*np.sqrt(T)
        
        if option_type == 'call':
            price = S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
        else:
            price = K*np.exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1)
            
        return max(price, 0.01)  # Ensure positive price
    
    def check_svi_butterfly_arbitrage_gk(self, k_val: float, svi_params: Dict, T_expiry: float) -> float:
        """
        Calculates g(k) from Gatheral's paper (Lemma 2.4) for a raw SVI slice
        using manual finite differences for derivatives.
        If g(k) < 0, there is butterfly arbitrage.
        
        Args:
            k_val: log-moneyness log(K/F)
            svi_params: dictionary {a, b, rho, m, sigma}
            T_expiry: time to expiry in years
            
        Returns:
            g(k) value - negative indicates butterfly arbitrage
        """
        def svi_total_variance(k):
            """SVI total variance function."""
            a, b, rho, m, sigma = svi_params['a'], svi_params['b'], svi_params['rho'], svi_params['m'], svi_params['sigma']
            return a + b * (rho * (k - m) + np.sqrt((k - m)**2 + sigma**2))
        
        try:
            w_k = svi_total_variance(k_val)
            if w_k <= 1e-9:  # Total variance must be positive
                return -1e9 
            
            eps = 1e-5  # Small step for numerical differentiation
            
            w_k_plus_eps = svi_total_variance(k_val + eps)
            w_k_minus_eps = svi_total_variance(k_val - eps)
            
            # First derivative: w'(k)
            dw_dk = (w_k_plus_eps - w_k_minus_eps) / (2 * eps)
            
            # Second derivative: w''(k)
            d2w_dk2 = (w_k_plus_eps - 2 * w_k + w_k_minus_eps) / (eps**2)
            
        except Exception as e:
            print(f"Error in numerical differentiation for g(k) (k={k_val}): {e}")
            return -1e9
        
        # Gatheral's g(k) formula from Lemma 2.4:
        # g(k) = (1 - k * w'(k) / (2*w(k)) )^2 - (w'(k)/2)^2 * (1/w(k) + 1/4) + w''(k)/2
        
        term1_denom = 2 * w_k
        if abs(term1_denom) < 1e-9:
            return -1e9 
            
        term1_num_factor = k_val * dw_dk / term1_denom
        term1 = (1 - term1_num_factor)**2
        
        term2_factor = (dw_dk / 2)**2
        term2_bracket_denom = w_k
        if abs(term2_bracket_denom) < 1e-9:
            return -1e9
        term2_bracket = (1 / term2_bracket_denom) + 0.25
        term2 = term2_factor * term2_bracket
        
        term3 = d2w_dk2 / 2.0
        
        g_k = term1 - term2 + term3
        return g_k

    def fit_svi_parameters(self, expiration: int) -> Dict:
        """
        Fit SVI parameters for a specific expiration with butterfly arbitrage checking.
        
        Args:
            expiration: Days to expiration
            
        Returns:
            Dictionary of SVI parameters with arbitrage checking
        """
        # Filter data for specific expiration
        exp_data = self.market_data[self.market_data['expiration'] == expiration].copy()
        
        if len(exp_data) == 0:
            return None
        
        # Calculate log-moneyness and implied volatility
        exp_data['log_moneyness'] = np.log(exp_data['strike'] / exp_data['spot_price'])
        exp_data['implied_vol'] = self._calculate_implied_volatility(exp_data)
        
        # Remove invalid data points
        exp_data = exp_data.dropna()
        
        if len(exp_data) < 5:
            return None
        
        # Fit SVI parameters
        def svi_function(k, a, b, rho, m, sigma):
            """SVI parameterization function."""
            return a + b * (rho * (k - m) + np.sqrt((k - m)**2 + sigma**2))
        
        def objective(params):
            a, b, rho, m, sigma = params
            if sigma <= 0 or abs(rho) >= 1:
                return 1e10
            
            # Roger Lee wing condition: b(1+|ρ|)T < 4
            if b * (1 + abs(rho)) * (expiration/365) > 4.0:
                return 1e10
            
            k = exp_data['log_moneyness'].values
            vol_squared = exp_data['implied_vol'].values**2
            
            predicted = svi_function(k, a, b, rho, m, sigma)
            error = np.sum((predicted - vol_squared)**2)
            
            # Penalty for butterfly arbitrage using g(k) at ATM (k=0)
            svi_params = {'a': a, 'b': b, 'rho': rho, 'm': m, 'sigma': sigma}
            g_atm = self.check_svi_butterfly_arbitrage_gk(0.0, svi_params, expiration/365)
            if g_atm < -1e-5:  # Negative g(k) implies arbitrage
                error += 1e6 * abs(g_atm)
            
            return error
        
        # Initial parameter guess
        initial_params = [0.01, 0.1, 0.0, 0.0, 0.1]
        
        # Bounds for parameters
        bounds = [
            (-0.1, 0.1),    # a
            (0.01, 1.0),    # b
            (-0.99, 0.99),  # rho
            (-1.0, 1.0),    # m
            (0.01, 1.0)     # sigma
        ]
        
        try:
            result = minimize(objective, initial_params, bounds=bounds, method='L-BFGS-B')
            
            if result.success:
                a, b, rho, m, sigma = result.x
                svi_params = {
                    'a': a, 'b': b, 'rho': rho, 'm': m, 'sigma': sigma,
                    'expiration': expiration,
                    'convergence': True
                }
                
                # Check for butterfly arbitrage
                g_atm = self.check_svi_butterfly_arbitrage_gk(0.0, svi_params, expiration/365)
                svi_params['butterfly_arbitrage'] = g_atm < -1e-5
                svi_params['g_atm'] = g_atm
                
                return svi_params
            else:
                return None
                
        except Exception as e:
            print(f"Error fitting SVI parameters for expiration {expiration}: {e}")
            return None
    
    def _calculate_implied_volatility(self, data: pd.DataFrame) -> pd.Series:
        """Calculate implied volatility for options data."""
        ivs = []
        
        for _, row in data.iterrows():
            try:
                iv = self._implied_volatility_newton(
                    row['price'], row['spot_price'], row['strike'],
                    row['expiration']/365, self.risk_free_rate, row['option_type']
                )
                ivs.append(iv)
            except:
                ivs.append(np.nan)
        
        return pd.Series(ivs, index=data.index)
    
    def _implied_volatility_newton(self, price, S, K, T, r, option_type, max_iter=100):
        """Calculate implied volatility using Newton-Raphson method."""
        if T <= 0:
            return np.nan
        
        # Initial guess
        sigma = 0.2
        
        for _ in range(max_iter):
            try:
                d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
                d2 = d1 - sigma*np.sqrt(T)
                
                if option_type == 'call':
                    bs_price = S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
                    vega = S*np.sqrt(T)*norm.pdf(d1)
                else:
                    bs_price = K*np.exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1)
                    vega = S*np.sqrt(T)*norm.pdf(d1)
                
                if vega == 0:
                    break
                
                diff = bs_price - price
                if abs(diff) < 1e-6:
                    break
                
                sigma = sigma - diff / vega
                sigma = max(0.001, min(5.0, sigma))  # Keep sigma in reasonable range
                
            except:
                return np.nan
        
        return sigma
    
    def calculate_implied_probabilities(self) -> Dict:
        """
        Calculate implied probabilities for all expiration dates.
        
        Returns:
            Dictionary with implied probabilities for each expiration
        """
        if self.market_data is None:
            raise ValueError("Market data not loaded. Please load data first.")
        
        unique_expirations = sorted(self.market_data['expiration'].unique())
        self.implied_probabilities = {}
        
        for expiration in unique_expirations:
            print(f"Calculating implied probabilities for {expiration} days to expiration...")
            
            # Fit SVI parameters
            svi_params = self.fit_svi_parameters(expiration)
            if svi_params is None:
                continue
            
            # Calculate implied probabilities
            probabilities = self._calculate_expiration_probabilities(expiration, svi_params)
            self.implied_probabilities[expiration] = {
                'probabilities': probabilities,
                'svi_params': svi_params
            }
        
        return self.implied_probabilities
    
    def _calculate_expiration_probabilities(self, expiration: int, svi_params: Dict) -> Dict:
        """Calculate implied probabilities for a specific expiration."""
        T = expiration / 365.0
        
        # Generate strike range
        spot_price = self.market_data['spot_price'].iloc[0]
        strikes = np.linspace(spot_price * 0.7, spot_price * 1.3, 50)
        
        probabilities = {
            'strikes': strikes,
            'call_probabilities': [],
            'put_probabilities': [],
            'risk_neutral_density': []
        }
        
        for strike in strikes:
            # Calculate implied volatility using SVI
            k = np.log(strike / spot_price)
            vol_squared = self._svi_volatility_squared(k, svi_params)
            vol = np.sqrt(max(vol_squared, 0.001))
            
            # Calculate d2 for probability calculation
            d2 = (np.log(spot_price/strike) + (self.risk_free_rate - self.dividend_yield - 0.5*vol**2)*T) / (vol*np.sqrt(T))
            
            # Call probability (probability of expiring ITM)
            call_prob = norm.cdf(d2)
            
            # Put probability
            put_prob = norm.cdf(-d2)
            
            # Risk-neutral density
            d1 = d2 + vol*np.sqrt(T)
            density = norm.pdf(d2) / (strike * vol * np.sqrt(T))
            
            probabilities['call_probabilities'].append(call_prob)
            probabilities['put_probabilities'].append(put_prob)
            probabilities['risk_neutral_density'].append(density)
        
        return probabilities
    
    def _svi_volatility_squared(self, k: float, svi_params: Dict) -> float:
        """Calculate volatility squared using SVI parameters."""
        a, b, rho, m, sigma = svi_params['a'], svi_params['b'], svi_params['rho'], svi_params['m'], svi_params['sigma']
        return a + b * (rho * (k - m) + np.sqrt((k - m)**2 + sigma**2))
    
    def plot_probability_surface(self, save_path: Optional[str] = None):
        """Plot implied probability surface for all expirations."""
        if not self.implied_probabilities:
            self.calculate_implied_probabilities()
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Implied Probability Analysis', fontsize=16)
        
        # Plot 1: Call probabilities heatmap
        self._plot_probability_heatmap(axes[0, 0], 'call_probabilities', 'Call Probabilities')
        
        # Plot 2: Put probabilities heatmap
        self._plot_probability_heatmap(axes[0, 1], 'put_probabilities', 'Put Probabilities')
        
        # Plot 3: Risk-neutral density
        self._plot_risk_neutral_density(axes[1, 0])
        
        # Plot 4: Probability evolution over time
        self._plot_probability_evolution(axes[1, 1])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def _plot_probability_heatmap(self, ax, prob_type: str, title: str):
        """Plot probability heatmap."""
        expirations = list(self.implied_probabilities.keys())
        strikes_data = []
        prob_data = []
        
        for exp in expirations:
            prob_info = self.implied_probabilities[exp]['probabilities']
            strikes_data.append(prob_info['strikes'])
            prob_data.append(prob_info[prob_type])
        
        # Create heatmap data
        max_strikes = max(len(strikes) for strikes in strikes_data)
        heatmap_data = np.zeros((len(expirations), max_strikes))
        
        for i, (strikes, probs) in enumerate(zip(strikes_data, prob_data)):
            # Interpolate to common strike grid
            common_strikes = np.linspace(min(strikes), max(strikes), max_strikes)
            interp_probs = np.interp(common_strikes, strikes, probs)
            heatmap_data[i, :] = interp_probs
        
        im = ax.imshow(heatmap_data, aspect='auto', cmap='viridis')
        ax.set_title(title)
        ax.set_xlabel('Strike Price Index')
        ax.set_ylabel('Days to Expiration')
        ax.set_yticks(range(len(expirations)))
        ax.set_yticklabels(expirations)
        
        plt.colorbar(im, ax=ax)
    
    def _plot_risk_neutral_density(self, ax):
        """Plot risk-neutral density for different expirations."""
        colors = plt.cm.viridis(np.linspace(0, 1, len(self.implied_probabilities)))
        
        for i, (expiration, prob_info) in enumerate(self.implied_probabilities.items()):
            strikes = prob_info['probabilities']['strikes']
            density = prob_info['probabilities']['risk_neutral_density']
            
            ax.plot(strikes, density, label=f'{expiration} days', color=colors[i], linewidth=2)
        
        ax.set_title('Risk-Neutral Density')
        ax.set_xlabel('Strike Price')
        ax.set_ylabel('Density')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_probability_evolution(self, ax):
        """Plot how probabilities evolve over time."""
        # Calculate average probabilities for each expiration
        avg_call_probs = []
        avg_put_probs = []
        expirations = []
        
        for exp, prob_info in self.implied_probabilities.items():
            call_probs = prob_info['probabilities']['call_probabilities']
            put_probs = prob_info['probabilities']['put_probabilities']
            
            avg_call_probs.append(np.mean(call_probs))
            avg_put_probs.append(np.mean(put_probs))
            expirations.append(exp)
        
        ax.plot(expirations, avg_call_probs, 'o-', label='Average Call Probability', linewidth=2)
        ax.plot(expirations, avg_put_probs, 's-', label='Average Put Probability', linewidth=2)
        
        ax.set_title('Probability Evolution Over Time')
        ax.set_xlabel('Days to Expiration')
        ax.set_ylabel('Average Probability')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def get_risk_metrics(self) -> Dict:
        """Calculate risk metrics from implied probabilities."""
        if not self.implied_probabilities:
            self.calculate_implied_probabilities()
        
        risk_metrics = {}
        
        for expiration, prob_info in self.implied_probabilities.items():
            strikes = prob_info['probabilities']['strikes']
            call_probs = prob_info['probabilities']['call_probabilities']
            put_probs = prob_info['probabilities']['put_probabilities']
            
            # Calculate various risk metrics
            atm_prob = call_probs[len(call_probs)//2]  # At-the-money probability
            
            # Probability of large moves
            large_up_move = np.mean([p for i, p in enumerate(call_probs) if strikes[i] > strikes[len(strikes)//2] * 1.1])
            large_down_move = np.mean([p for i, p in enumerate(put_probs) if strikes[i] < strikes[len(strikes)//2] * 0.9])
            
            risk_metrics[expiration] = {
                'atm_probability': atm_prob,
                'large_up_move_prob': large_up_move,
                'large_down_move_prob': large_down_move,
                'volatility_skew': self._calculate_volatility_skew(expiration),
                'tail_risk': self._calculate_tail_risk(expiration)
            }
        
        return risk_metrics
    
    def _calculate_volatility_skew(self, expiration: int) -> float:
        """Calculate volatility skew for given expiration."""
        svi_params = self.implied_probabilities[expiration]['svi_params']
        return svi_params['rho']  # SVI rho parameter represents skew
    
    def _calculate_tail_risk(self, expiration: int) -> float:
        """Calculate tail risk metric."""
        prob_info = self.implied_probabilities[expiration]['probabilities']
        strikes = prob_info['strikes']
        density = prob_info['risk_neutral_density']
        
        # Calculate tail risk as the ratio of tail density to center density
        center_idx = len(strikes) // 2
        center_density = density[center_idx]
        
        # Average tail density
        tail_density = np.mean([density[i] for i in range(len(density)) if abs(i - center_idx) > len(density) // 4])
        
        return tail_density / center_density if center_density > 0 else 0
    
    def export_results(self, filename: str):
        """Export results to CSV file."""
        if not self.implied_probabilities:
            self.calculate_implied_probabilities()
        
        # Prepare data for export
        export_data = []
        
        for expiration, prob_info in self.implied_probabilities.items():
            strikes = prob_info['probabilities']['strikes']
            call_probs = prob_info['probabilities']['call_probabilities']
            put_probs = prob_info['probabilities']['put_probabilities']
            density = prob_info['probabilities']['risk_neutral_density']
            
            for i, strike in enumerate(strikes):
                export_data.append({
                    'expiration': expiration,
                    'strike': strike,
                    'call_probability': call_probs[i],
                    'put_probability': put_probs[i],
                    'risk_neutral_density': density[i]
                })
        
        df = pd.DataFrame(export_data)
        df.to_csv(filename, index=False)
        print(f"Results exported to {filename}")


# Example usage and testing
if __name__ == "__main__":
    # Initialize the enhanced SVI model
    svi = SVIEnhanced()
    
    # Load or generate market data
    svi.load_market_data('synthetic_data')
    
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
    
    # Plot results
    print("\nGenerating plots...")
    svi.plot_probability_surface('probability_analysis.png')
    
    # Export results
    svi.export_results('implied_probabilities.csv')
    
    print("\nAnalysis complete!")
