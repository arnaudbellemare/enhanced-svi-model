"""
Advanced Visualization Tools for SVI Implied Probability Analysis

This module provides comprehensive visualization capabilities for analyzing
implied probabilities across different expiration dates and strike prices.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

class SVIVisualizer:
    """
    Advanced visualization tools for SVI implied probability analysis.
    """
    
    def __init__(self, svi_model):
        """
        Initialize visualizer with SVI model.
        
        Args:
            svi_model: SVIEnhanced model instance
        """
        self.svi_model = svi_model
        self.colors = plt.cm.viridis(np.linspace(0, 1, 10))
        
    def plot_3d_probability_surface(self, save_path: Optional[str] = None):
        """
        Create 3D surface plot of implied probabilities.
        
        Args:
            save_path: Optional path to save the plot
        """
        if not self.svi_model.implied_probabilities:
            self.svi_model.calculate_implied_probabilities()
        
        fig = plt.figure(figsize=(15, 10))
        
        # Create 3D subplot
        ax = fig.add_subplot(111, projection='3d')
        
        # Prepare data for 3D plotting
        expirations = sorted(self.svi_model.implied_probabilities.keys())
        all_strikes = []
        all_probs = []
        all_expirations = []
        
        for exp in expirations:
            prob_info = self.svi_model.implied_probabilities[exp]['probabilities']
            strikes = prob_info['strikes']
            call_probs = prob_info['call_probabilities']
            
            all_strikes.extend(strikes)
            all_probs.extend(call_probs)
            all_expirations.extend([exp] * len(strikes))
        
        # Create surface plot
        exp_array = np.array(all_expirations)
        strike_array = np.array(all_strikes)
        prob_array = np.array(all_probs)
        
        # Create meshgrid for surface
        unique_exp = np.unique(exp_array)
        unique_strikes = np.linspace(min(strike_array), max(strike_array), 50)
        
        X, Y = np.meshgrid(unique_strikes, unique_exp)
        Z = np.zeros_like(X)
        
        # Interpolate probabilities onto grid
        for i, exp in enumerate(unique_exp):
            exp_data = self.svi_model.implied_probabilities[exp]['probabilities']
            interp_probs = np.interp(unique_strikes, exp_data['strikes'], exp_data['call_probabilities'])
            Z[i, :] = interp_probs
        
        # Plot surface
        surf = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8, linewidth=0, antialiased=True)
        
        ax.set_xlabel('Strike Price')
        ax.set_ylabel('Days to Expiration')
        ax.set_zlabel('Call Probability')
        ax.set_title('3D Implied Probability Surface')
        
        # Add colorbar
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_interactive_probability_heatmap(self, save_path: Optional[str] = None):
        """
        Create interactive probability heatmap using Plotly.
        
        Args:
            save_path: Optional path to save the plot
        """
        if not self.svi_model.implied_probabilities:
            self.svi_model.calculate_implied_probabilities()
        
        # Prepare data
        expirations = sorted(self.svi_model.implied_probabilities.keys())
        all_data = []
        
        for exp in expirations:
            prob_info = self.svi_model.implied_probabilities[exp]['probabilities']
            strikes = prob_info['strikes']
            call_probs = prob_info['call_probabilities']
            put_probs = prob_info['put_probabilities']
            
            for i, strike in enumerate(strikes):
                all_data.append({
                    'expiration': exp,
                    'strike': strike,
                    'call_probability': call_probs[i],
                    'put_probability': put_probs[i]
                })
        
        df = pd.DataFrame(all_data)
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Call Probabilities', 'Put Probabilities', 
                          'Probability Difference', 'Risk-Neutral Density'),
            specs=[[{"type": "heatmap"}, {"type": "heatmap"}],
                   [{"type": "heatmap"}, {"type": "heatmap"}]]
        )
        
        # Call probabilities heatmap
        call_pivot = df.pivot_table(values='call_probability', index='expiration', 
                                   columns='strike', aggfunc='mean')
        fig.add_trace(
            go.Heatmap(z=call_pivot.values, x=call_pivot.columns, y=call_pivot.index,
                      colorscale='Viridis', name='Call Prob'),
            row=1, col=1
        )
        
        # Put probabilities heatmap
        put_pivot = df.pivot_table(values='put_probability', index='expiration', 
                                  columns='strike', aggfunc='mean')
        fig.add_trace(
            go.Heatmap(z=put_pivot.values, x=put_pivot.columns, y=put_pivot.index,
                      colorscale='Viridis', name='Put Prob'),
            row=1, col=2
        )
        
        # Probability difference
        df['prob_diff'] = df['call_probability'] - df['put_probability']
        diff_pivot = df.pivot_table(values='prob_diff', index='expiration', 
                                   columns='strike', aggfunc='mean')
        fig.add_trace(
            go.Heatmap(z=diff_pivot.values, x=diff_pivot.columns, y=diff_pivot.index,
                      colorscale='RdBu', name='Prob Diff'),
            row=2, col=1
        )
        
        # Risk-neutral density
        density_data = []
        for exp in expirations:
            prob_info = self.svi_model.implied_probabilities[exp]['probabilities']
            strikes = prob_info['strikes']
            density = prob_info['risk_neutral_density']
            
            for i, strike in enumerate(strikes):
                density_data.append({
                    'expiration': exp,
                    'strike': strike,
                    'density': density[i]
                })
        
        density_df = pd.DataFrame(density_data)
        density_pivot = density_df.pivot_table(values='density', index='expiration', 
                                             columns='strike', aggfunc='mean')
        fig.add_trace(
            go.Heatmap(z=density_pivot.values, x=density_pivot.columns, y=density_pivot.index,
                      colorscale='Plasma', name='Density'),
            row=2, col=2
        )
        
        fig.update_layout(
            title_text="Interactive Implied Probability Analysis",
            height=800,
            showlegend=False
        )
        
        if save_path:
            fig.write_html(save_path)
        
        fig.show()
    
    def plot_probability_evolution_animation(self, save_path: Optional[str] = None):
        """
        Create animated plot showing probability evolution over time.
        
        Args:
            save_path: Optional path to save the animation
        """
        if not self.svi_model.implied_probabilities:
            self.svi_model.calculate_implied_probabilities()
        
        expirations = sorted(self.svi_model.implied_probabilities.keys())
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot 1: Probability curves
        for i, exp in enumerate(expirations):
            prob_info = self.svi_model.implied_probabilities[exp]['probabilities']
            strikes = prob_info['strikes']
            call_probs = prob_info['call_probabilities']
            put_probs = prob_info['put_probabilities']
            
            ax1.plot(strikes, call_probs, 'o-', label=f'Call {exp}d', 
                    color=self.colors[i], linewidth=2, markersize=4)
            ax1.plot(strikes, put_probs, 's--', label=f'Put {exp}d', 
                    color=self.colors[i], linewidth=2, markersize=4, alpha=0.7)
        
        ax1.set_xlabel('Strike Price')
        ax1.set_ylabel('Probability')
        ax1.set_title('Probability Curves by Expiration')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Risk-neutral density
        for i, exp in enumerate(expirations):
            prob_info = self.svi_model.implied_probabilities[exp]['probabilities']
            strikes = prob_info['strikes']
            density = prob_info['risk_neutral_density']
            
            ax2.plot(strikes, density, label=f'{exp} days', 
                    color=self.colors[i], linewidth=2)
        
        ax2.set_xlabel('Strike Price')
        ax2.set_ylabel('Risk-Neutral Density')
        ax2.set_title('Risk-Neutral Density by Expiration')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_risk_metrics_dashboard(self, save_path: Optional[str] = None):
        """
        Create comprehensive risk metrics dashboard.
        
        Args:
            save_path: Optional path to save the plot
        """
        if not self.svi_model.implied_probabilities:
            self.svi_model.calculate_implied_probabilities()
        
        risk_metrics = self.svi_model.get_risk_metrics()
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Risk Metrics Dashboard', fontsize=16)
        
        expirations = sorted(risk_metrics.keys())
        
        # ATM Probability
        atm_probs = [risk_metrics[exp]['atm_probability'] for exp in expirations]
        axes[0, 0].plot(expirations, atm_probs, 'o-', linewidth=2, markersize=8)
        axes[0, 0].set_title('ATM Probability')
        axes[0, 0].set_xlabel('Days to Expiration')
        axes[0, 0].set_ylabel('Probability')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Volatility Skew
        skews = [risk_metrics[exp]['volatility_skew'] for exp in expirations]
        axes[0, 1].plot(expirations, skews, 's-', linewidth=2, markersize=8, color='red')
        axes[0, 1].set_title('Volatility Skew')
        axes[0, 1].set_xlabel('Days to Expiration')
        axes[0, 1].set_ylabel('Skew')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Tail Risk
        tail_risks = [risk_metrics[exp]['tail_risk'] for exp in expirations]
        axes[0, 2].plot(expirations, tail_risks, '^-', linewidth=2, markersize=8, color='green')
        axes[0, 2].set_title('Tail Risk')
        axes[0, 2].set_xlabel('Days to Expiration')
        axes[0, 2].set_ylabel('Tail Risk Ratio')
        axes[0, 2].grid(True, alpha=0.3)
        
        # Large Move Probabilities
        up_moves = [risk_metrics[exp]['large_up_move_prob'] for exp in expirations]
        down_moves = [risk_metrics[exp]['large_down_move_prob'] for exp in expirations]
        
        axes[1, 0].plot(expirations, up_moves, 'o-', label='Up Moves', linewidth=2)
        axes[1, 0].plot(expirations, down_moves, 's-', label='Down Moves', linewidth=2)
        axes[1, 0].set_title('Large Move Probabilities')
        axes[1, 0].set_xlabel('Days to Expiration')
        axes[1, 0].set_ylabel('Probability')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Risk Summary Heatmap
        risk_data = []
        for exp in expirations:
            risk_data.append([
                risk_metrics[exp]['atm_probability'],
                risk_metrics[exp]['volatility_skew'],
                risk_metrics[exp]['tail_risk']
            ])
        
        risk_array = np.array(risk_data)
        im = axes[1, 1].imshow(risk_array.T, aspect='auto', cmap='RdYlBu_r')
        axes[1, 1].set_title('Risk Metrics Heatmap')
        axes[1, 1].set_xlabel('Expiration')
        axes[1, 1].set_ylabel('Risk Metric')
        axes[1, 1].set_xticks(range(len(expirations)))
        axes[1, 1].set_xticklabels(expirations)
        axes[1, 1].set_yticks(range(3))
        axes[1, 1].set_yticklabels(['ATM Prob', 'Skew', 'Tail Risk'])
        plt.colorbar(im, ax=axes[1, 1])
        
        # Probability Distribution Comparison
        for i, exp in enumerate(expirations):
            prob_info = self.svi_model.implied_probabilities[exp]['probabilities']
            strikes = prob_info['strikes']
            call_probs = prob_info['call_probabilities']
            
            # Normalize for comparison
            normalized_probs = np.array(call_probs) / np.max(call_probs)
            axes[1, 2].plot(strikes, normalized_probs, label=f'{exp}d', 
                           color=self.colors[i], linewidth=2, alpha=0.8)
        
        axes[1, 2].set_title('Normalized Probability Distributions')
        axes[1, 2].set_xlabel('Strike Price')
        axes[1, 2].set_ylabel('Normalized Probability')
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_svi_parameter_analysis(self, save_path: Optional[str] = None):
        """
        Plot SVI parameter analysis across expirations.
        
        Args:
            save_path: Optional path to save the plot
        """
        if not self.svi_model.implied_probabilities:
            self.svi_model.calculate_implied_probabilities()
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('SVI Parameter Analysis', fontsize=16)
        
        expirations = sorted(self.svi_model.implied_probabilities.keys())
        
        # Extract SVI parameters
        a_params = []
        b_params = []
        rho_params = []
        m_params = []
        sigma_params = []
        
        for exp in expirations:
            svi_params = self.svi_model.implied_probabilities[exp]['svi_params']
            a_params.append(svi_params['a'])
            b_params.append(svi_params['b'])
            rho_params.append(svi_params['rho'])
            m_params.append(svi_params['m'])
            sigma_params.append(svi_params['sigma'])
        
        # Plot each parameter
        params = [a_params, b_params, rho_params, m_params, sigma_params]
        param_names = ['a', 'b', 'rho', 'm', 'sigma']
        colors = ['blue', 'red', 'green', 'orange', 'purple']
        
        for i, (param, name, color) in enumerate(zip(params, param_names, colors)):
            row = i // 3
            col = i % 3
            
            axes[row, col].plot(expirations, param, 'o-', color=color, 
                              linewidth=2, markersize=8)
            axes[row, col].set_title(f'SVI Parameter: {name}')
            axes[row, col].set_xlabel('Days to Expiration')
            axes[row, col].set_ylabel(f'{name} Value')
            axes[row, col].grid(True, alpha=0.3)
        
        # Remove empty subplot
        axes[1, 2].remove()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def create_comprehensive_report(self, save_path: str = 'svi_analysis_report.html'):
        """
        Create comprehensive HTML report with all visualizations.
        
        Args:
            save_path: Path to save the HTML report
        """
        import plotly.offline as pyo
        
        # Create comprehensive report
        report_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>SVI Implied Probability Analysis Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1, h2 {{ color: #333; }}
                .section {{ margin: 30px 0; }}
                .metric {{ background-color: #f5f5f5; padding: 10px; margin: 10px 0; border-radius: 5px; }}
            </style>
        </head>
        <body>
            <h1>SVI Implied Probability Analysis Report</h1>
            
            <div class="section">
                <h2>Analysis Summary</h2>
                <p>This report provides a comprehensive analysis of implied probabilities 
                across all expiration dates using the Stochastic Volatility Inspired (SVI) model.</p>
            </div>
            
            <div class="section">
                <h2>Key Metrics</h2>
        """
        
        if self.svi_model.implied_probabilities:
            risk_metrics = self.svi_model.get_risk_metrics()
            
            for exp, metrics in risk_metrics.items():
                report_html += f"""
                <div class="metric">
                    <h3>{exp} Days to Expiration</h3>
                    <p><strong>ATM Probability:</strong> {metrics['atm_probability']:.3f}</p>
                    <p><strong>Volatility Skew:</strong> {metrics['volatility_skew']:.3f}</p>
                    <p><strong>Tail Risk:</strong> {metrics['tail_risk']:.3f}</p>
                    <p><strong>Large Up Move Probability:</strong> {metrics['large_up_move_prob']:.3f}</p>
                    <p><strong>Large Down Move Probability:</strong> {metrics['large_down_move_prob']:.3f}</p>
                </div>
                """
        
        report_html += """
            </div>
            
            <div class="section">
                <h2>Visualizations</h2>
                <p>Interactive visualizations and detailed analysis charts are available 
                through the Python visualization tools.</p>
            </div>
            
        </body>
        </html>
        """
        
        with open(save_path, 'w') as f:
            f.write(report_html)
        
        print(f"Comprehensive report saved to {save_path}")
    
    def plot_interactive_probability_heatmap(self, save_path: Optional[str] = None):
        """
        Create interactive probability heatmap using Plotly.
        
        Args:
            save_path: Optional path to save the HTML file
        """
        if not self.svi_model.implied_probabilities:
            self.svi_model.calculate_implied_probabilities()
        
        # Prepare data for heatmap
        expirations = sorted(self.svi_model.implied_probabilities.keys())
        all_strikes = set()
        
        for exp in expirations:
            strikes = self.svi_model.implied_probabilities[exp]['probabilities']['strikes']
            all_strikes.update(strikes)
        
        all_strikes = sorted(list(all_strikes))
        
        # Create probability matrix
        prob_matrix = []
        for exp in expirations:
            prob_info = self.svi_model.implied_probabilities[exp]['probabilities']
            strikes = prob_info['strikes']
            call_probs = prob_info['call_probabilities']
            
            # Interpolate probabilities for all strikes
            interp_probs = np.interp(all_strikes, strikes, call_probs)
            prob_matrix.append(interp_probs)
        
        # Create interactive heatmap
        fig = go.Figure(data=go.Heatmap(
            z=prob_matrix,
            x=all_strikes,
            y=expirations,
            colorscale='Viridis',
            hoverongaps=False,
            hovertemplate='Strike: %{x}<br>Expiration: %{y} days<br>Probability: %{z:.3f}<extra></extra>'
        ))
        
        fig.update_layout(
            title='Interactive Probability Heatmap',
            xaxis_title='Strike Price',
            yaxis_title='Days to Expiration',
            width=800,
            height=600
        )
        
        if save_path:
            fig.write_html(save_path)
        
        return fig
    
    def plot_implied_probability_density_with_price(self, save_path: Optional[str] = None):
        """
        Create implied probability density function with actual price bar.
        
        Args:
            save_path: Optional path to save the plot
        """
        if not self.svi_model.implied_probabilities:
            self.svi_model.calculate_implied_probabilities()
        
        # Get current price from market data
        current_price = self.svi_model.market_data['spot_price'].iloc[0]
        
        # Get the first expiration for main plot
        first_exp = list(self.svi_model.implied_probabilities.keys())[0]
        prob_info = self.svi_model.implied_probabilities[first_exp]['probabilities']
        
        strikes = prob_info['strikes']
        density = prob_info['risk_neutral_density']
        
        # Create the plot
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Calculate cumulative density for tail risk visualization
        cumulative_density = np.cumsum(density) * (strikes[1] - strikes[0])  # Approximate integration
        cumulative_density = cumulative_density / cumulative_density[-1]  # Normalize to 1
        
        # Find 16th and 84th percentiles (16% tails, 68% middle)
        tail_lower_idx = np.argmin(np.abs(cumulative_density - 0.16))
        tail_upper_idx = np.argmin(np.abs(cumulative_density - 0.84))
        
        tail_lower_strike = strikes[tail_lower_idx]
        tail_upper_strike = strikes[tail_upper_idx]
        
        # Plot the probability density with tail risk coloring
        ax.plot(strikes, density, 'k-', linewidth=2, label='Risk-Neutral Density')
        
        # Color the 16% tails in red (high tail risk)
        left_tail_mask = strikes <= tail_lower_strike
        right_tail_mask = strikes >= tail_upper_strike
        
        if np.any(left_tail_mask):
            left_strikes = strikes[left_tail_mask]
            left_density = density[left_tail_mask]
            ax.fill_between(left_strikes, left_density, 
                           alpha=0.7, color='red', label='16% Left Tail (Downside Risk)')
        
        if np.any(right_tail_mask):
            right_strikes = strikes[right_tail_mask]
            right_density = density[right_tail_mask]
            ax.fill_between(right_strikes, right_density, 
                           alpha=0.7, color='red', label='16% Right Tail (Upside Risk)')
        
        # Color the 68% middle in green (normal distribution)
        middle_mask = (strikes > tail_lower_strike) & (strikes < tail_upper_strike)
        if np.any(middle_mask):
            middle_strikes = strikes[middle_mask]
            middle_density = density[middle_mask]
            ax.fill_between(middle_strikes, middle_density, 
                           alpha=0.5, color='green', label='68% Middle (Normal Range)')
        
        # Add vertical line for current price
        ax.axvline(x=current_price, color='blue', linestyle='--', linewidth=3, 
                  label=f'Current Price: ${current_price:,.2f}')
        
        # Add vertical lines for tail boundaries
        ax.axvline(x=tail_lower_strike, color='red', linestyle=':', linewidth=2, alpha=0.7,
                  label=f'16% Tail Boundary: ${tail_lower_strike:,.2f}')
        ax.axvline(x=tail_upper_strike, color='red', linestyle=':', linewidth=2, alpha=0.7,
                  label=f'84% Tail Boundary: ${tail_upper_strike:,.2f}')
        
        # Add probability at current price
        price_density = np.interp(current_price, strikes, density)
        ax.plot(current_price, price_density, 'ro', markersize=10, 
               label=f'Density at Price: {price_density:.4f}')
        
        ax.set_xlabel('Strike Price', fontsize=12)
        ax.set_ylabel('Probability Density', fontsize=12)
        ax.set_title(f'Implied Probability Density Function - {first_exp} Days to Expiration\n'
                    f'Current Price: ${current_price:,.2f}', fontsize=14)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Add text box with key information including tail risk
        textstr = f'Current Price: ${current_price:,.2f}\n'
        textstr += f'Density at Price: {price_density:.4f}\n'
        textstr += f'Expiration: {first_exp} days\n'
        textstr += f'16% Tail Boundaries:\n'
        textstr += f'  Lower: ${tail_lower_strike:,.2f}\n'
        textstr += f'  Upper: ${tail_upper_strike:,.2f}\n'
        textstr += f'Tail Risk: {((tail_lower_strike - current_price) / current_price * 100):.1f}% to {((tail_upper_strike - current_price) / current_price * 100):.1f}%'
        
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=9,
                verticalalignment='top', bbox=props)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig


# Example usage
if __name__ == "__main__":
    from svi_enhanced import SVIEnhanced
    
    # Initialize model and visualizer
    svi = SVIEnhanced()
    svi.load_market_data('synthetic_data')
    svi.calculate_implied_probabilities()
    
    visualizer = SVIVisualizer(svi)
    
    # Create all visualizations
    print("Creating 3D probability surface...")
    visualizer.plot_3d_probability_surface('3d_surface.png')
    
    print("Creating interactive heatmap...")
    visualizer.plot_interactive_probability_heatmap('interactive_heatmap.html')
    
    print("Creating probability evolution plot...")
    visualizer.plot_probability_evolution_animation('evolution.png')
    
    print("Creating risk metrics dashboard...")
    visualizer.plot_risk_metrics_dashboard('risk_dashboard.png')
    
    print("Creating SVI parameter analysis...")
    visualizer.plot_svi_parameter_analysis('svi_parameters.png')
    
    print("Creating comprehensive report...")
    visualizer.create_comprehensive_report('analysis_report.html')
    
    print("All visualizations completed!")
