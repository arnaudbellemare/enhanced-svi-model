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
        Create beautiful implied probability density function with tail risk visualization using Plotly.
        
        Args:
            save_path: Optional path to save the HTML file
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
        
        # Convert to numpy arrays to ensure proper indexing
        strikes = np.array(strikes)
        density = np.array(density)
        
        # Calculate cumulative density for tail risk visualization
        cumulative_density = np.cumsum(density) * (strikes[1] - strikes[0])  # Approximate integration
        cumulative_density = cumulative_density / cumulative_density[-1]  # Normalize to 1
        
        # Find 16th and 84th percentiles (16% tails, 68% middle)
        tail_lower_idx = np.argmin(np.abs(cumulative_density - 0.16))
        tail_upper_idx = np.argmin(np.abs(cumulative_density - 0.84))
        
        tail_lower_strike = strikes[tail_lower_idx]
        tail_upper_strike = strikes[tail_upper_idx]
        
        # Create beautiful Plotly figure
        fig = go.Figure()
        
        # Add main density curve
        fig.add_trace(go.Scatter(
            x=strikes,
            y=density,
            mode='lines',
            name='Risk-Neutral Density',
            line=dict(color='#1f77b4', width=3),
            hovertemplate='Strike: $%{x:,.0f}<br>Density: %{y:.6f}<extra></extra>'
        ))
        
        # Add left tail (16% downside risk) - RED
        left_tail_mask = strikes <= tail_lower_strike
        if np.any(left_tail_mask):
            fig.add_trace(go.Scatter(
                x=strikes[left_tail_mask],
                y=density[left_tail_mask],
                mode='lines',
                fill='tozeroy',
                fillcolor='rgba(220, 38, 38, 0.7)',  # Clear red
                line=dict(color='rgba(220, 38, 38, 1)', width=2),
                name='16% Left Tail (Downside Risk)',
                hovertemplate='Strike: $%{x:,.0f}<br>Density: %{y:.6f}<br>Risk Zone: Downside<extra></extra>'
            ))
        
        # Add right tail (16% upside risk) - ORANGE
        right_tail_mask = strikes >= tail_upper_strike
        if np.any(right_tail_mask):
            fig.add_trace(go.Scatter(
                x=strikes[right_tail_mask],
                y=density[right_tail_mask],
                mode='lines',
                fill='tozeroy',
                fillcolor='rgba(251, 146, 60, 0.7)',  # Clear orange
                line=dict(color='rgba(251, 146, 60, 1)', width=2),
                name='16% Right Tail (Upside Risk)',
                hovertemplate='Strike: $%{x:,.0f}<br>Density: %{y:.6f}<br>Risk Zone: Upside<extra></extra>'
            ))
        
        # Add middle section (68% normal range) - GREEN
        middle_mask = (strikes > tail_lower_strike) & (strikes < tail_upper_strike)
        if np.any(middle_mask):
            fig.add_trace(go.Scatter(
                x=strikes[middle_mask],
                y=density[middle_mask],
                mode='lines',
                fill='tozeroy',
                fillcolor='rgba(34, 197, 94, 0.5)',  # Clear green
                line=dict(color='rgba(34, 197, 94, 1)', width=2),
                name='68% Middle (Normal Range)',
                hovertemplate='Strike: $%{x:,.0f}<br>Density: %{y:.6f}<br>Risk Zone: Normal<extra></extra>'
            ))
        
        # Add current price line
        price_density = np.interp(current_price, strikes, density)
        fig.add_vline(
            x=current_price,
            line=dict(color='#2E86AB', width=3, dash='dash'),
            annotation_text=f'Current Price: ${current_price:,.2f}',
            annotation_position="top right"
        )
        
        # Add tail boundary lines
        fig.add_vline(
            x=tail_lower_strike,
            line=dict(color='red', width=2, dash='dot'),
            annotation_text=f'16% Tail: ${tail_lower_strike:,.0f}',
            annotation_position="bottom"
        )
        
        fig.add_vline(
            x=tail_upper_strike,
            line=dict(color='red', width=2, dash='dot'),
            annotation_text=f'84% Tail: ${tail_upper_strike:,.0f}',
            annotation_position="bottom"
        )
        
        # Add current price point
        fig.add_trace(go.Scatter(
            x=[current_price],
            y=[price_density],
            mode='markers',
            marker=dict(color='#2E86AB', size=12, symbol='diamond'),
            name=f'Current Price: ${current_price:,.2f}',
            hovertemplate=f'Current Price: ${current_price:,.2f}<br>Density: {price_density:.6f}<extra></extra>'
        ))
        
        # Update layout for beautiful appearance
        fig.update_layout(
            title=dict(
                text=f'<b>Implied Probability Density Function</b><br><sub>{first_exp} Days to Expiration | Current Price: ${current_price:,.2f}</sub>',
                x=0.5,
                font=dict(size=20, color='#2c3e50')
            ),
            xaxis=dict(
                title=dict(text='Strike Price ($)', font=dict(size=14, color='#2c3e50')),
                tickfont=dict(size=12),
                gridcolor='rgba(128,128,128,0.2)',
                showgrid=True
            ),
            yaxis=dict(
                title=dict(text='Probability Density', font=dict(size=14, color='#2c3e50')),
                tickfont=dict(size=12),
                gridcolor='rgba(128,128,128,0.2)',
                showgrid=True
            ),
            plot_bgcolor='white',
            paper_bgcolor='white',
            width=1200,
            height=700,
            margin=dict(l=120, r=50, t=120, b=80),  # More left margin for metrics box
            legend=dict(
                orientation="v",
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.99,  # Move legend to right side
                bgcolor='rgba(255,255,255,0.95)',
                bordercolor='rgba(0,0,0,0.3)',
                borderwidth=1,
                font=dict(size=11)
            ),
            hovermode='x unified'
        )
        
        # Add clean metrics box with better positioning
        tail_risk_down = ((tail_lower_strike - current_price) / current_price * 100)
        tail_risk_up = ((tail_upper_strike - current_price) / current_price * 100)
        
        fig.add_annotation(
            x=0.01,
            y=0.95,
            xref='paper',
            yref='paper',
            text=f'<b>📊 KEY METRICS</b><br>'
                 f'💰 Current Price: <b>${current_price:,.0f}</b><br>'
                 f'📈 Density at Price: <b>{price_density:.6f}</b><br>'
                 f'📅 Expiration: <b>{first_exp} days</b><br>'
                 f'<br><b>🎯 TAIL BOUNDARIES</b><br>'
                 f'🔴 Lower: <b>${tail_lower_strike:,.0f}</b><br>'
                 f'🟠 Upper: <b>${tail_upper_strike:,.0f}</b><br>'
                 f'<br><b>⚠️ TAIL RISK</b><br>'
                 f'<b>{tail_risk_down:.1f}% to {tail_risk_up:.1f}%</b>',
            showarrow=False,
            align='left',
            bgcolor='rgba(248, 250, 252, 0.95)',
            bordercolor='rgba(59, 130, 246, 0.3)',
            borderwidth=2,
            font=dict(size=12, color='#1e293b'),
            xanchor='left',
            yanchor='top'
        )
        
        if save_path:
            fig.write_html(save_path)
        
        return fig
    
    def plot_aggregate_probability_density_with_price(self, save_path: Optional[str] = None):
        """
        Create beautiful aggregate probability density function across ALL expirations using Plotly.
        
        Args:
            save_path: Optional path to save the HTML file
        """
        if not self.svi_model.implied_probabilities:
            self.svi_model.calculate_implied_probabilities()
        
        # Get current price from market data
        current_price = self.svi_model.market_data['spot_price'].iloc[0]
        
        # Aggregate data across all expirations
        all_strikes = []
        all_densities = []
        expiration_weights = []
        
        for exp, data in self.svi_model.implied_probabilities.items():
            prob_info = data['probabilities']
            strikes = np.array(prob_info['strikes'])
            density = np.array(prob_info['risk_neutral_density'])
            
            # Weight by time to expiration (longer expirations get more weight)
            weight = max(1, exp)  # Avoid zero weight for same-day expirations
            
            all_strikes.extend(strikes)
            all_densities.extend(density * weight)
            expiration_weights.extend([weight] * len(strikes))
        
        # Convert to numpy arrays
        all_strikes = np.array(all_strikes)
        all_densities = np.array(all_densities)
        expiration_weights = np.array(expiration_weights)
        
        # Create a grid for interpolation
        min_strike = np.min(all_strikes)
        max_strike = np.max(all_strikes)
        strike_grid = np.linspace(min_strike, max_strike, 200)
        
        # Interpolate and aggregate densities
        aggregated_density = np.zeros_like(strike_grid)
        for i, exp in enumerate(self.svi_model.implied_probabilities.keys()):
            prob_info = self.svi_model.implied_probabilities[exp]['probabilities']
            strikes = np.array(prob_info['strikes'])
            density = np.array(prob_info['risk_neutral_density'])
            
            # Interpolate to common grid
            interp_density = np.interp(strike_grid, strikes, density)
            
            # Weight by expiration (longer expirations get more weight)
            weight = max(1, exp)
            aggregated_density += interp_density * weight
        
        # Normalize the aggregated density
        aggregated_density = aggregated_density / np.trapz(aggregated_density, strike_grid)
        
        # Calculate cumulative density for tail risk visualization
        cumulative_density = np.cumsum(aggregated_density) * (strike_grid[1] - strike_grid[0])
        cumulative_density = cumulative_density / cumulative_density[-1]
        
        # Find 16th and 84th percentiles (16% tails, 68% middle)
        tail_lower_idx = np.argmin(np.abs(cumulative_density - 0.16))
        tail_upper_idx = np.argmin(np.abs(cumulative_density - 0.84))
        
        tail_lower_strike = strike_grid[tail_lower_idx]
        tail_upper_strike = strike_grid[tail_upper_idx]
        
        # Create beautiful Plotly figure
        fig = go.Figure()
        
        # Add main aggregated density curve
        fig.add_trace(go.Scatter(
            x=strike_grid,
            y=aggregated_density,
            mode='lines',
            name='Aggregated Risk-Neutral Density',
            line=dict(color='#1f77b4', width=4),
            hovertemplate='Strike: $%{x:,.0f}<br>Aggregated Density: %{y:.6f}<extra></extra>'
        ))
        
        # Add left tail (16% downside risk) - RED
        left_tail_mask = strike_grid <= tail_lower_strike
        if np.any(left_tail_mask):
            fig.add_trace(go.Scatter(
                x=strike_grid[left_tail_mask],
                y=aggregated_density[left_tail_mask],
                mode='lines',
                fill='tozeroy',
                fillcolor='rgba(220, 38, 38, 0.7)',  # Clear red
                line=dict(color='rgba(220, 38, 38, 1)', width=2),
                name='16% Left Tail (Downside Risk)',
                hovertemplate='Strike: $%{x:,.0f}<br>Density: %{y:.6f}<br>Risk Zone: Downside<extra></extra>'
            ))
        
        # Add right tail (16% upside risk) - ORANGE
        right_tail_mask = strike_grid >= tail_upper_strike
        if np.any(right_tail_mask):
            fig.add_trace(go.Scatter(
                x=strike_grid[right_tail_mask],
                y=aggregated_density[right_tail_mask],
                mode='lines',
                fill='tozeroy',
                fillcolor='rgba(251, 146, 60, 0.7)',  # Clear orange
                line=dict(color='rgba(251, 146, 60, 1)', width=2),
                name='16% Right Tail (Upside Risk)',
                hovertemplate='Strike: $%{x:,.0f}<br>Density: %{y:.6f}<br>Risk Zone: Upside<extra></extra>'
            ))
        
        # Add middle section (68% normal range) - GREEN
        middle_mask = (strike_grid > tail_lower_strike) & (strike_grid < tail_upper_strike)
        if np.any(middle_mask):
            fig.add_trace(go.Scatter(
                x=strike_grid[middle_mask],
                y=aggregated_density[middle_mask],
                mode='lines',
                fill='tozeroy',
                fillcolor='rgba(34, 197, 94, 0.5)',  # Clear green
                line=dict(color='rgba(34, 197, 94, 1)', width=2),
                name='68% Middle (Normal Range)',
                hovertemplate='Strike: $%{x:,.0f}<br>Density: %{y:.6f}<br>Risk Zone: Normal<extra></extra>'
            ))
        
        # Add current price line
        price_density = np.interp(current_price, strike_grid, aggregated_density)
        fig.add_vline(
            x=current_price,
            line=dict(color='#2E86AB', width=4, dash='dash'),
            annotation_text=f'Current Price: ${current_price:,.2f}',
            annotation_position="top right"
        )
        
        # Add tail boundary lines
        fig.add_vline(
            x=tail_lower_strike,
            line=dict(color='red', width=2, dash='dot'),
            annotation_text=f'16% Tail: ${tail_lower_strike:,.0f}',
            annotation_position="bottom"
        )
        
        fig.add_vline(
            x=tail_upper_strike,
            line=dict(color='red', width=2, dash='dot'),
            annotation_text=f'84% Tail: ${tail_upper_strike:,.0f}',
            annotation_position="bottom"
        )
        
        # Add current price point
        fig.add_trace(go.Scatter(
            x=[current_price],
            y=[price_density],
            mode='markers',
            marker=dict(color='#2E86AB', size=15, symbol='diamond'),
            name=f'Current Price: ${current_price:,.2f}',
            hovertemplate=f'Current Price: ${current_price:,.2f}<br>Aggregated Density: {price_density:.6f}<extra></extra>'
        ))
        
        # Get all expirations for display
        all_expirations = sorted(self.svi_model.implied_probabilities.keys())
        
        # Update layout for beautiful appearance
        fig.update_layout(
            title=dict(
                text=f'<b>Aggregated Implied Probability Density Function</b><br><sub>All Expirations: {all_expirations[0]} to {all_expirations[-1]} days | Current Price: ${current_price:,.2f}</sub>',
                x=0.5,
                font=dict(size=20, color='#2c3e50')
            ),
            xaxis=dict(
                title=dict(text='Strike Price ($)', font=dict(size=14, color='#2c3e50')),
                tickfont=dict(size=12),
                gridcolor='rgba(128,128,128,0.2)',
                showgrid=True
            ),
            yaxis=dict(
                title=dict(text='Aggregated Probability Density', font=dict(size=14, color='#2c3e50')),
                tickfont=dict(size=12),
                gridcolor='rgba(128,128,128,0.2)',
                showgrid=True
            ),
            plot_bgcolor='white',
            paper_bgcolor='white',
            width=1400,
            height=800,
            margin=dict(l=120, r=50, t=120, b=80),  # More left margin for metrics box
            legend=dict(
                orientation="v",
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.99,  # Move legend to right side
                bgcolor='rgba(255,255,255,0.95)',
                bordercolor='rgba(0,0,0,0.3)',
                borderwidth=1,
                font=dict(size=11)
            ),
            hovermode='x unified'
        )
        
        # Add clean metrics box with better positioning
        tail_risk_down = ((tail_lower_strike - current_price) / current_price * 100)
        tail_risk_up = ((tail_upper_strike - current_price) / current_price * 100)
        
        fig.add_annotation(
            x=0.01,
            y=0.95,
            xref='paper',
            yref='paper',
            text=f'<b>📊 AGGREGATED METRICS</b><br>'
                 f'💰 Current Price: <b>${current_price:,.0f}</b><br>'
                 f'📈 Density at Price: <b>{price_density:.6f}</b><br>'
                 f'📅 Expirations: <b>{all_expirations[0]}-{all_expirations[-1]} days</b><br>'
                 f'<br><b>🎯 TAIL BOUNDARIES</b><br>'
                 f'🔴 Lower: <b>${tail_lower_strike:,.0f}</b><br>'
                 f'🟠 Upper: <b>${tail_upper_strike:,.0f}</b><br>'
                 f'<br><b>⚠️ TAIL RISK</b><br>'
                 f'<b>{tail_risk_down:.1f}% to {tail_risk_up:.1f}%</b>',
            showarrow=False,
            align='left',
            bgcolor='rgba(248, 250, 252, 0.95)',
            bordercolor='rgba(59, 130, 246, 0.3)',
            borderwidth=2,
            font=dict(size=12, color='#1e293b'),
            xanchor='left',
            yanchor='top'
        )
        
        if save_path:
            fig.write_html(save_path)
        
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
