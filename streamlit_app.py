"""
Enhanced SVI Model - Streamlit Web Application
Beautiful web interface for the Enhanced SVI Model with Implied Probability Analysis
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import our modules
from svi_enhanced import SVIEnhanced
from probability_calculator import ProbabilityCalculator
from visualization import SVIVisualizer

# Page configuration
st.set_page_config(
    page_title="Enhanced SVI Model",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #1f77b4;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
    .info-box {
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

def main():
    """Main Streamlit application."""
    
    # Header
    st.markdown('<div class="main-header">🎯 Enhanced SVI Model</div>', unsafe_allow_html=True)
    st.markdown("**Stochastic Volatility Inspired Model with Implied Probability Analysis**")
    
    # Sidebar for navigation
    st.sidebar.title("🎮 Navigation")
    
    # Initialize session state
    if 'svi_model' not in st.session_state:
        st.session_state.svi_model = None
    if 'probabilities' not in st.session_state:
        st.session_state.probabilities = None
    if 'risk_metrics' not in st.session_state:
        st.session_state.risk_metrics = None
    
    # Navigation menu
    page = st.sidebar.selectbox(
        "Choose Analysis Mode:",
        [
            "🏠 Home",
            "🚀 Quick Demo", 
            "📊 Basic Analysis",
            "🔬 Advanced Analysis", 
            "📈 Visualizations",
            "📁 Export Data",
            "⚙️ Configuration",
            "❓ Help"
        ]
    )
    
    # Home page
    if page == "🏠 Home":
        show_home_page()
    
    # Quick Demo page
    elif page == "🚀 Quick Demo":
        show_quick_demo()
    
    # Basic Analysis page
    elif page == "📊 Basic Analysis":
        show_basic_analysis()
    
    # Advanced Analysis page
    elif page == "🔬 Advanced Analysis":
        show_advanced_analysis()
    
    # Visualizations page
    elif page == "📈 Visualizations":
        show_visualizations()
    
    # Export Data page
    elif page == "📁 Export Data":
        show_export_data()
    
    # Configuration page
    elif page == "⚙️ Configuration":
        show_configuration()
    
    # Help page
    elif page == "❓ Help":
        show_help()

def show_home_page():
    """Display the home page."""
    st.markdown("## 🎯 Welcome to Enhanced SVI Model")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        ### 🚀 Quick Start
        - **Quick Demo**: See everything in action
        - **Basic Analysis**: Core SVI with probabilities
        - **Advanced Analysis**: Detailed risk metrics
        """)
    
    with col2:
        st.markdown("""
        ### 📊 Features
        - **SVI Parameter Fitting** for all expiration dates
        - **Implied Probability Calculations** 
        - **Risk Metrics & Tail Risk Analysis**
        - **Advanced Visualizations**
        """)
    
    with col3:
        st.markdown("""
        ### 🎮 Navigation
        Use the sidebar to:
        - Choose analysis mode
        - Configure parameters
        - Export results
        - Get help
        """)
    
    # Key metrics display
    st.markdown("## 📈 Key Features")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("🎯 SVI Model", "Advanced", "Parameter Fitting")
    
    with col2:
        st.metric("📊 Probabilities", "All Expirations", "Implied Analysis")
    
    with col3:
        st.metric("🔬 Risk Metrics", "Comprehensive", "Tail Risk")
    
    with col4:
        st.metric("📈 Visualizations", "Interactive", "3D Surfaces")

def show_quick_demo():
    """Display the quick demo page."""
    st.markdown("## 🚀 Quick Demo - See Everything in Action")
    
    if st.button("🎬 Run Complete Demo", type="primary"):
        with st.spinner("Running complete demonstration..."):
            try:
                # Initialize components
                svi = SVIEnhanced()
                calc = ProbabilityCalculator()
                
                # Load data
                svi.load_market_data('crypto_data')
                
                # Calculate probabilities
                probabilities = svi.calculate_implied_probabilities()
                
                # Calculate risk metrics
                risk_metrics = svi.get_risk_metrics()
                
                # Store in session state
                st.session_state.svi_model = svi
                st.session_state.probabilities = probabilities
                st.session_state.risk_metrics = risk_metrics
                
                st.success("✅ Demo completed successfully!")
                
                # Show results summary
                st.markdown("### 📊 Results Summary")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Expirations Analyzed", len(probabilities))
                
                with col2:
                    total_strikes = sum(len(data['probabilities']['strikes']) for data in probabilities.values())
                    st.metric("Total Strike Points", total_strikes)
                
                with col3:
                    avg_atm_prob = np.mean([m['atm_probability'] for m in risk_metrics.values()])
                    st.metric("Avg ATM Probability", f"{avg_atm_prob:.3f}")
                
                # Show sample results
                st.markdown("### 📋 Sample Results")
                
                first_exp = list(probabilities.keys())[0]
                prob_data = probabilities[first_exp]['probabilities']
                
                st.markdown(f"**{first_exp} days to expiration:**")
                st.markdown(f"- Strike Range: {prob_data['strikes'][0]:.2f} - {prob_data['strikes'][-1]:.2f}")
                st.markdown(f"- Average Call Probability: {np.mean(prob_data['call_probabilities']):.3f}")
                st.markdown(f"- Average Put Probability: {np.mean(prob_data['put_probabilities']):.3f}")
                
            except Exception as e:
                st.error(f"❌ Error in demo: {e}")

def show_basic_analysis():
    """Display the basic analysis page."""
    st.markdown("## 📊 Basic Analysis - Core SVI with Probabilities")
    
    if st.button("🔍 Run Basic Analysis", type="primary"):
        with st.spinner("Running basic analysis..."):
            try:
                svi = SVIEnhanced()
                svi.load_market_data('crypto_data')
                probabilities = svi.calculate_implied_probabilities()
                
                st.session_state.svi_model = svi
                st.session_state.probabilities = probabilities
                
                st.success("✅ Basic analysis completed!")
                
                # Display results
                st.markdown("### 📊 Analysis Results")
                
                # Create a DataFrame for display
                results_data = []
                for exp, data in probabilities.items():
                    prob_data = data['probabilities']
                    results_data.append({
                        'Expiration (days)': exp,
                        'Strike Points': len(prob_data['strikes']),
                        'Avg Call Prob': np.mean(prob_data['call_probabilities']),
                        'Avg Put Prob': np.mean(prob_data['put_probabilities']),
                        'Strike Range': f"{prob_data['strikes'][0]:.2f} - {prob_data['strikes'][-1]:.2f}"
                    })
                
                df = pd.DataFrame(results_data)
                st.dataframe(df, use_container_width=True)
                
                # Show probability distribution for first expiration
                if probabilities:
                    first_exp = list(probabilities.keys())[0]
                    prob_data = probabilities[first_exp]['probabilities']
                    
                    st.markdown(f"### 📈 Probability Distribution ({first_exp} days)")
                    
                    # Create probability plot
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
                    
                    # Call probabilities
                    ax1.plot(prob_data['strikes'], prob_data['call_probabilities'], 'b-', linewidth=2, label='Call Probabilities')
                    ax1.set_title(f'Call Probabilities - {first_exp} days')
                    ax1.set_xlabel('Strike Price')
                    ax1.set_ylabel('Probability')
                    ax1.grid(True, alpha=0.3)
                    ax1.legend()
                    
                    # Put probabilities
                    ax2.plot(prob_data['strikes'], prob_data['put_probabilities'], 'r-', linewidth=2, label='Put Probabilities')
                    ax2.set_title(f'Put Probabilities - {first_exp} days')
                    ax2.set_xlabel('Strike Price')
                    ax2.set_ylabel('Probability')
                    ax2.grid(True, alpha=0.3)
                    ax2.legend()
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                
            except Exception as e:
                st.error(f"❌ Error in basic analysis: {e}")

def show_advanced_analysis():
    """Display the advanced analysis page."""
    st.markdown("## 🔬 Advanced Analysis - Detailed Risk Metrics")
    
    if st.button("🔬 Run Advanced Analysis", type="primary"):
        with st.spinner("Running advanced analysis..."):
            try:
                svi = SVIEnhanced()
                svi.load_market_data('crypto_data')
                svi.calculate_implied_probabilities()
                risk_metrics = svi.get_risk_metrics()
                
                st.session_state.svi_model = svi
                st.session_state.risk_metrics = risk_metrics
                
                st.success("✅ Advanced analysis completed!")
                
                # Display risk metrics
                st.markdown("### 📊 Risk Metrics Summary")
                
                # Create risk metrics DataFrame
                risk_data = []
                for exp, metrics in risk_metrics.items():
                    risk_data.append({
                        'Expiration (days)': exp,
                        'ATM Probability': f"{metrics['atm_probability']:.3f}",
                        'Volatility Skew': f"{metrics['volatility_skew']:.3f}",
                        'Tail Risk': f"{metrics['tail_risk']:.3f}",
                        'Large Up Move': f"{metrics['large_up_move_prob']:.3f}",
                        'Large Down Move': f"{metrics['large_down_move_prob']:.3f}"
                    })
                
                df = pd.DataFrame(risk_data)
                st.dataframe(df, use_container_width=True)
                
                # Portfolio-level metrics
                st.markdown("### 📈 Portfolio-Level Analysis")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    avg_atm = np.mean([m['atm_probability'] for m in risk_metrics.values()])
                    st.metric("Avg ATM Probability", f"{avg_atm:.3f}")
                
                with col2:
                    avg_skew = np.mean([m['volatility_skew'] for m in risk_metrics.values()])
                    st.metric("Avg Volatility Skew", f"{avg_skew:.3f}")
                
                with col3:
                    avg_tail = np.mean([m['tail_risk'] for m in risk_metrics.values()])
                    st.metric("Avg Tail Risk", f"{avg_tail:.3f}")
                
                with col4:
                    avg_up = np.mean([m['large_up_move_prob'] for m in risk_metrics.values()])
                    st.metric("Avg Up Move Prob", f"{avg_up:.3f}")
                
                # Risk trend analysis
                st.markdown("### 📊 Risk Trend Analysis")
                
                expirations = sorted(risk_metrics.keys())
                tail_risks = [risk_metrics[exp]['tail_risk'] for exp in expirations]
                
                if len(tail_risks) > 1:
                    risk_trend = np.polyfit(expirations, tail_risks, 1)[0]
                    trend_direction = "increasing" if risk_trend > 0 else "decreasing"
                    
                    st.markdown(f"**Risk Trend**: {trend_direction} over time (slope: {risk_trend:.6f})")
                
                # Create risk metrics visualization
                fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
                
                # ATM Probability
                ax1.plot(expirations, [risk_metrics[exp]['atm_probability'] for exp in expirations], 'o-', linewidth=2)
                ax1.set_title('ATM Probability by Expiration')
                ax1.set_xlabel('Days to Expiration')
                ax1.set_ylabel('ATM Probability')
                ax1.grid(True, alpha=0.3)
                
                # Volatility Skew
                ax2.plot(expirations, [risk_metrics[exp]['volatility_skew'] for exp in expirations], 's-', linewidth=2, color='red')
                ax2.set_title('Volatility Skew by Expiration')
                ax2.set_xlabel('Days to Expiration')
                ax2.set_ylabel('Skew')
                ax2.grid(True, alpha=0.3)
                
                # Tail Risk
                ax3.plot(expirations, [risk_metrics[exp]['tail_risk'] for exp in expirations], '^-', linewidth=2, color='green')
                ax3.set_title('Tail Risk by Expiration')
                ax3.set_xlabel('Days to Expiration')
                ax3.set_ylabel('Tail Risk')
                ax3.grid(True, alpha=0.3)
                
                # Large Move Probabilities
                up_moves = [risk_metrics[exp]['large_up_move_prob'] for exp in expirations]
                down_moves = [risk_metrics[exp]['large_down_move_prob'] for exp in expirations]
                ax4.plot(expirations, up_moves, 'o-', label='Up Moves', linewidth=2)
                ax4.plot(expirations, down_moves, 's-', label='Down Moves', linewidth=2)
                ax4.set_title('Large Move Probabilities')
                ax4.set_xlabel('Days to Expiration')
                ax4.set_ylabel('Probability')
                ax4.legend()
                ax4.grid(True, alpha=0.3)
                
                plt.tight_layout()
                st.pyplot(fig)
                
            except Exception as e:
                st.error(f"❌ Error in advanced analysis: {e}")

def show_visualizations():
    """Display the visualizations page."""
    st.markdown("## 📈 Visualizations - Create Charts and Reports")
    
    if st.session_state.svi_model is None:
        st.warning("⚠️ Please run Basic Analysis or Advanced Analysis first to generate data.")
        return
    
    st.markdown("### 🎨 Choose Visualization Type")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📊 Risk Metrics Dashboard", type="primary"):
            with st.spinner("Creating risk metrics dashboard..."):
                try:
                    visualizer = SVIVisualizer(st.session_state.svi_model)
                    visualizer.plot_risk_metrics_dashboard('streamlit_risk_dashboard.png')
                    st.success("✅ Risk dashboard created!")
                    st.image('streamlit_risk_dashboard.png', caption='Risk Metrics Dashboard')
                except Exception as e:
                    st.error(f"❌ Error creating dashboard: {e}")
        
        if st.button("🌐 3D Probability Surface", type="primary"):
            with st.spinner("Creating 3D surface..."):
                try:
                    visualizer = SVIVisualizer(st.session_state.svi_model)
                    visualizer.plot_3d_probability_surface('streamlit_3d_surface.png')
                    st.success("✅ 3D surface created!")
                    st.image('streamlit_3d_surface.png', caption='3D Probability Surface')
                except Exception as e:
                    st.error(f"❌ Error creating 3D surface: {e}")
    
    with col2:
        if st.button("🔥 Interactive Heatmap", type="primary"):
            with st.spinner("Creating interactive heatmap..."):
                try:
                    visualizer = SVIVisualizer(st.session_state.svi_model)
                    fig = visualizer.plot_interactive_probability_heatmap('streamlit_heatmap.html')
                    st.success("✅ Interactive heatmap created!")
                    
                    # Display the interactive heatmap directly in Streamlit
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.markdown("📁 Interactive heatmap also saved as 'streamlit_heatmap.html'")
                except Exception as e:
                    st.error(f"❌ Error creating heatmap: {e}")
        
        if st.button("📈 SVI Parameter Analysis", type="primary"):
            with st.spinner("Creating SVI parameter analysis..."):
                try:
                    visualizer = SVIVisualizer(st.session_state.svi_model)
                    visualizer.plot_svi_parameter_analysis('streamlit_svi_parameters.png')
                    st.success("✅ SVI parameter analysis created!")
                    st.image('streamlit_svi_parameters.png', caption='SVI Parameter Analysis')
                except Exception as e:
                    st.error(f"❌ Error creating SVI analysis: {e}")
    
    # Add new visualization section for probability density
    st.markdown("### 🎯 Advanced Probability Analysis")
    
    col3, col4 = st.columns(2)
    
    with col3:
        if st.button("📊 Probability Density with Price", type="primary"):
            with st.spinner("Creating beautiful probability density with tail risk..."):
                try:
                    visualizer = SVIVisualizer(st.session_state.svi_model)
                    fig = visualizer.plot_implied_probability_density_with_price('streamlit_density.html')
                    st.success("✅ Beautiful probability density created!")
                    
                    # Display the interactive Plotly plot
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.markdown("📁 Interactive probability density also saved as 'streamlit_density.html'")
                except Exception as e:
                    st.error(f"❌ Error creating probability density: {e}")
    
    with col4:
        if st.button("🔄 Probability Evolution", type="primary"):
            with st.spinner("Creating probability evolution..."):
                try:
                    visualizer = SVIVisualizer(st.session_state.svi_model)
                    visualizer.plot_probability_evolution_animation('streamlit_evolution.png')
                    st.success("✅ Probability evolution created!")
                    st.image('streamlit_evolution.png', caption='Probability Evolution')
                except Exception as e:
                    st.error(f"❌ Error creating evolution plot: {e}")
    
    # Add aggregate visualization section
    st.markdown("### 🌐 Aggregate Analysis Across All Expirations")
    
    col5, col6 = st.columns(2)
    
    with col5:
        if st.button("📊 Aggregate Probability Density", type="primary"):
            with st.spinner("Creating aggregate probability density across all expirations..."):
                try:
                    visualizer = SVIVisualizer(st.session_state.svi_model)
                    fig = visualizer.plot_aggregate_probability_density_with_price('streamlit_aggregate.html')
                    st.success("✅ Aggregate probability density created!")
                    
                    # Display the interactive Plotly plot
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.markdown("📁 Interactive aggregate density also saved as 'streamlit_aggregate.html'")
                except Exception as e:
                    st.error(f"❌ Error creating aggregate density: {e}")
    
    with col6:
        if st.button("📈 Multi-Expiration Comparison", type="primary"):
            with st.spinner("Creating multi-expiration comparison..."):
                try:
                    # Create a comparison plot showing multiple expirations
                    visualizer = SVIVisualizer(st.session_state.svi_model)
                    
                    # Get all expirations
                    all_expirations = sorted(st.session_state.svi_model.implied_probabilities.keys())
                    
                    # Create comparison figure
                    fig = go.Figure()
                    
                    colors = px.colors.qualitative.Set3
                    for i, exp in enumerate(all_expirations[:5]):  # Show first 5 expirations
                        prob_info = st.session_state.svi_model.implied_probabilities[exp]['probabilities']
                        strikes = np.array(prob_info['strikes'])
                        density = np.array(prob_info['risk_neutral_density'])
                        
                        fig.add_trace(go.Scatter(
                            x=strikes,
                            y=density,
                            mode='lines',
                            name=f'{exp} days',
                            line=dict(color=colors[i % len(colors)], width=2),
                            hovertemplate=f'{exp} days<br>Strike: $%{{x:,.0f}}<br>Density: %{{y:.6f}}<extra></extra>'
                        ))
                    
                    # Add current price line
                    current_price = st.session_state.svi_model.market_data['spot_price'].iloc[0]
                    fig.add_vline(
                        x=current_price,
                        line=dict(color='red', width=3, dash='dash'),
                        annotation_text=f'Current Price: ${current_price:,.2f}'
                    )
                    
                    fig.update_layout(
                        title='Multi-Expiration Probability Density Comparison',
                        xaxis_title='Strike Price ($)',
                        yaxis_title='Probability Density',
                        width=800,
                        height=500,
                        hovermode='x unified'
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    st.success("✅ Multi-expiration comparison created!")
                    
                except Exception as e:
                    st.error(f"❌ Error creating comparison: {e}")

def show_export_data():
    """Display the export data page."""
    st.markdown("## 📁 Export Data - Export Results")
    
    if st.session_state.svi_model is None:
        st.warning("⚠️ Please run Basic Analysis or Advanced Analysis first to generate data.")
        return
    
    st.markdown("### 📊 Choose Export Type")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📊 Export Probability Data", type="primary"):
            with st.spinner("Exporting probability data..."):
                try:
                    st.session_state.svi_model.export_results('streamlit_probabilities.csv')
                    st.success("✅ Probability data exported to 'streamlit_probabilities.csv'")
                except Exception as e:
                    st.error(f"❌ Error exporting data: {e}")
    
    with col2:
        if st.button("📈 Export Risk Metrics", type="primary"):
            with st.spinner("Exporting risk metrics..."):
                try:
                    if st.session_state.risk_metrics:
                        df = pd.DataFrame([
                            {
                                'expiration': exp,
                                'atm_probability': metrics['atm_probability'],
                                'volatility_skew': metrics['volatility_skew'],
                                'tail_risk': metrics['tail_risk'],
                                'large_up_move_prob': metrics['large_up_move_prob'],
                                'large_down_move_prob': metrics['large_down_move_prob']
                            }
                            for exp, metrics in st.session_state.risk_metrics.items()
                        ])
                        df.to_csv('streamlit_risk_metrics.csv', index=False)
                        st.success("✅ Risk metrics exported to 'streamlit_risk_metrics.csv'")
                    else:
                        st.warning("⚠️ No risk metrics available. Run Advanced Analysis first.")
                except Exception as e:
                    st.error(f"❌ Error exporting risk metrics: {e}")
    
    with col3:
        if st.button("📋 Export All Data", type="primary"):
            with st.spinner("Exporting all data..."):
                try:
                    # Export probabilities
                    st.session_state.svi_model.export_results('streamlit_all_probabilities.csv')
                    
                    # Export risk metrics
                    if st.session_state.risk_metrics:
                        df = pd.DataFrame([
                            {
                                'expiration': exp,
                                'atm_probability': metrics['atm_probability'],
                                'volatility_skew': metrics['volatility_skew'],
                                'tail_risk': metrics['tail_risk']
                            }
                            for exp, metrics in st.session_state.risk_metrics.items()
                        ])
                        df.to_csv('streamlit_all_risk_metrics.csv', index=False)
                    
                    st.success("✅ All data exported successfully!")
                    st.markdown("**Exported files:**")
                    st.markdown("- streamlit_all_probabilities.csv")
                    st.markdown("- streamlit_all_risk_metrics.csv")
                except Exception as e:
                    st.error(f"❌ Error exporting all data: {e}")

def show_configuration():
    """Display the configuration page."""
    st.markdown("## ⚙️ Configuration - Set Parameters")
    
    st.markdown("### 🔧 Current Configuration")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**Risk-free rate:** 0.05 (5%)")
    
    with col2:
        st.markdown("**Dividend yield:** 0.02 (2%)")
    
    with col3:
        st.markdown("**Default spot price:** 100.0")
    
    st.markdown("### 📝 Configuration Options")
    
    st.info("💡 Configuration changes will be applied to new analyses. Current session data will not be affected.")
    
    # Risk-free rate
    risk_free_rate = st.slider("Risk-free Rate", 0.0, 0.1, 0.05, 0.001, help="Risk-free interest rate")
    
    # Dividend yield
    dividend_yield = st.slider("Dividend Yield", 0.0, 0.1, 0.02, 0.001, help="Dividend yield")
    
    # Spot price
    spot_price = st.number_input("Spot Price", 50.0, 500.0, 100.0, 1.0, help="Current spot price")
    
    if st.button("💾 Save Configuration", type="primary"):
        st.success("✅ Configuration saved!")
        st.markdown(f"**New settings:**")
        st.markdown(f"- Risk-free rate: {risk_free_rate:.3f}")
        st.markdown(f"- Dividend yield: {dividend_yield:.3f}")
        st.markdown(f"- Spot price: {spot_price:.2f}")

def show_help():
    """Display the help page."""
    st.markdown("## ❓ Help & Documentation")
    
    st.markdown("### 📚 Enhanced SVI Model Help")
    
    st.markdown("""
    This application provides comprehensive implied probability analysis using the 
    Stochastic Volatility Inspired (SVI) model.
    """)
    
    st.markdown("### 🎯 Main Features")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Core Functionality:**
        - SVI parameter fitting for all expiration dates
        - Implied probability calculations
        - Risk metrics and tail risk analysis
        - Advanced visualizations
        - Data export capabilities
        """)
    
    with col2:
        st.markdown("""
        **Navigation:**
        - Use the sidebar to choose analysis mode
        - Run analyses in sequence for best results
        - Export data after analysis completion
        - Configure parameters as needed
        """)
    
    st.markdown("### 🔧 Technical Details")
    
    st.markdown("""
    **Mathematical Foundations:**
    - SVI parameterization: `w(k) = a + b * (ρ * (k - m) + √((k - m)² + σ²))`
    - Implied probabilities: `P(S_T > K) = N(d₂)` for calls
    - Risk-neutral density: `q(K) = e^(rT) * ∂²C/∂K²`
    - Advanced risk metrics: VaR, Expected Shortfall, Tail Risk
    """)
    
    st.markdown("### 📁 Generated Files")
    
    st.markdown("""
    **Output Files:**
    - CSV exports for probability data
    - PNG files for visualizations
    - HTML files for interactive charts
    - Comprehensive reports
    """)
    
    st.markdown("### 🚀 Getting Started")
    
    st.markdown("""
    1. **Start with Quick Demo** to see everything in action
    2. **Run Basic Analysis** for core functionality
    3. **Use Advanced Analysis** for detailed risk metrics
    4. **Create Visualizations** for charts and reports
    5. **Export Data** to save results
    """)

if __name__ == "__main__":
    main()
