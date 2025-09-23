# SVI Enhanced Model Documentation

## Overview

The Enhanced SVI Model extends the Stochastic Volatility Inspired (SVI) parameterization to include comprehensive implied probability calculations for all expiration dates. This provides deeper market insights through probability-based analysis.

## Key Features

### 1. SVI Parameterization
The SVI model parameterizes the implied volatility smile using the following form:

```
w(k) = a + b * (ρ * (k - m) + √((k - m)² + σ²))
```

Where:
- `w(k)` is the total variance (volatility squared)
- `k` is the log-moneyness (ln(K/S))
- `a`, `b`, `ρ`, `m`, `σ` are the SVI parameters

### 2. Implied Probability Calculations

#### Call Option Probability
The probability that a call option expires in-the-money is given by:

```
P(S_T > K) = N(d₂)
```

Where:
- `d₂ = (ln(S/K) + (r - q - σ²/2)T) / (σ√T)`
- `N(·)` is the cumulative standard normal distribution
- `S` is the current spot price
- `K` is the strike price
- `r` is the risk-free rate
- `q` is the dividend yield
- `σ` is the implied volatility
- `T` is the time to expiration

#### Put Option Probability
The probability that a put option expires in-the-money is:

```
P(S_T < K) = N(-d₂)
```

### 3. Risk-Neutral Density

The risk-neutral density is calculated using the Breeden-Litzenberger formula:

```
q(K) = e^(rT) * ∂²C/∂K²
```

Where `C` is the call option price and `K` is the strike price.

## Model Components

### SVIEnhanced Class

The main class that orchestrates the entire analysis:

```python
class SVIEnhanced:
    def __init__(self):
        self.market_data = None
        self.svi_params = {}
        self.implied_probabilities = {}
        self.volatility_surface = {}
        self.risk_free_rate = 0.05
        self.dividend_yield = 0.0
```

#### Key Methods

1. **`load_market_data(data_source)`**
   - Loads market data from CSV files or generates synthetic data
   - Handles data validation and preprocessing

2. **`fit_svi_parameters(expiration)`**
   - Fits SVI parameters for a specific expiration date
   - Uses constrained optimization to ensure parameter validity

3. **`calculate_implied_probabilities()`**
   - Calculates implied probabilities for all expiration dates
   - Returns comprehensive probability analysis

4. **`get_risk_metrics()`**
   - Calculates various risk metrics from implied probabilities
   - Includes tail risk, volatility skew, and probability distributions

### ProbabilityCalculator Class

Advanced probability calculation engine:

```python
class ProbabilityCalculator:
    def __init__(self, risk_free_rate=0.05, dividend_yield=0.0):
        self.risk_free_rate = risk_free_rate
        self.dividend_yield = dividend_yield
```

#### Key Methods

1. **`calculate_implied_probability_density()`**
   - Calculates probability density from option prices
   - Uses Breeden-Litzenberger formula

2. **`calculate_moment_risk_metrics()`**
   - Calculates moment-based risk metrics
   - Includes skewness, kurtosis, and tail risk

3. **`calculate_butterfly_spread_probability()`**
   - Calculates butterfly spread probabilities
   - Useful for range-bound strategies

### SVIVisualizer Class

Comprehensive visualization tools:

```python
class SVIVisualizer:
    def __init__(self, svi_model):
        self.svi_model = svi_model
```

#### Key Methods

1. **`plot_3d_probability_surface()`**
   - Creates 3D surface plots of implied probabilities
   - Shows probability evolution across strikes and expirations

2. **`plot_interactive_probability_heatmap()`**
   - Interactive heatmaps using Plotly
   - Allows zooming and detailed analysis

3. **`plot_risk_metrics_dashboard()`**
   - Comprehensive risk metrics visualization
   - Multiple subplots for different risk measures

## Usage Examples

### Basic Usage

```python
from svi_enhanced import SVIEnhanced

# Initialize the model
svi = SVIEnhanced()

# Load market data
svi.load_market_data('market_data.csv')

# Calculate implied probabilities
probabilities = svi.calculate_implied_probabilities()

# Get risk metrics
risk_metrics = svi.get_risk_metrics()

# Plot results
svi.plot_probability_surface()
```

### Advanced Analysis

```python
from probability_calculator import ProbabilityCalculator
from visualization import SVIVisualizer

# Initialize components
calc = ProbabilityCalculator(risk_free_rate=0.05)
visualizer = SVIVisualizer(svi)

# Calculate detailed probability metrics
prob_data = calc.calculate_implied_probability_density(
    strikes, call_prices, put_prices, spot_price, time_to_expiry
)

# Create comprehensive visualizations
visualizer.plot_3d_probability_surface()
visualizer.plot_interactive_probability_heatmap()
visualizer.plot_risk_metrics_dashboard()
```

## Risk Metrics

### 1. ATM Probability
The probability of the underlying expiring at-the-money, indicating market uncertainty.

### 2. Volatility Skew
The SVI parameter `ρ` represents the volatility skew, indicating market sentiment about direction.

### 3. Tail Risk
Measures the probability of extreme moves, calculated as the ratio of tail density to center density.

### 4. Large Move Probabilities
Probabilities of significant up or down moves, useful for risk management.

### 5. Value at Risk (VaR)
Quantile-based risk measures at different confidence levels.

### 6. Expected Shortfall
Conditional expected loss beyond VaR thresholds.

## Mathematical Foundations

### SVI Parameter Constraints

The SVI parameters must satisfy certain constraints to ensure arbitrage-free pricing:

1. **No-arbitrage condition**: `b(1 + |ρ|) ≤ 4`
2. **Butterfly arbitrage**: The density must be non-negative
3. **Calendar spread arbitrage**: The total variance must be increasing in time

### Probability Density Properties

The risk-neutral density must satisfy:

1. **Non-negativity**: `q(K) ≥ 0` for all K
2. **Normalization**: `∫ q(K) dK = 1`
3. **Martingale condition**: `∫ K q(K) dK = S₀ e^(r-q)T`

## Implementation Details

### Optimization Algorithm

The SVI parameter fitting uses the L-BFGS-B algorithm with the following bounds:

- `a ∈ [-0.1, 0.1]`
- `b ∈ [0.01, 1.0]`
- `ρ ∈ [-0.99, 0.99]`
- `m ∈ [-1.0, 1.0]`
- `σ ∈ [0.01, 1.0]`

### Numerical Stability

The implementation includes several numerical stability measures:

1. **Volatility bounds**: Implied volatilities are constrained to [0.001, 5.0]
2. **Density normalization**: Risk-neutral densities are normalized to ensure proper probability measures
3. **Interpolation**: Smooth interpolation is used for missing data points

### Performance Considerations

- **Vectorization**: All calculations use NumPy vectorization for speed
- **Caching**: SVI parameters are cached to avoid redundant calculations
- **Memory management**: Large datasets are processed in chunks to manage memory usage

## Validation and Testing

### Unit Tests

The model includes comprehensive unit tests for:

1. **Parameter fitting accuracy**
2. **Probability calculation correctness**
3. **Risk metric validation**
4. **Visualization functionality**

### Benchmarking

Performance benchmarks are provided for:

1. **Large dataset processing**
2. **Real-time calculation speed**
3. **Memory usage optimization**

## Extensions and Future Work

### Planned Features

1. **Multi-asset correlation analysis**
2. **Dynamic hedging strategies**
3. **Machine learning integration**
4. **Real-time data feeds**

### Research Applications

1. **Market microstructure analysis**
2. **Volatility forecasting**
3. **Risk management applications**
4. **Trading strategy development**

## References

1. Gatheral, J. (2004). "A parsimonious arbitrage-free implied volatility parameterization"
2. Breeden, D. T., & Litzenberger, R. H. (1978). "Prices of state-contingent claims implicit in option prices"
3. Carr, P., & Madan, D. (2001). "Towards a theory of volatility trading"
4. Cont, R., & Tankov, P. (2004). "Financial modelling with jump processes"
