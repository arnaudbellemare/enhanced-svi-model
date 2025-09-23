# Enhanced SVI Model with Implied Probability Analysis

This repository extends the Stochastic Volatility Inspired (SVI) model to include comprehensive implied probability calculations for all expiration dates, providing deeper market insights through probability-based analysis.

## Features

- **SVI Model Implementation**: Advanced stochastic volatility modeling
- **Implied Probability Calculations**: Market probability views for all expiration dates
- **Multi-Expiration Analysis**: Comprehensive analysis across all available expirations
- **Visualization Tools**: Heatmaps and charts for probability surfaces
- **Risk Metrics**: Advanced risk assessment through probability analysis

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```python
from svi_enhanced import SVIEnhanced
import numpy as np

# Initialize the enhanced SVI model
svi = SVIEnhanced()

# Load market data
svi.load_market_data('market_data.csv')

# Calculate implied probabilities for all expirations
probabilities = svi.calculate_implied_probabilities()

# Visualize probability surface
svi.plot_probability_surface()
```

## Key Components

- `svi_enhanced.py`: Main SVI model with implied probability calculations
- `probability_calculator.py`: Core probability calculation engine
- `visualization.py`: Advanced plotting and visualization tools
- `data_handler.py`: Market data processing and management
- `risk_metrics.py`: Risk assessment and metrics calculation

## Documentation

See the `docs/` directory for detailed documentation on:
- Model implementation
- Probability calculation methodology
- Visualization options
- Risk metrics interpretation

## License

MIT License
