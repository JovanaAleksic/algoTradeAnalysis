# ThinkOrSwim Trading Strategy Analysis Tool

## Overview
A Python-based analysis tool that generates comprehensive performance reports from ThinkOrSwim (TOS) trading data exports. The tool creates detailed PDF reports with statistical analysis, visualizations, and performance metrics.

## Features
The generated report includes:

### Core Analysis
- Trade P/L Distribution
- Cumulative P/L Over Time
- Win/Loss Rates & Statistics
- Profit Factor Analysis
- Maximum/Average Gains and Losses

### Advanced Analytics
- Time-Based Analysis (Hourly/Daily/Monthly)
- Trade Duration Analysis
- Drawdown Analysis
- Consecutive Loss Analysis
- Risk/Reward Metrics
- Market Movement Impact
- Trade Clustering Patterns
- Strategy Robustness Scoring
- Entry/Exit Efficiency

### Visualizations
- P/L Distribution Plots
- Equity Curve
- Drawdown Charts
- Duration Analysis Charts
- Time-Based Performance Heatmaps
- Market Movement Impact Charts
- Clustering Analysis Graphs

## Requirements
```
Python 3.x
pandas
numpy
matplotlib
seaborn
reportlab
```

## File Structure
- `analysisTos.py` - Main script for processing TOS data
- `utils.py` - Utility functions for analysis and report generation

## Input Data Format
Expects ThinkOrSwim trade history export with the following columns:
- Date/Time
- Strategy
- Side
- Amount
- Price
- Trade P/L
- Cumulative P/L

## Usage
```bash
python analysisTos.py [input_file]
```

## Output
Generates a detailed PDF report named `trading_report.pdf` containing all analyses and visualizations.

## Note
This tool is designed to work with ThinkOrSwim's trade history export format. The trade data should be exported from TOS platform with all trades (both opening and closing transactions) included.

## Disclaimer
This tool is for analysis purposes only. Trading involves risk of loss. Past performance does not guarantee future results.

