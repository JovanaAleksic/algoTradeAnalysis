import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch

def analyze_consecutive_losses(df):
    # Initialize variables for counting consecutive losses
    current_streak = 0
    loss_streaks = []
    
    # Loop through trades
    for pl in df['Trade_PL_numeric']:
        if pl < 0:
            current_streak += 1
        else:
            if current_streak > 2:  # Only record streaks > 2
                loss_streaks.append(current_streak)
            current_streak = 0
    
    # Don't forget to add the last streak if it exists
    if current_streak > 2:
        loss_streaks.append(current_streak)
    
    return loss_streaks

# Add to the analysis
def time_based_analysis(df):
    # Trading by time of day
    df['Hour'] = df['Entry_Time'].dt.hour
    hourly_returns = df.groupby('Hour')['Trade_PL_numeric'].agg([
        'mean', 'count', 'sum'
    ]).round(2)
    
    # Trading by day of week
    df['DayOfWeek'] = df['Entry_Time'].dt.day_name()
    daily_returns = df.groupby('DayOfWeek')['Trade_PL_numeric'].agg([
        'mean', 'count', 'sum'
    ]).round(2)
    
    # Monthly analysis
    df['Month'] = df['Entry_Time'].dt.to_period('M')
    monthly_returns = df.groupby('Month')['Trade_PL_numeric'].sum()
    
    return hourly_returns, daily_returns, monthly_returns


def duration_analysis(df):
    # Calculate average duration for winning vs losing trades
    winning_durations = df[df['Trade_PL_numeric'] > 0]['Duration']
    losing_durations = df[df['Trade_PL_numeric'] < 0]['Duration']
    
    duration_stats = {
        'Avg Winner Duration': winning_durations.mean(),
        'Avg Loser Duration': losing_durations.mean(),
        'Max Duration': df['Duration'].max(),
        'Min Duration': df['Duration'].min()
    }
    
    # Group trades by duration bins
    df['Duration_Minutes'] = df['Duration'].dt.total_seconds() / 60
    duration_bins = pd.qcut(df['Duration_Minutes'], q=5)
    duration_performance = df.groupby(duration_bins)['Trade_PL_numeric'].agg([
        'mean', 'count', 'sum'
    ])
    
    return duration_stats, duration_performance


def drawdown_analysis(df):
    # Calculate drawdowns
    cumulative = df['Cumulative_PL_numeric']
    running_max = cumulative.cummax()
    drawdowns = cumulative - running_max
    
    drawdown_stats = {
        'Max Drawdown': drawdowns.min(),
        'Average Drawdown': drawdowns[drawdowns < 0].mean(),
        'Number of Drawdowns': len(drawdowns[drawdowns < 0].unique()),
        'Current Drawdown': drawdowns.iloc[-1]
    }
    
    return drawdown_stats, drawdowns



def risk_reward_analysis(df):
    # Calculate risk-reward ratios
    avg_win = df[df['Trade_PL_numeric'] > 0]['Trade_PL_numeric'].mean()
    avg_loss = abs(df[df['Trade_PL_numeric'] < 0]['Trade_PL_numeric'].mean())
    risk_reward = avg_win / avg_loss
    
    # Calculate Sharpe-like ratio (assuming risk-free rate = 0)
    returns = df['Trade_PL_numeric']
    sharpe = returns.mean() / returns.std() * np.sqrt(252)  # Annualized
    
    # Calculate win rate by trade size
    df['Trade_Size'] = pd.qcut(abs(df['Trade_PL_numeric']), q=5)
    size_performance = df.groupby('Trade_Size')['Trade_PL_numeric'].agg([
        'mean', 'count', 'sum'
    ])
    
    return {
        'Risk_Reward_Ratio': risk_reward,
        'Sharpe_Ratio': sharpe,
        'Size_Performance': size_performance
    }



def strategy_consistency(df, window=20):
    # Rolling metrics
    df['Rolling_Win_Rate'] = df['Trade_PL_numeric'].apply(
        lambda x: 1 if x > 0 else 0
    ).rolling(window).mean() * 100
    
    df['Rolling_PL'] = df['Trade_PL_numeric'].rolling(window).mean()
    df['Rolling_Std'] = df['Trade_PL_numeric'].rolling(window).std()
    df['Rolling_Sharpe'] = df['Rolling_PL'] / df['Rolling_Std']
    
    # Monthly analysis
    monthly_stats = df.groupby(df['Entry_Time'].dt.to_period('M'))['Trade_PL_numeric'].agg([
        'mean', 'std', 'count', 'sum'
    ])
    monthly_stats['Sharpe'] = monthly_stats['mean'] / monthly_stats['std']
    monthly_stats['Win_Rate'] = df.groupby(df['Entry_Time'].dt.to_period('M'))['Trade_PL_numeric'].apply(
        lambda x: (x > 0).mean() * 100
    )
    
    # Calculate consistency metrics
    consistency_metrics = {
        'Profitable_Months': (monthly_stats['sum'] > 0).mean() * 100,
        'Best_Month': monthly_stats['sum'].max(),
        'Worst_Month': monthly_stats['sum'].min(),
        'Avg_Monthly_Trades': monthly_stats['count'].mean(),
        'Highest_Monthly_Sharpe': monthly_stats['Sharpe'].max(),
        'Lowest_Monthly_Sharpe': monthly_stats['Sharpe'].min(),
        'Monthly_Win_Rate_Std': monthly_stats['Win_Rate'].std(),  # Lower is more consistent
        'Consecutive_Profitable_Months': get_max_consecutive(monthly_stats['sum'] > 0)
    }
    
    return monthly_stats, consistency_metrics, df[['Rolling_Win_Rate', 'Rolling_PL', 'Rolling_Sharpe']]

def get_max_consecutive(series):
    # Helper function to get maximum consecutive True values
    max_consecutive = current_consecutive = 0
    for value in series:
        if value:
            current_consecutive += 1
            max_consecutive = max(max_consecutive, current_consecutive)
        else:
            current_consecutive = 0
    return max_consecutive


def duration_analysis(df):
    # Create custom bins for duration (in minutes)
    bins = [0, 5, 15, 30, 60, 120, 240, 480, float('inf')]
    labels = ['0-5m', '5-15m', '15-30m', '30-60m', '1-2h', '2-4h', '4-8h', '>8h']
    
    # Convert duration to minutes for each trade
    df['Duration_Minutes'] = df['Duration'].dt.total_seconds() / 60
    df['Duration_Category'] = pd.cut(df['Duration_Minutes'], 
                                   bins=bins, 
                                   labels=labels, 
                                   include_lowest=True)
    
    # Analysis by duration category
    duration_stats = df.groupby('Duration_Category').agg({
        'Trade_PL_numeric': ['count', 'mean', 'sum', 
                            lambda x: (x > 0).mean() * 100],  # win rate
        'Duration_Minutes': 'mean'
    }).round(2)
    
    duration_stats.columns = ['Count', 'Avg P/L', 'Total P/L', 'Win Rate %', 'Avg Minutes']
    
    # Calculate additional statistics
    winning_durations = df[df['Trade_PL_numeric'] > 0]['Duration_Minutes']
    losing_durations = df[df['Trade_PL_numeric'] < 0]['Duration_Minutes']
    
    additional_stats = {
        'Avg Winner Duration': f"{winning_durations.mean():.2f} minutes",
        'Avg Loser Duration': f"{losing_durations.mean():.2f} minutes",
        'Max Duration': f"{df['Duration_Minutes'].max():.2f} minutes",
        'Min Duration': f"{df['Duration_Minutes'].min():.2f} minutes",
        'Most Profitable Duration': df.groupby('Duration_Category')['Trade_PL_numeric'].sum().idxmax(),
        'Highest Win Rate Duration': df.groupby('Duration_Category').apply(
            lambda x: (x['Trade_PL_numeric'] > 0).mean() * 100).idxmax()
    }
    
    return duration_stats, additional_stats

def create_duration_plots(df):
    # Create figure with multiple subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot 1: Distribution of trade durations
    duration_counts = df['Duration_Category'].value_counts().sort_index()
    duration_counts.plot(kind='bar', ax=ax1)
    ax1.set_title('Distribution of Trade Durations')
    ax1.set_xlabel('Duration Category')
    ax1.set_ylabel('Number of Trades')
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
    
    # Plot 2: Average P/L by duration
    avg_pl = df.groupby('Duration_Category')['Trade_PL_numeric'].mean()
    avg_pl.plot(kind='bar', ax=ax2, color='green')
    ax2.set_title('Average P/L by Duration')
    ax2.set_xlabel('Duration Category')
    ax2.set_ylabel('Average P/L ($)')
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
    
    plt.tight_layout()
    
    # Save plots to buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
    buf.seek(0)
    plt.close()
    
    return buf


def create_consistency_plots(df, monthly_stats, rolling_metrics):
    fig = plt.figure(figsize=(15, 12))
    
    # Plot 1: Rolling Win Rate
    ax1 = plt.subplot(311)
    ax1.plot(range(len(rolling_metrics)), rolling_metrics['Rolling_Win_Rate'], label='Rolling Win Rate')
    ax1.axhline(y=50, color='r', linestyle='--', alpha=0.5)
    ax1.set_title('20-Trade Rolling Win Rate')
    ax1.set_ylabel('Win Rate (%)')
    ax1.grid(True)
    
    # Plot 2: Rolling P/L
    ax2 = plt.subplot(312)
    ax2.plot(range(len(rolling_metrics)), rolling_metrics['Rolling_PL'], label='Rolling Average P/L', color='green')
    ax2.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    ax2.set_title('20-Trade Rolling Average P/L')
    ax2.set_ylabel('P/L ($)')
    ax2.grid(True)
    
    # Plot 3: Monthly Performance
    ax3 = plt.subplot(313)
    monthly_returns = monthly_stats['sum']
    colors = ['green' if x >= 0 else 'red' for x in monthly_returns]
    monthly_returns.plot(kind='bar', ax=ax3, color=colors)
    ax3.set_title('Monthly Performance')
    ax3.set_ylabel('P/L ($)')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    
    # Save plots to buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
    buf.seek(0)
    plt.close()
    
    return buf

def market_movement_analysis(df):
    # Calculate daily range percentage
    df['Daily_Range'] = (df['Exit_Price'] - df['Entry_Price']) / df['Entry_Price'] * 100
    df['Range_Category'] = pd.qcut(df['Daily_Range'].abs(), q=5, labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
    
    # Analyze performance by market movement
    movement_stats = df.groupby('Range_Category').agg({
        'Trade_PL_numeric': ['count', 'mean', 'sum', lambda x: (x > 0).mean() * 100],
        'Duration': 'mean'
    }).round(2)
    
    return movement_stats


def trade_clustering_analysis(df):
    # Calculate time between trades
    df['Time_Between_Trades'] = df['Entry_Time'].diff()
    df['Trade_Cluster'] = (df['Time_Between_Trades'] > pd.Timedelta(hours=2)).cumsum()
    
    # Analyze performance by cluster
    cluster_stats = df.groupby('Trade_Cluster').agg({
        'Trade_PL_numeric': ['count', 'mean', 'sum', lambda x: (x > 0).mean() * 100],
        'Time_Between_Trades': 'mean'
    })
    
    return cluster_stats



def recovery_analysis(df):
    # Initialize variables for tracking drawdown periods
    equity_curve = df['Cumulative_PL_numeric']
    peak = equity_curve.iloc[0]
    drawdown_periods = []
    current_drawdown = {'start': 0, 'depth': 0, 'recovery_time': 0}
    
    for i in range(len(equity_curve)):
        if equity_curve.iloc[i] > peak:
            # New peak
            if current_drawdown['depth'] != 0:
                # End of drawdown period
                current_drawdown['end'] = i
                current_drawdown['recovery_time'] = i - current_drawdown['start']
                drawdown_periods.append(current_drawdown)
                current_drawdown = {'start': i, 'depth': 0, 'recovery_time': 0}
            peak = equity_curve.iloc[i]
        else:
            # In drawdown
            drawdown = (equity_curve.iloc[i] - peak) / peak
            if drawdown < current_drawdown['depth']:
                current_drawdown['depth'] = drawdown
                
    return pd.DataFrame(drawdown_periods)




def adaptation_analysis(df):
    # Calculate various performance metrics in different time windows
    windows = [20, 50, 100]
    metrics = {}
    
    for window in windows:
        metrics[window] = {
            'rolling_win_rate': df['Trade_PL_numeric'].apply(lambda x: x > 0).rolling(window).mean(),
            'rolling_avg_win': df[df['Trade_PL_numeric'] > 0]['Trade_PL_numeric'].rolling(window).mean(),
            'rolling_avg_loss': df[df['Trade_PL_numeric'] < 0]['Trade_PL_numeric'].rolling(window).mean(),
            'rolling_profit_factor': abs(
                df[df['Trade_PL_numeric'] > 0]['Trade_PL_numeric'].rolling(window).sum() /
                df[df['Trade_PL_numeric'] < 0]['Trade_PL_numeric'].rolling(window).sum()
            )
        }
    
    return metrics


def calculate_robustness_score(df):
    # Components for robustness score
    win_rate = len(df[df['Trade_PL_numeric'] > 0]) / len(df)
    profit_factor = abs(df[df['Trade_PL_numeric'] > 0]['Trade_PL_numeric'].sum() / 
                       df[df['Trade_PL_numeric'] < 0]['Trade_PL_numeric'].sum())
    avg_win_loss_ratio = abs(df[df['Trade_PL_numeric'] > 0]['Trade_PL_numeric'].mean() / 
                            df[df['Trade_PL_numeric'] < 0]['Trade_PL_numeric'].mean())
    
    # Monthly consistency
    monthly_returns = df.groupby(df['Entry_Time'].dt.to_period('M'))['Trade_PL_numeric'].sum()
    monthly_consistency = (monthly_returns > 0).mean()
    
    # Calculate max drawdown percentage
    cumulative = df['Cumulative_PL_numeric']
    rolling_max = cumulative.cummax()
    drawdown = (cumulative - rolling_max) / rolling_max
    max_drawdown = abs(drawdown.min())
    
    # Combine into single score (you can adjust weights)
    robustness_score = (
        0.2 * win_rate +
        0.2 * min(profit_factor / 3, 1) +  # Cap at 1
        0.2 * min(avg_win_loss_ratio / 2, 1) +  # Cap at 1
        0.2 * monthly_consistency +
        0.2 * (1 - min(max_drawdown, 1))  # Lower drawdown is better
    ) * 100
    
    return robustness_score, {
        'win_rate': win_rate,
        'profit_factor': profit_factor,
        'avg_win_loss_ratio': avg_win_loss_ratio,
        'monthly_consistency': monthly_consistency,
        'max_drawdown': max_drawdown
    }




def entry_exit_efficiency(df):
    # Calculate potential profit based on price movement during trade
    df['Price_Movement'] = df['Exit_Price'] - df['Entry_Price']
    df['Entry_Efficiency'] = np.where(
        df['Trade_PL_numeric'] > 0,
        df['Trade_PL_numeric'] / df['Price_Movement'].abs(),
        1 - abs(df['Trade_PL_numeric'] / df['Price_Movement'])
    )
    
    return df['Entry_Efficiency'].mean()



########################################################

def create_trading_report(consolidated_df, filename="trading_report.pdf"):


	# Add consecutive losses analysis
    loss_streaks = analyze_consecutive_losses(consolidated_df)
    
    plt.figure(figsize=(10, 6))
    plt.hist(loss_streaks, bins=range(min(loss_streaks), max(loss_streaks) + 2, 1), 
             align='left', rwidth=0.8)
    plt.title('Distribution of Consecutive Losing Trades (>2)')
    plt.xlabel('Number of Consecutive Losses')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    
    # Save plot to bytes buffer
    buf3 = io.BytesIO()
    plt.savefig(buf3, format='png', dpi=300, bbox_inches='tight')
    buf3.seek(0)
    consec_plot = Image(buf3)
    consec_plot.drawHeight = 3.5*inch
    consec_plot.drawWidth = 6*inch
    plt.close()


    # Save plots to memory
    plt.figure(figsize=(10, 6))
    # Use the actual column name that contains P/L data
    sns.histplot(data=consolidated_df['Trade_PL_numeric'], bins=50)
    plt.axvline(x=0, color='r', linestyle='--', alpha=0.5)
    plt.title('Distribution of Trade P/L')
    plt.xlabel('Profit/Loss ($)')
    plt.ylabel('Frequency')
    
    # Save plot to bytes buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
    buf.seek(0)
    dist_plot = Image(buf)
    dist_plot.drawHeight = 3.5*inch
    dist_plot.drawWidth = 6*inch
    plt.close()

    # Create second plot
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(consolidated_df)), consolidated_df['Cumulative_PL_numeric'])
    plt.title('Cumulative P/L Over Time')
    plt.xlabel('Trade Number')
    plt.ylabel('Cumulative P/L ($)')
    plt.grid(True)
    
    buf2 = io.BytesIO()
    plt.savefig(buf2, format='png', dpi=300, bbox_inches='tight')
    buf2.seek(0)
    cum_plot = Image(buf2)
    cum_plot.drawHeight = 3.5*inch
    cum_plot.drawWidth = 6*inch
    plt.close()

    # Calculate statistics
    winning_trades = consolidated_df[consolidated_df['Trade_PL_numeric'] > 0]
    losing_trades = consolidated_df[consolidated_df['Trade_PL_numeric'] < 0]
    breakeven_trades = consolidated_df[consolidated_df['Trade_PL_numeric'] == 0]

    max_win = winning_trades['Trade_PL_numeric'].max()
    max_loss = losing_trades['Trade_PL_numeric'].min()
    avg_win = winning_trades['Trade_PL_numeric'].mean()
    median_win = winning_trades['Trade_PL_numeric'].median()
    avg_loss = losing_trades['Trade_PL_numeric'].mean()
    win_rate = len(winning_trades) / len(consolidated_df) * 100
    loss_rate = len(losing_trades) / len(consolidated_df) * 100
    breakeven_rate = len(breakeven_trades) / len(consolidated_df) * 100
    profit_factor = abs(winning_trades['Trade_PL_numeric'].sum() / losing_trades['Trade_PL_numeric'].sum())

    # Create PDF
    doc = SimpleDocTemplate(filename, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []

    # Title
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        spaceAfter=30
    )
    story.append(Paragraph("Trading Strategy Analysis Report", title_style))
    story.append(Spacer(1, 12))

    # Key Statistics Table
    data = [
        ['Metric', 'Value'],
        ['Total Trades', len(consolidated_df)],
        ['Winning Trades', len(winning_trades)],
        ['Losing Trades', len(losing_trades)],
        ['Breakeven Trades', len(breakeven_trades)],
        ['Win Rate', f"{win_rate:.2f}%"],
        ['Loss Rate', f"{loss_rate:.2f}%"],
        ['Breakeven Rate', f"{breakeven_rate:.2f}%"],
        ['Maximum Win', f"${max_win:.2f}"],
        ['Maximum Loss', f"${max_loss:.2f}"],
        ['Average Win', f"${avg_win:.2f}"],
        ['Median Win', f"${median_win:.2f}"],
        ['Average Loss', f"${avg_loss:.2f}"],
        ['Profit Factor', f"{profit_factor:.2f}"]
    ]

    table = Table(data)
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 14),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.white),
        ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 12),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))

    story.append(table)
    story.append(Spacer(1, 20))

    # Add plots
    # Add consecutive losses analysis to story after other plots
    story.append(Paragraph("Consecutive Losing Trades Analysis", styles['Heading2']))
    story.append(consec_plot)
    story.append(Spacer(1, 20))

    story.append(Paragraph("Trade P/L Distribution", styles['Heading2']))
    story.append(dist_plot)
    story.append(Spacer(1, 20))
    
    story.append(Paragraph("Cumulative P/L Over Time", styles['Heading2']))
    story.append(cum_plot)
    story.append(Spacer(1, 20))

    # Add trade statistics
    story.append(Paragraph("Trade Statistics", styles['Heading2']))
    stats = consolidated_df['Trade_PL_numeric'].describe()
    stats_data = [[i, f"${stats[i]:.2f}"] for i in stats.index]
    stats_table = Table([['Statistic', 'Value']] + stats_data)
    stats_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 14),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.white),
        ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 12),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    story.append(stats_table)


    story.append(Spacer(1, 20))

    consec_stats = [
        ['Metric', 'Value'],
        ['Maximum Consecutive Losses', max(loss_streaks)],
        ['Average Losing Streak', f"{sum(loss_streaks)/len(loss_streaks):.2f}"],
        ['Number of Losing Streaks >2', len(loss_streaks)],
        ['Most Common Streak Length', max(set(loss_streaks), key=loss_streaks.count)]]
    
    consec_table = Table(consec_stats)
    consec_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 14),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.white),
        ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 12),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    story.append(Spacer(1, 20))    
    story.append(consec_table)

    story.append(Spacer(1, 20))  

    # Time-Based Analysis
    story.append(Paragraph("Time-Based Analysis", styles['Heading2']))
    hourly_returns, daily_returns, monthly_returns = time_based_analysis(consolidated_df)
    
    # Hourly returns plot
    plt.figure(figsize=(10, 6))
    hourly_returns['sum'].plot(kind='bar')
    plt.title('Profit by Hour of Day')
    plt.xlabel('Hour')
    plt.ylabel('Total Profit/Loss ($)')
    plt.xticks(rotation=45)
    buf_hourly = io.BytesIO()
    plt.savefig(buf_hourly, format='png', dpi=300, bbox_inches='tight')
    buf_hourly.seek(0)
    hourly_plot = Image(buf_hourly)
    hourly_plot.drawHeight = 3.5*inch
    hourly_plot.drawWidth = 6*inch
    plt.close()
    story.append(hourly_plot)
    story.append(Spacer(1, 20))

    # Daily returns plot
    plt.figure(figsize=(10, 6))
    daily_returns['sum'].plot(kind='bar')
    plt.title('Profit by Day of Week')
    plt.xlabel('Day')
    plt.ylabel('Total Profit/Loss ($)')
    plt.xticks(rotation=45)
    buf_daily = io.BytesIO()
    plt.savefig(buf_daily, format='png', dpi=300, bbox_inches='tight')
    buf_daily.seek(0)
    daily_plot = Image(buf_daily)
    daily_plot.drawHeight = 3.5*inch
    daily_plot.drawWidth = 6*inch
    plt.close()
    story.append(daily_plot)
    story.append(Spacer(1, 20))

    # Add Drawdown Analysis
    story.append(Paragraph("Drawdown Analysis", styles['Heading2']))
    drawdown_stats, drawdowns = drawdown_analysis(consolidated_df)
    
    # Drawdown plot
    plt.figure(figsize=(10, 6))
    drawdowns.plot()
    plt.title('Drawdown Over Time')
    plt.xlabel('Trade Number')
    plt.ylabel('Drawdown ($)')
    plt.grid(True)
    buf_dd = io.BytesIO()
    plt.savefig(buf_dd, format='png', dpi=300, bbox_inches='tight')
    buf_dd.seek(0)
    dd_plot = Image(buf_dd)
    dd_plot.drawHeight = 3.5*inch
    dd_plot.drawWidth = 6*inch
    plt.close()
    story.append(dd_plot)
    story.append(Spacer(1, 20))

    # Add drawdown statistics table
    drawdown_table_data = [
        ['Metric', 'Value'],
        ['Maximum Drawdown', f"${drawdown_stats['Max Drawdown']:.2f}"],
        ['Average Drawdown', f"${drawdown_stats['Average Drawdown']:.2f}"],
        ['Number of Drawdowns', drawdown_stats['Number of Drawdowns']],
        ['Current Drawdown', f"${drawdown_stats['Current Drawdown']:.2f}"]
    ]
    
    drawdown_table = Table(drawdown_table_data)
    drawdown_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 14),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.white),
        ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 12),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    story.append(drawdown_table)
    story.append(Spacer(1, 20))

    # Add Risk/Reward Analysis
    story.append(Paragraph("Risk/Reward Analysis", styles['Heading2']))
    risk_metrics = risk_reward_analysis(consolidated_df)
    
    # Create risk/reward statistics table
    risk_reward_data = [
        ['Metric', 'Value'],
        ['Risk/Reward Ratio', f"{risk_metrics['Risk_Reward_Ratio']:.2f}"],
        ['Sharpe Ratio', f"{risk_metrics['Sharpe_Ratio']:.2f}"]
    ]
    
    risk_reward_table = Table(risk_reward_data)
    risk_reward_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 14),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.white),
        ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 12),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    story.append(risk_reward_table)

    story.append(Paragraph("Trade Duration Analysis", styles['Heading2']))
    
    # Get duration analysis
    duration_stats, additional_stats = duration_analysis(consolidated_df)
    
    # Add duration plots
    buf = create_duration_plots(consolidated_df)
    duration_plot = Image(buf)
    duration_plot.drawHeight = 5*inch
    duration_plot.drawWidth = 8*inch
    story.append(duration_plot)
    story.append(Spacer(1, 30))
    
    # Add duration statistics table
    duration_data = [['Duration', 'Count', 'Avg P/L', 'Total P/L', 'Win Rate %', 'Avg Minutes']]
    
    for idx in duration_stats.index:
        row = [
            idx,
            int(duration_stats.loc[idx, 'Count']),
            f"${duration_stats.loc[idx, 'Avg P/L']:.2f}",
            f"${duration_stats.loc[idx, 'Total P/L']:.2f}",
            f"{duration_stats.loc[idx, 'Win Rate %']:.1f}%",
            f"{duration_stats.loc[idx, 'Avg Minutes']:.1f}"
        ]
        duration_data.append(row)
    
    duration_table = Table(duration_data)
    duration_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.white),
        ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 10),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    
    story.append(duration_table)
    story.append(Spacer(1, 20))
    
    # Add additional statistics table
    additional_data = [['Metric', 'Value']]
    for key, value in additional_stats.items():
        additional_data.append([key, value])
    
    additional_table = Table(additional_data)
    additional_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.white),
        ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 10),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    
    story.append(additional_table)

    story.append(Paragraph("Strategy Consistency Analysis", styles['Heading2']))
    
    # Get consistency analysis
    monthly_stats, consistency_metrics, rolling_metrics = strategy_consistency(consolidated_df)
    
    # Add consistency plots
    buf = create_consistency_plots(consolidated_df, monthly_stats, rolling_metrics)
    consistency_plot = Image(buf)
    consistency_plot.drawHeight = 8*inch
    consistency_plot.drawWidth = 8*inch
    story.append(consistency_plot)
    story.append(Spacer(1, 20))
    
    # Add consistency metrics table
    metrics_data = [['Metric', 'Value']]
    metrics_data.extend([
        ['Profitable Months (%)', f"{consistency_metrics['Profitable_Months']:.1f}%"],
        ['Best Month', f"${consistency_metrics['Best_Month']:.2f}"],
        ['Worst Month', f"${consistency_metrics['Worst_Month']:.2f}"],
        ['Average Monthly Trades', f"{consistency_metrics['Avg_Monthly_Trades']:.1f}"],
        ['Highest Monthly Sharpe', f"{consistency_metrics['Highest_Monthly_Sharpe']:.2f}"],
        ['Lowest Monthly Sharpe', f"{consistency_metrics['Lowest_Monthly_Sharpe']:.2f}"],
        ['Monthly Win Rate Std', f"{consistency_metrics['Monthly_Win_Rate_Std']:.2f}%"],
        ['Max Consecutive Profitable Months', str(consistency_metrics['Consecutive_Profitable_Months'])]
    ])
    
    metrics_table = Table(metrics_data)
    metrics_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.white),
        ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 10),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    
    story.append(metrics_table)
    story.append(Spacer(1, 20))
    
    # Add monthly statistics table
    monthly_data = [['Month', 'Total P/L', 'Trades', 'Win Rate', 'Sharpe']]
    for idx in monthly_stats.index:
        row = [
            str(idx),
            f"${monthly_stats.loc[idx, 'sum']:.2f}",
            int(monthly_stats.loc[idx, 'count']),
            f"{monthly_stats.loc[idx, 'Win_Rate']:.1f}%",
            f"{monthly_stats.loc[idx, 'Sharpe']:.2f}"
        ]
        monthly_data.append(row)


    story.append(Paragraph("Advanced Strategy Analysis", styles['Heading2']))
    story.append(Spacer(1, 20))

    # 1. Market Movement Analysis
    story.append(Paragraph("Market Movement Analysis", styles['Heading3']))
    movement_stats = market_movement_analysis(consolidated_df)
    
    # Create visualization for market movement
    plt.figure(figsize=(10, 6))
    movement_stats['Trade_PL_numeric']['mean'].plot(kind='bar')
    plt.title('Average P/L by Market Movement')
    plt.xlabel('Market Movement Category')
    plt.ylabel('Average P/L ($)')
    plt.xticks(rotation=45)
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
    buf.seek(0)
    plt.close()
    
    story.append(Image(buf, width=6*inch, height=4*inch))
    story.append(Spacer(1, 20))

    # Add market movement statistics table
    movement_data = [['Movement Category', 'Count', 'Avg P/L', 'Win Rate %']]
    for idx in movement_stats.index:
        row = [
            str(idx),
            int(movement_stats.loc[idx, ('Trade_PL_numeric', 'count')]),
            f"${movement_stats.loc[idx, ('Trade_PL_numeric', 'mean')]:.2f}",
            f"{movement_stats.loc[idx, ('Trade_PL_numeric', '<lambda_0>')]:.1f}%"
        ]
        movement_data.append(row)
    
    movement_table = Table(movement_data)
    movement_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.white),
        ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    story.append(movement_table)
    story.append(Spacer(1, 30))

    # 2. Trade Clustering Analysis
    story.append(Paragraph("Trade Clustering Analysis", styles['Heading3']))
    cluster_stats = trade_clustering_analysis(consolidated_df)
    
    # Create visualization for clustering
    plt.figure(figsize=(10, 6))
    plt.plot(cluster_stats.index, cluster_stats['Trade_PL_numeric']['mean'], marker='o')
    plt.title('Average P/L by Trade Cluster')
    plt.xlabel('Cluster Number')
    plt.ylabel('Average P/L ($)')
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
    buf.seek(0)
    plt.close()
    
    story.append(Image(buf, width=6*inch, height=4*inch))
    story.append(Spacer(1, 20))

    # Add clustering statistics table
    cluster_data = [['Cluster', 'Trades', 'Avg P/L', 'Win Rate %', 'Avg Time Between']]
    for idx in cluster_stats.index[:10]:  # Show first 10 clusters
        row = [
            str(idx),
            int(cluster_stats.loc[idx, ('Trade_PL_numeric', 'count')]),
            f"${cluster_stats.loc[idx, ('Trade_PL_numeric', 'mean')]:.2f}",
            f"{cluster_stats.loc[idx, ('Trade_PL_numeric', '<lambda_0>')]:.1f}%",
            str(cluster_stats.loc[idx, ('Time_Between_Trades', 'mean')]).split('.')[0]
        ]
        cluster_data.append(row)
    
    cluster_table = Table(cluster_data)
    cluster_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    story.append(cluster_table)
    story.append(Spacer(1, 30))

    # 3. Recovery Analysis
    story.append(Paragraph("Drawdown Recovery Analysis", styles['Heading3']))
    recovery_stats = recovery_analysis(consolidated_df)
    
    # Create visualization for recovery periods
    plt.figure(figsize=(10, 6))
    plt.scatter(recovery_stats['depth'], recovery_stats['recovery_time'])
    plt.title('Drawdown Depth vs Recovery Time')
    plt.xlabel('Drawdown Depth (%)')
    plt.ylabel('Recovery Time (trades)')
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
    buf.seek(0)
    plt.close()
    
    story.append(Image(buf, width=6*inch, height=4*inch))
    story.append(Spacer(1, 20))

    # 4. Strategy Robustness
    story.append(Paragraph("Strategy Robustness Analysis", styles['Heading3']))
    robustness_score, robustness_components = calculate_robustness_score(consolidated_df)
    
    # Create robustness components table
    robustness_data = [
        ['Component', 'Value'],
        ['Overall Robustness Score', f"{robustness_score:.2f}/100"],
        ['Win Rate', f"{robustness_components['win_rate']:.2%}"],
        ['Profit Factor', f"{robustness_components['profit_factor']:.2f}"],
        ['Win/Loss Ratio', f"{robustness_components['avg_win_loss_ratio']:.2f}"],
        ['Monthly Consistency', f"{robustness_components['monthly_consistency']:.2%}"],
        ['Maximum Drawdown', f"{robustness_components['max_drawdown']:.2%}"]
    ]
    
    robustness_table = Table(robustness_data)
    robustness_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    story.append(robustness_table)
    story.append(Spacer(1, 30))

    # 5. Entry/Exit Efficiency
    story.append(Paragraph("Entry/Exit Efficiency Analysis", styles['Heading3']))
    efficiency_score = entry_exit_efficiency(consolidated_df)
    
    # Create efficiency visualization
    plt.figure(figsize=(10, 6))
    plt.hist(consolidated_df['Entry_Efficiency'], bins=20)
    plt.title('Distribution of Entry/Exit Efficiency')
    plt.xlabel('Efficiency Score')
    plt.ylabel('Frequency')
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
    buf.seek(0)
    plt.close()
    
    story.append(Image(buf, width=6*inch, height=4*inch))
    story.append(Spacer(1, 20))

    # Add efficiency statistics
    efficiency_data = [
        ['Metric', 'Value'],
        ['Average Entry/Exit Efficiency', f"{efficiency_score:.2%}"],
        ['Best Entry Efficiency', f"{consolidated_df['Entry_Efficiency'].max():.2%}"],
        ['Worst Entry Efficiency', f"{consolidated_df['Entry_Efficiency'].min():.2%}"]
    ]
    
    efficiency_table = Table(efficiency_data)
    efficiency_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    story.append(efficiency_table)

    # Add summary and recommendations
    story.append(Spacer(1, 30))
    story.append(Paragraph("Strategy Insights and Recommendations", styles['Heading3']))
    
    insights = [
        f"• Strategy shows highest efficiency in {movement_stats['Trade_PL_numeric']['mean'].idxmax()} market movements",
        f"• Average recovery time from drawdowns: {recovery_stats['recovery_time'].mean():.1f} trades",
        f"• Strategy robustness score of {robustness_score:.1f} indicates {'strong' if robustness_score > 70 else 'moderate' if robustness_score > 50 else 'weak'} performance stability",
        f"• Entry/Exit efficiency of {efficiency_score:.1%} suggests {'excellent' if efficiency_score > 0.7 else 'good' if efficiency_score > 0.5 else 'potential for improvement in'} trade timing"
    ]
    
    for insight in insights:
        story.append(Paragraph(insight, styles['Normal']))
        story.append(Spacer(1, 12))

    
    # story.append(Paragraph(f"Strategy Robustness Score: {robustness_score:.2f}/100", styles['Heading3']))

        # Build PDF
    doc.build(story)