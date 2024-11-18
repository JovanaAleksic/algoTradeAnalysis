import sys
import pandas as pd
import numpy as np
from datetime import timedelta
from utils import *

input_file = sys.argv[1] 

df = pd.read_csv(input_file, skiprows=3, skipfooter=8, sep=";")
df.drop(columns=['Id', 'Unnamed: 9'], inplace=True)
df['Strategy'] = df['Strategy'].apply(lambda x: x.split("(")[1][:-1])
df['Open_Close'] = df['Side'].apply(lambda x: x.split(" ")[2])
df['Date/Time'] = pd.to_datetime(df['Date/Time'], format='%m/%d/%y, %I:%M %p')
print(df)

trades = []
for i in range(0, len(df), 2):  # Step by 2 to get open/close pairs
    open_trade = df.iloc[i]
    close_trade = df.iloc[i+1]
    
    # Calculate duration
    duration = close_trade['Date/Time'] - open_trade['Date/Time']
    
    trade = {
        'Strategy': open_trade['Strategy'],
        'Entry_Side': open_trade['Side'],
        'Exit_Side': close_trade['Side'],
        'Amount': open_trade['Amount'],
        'Entry_Price': open_trade['Price'].replace('$',''),
        'Exit_Price': close_trade['Price'].replace('$',''),
        'Entry_Time': open_trade['Date/Time'],
        'Exit_Time': close_trade['Date/Time'],
        'Duration': duration,
        'Trade_PL': close_trade['Trade P/L'],
        'Cumulative_PL': close_trade['P/L']
    }
    trades.append(trade)

# Create consolidated dataframe
consolidated_df = pd.DataFrame(trades)

# Convert prices to numeric 
consolidated_df['Entry_Price'] = pd.to_numeric(consolidated_df['Entry_Price'])
consolidated_df['Exit_Price'] = pd.to_numeric(consolidated_df['Exit_Price'])

# Convert Trade_PL to numeric, handling commas and currency formatting
consolidated_df['Trade_PL_numeric'] = consolidated_df['Trade_PL'].replace({'\$':'','\(':'-','\)':'',',':''}, regex=True).astype(float)

# Cumulative P/L over time
consolidated_df['Cumulative_PL_numeric'] = consolidated_df['Cumulative_PL'].replace({'\$': '','\(': '-','\)': '',',': ''}, regex=True).astype(float)

# #########################################################

# Generate the report
create_trading_report(consolidated_df)