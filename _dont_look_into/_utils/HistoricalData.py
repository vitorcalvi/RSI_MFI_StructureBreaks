#!/usr/bin/env python3
"""
Streamlined CLI tool for fetching historical crypto data to CSV
Usage: python3 HistoricalData.py -s BTCUSDT -i 60 -t 7d
"""

import argparse
import csv
import os
import sys
from datetime import datetime, timedelta
import requests
import time

class BybitClient:
    def __init__(self):
        self.base_url = "https://api.bybit.com"

    def get_klines(self, symbol: str, interval: str, start_time: int, end_time: int, limit: int = 1000) -> list:
        """Fetch kline data from Bybit"""
        endpoint = "/v5/market/kline"
        
        params = {
            "category": "spot",
            "symbol": symbol,
            "interval": interval,
            "start": start_time,
            "end": end_time,
            "limit": limit
        }
        
        response = requests.get(f"{self.base_url}{endpoint}", params=params)
        
        if response.status_code != 200:
            raise Exception(f"API request failed: {response.status_code}")
        
        data = response.json()
        if data.get("retCode") != 0:
            raise Exception(f"API error: {data.get('retMsg', 'Unknown error')}")
        
        return data.get("result", {}).get("list", [])

def parse_timeframe(timeframe: str) -> timedelta:
    """Parse timeframe string to timedelta"""
    timeframe = timeframe.lower()
    
    if timeframe.endswith('d'):
        return timedelta(days=int(timeframe[:-1]))
    elif timeframe.endswith('h'):
        return timedelta(hours=int(timeframe[:-1]))
    elif timeframe.endswith('m'):
        return timedelta(minutes=int(timeframe[:-1]))
    elif timeframe.endswith('w'):
        return timedelta(weeks=int(timeframe[:-1]))
    else:
        raise ValueError(f"Invalid timeframe: {timeframe}")

def fetch_historical_data(symbol: str, interval: str, timeframe: str) -> list:
    """Fetch historical data"""
    valid_intervals = ['1', '3', '5', '15', '30', '60', '120', '240', '360', '720', 'D', 'W', 'M']
    if interval not in valid_intervals:
        raise ValueError(f"Invalid interval: {interval}")
    
    end_time = datetime.now()
    start_time = end_time - parse_timeframe(timeframe)
    
    start_timestamp = int(start_time.timestamp() * 1000)
    end_timestamp = int(end_time.timestamp() * 1000)
    
    print(f"Fetching {symbol} data from {start_time.strftime('%Y-%m-%d %H:%M:%S')} to {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    client = BybitClient()
    all_data = []
    current_end = end_timestamp
    
    while current_end > start_timestamp:
        try:
            klines = client.get_klines(symbol, interval, start_timestamp, current_end)
            
            if not klines:
                break
                
            filtered_klines = [k for k in klines if int(k[0]) >= start_timestamp]
            all_data.extend(filtered_klines)
            
            if len(klines) < 1000:
                break
                
            current_end = int(klines[-1][0]) - 1
            time.sleep(0.5)
            
        except Exception as e:
            print(f"Error fetching data: {e}")
            break
    
    # Remove duplicates and sort
    unique_data = {int(item[0]): item for item in all_data}
    return sorted(unique_data.values(), key=lambda x: int(x[0]))

def save_to_csv(data: list, symbol: str, interval: str, timeframe: str) -> None:
    """Save data to CSV"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"data/{symbol}_{interval}_{timeframe}_{timestamp}.csv"
    os.makedirs("data", exist_ok=True)
    
    headers = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover']
    
    with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(headers)
        
        for row in data:
            dt = datetime.fromtimestamp(int(row[0]) / 1000)
            formatted_row = [
                dt.strftime('%Y-%m-%d %H:%M:%S'),
                float(row[1]),  # open
                float(row[2]),  # high
                float(row[3]),  # low
                float(row[4]),  # close
                float(row[5]),  # volume
                float(row[6])   # turnover
            ]
            writer.writerow(formatted_row)
    
    print(f"Data saved to: {filename}")
    print(f"Total records: {len(data)}")

def main():
    parser = argparse.ArgumentParser(
        description="Fetch historical crypto data to CSV",
        epilog="""
Examples:
  python3 HistoricalData.py -s BTCUSDT -i 60 -t 7d
  python3 HistoricalData.py -s ADAUSDT -i D -t 30d
  python3 HistoricalData.py -s ADAUSDT -i 15 -t 24h

Intervals: 1,3,5,15,30,60,120,240,360,720,D,W,M
Timeframes: 1h, 24h, 7d, 30d, 1w, etc.
        """
    )
    
    parser.add_argument('-s', '--symbol', required=True, help='Trading symbol (e.g., BTCUSDT)')
    parser.add_argument('-i', '--interval', required=True, help='Kline interval')
    parser.add_argument('-t', '--timeframe', required=True, help='Time period (e.g., 7d, 24h, 1w)')
    
    args = parser.parse_args()
    
    try:
        data = fetch_historical_data(args.symbol, args.interval, args.timeframe)
        
        if not data:
            print("No data found")
            sys.exit(1)
        
        save_to_csv(data, args.symbol, args.interval, args.timeframe)
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()