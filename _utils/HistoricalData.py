#!/usr/bin/env python3
"""
Streamlined CLI tool for fetching historical crypto data to CSV
Usage: python3 getHistorical.py -s BTCUSDT -i 60 -t 7d
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

    def get_klines(self, symbol: str, interval: str, start_time: int, end_time: int, limit: int = 1000) -> list[list[str]]:
        """Fetch kline/candlestick data from Bybit (public endpoint)"""
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
            raise Exception(f"API request failed: {response.status_code} - {response.text}")
        
        data = response.json()
        if data.get("retCode") != 0:
            raise Exception(f"API error: {data.get('retMsg', 'Unknown error')}")
        
        return data.get("result", {}).get("list", [])


def parse_timeframe(timeframe: str) -> timedelta:
    """Parse timeframe string to timedelta object"""
    timeframe = timeframe.lower()
    
    if timeframe.endswith('d'):
        days = int(timeframe[:-1])
        return timedelta(days=days)
    elif timeframe.endswith('h'):
        hours = int(timeframe[:-1])
        return timedelta(hours=hours)
    elif timeframe.endswith('m'):
        minutes = int(timeframe[:-1])
        return timedelta(minutes=minutes)
    elif timeframe.endswith('w'):
        weeks = int(timeframe[:-1])
        return timedelta(weeks=weeks)
    else:
        raise ValueError(f"Invalid timeframe format: {timeframe}. Use format like '7d', '24h', '60m', '1w'")


def validate_interval(interval: str) -> bool:
    """Validate if interval is supported by Bybit"""
    valid_intervals = ['1', '3', '5', '15', '30', '60', '120', '240', '360', '720', 'D', 'W', 'M']
    return interval in valid_intervals


def fetch_historical_data(symbol: str, interval: str, timeframe: str) -> list[list[str]]:
    """Fetch historical data for given parameters"""
    if not validate_interval(interval):
        raise ValueError(f"Invalid interval: {interval}. Valid intervals: 1,3,5,15,30,60,120,240,360,720,D,W,M")
    
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
            time.sleep(0.5)  # Increased sleep for rate limiting
            
        except Exception as e:
            print(f"Error fetching data: {e}")
            break
    
    # Remove duplicates and sort by timestamp (ascending)
    unique_data = {int(item[0]): item for item in all_data}
    sorted_data = sorted(unique_data.values(), key=lambda x: int(x[0]))
    
    return sorted_data


def save_to_csv(data: list[list[str]], symbol: str, interval: str, timeframe: str) -> None:
    """Save data to CSV file"""
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
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 getHistorical.py -s BTCUSDT -i 60 -t 7d
  python3 getHistorical.py -s ADAUSDT -i D -t 30d
  python3 getHistorical.py -s ADAUSDT -i 15 -t 24h

Intervals: 1,3,5,15,30,60,120,240,360,720,D,W,M (minutes or D/W/M)
Timeframes: 1h, 24h, 7d, 30d, 1w, etc.
        """
    )
    
    parser.add_argument('-s', '--symbol', required=True, 
                        help='Trading symbol (e.g., BTCUSDT)')
    parser.add_argument('-i', '--interval', required=True,
                        help='Kline interval (1,3,5,15,30,60,120,240,360,720,D,W,M)')
    parser.add_argument('-t', '--timeframe', required=True,
                        help='Time period to fetch (e.g., 7d, 24h, 1w)')
    
    args = parser.parse_args()
    
    try:
        data = fetch_historical_data(args.symbol, args.interval, args.timeframe)
        
        if not data:
            print("No data found for the specified parameters")
            sys.exit(1)
        
        save_to_csv(data, args.symbol, args.interval, args.timeframe)
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
