import logging
from datetime import datetime
import pandas as pd

logger = logging.getLogger(__name__)

def fetch_historical_data(symbol: str, from_date: datetime, to_date: datetime, interval="day") -> list[dict]:
    import yfinance as yf
    
    yf_symbol = symbol + ".NS"
    
    if interval == "day":
        yf_interval = "1d"
    elif interval == "week":
        yf_interval = "1wk"
    elif interval == "minute" or interval == "1m":
        yf_interval = "1m"
    else:
        yf_interval = "1d"
    
    try:
        logger.info("Fetching yfinance for %s from %s to %s (interval=%s)", yf_symbol, from_date, to_date, yf_interval)
        df = yf.download(yf_symbol, start=from_date, end=to_date, interval=yf_interval, progress=False)
        if df.empty:
            return []
            
        df = df.reset_index()
        
        # Determine date column
        date_col = None
        for col in df.columns:
            if isinstance(col, tuple):
                if col[0] in ["Date", "Datetime"]:
                    date_col = col
                    break
            else:
                if col in ["Date", "Datetime"]:
                    date_col = col
                    break
                    
        if not date_col:
            date_col = df.columns[0]
            
        records = []
        for _, row in df.iterrows():
            def get_val(name):
                for c in df.columns:
                    if isinstance(c, tuple) and c[0] == name:
                        return row[c]
                    elif not isinstance(c, tuple) and c == name:
                        return row[c]
                return 0.0

            dt_val = row[date_col]
            records.append({
                "date": dt_val.to_pydatetime() if hasattr(dt_val, "to_pydatetime") else dt_val,
                "open": float(get_val("Open")),
                "high": float(get_val("High")),
                "low": float(get_val("Low")),
                "close": float(get_val("Close")),
                "volume": float(get_val("Volume")),
            })
        return records
    except Exception as e:
        logger.error("Error fetching yfinance history for %s: %s", symbol, e)
        return []
