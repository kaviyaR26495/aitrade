import requests
import json

URL = "http://localhost:8000/api/pipeline"

symbols = [
        "ADANIENT",
        "ADANIPORTS",
        "APOLLOHOSP",
        "ASIANPAINT",
        "AXISBANK",
        "BAJAJ-AUTO",
        "BAJAJFINSV",
        "BAJFINANCE",
        "BEL",
        "BHARTIARTL",
        "CIPLA",
        "COALINDIA",
        "DRREDDY",
        "EICHERMOT",
        "ETERNAL",
        "GRASIM",
        "HCLTECH",
        "HDFCBANK",
        "HDFCLIFE",
        "HINDALCO",
        "HINDUNILVR",
        "ICICIBANK",
        "INDIGO",
        "INFY",
        "ITC",
        "JIOFIN",
        "JSWSTEEL",
        "KOTAKBANK",
        "LT",
        "M&M",
        "MARUTI",
        "MAXHEALTH",
        "NESTLEIND",
        "NTPC",
        "ONGC",
        "POWERGRID",
        "RELIANCE",
        "SBILIFE",
        "SBIN",
        "SHRIRAMFIN",
        "SUNPHARMA",
        "TATACONSUM",
        "TATASTEEL",
        "TCS",
        "TECHM",
        "TITAN",
        "TMPV",
        "TRENT",
        "ULTRACEMCO",
        "WIPRO"
    ]

payload = {
    "symbols": symbols,
    "skip_sync": True,
    "force_sync": False,
    "use_regime_pooling": True,
    "resume_job_id": "3e427621-c58a-4d91-988f-d89a96adc8a7"
}

r = requests.post(f"{URL}/start", json=payload)
print(r.status_code, r.text)
if r.ok:
    job = r.json()
    print("New job id:", job["job_id"])
