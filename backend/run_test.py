import asyncio
import json
import urllib.request
from datetime import date

async def main():
    req_data = {
        'model_type': 'ensemble',
        'stock_id': 3,
        'interval': 'day',
        'start_date': '2026-04-01',
        'end_date': '2026-04-22',
        'initial_capital': 100000.0,
        'min_confidence': 0.50
    }

    req = urllib.request.Request(
        'http://localhost:8000/api/backtest/run',
        data=json.dumps(req_data).encode('utf-8'),
        headers={'Content-Type': 'application/json'}
    )

    try:
        with urllib.request.urlopen(req) as response:
            result = json.loads(response.read().decode('utf-8'))
            print('Return:', result.get('total_return'))
            print('Trades:', result.get('trades_count'))
            print('Buy hold return:', result.get('buy_hold_return'))
    except Exception as e:
        print('Error:', e)

asyncio.run(main())
