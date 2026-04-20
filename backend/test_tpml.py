import asyncio
from app.db.database import async_session_factory
from app.ml.predictor import run_target_price_predictions

async def run():
    async with async_session_factory() as session:
        res = await run_target_price_predictions(session, pop_threshold=0.55)
        print(f"Total stocks: {res['total_stocks']}")
        print(f"Created: {res['signals_created']}")
        print(f"Rejected: {res['signals_rejected']}")
        print(f"Errors: {res['errors']}")
        
        # Count rejection reasons
        reasons = {}
        for r in res['rejected']:
            rsn = r['reason'].split()[0] if 'reason' in r else 'Unknown'
            reasons[rsn] = reasons.get(rsn, 0) + 1
            if rsn.startswith('PoP'):
                pass
        
        for k, v in sorted(reasons.items(), key=lambda x: -x[1])[:10]:
            print(f"  {k}: {v}")
            
        if res['errors']:
            print("First error:", res['error_details'][0])

asyncio.run(run())
