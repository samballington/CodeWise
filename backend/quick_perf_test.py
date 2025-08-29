#!/usr/bin/env python3
import asyncio
import time
try:
    from smart_search import smart_search
except ImportError:
    from backend.smart_search import smart_search

async def quick_test():
    start = time.time()
    result = await smart_search('system architecture overview', k=10, mentioned_projects=['SWE_Project'])
    end = time.time()
    print(f'Architecture query time: {end-start:.2f}s')
    print(f'Results: {len(result["results"])}')

asyncio.run(quick_test())