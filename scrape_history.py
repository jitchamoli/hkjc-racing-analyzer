"""Backwards-compatible shim — import from data.scraper."""
from data.scraper import *  # noqa: F401,F403

if __name__ == '__main__':
    import asyncio
    from data.scraper import main
    asyncio.run(main())
