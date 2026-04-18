"""Entry point: ``python -m lexy_core``."""

import asyncio

from lexy_core.app import main

if __name__ == "__main__":
    asyncio.run(main())
