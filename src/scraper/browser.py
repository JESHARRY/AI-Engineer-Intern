"""Playwright browser lifecycle and context manager."""

import logging
from typing import Optional
from playwright.async_api import (
    async_playwright,
    Browser,
    BrowserContext,
    Page,
    Playwright,
)
from config import get_settings

logger = logging.getLogger(__name__)


class BrowserManager:
    """Manages Playwright browser instance and context lifecycles cleanly."""

    def __init__(
        self,
        headless: bool = True,
        timeout_seconds: Optional[int] = None,
        user_agent: Optional[str] = None,
    ):
        settings = get_settings()
        self.headless = headless
        self.timeout_seconds = timeout_seconds or settings.request_timeout_seconds
        self.timeout_ms = self.timeout_seconds * 1000
        self.user_agent = user_agent or (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        )
        self._playwright: Optional[Playwright] = None
        self._browser: Optional[Browser] = None
        self._context: Optional[BrowserContext] = None

    async def start(self) -> None:
        """Launch Chromium browser and establish isolated browser context."""
        if not self._playwright:
            self._playwright = await async_playwright().start()
            self._browser = await self._playwright.chromium.launch(
                headless=self.headless
            )
            self._context = await self._browser.new_context(
                user_agent=self.user_agent,
                viewport={"width": 1280, "height": 800},
                ignore_https_errors=True,
            )
            self._context.set_default_navigation_timeout(self.timeout_ms)
            self._context.set_default_timeout(self.timeout_ms)
            logger.debug("Playwright Chromium browser and context started.")

    async def get_page(self) -> Page:
        """Create and return a new Page in the active browser context."""
        if not self._context:
            await self.start()
        assert self._context is not None
        return await self._context.new_page()

    async def close(self) -> None:
        """Close browser context, browser instance, and stop Playwright cleanly."""
        if self._context:
            try:
                await self._context.close()
            except Exception as e:
                logger.warning(f"Error closing browser context: {e}")
            self._context = None

        if self._browser:
            try:
                await self._browser.close()
            except Exception as e:
                logger.warning(f"Error closing browser instance: {e}")
            self._browser = None

        if self._playwright:
            try:
                await self._playwright.stop()
            except Exception as e:
                logger.warning(f"Error stopping Playwright: {e}")
            self._playwright = None

        logger.debug("Playwright session terminated cleanly.")

    async def __aenter__(self) -> "BrowserManager":
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.close()
