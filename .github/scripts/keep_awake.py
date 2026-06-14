"""Keep the Streamlit Community Cloud app awake with a real browser session.

A plain HTTP request (e.g. curl) does NOT reset Streamlit Community Cloud's
inactivity timer — Streamlit measures activity by genuine *viewer sessions*
(a browser that opens the page and holds its WebSocket connection). So we drive
a headless Chromium to load the app, and:

  * if the app is asleep, click the "Yes, get this app back up!" button and
    wait for the real app to render;
  * then hold the session open briefly so it registers as real activity.

The script logs verbosely (unbuffered) and exits non-zero if it detects the
sleep page but cannot confirm the app woke up, so a failure shows as a red run
instead of a misleading green one.
"""

import re
import sys

from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
from playwright.sync_api import sync_playwright

APP_URL = (
    "https://cathay-pacific-sentiment-analysis-jxhdwetuymk7g2ftgq7n7e.streamlit.app"
)

# How long to keep the browser session open so Streamlit registers a viewer.
HOLD_MS = 60_000
# Max time to wait for the real app to render after waking it.
WAKE_TIMEOUT_MS = 150_000

# Text that only appears on Streamlit's "this app has gone to sleep" page.
SLEEP_TEXT = re.compile("gone to sleep", re.IGNORECASE)
WAKE_BUTTON_TEXT = re.compile("get this app back up", re.IGNORECASE)


def log(message: str) -> None:
    print(message, flush=True)


def is_asleep(page) -> bool:
    """True if the Streamlit sleep page is showing."""
    try:
        if page.get_by_text(SLEEP_TEXT).count() > 0:
            return True
        if page.get_by_text(WAKE_BUTTON_TEXT).count() > 0:
            return True
    except Exception as exc:
        log(f"Sleep-detection check raised: {exc}")
    return False


def click_wake_button(page) -> bool:
    """Try several locators to click the wake button. Returns True if clicked."""
    candidates = [
        page.get_by_role("button", name=WAKE_BUTTON_TEXT),
        page.get_by_text(WAKE_BUTTON_TEXT),
        page.locator("button:has-text('get this app back up')"),
    ]
    for locator in candidates:
        try:
            if locator.count() > 0:
                locator.first.click(timeout=15_000)
                return True
        except Exception as exc:
            log(f"  wake-button candidate failed: {exc}")
    return False


def wait_for_app(page) -> bool:
    """Wait for the real Streamlit app container to render."""
    try:
        page.wait_for_selector(
            '[data-testid="stAppViewContainer"], [data-testid="stSidebar"]',
            timeout=WAKE_TIMEOUT_MS,
        )
        return True
    except PlaywrightTimeoutError:
        return False


def main() -> int:
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        page = browser.new_page()
        log(f"Loading {APP_URL}")

        try:
            page.goto(APP_URL, wait_until="domcontentloaded", timeout=60_000)
        except PlaywrightTimeoutError:
            log("ERROR: timed out loading the app.")
            browser.close()
            return 1

        # Give the page a moment to settle so the sleep screen (if any) renders.
        page.wait_for_timeout(5_000)
        log(f"Page title: {page.title()!r}")
        log(f"Current URL: {page.url}")

        exit_code = 0
        if is_asleep(page):
            log("App is ASLEEP — attempting to wake it.")
            if not click_wake_button(page):
                log("ERROR: sleep page detected but no wake button could be clicked.")
                browser.close()
                return 1
            log("Clicked the wake button — waiting for the app to render...")
            if wait_for_app(page):
                log("App rendered — wake confirmed.")
            else:
                log("ERROR: app did not render within the timeout after waking.")
                exit_code = 1
        else:
            # Confirm we're actually looking at the app, not an error page.
            if wait_for_app(page):
                log("App is already awake.")
            else:
                log("WARNING: app container not found and no sleep page detected.")
                exit_code = 1

        # Hold the session so Streamlit counts an active viewer.
        log(f"Holding session for {HOLD_MS // 1000}s...")
        page.wait_for_timeout(HOLD_MS)
        log("Session held — done.")
        browser.close()

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
