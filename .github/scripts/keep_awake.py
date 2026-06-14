"""Keep the Streamlit Community Cloud app awake with a real browser session.

A plain HTTP request (e.g. curl) does NOT reset Streamlit Community Cloud's
inactivity timer — Streamlit measures activity by genuine *viewer sessions*
(a browser that opens the page and holds its WebSocket connection). So we drive
a headless Chromium to load the app, click the "wake up" button if the app has
already gone to sleep, and hold the session open briefly so it registers as
real activity.
"""

import sys

from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
from playwright.sync_api import sync_playwright

APP_URL = (
    "https://cathay-pacific-sentiment-analysis-jxhdwetuymk7g2ftgq7n7e.streamlit.app"
)

# How long to keep the browser session open so Streamlit registers a viewer.
HOLD_MS = 45_000
# Extra time to let the app spin up after clicking the wake button.
WAKE_SPINUP_MS = 20_000


def main() -> int:
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        page = browser.new_page()
        print(f"Loading {APP_URL}")

        try:
            page.goto(APP_URL, wait_until="domcontentloaded", timeout=60_000)
        except PlaywrightTimeoutError:
            print("ERROR: timed out loading the app.")
            browser.close()
            return 1

        # If the app is asleep, Streamlit shows a "Yes, get this app back up!"
        # button. Clicking it (a real interaction) wakes the app; an HTTP ping
        # cannot do this.
        try:
            wake_button = page.get_by_role(
                "button", name="Yes, get this app back up!"
            )
            if wake_button.count() > 0:
                print("App was asleep — clicking the wake button.")
                wake_button.first.click()
                page.wait_for_timeout(WAKE_SPINUP_MS)
            else:
                print("App is already awake.")
        except Exception as exc:  # best-effort; never fail the run on this
            print(f"Wake-button step skipped: {exc}")

        # Hold the session so Streamlit counts an active viewer and resets the
        # inactivity timer.
        page.wait_for_timeout(HOLD_MS)
        print("Session held — app should be awake.")
        browser.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())
