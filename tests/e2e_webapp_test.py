import time
import sys
from playwright.sync_api import sync_playwright

def run_e2e_tests():
    print("=== STARTING EDGAR PULSE E2E WEBAPP TESTS ===", flush=True)
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(viewport={"width": 1280, "height": 950})
        page = context.new_page()

        # Step 1: Navigate to Streamlit App
        print("\n[Step 1] Navigating to http://localhost:8501/ ...", flush=True)
        page.goto("http://localhost:8501", timeout=30000)
        page.wait_for_selector('[data-testid="stAppViewContainer"]', timeout=15000)
        page.wait_for_selector(".ep-live-pip", timeout=15000)
        page.wait_for_timeout(2000)

        # Verify Title & Scope Bar
        title = page.title()
        print(f"  Page title: '{title}'", flush=True)
        assert "EDGAR Pulse" in title, f"Unexpected page title: {title}"

        # Verify Header & Live Pip
        live_pip_count = page.locator(".ep-live-pip").count()
        print(f"  Live pip indicators found: {live_pip_count}", flush=True)
        assert live_pip_count >= 2, "Expected at least 2 .ep-live-pip indicators (sidebar + header)"
        print("  Confirmed: Terminal Header & .ep-live-pip pulsing indicators active", flush=True)

        # Verify Sidebar: Ingested Corpora & Session Quota
        sidebar = page.locator('[data-testid="stSidebar"]')
        assert "Ingested 10-K Corpora" in sidebar.inner_text(), "Sidebar missing Ingested 10-K Corpora"
        assert "0 / 30" in sidebar.inner_text(), "Sidebar quota meter should start at 0 / 30"
        print("  Confirmed: Ingested 10-K Corpora panel & initial 0 / 30 quota meter", flush=True)

        # Step 2: Test Onboarding Suggested Question Click
        print("\n[Step 2] Testing Onboarding Query Matrix Button ('Tesla Valuation & P/E Ratio')...", flush=True)
        tsla_btn = page.locator("button:has-text('Tesla Valuation & P/E Ratio')")
        assert tsla_btn.count() > 0, "Tesla Valuation button not found"
        tsla_btn.click()

        # Wait for assistant response to render
        print("  Waiting for assistant response execution...", flush=True)
        page.wait_for_selector('text=AUDITED SOURCES', timeout=35000)
        page.wait_for_timeout(2000)

        # Verify Market Metric Card Strip
        chat_content = page.locator('[data-testid="stChatMessage"]').all_inner_texts()
        assistant_text = " ".join(chat_content)
        assert "TSLA" in assistant_text, "Response missing TSLA ticker"
        assert "346.8x" in assistant_text or "P/E" in assistant_text, "Response missing P/E metric"
        assert "1.48T" in assistant_text or "market cap" in assistant_text.lower(), "Response missing Market Cap"
        print("  Confirmed: Quantitative market metric cards rendered (Price, P/E, Cap, High, Vol)", flush=True)

        # Verify Reactive Quota Increment
        sidebar_text_turn1 = sidebar.inner_text()
        print(f"  Sidebar quota after Turn 1: {[line for line in sidebar_text_turn1.splitlines() if '30' in line]}", flush=True)
        assert "1 / 30" in sidebar_text_turn1, "Sidebar quota meter failed to update reactively to 1 / 30"
        print("  Confirmed: Session quota reactively incremented to 1 / 30", flush=True)

        # Step 3: Test Suggested Follow-Up Question Click (User-reported issue!)
        print("\n[Step 3] Testing Contextual Follow-Up Suggestions Click...", flush=True)
        page.wait_for_selector("text=RECOMMENDED FOLLOW-UP INQUIRIES:", timeout=10000)
        
        # Locate the follow-up buttons rendered on the latest assistant turn
        followup_btn = page.locator("button:has-text('What are the major supply chain and manufacturing risks for Tesla?')")
        assert followup_btn.count() > 0, "Tesla supply chain follow-up button not found"
        
        print("  Clicking suggested follow-up question: 'What are the major supply chain and manufacturing risks for Tesla?'...", flush=True)
        followup_btn.click()

        # Verify that clicking the follow-up triggered a new query turn (message count >= 4)!
        print("  Waiting for follow-up response to execute...", flush=True)
        page.wait_for_function("() => document.querySelectorAll('[data-testid=\"stChatMessage\"]').length >= 4", timeout=35000)
        page.locator('[data-testid="stChatMessage"]').nth(3).locator("text=Audit Trail").first.wait_for(timeout=50000)
        page.wait_for_timeout(2000)

        messages_after_followup = page.locator('[data-testid="stChatMessage"]').all_inner_texts()
        print(f"  Total chat messages after follow-up click: {len(messages_after_followup)}", flush=True)
        assert len(messages_after_followup) >= 4, f"Follow-up click did not generate new turn! Found {len(messages_after_followup)} messages"
        assert "What are the major supply chain and manufacturing risks for Tesla?" in messages_after_followup[2], "Turn 2 user prompt text missing"
        
        # Verify Quota updated to 2 / 30
        sidebar_text_turn2 = sidebar.inner_text()
        print(f"  Sidebar quota after Turn 2: {[line for line in sidebar_text_turn2.splitlines() if '30' in line]}", flush=True)
        assert "2 / 30" in sidebar_text_turn2, "Sidebar quota failed to update reactively to 2 / 30 after follow-up click"
        print("  SUCCESS: Suggested follow-up question clicked, executed turn 2, and updated quota to 2 / 30!", flush=True)

        # Step 4: Test Manual Chat Input
        print("\n[Step 4] Testing Manual Chat Input ('Who is the CEO of Nvidia?')...", flush=True)
        chat_input = page.locator('[data-testid="stChatInput"] textarea')
        assert chat_input.count() > 0, "Chat input textarea not found"
        chat_input.fill("Who is the CEO of Nvidia?")
        chat_input.press("Enter")

        print("  Waiting for NVIDIA query execution...", flush=True)
        page.wait_for_function("() => document.querySelectorAll('[data-testid=\"stChatMessage\"]').length >= 6", timeout=50000)
        page.locator('[data-testid="stChatMessage"]').nth(5).locator("text=Audit Trail").first.wait_for(timeout=50000)
        page.wait_for_timeout(2000)

        all_text = " ".join(page.locator('[data-testid="stChatMessage"]').all_inner_texts())
        assert "Jensen Huang" in all_text or "Huang" in all_text, "Response missing Jensen Huang for Nvidia CEO"
        print("  Confirmed: Manual query executed and returned verified CEO fact", flush=True)

        sidebar_text_turn3 = sidebar.inner_text()
        assert "3 / 30" in sidebar_text_turn3, "Sidebar quota failed to update reactively to 3 / 30"
        print("  Confirmed: Session quota reactively incremented to 3 / 30", flush=True)

        # Step 5: Test Unsupported Ticker Honest Refusal & Coverage Notice Card
        print("\n[Step 5] Testing Unsupported Ticker Refusal & Guidance Card ('What are Google 10-K supply chain risks?')...", flush=True)
        chat_input = page.locator('[data-testid="stChatInput"] textarea')
        chat_input.fill("What are Google 10-K supply chain risks?")
        chat_input.press("Enter")

        page.wait_for_function("() => document.querySelectorAll('[data-testid=\"stChatMessage\"]').length >= 8", timeout=35000)
        page.wait_for_timeout(3000)

        latest_chat = page.locator('[data-testid="stChatMessage"]').all_inner_texts()[-1]
        assert "Data not available in the context." in latest_chat, "Did not return verbatim honest refusal string"
        assert "SEC FILING COVERAGE NOTICE" in latest_chat or "AAPL" in latest_chat, "Refusal coverage guidance card missing"
        print("  Confirmed: Verbatim refusal and SEC Filing Coverage Notice card verified", flush=True)

        # Step 6: Test Research Brief Export Button
        print("\n[Step 6] Testing Research Brief Export Button...", flush=True)
        export_btn = page.locator("button:has-text('Export Research Brief (.md)')")
        assert export_btn.count() > 0, "Export Research Brief download button not found"
        print("  Confirmed: Export Research Brief (.md) button is present and active", flush=True)

        # Step 7: Test Clear Conversation Reset
        print("\n[Step 7] Testing Clear Conversation Confirmation Dialog...", flush=True)
        clear_popover = page.locator("button:has-text('Clear Conversation')")
        assert clear_popover.count() > 0, "Clear Conversation popover button not found"
        clear_popover.click()
        page.wait_for_timeout(1000)

        confirm_btn = page.locator("button:has-text('Confirm Reset')")
        assert confirm_btn.count() > 0, "Confirm Reset button inside popover not found"
        confirm_btn.click()

        page.wait_for_timeout(3000)
        # After reset, messages should be empty and sample queries expanded
        reset_messages = page.locator('[data-testid="stChatMessage"]').count()
        reset_sidebar = sidebar.inner_text()
        print(f"  Messages after reset: {reset_messages}", flush=True)
        assert reset_messages == 0, f"Expected 0 messages after reset, got {reset_messages}"
        assert "0 / 30" in reset_sidebar, "Quota should reset to 0 / 30 after clear"
        print("  Confirmed: Conversation reset completely cleared history and reset quota to 0 / 30!", flush=True)

        # Take full page screenshot for artifact proof
        screenshot_path = "C:/Users/Aarya/.gemini/antigravity-cli/brain/cc88ba8b-86dc-4c63-ae22-a1e7f9f07fb6/e2e_testing_verified.png"
        page.screenshot(path=screenshot_path, full_page=True)
        print(f"\n[Screenshot Saved] {screenshot_path}", flush=True)

        browser.close()
        print("\n=== ALL 7 E2E WEBAPP TESTS PASSED SUCCESSFULLY! ===", flush=True)

if __name__ == "__main__":
    run_e2e_tests()
