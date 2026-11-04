"""Real Chromium comparison UI and portable result export, without external services."""

import argparse
import threading
from pathlib import Path

from atomic_model_project_ui.server import create_server
from atomic_model_results import bundle
from atomic_model_results.backend import Backend
from atomic_model_workspace.common import atomic_json
from examples.eos_results_workflow import create_runs


def main():
    from playwright.sync_api import expect, sync_playwright

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--references", type=Path, default=Path("examples/eos_reference"))
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    folder, states = create_runs(a.references, a.output / "workflow")
    backend = Backend(folder)
    server = create_server(folder, backend=backend, exports=a.output / "exports")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    errors = []
    try:
        with sync_playwright() as api:
            browser = api.chromium.launch(headless=True, args=["--no-sandbox"])
            page = browser.new_page(viewport=dict(width=1280, height=1000))
            page.set_default_timeout(90000)
            page.on("pageerror", lambda error: errors.append(str(error)))
            page.goto(f"http://127.0.0.1:{server.server_port}", wait_until="networkidle")

            def ready():
                expect(page.locator("main")).to_have_attribute("aria-busy", "false", timeout=90000)

            ready()
            expect(page.locator("#results-panel")).to_be_visible()
            assert page.locator("#results-selection option").count() == 4
            page.select_option("#results-selection", [s["evaluation_id"] for s in states.values()])
            page.click("#results-compare")
            ready()
            expect(page.locator("#results-output table")).to_have_count(2)
            assert page.locator("#results-output tbody tr").count() == 4
            for cell in page.locator("#results-output tbody tr td:last-child").all():
                displayed = cell.inner_text()
                original = float(cell.get_attribute("title"))
                assert len(displayed) <= 10
                assert abs(float(displayed) - original) <= max(1e-30, abs(original) * 0.051)
            page.locator("#results-panel").scroll_into_view_if_needed()
            page.screenshot(path=str(a.output / "results-desktop.png"))
            page.click("#results-export")
            ready()
            expect(page.locator("#results-message")).to_contain_text("bundle.json")
            exported = next((a.output / "exports").glob("results-*/bundle.json"))
            summary = bundle.verify(exported.parent)
            assert len(summary["groups"]) == 2
            page.set_viewport_size(dict(width=390, height=844))
            page.locator("#results-output").scroll_into_view_if_needed()
            assert page.evaluate("document.documentElement.scrollWidth<=innerWidth+1")
            page.screenshot(path=str(a.output / "results-mobile.png"))
            assert not errors, errors
            atomic_json(
                a.output / "browser_proof.json",
                dict(
                    schema_version=1,
                    browser=browser.version,
                    errors=errors,
                    rows=4,
                    compact_metric_display=True,
                    raw_precision_retained_in_tooltips=True,
                    protocol_groups=2,
                    export_verified=True,
                    mobile_no_overflow=True,
                    validated=False,
                    plant_referenced=False,
                ),
            )
            browser.close()
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
        backend.close()


if __name__ == "__main__":
    main()
