"""Optional real Chromium study workflow, including import, curves and stale results."""

import argparse
import json
import threading
from pathlib import Path

from atomic_model_project_ui.server import create_server
from atomic_model_study import store
from atomic_model_study.backend import Backend
from atomic_model_workspace.common import atomic_json
from examples.eos_study_workflow import create_fixture


def main():
    from playwright.sync_api import expect, sync_playwright

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--references", type=Path, default=Path("examples/eos_reference"))
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    project = create_fixture(a.references, a.output / "input")
    folder = a.output / "study"
    store.create(project, folder)
    backend = Backend(folder)
    server = create_server(folder, backend=backend, exports=a.output / "exports")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    errors = []

    def ready(page):
        expect(page.locator("main")).to_have_attribute("aria-busy", "false")

    try:
        with sync_playwright() as api:
            browser = api.chromium.launch(headless=True, args=["--no-sandbox"])
            page = browser.new_page(viewport=dict(width=1280, height=1000))
            page.set_default_timeout(30000)
            page.on("pageerror", lambda error: errors.append(str(error)))
            page.goto(f"http://127.0.0.1:{server.server_port}", wait_until="networkidle")
            ready(page)
            expect(page.locator("#study-panel")).to_be_visible()
            for id, metadata, csv in [
                ("energy", "energy_reference.json", "energy.csv"),
                ("thermal", "reference.json", "constant.csv"),
            ]:
                page.fill("#data-id", id)
                page.set_input_files("#data-declaration", a.references / metadata)
                page.set_input_files("#data-csv", a.references / csv)
                page.click("#data-preview")
                ready(page)
                expect(page.locator("#data-attach")).to_be_enabled()
                # Changing the label invalidates the admitted upload.
                page.fill("#data-id", id + "-changed")
                expect(page.locator("#data-attach")).to_be_disabled()
                page.fill("#data-id", id)
                page.click("#data-preview")
                ready(page)
                page.click("#data-attach")
                ready(page)
            page.select_option("#data-binding", "energy")
            page.click("#evaluation-start")
            ready(page)
            expect(page.locator("#evaluation-state")).to_contain_text('"status": "completed"')
            positive_id = page.input_value("#evaluation-id")
            page.click("#evaluation-summary")
            ready(page)
            expect(page.locator("#evaluation-report")).to_contain_text('"passed": true')
            page.click("#evaluation-curve")
            ready(page)
            expect(page.locator("#evaluation-plot polyline")).to_have_count(2)
            page.screenshot(path=str(a.output / "study-desktop.png"), full_page=True)
            # Objective change leaves physical binding intact but makes old result stale.
            page.fill("#target", "301")
            page.fill("#note", '<img src=x onerror="window.unwanted=1">')
            page.click("#preview-scenario")
            ready(page)
            page.click("#commit")
            ready(page)
            expect(page.locator("#evaluation-state")).to_contain_text('"stale": true')
            assert page.locator("#history img").count() == 0
            assert page.locator("#evaluation-plot polyline").count() == 0
            expect(page.locator("#curve-caption")).to_contain_text("stale=true")
            page.select_option("#data-binding", "thermal")
            page.click("#evaluation-start")
            ready(page)
            expect(page.locator("#evaluation-state")).to_contain_text('"status": "completed"')
            page.click("#evaluation-summary")
            ready(page)
            expect(page.locator("#evaluation-report")).to_contain_text('"passed": false')
            page.click("#evaluation-curve")
            ready(page)
            page.set_viewport_size(dict(width=390, height=844))
            assert page.evaluate("document.documentElement.scrollWidth <= innerWidth + 1")
            page.locator("#evaluation-plot").scroll_into_view_if_needed()
            page.screenshot(path=str(a.output / "study-mobile.png"))
            page.click("#export")
            ready(page)
            exports = list((a.output / "exports").glob("*/study.json"))
            assert len(exports) == 1
            store.import_snapshot(exports[0], a.output / "imported")
            assert set(store.inspect(a.output / "imported")["references"]) == {"energy", "thermal"}
            assert not errors, errors
            proof = dict(
                schema_version=1,
                browser=browser.version,
                errors=errors,
                positive_evaluation_id=positive_id,
                negative_evaluation_id=page.input_value("#evaluation-id"),
                preview_invalidated=True,
                stale_result_visible=True,
                mobile_no_overflow=True,
                portable_snapshot_imported=True,
                validated=False,
                plant_referenced=False,
            )
            atomic_json(a.output / "browser_proof.json", proof)
            print(json.dumps(proof))
            browser.close()
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
        backend.close()


if __name__ == "__main__":
    main()
