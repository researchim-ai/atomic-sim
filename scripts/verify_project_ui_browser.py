"""Explicit optional Chromium QA of the local project editor; no external URLs."""

import argparse
import hashlib
import json
import threading
from pathlib import Path


def main():
    from playwright.sync_api import sync_playwright

    import atomic_model_project_ui
    from atomic_model_project_ui.server import create_server
    from atomic_model_projects import store
    from atomic_model_runtime import create_eos_episode
    from atomic_model_workspace.campaign import create_campaign, load_campaign
    from atomic_model_workspace.jobs import run_job
    from atomic_model_workspace.project import load_project

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--project", type=Path, default=Path("examples/eos_workspace/project.json"))
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    output = a.output
    output.mkdir(parents=True, exist_ok=False)
    folder = output / "store"
    store.create(a.project, folder)
    server = create_server(folder, port=0, exports=output / "exports")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    errors = []

    def ready(page):
        page.locator("main[aria-busy='false']").wait_for(state="visible")
        assert "eos-thermal" in page.locator("#model").inner_text()

    try:
        with sync_playwright() as browser_api:
            browser = browser_api.chromium.launch(
                headless=True, args=["--no-sandbox"], timeout=20000
            )
            page = browser.new_page(viewport=dict(width=1280, height=1000))
            page.set_default_timeout(20000)
            page.on("pageerror", lambda e: errors.append(str(e)))
            page.goto(f"http://127.0.0.1:{server.server_port}", wait_until="networkidle")
            ready(page)
            page.locator("details summary").click()
            assert "source_sha256" in page.locator("#provenance").inner_text()
            page.locator("details summary").click()
            page.select_option("#scenario", "mixed_dev")
            page.fill("#horizon", "0.04")
            page.click("#preview-scenario")
            ready(page)
            assert page.locator("#commit").is_enabled()
            assert store.inspect(folder)["revision"] == 0
            page.screenshot(path=str(output / "preview-desktop.png"), full_page=True)
            # Editing after preview must invalidate the previously admitted request.
            page.fill("#horizon", "0.05")
            assert page.locator("#commit").is_disabled()
            page.fill("#horizon", "0.04")
            page.click("#preview-scenario")
            ready(page)
            page.click("#commit")
            ready(page)
            first = store.inspect(folder)
            assert (
                first["revision"] == 1
                and first["project"]["scenarios"]["mixed_dev"]["config"]["episode_horizon_s"]
                == 0.04
            )
            page.click("#export")
            ready(page)
            exported = Path(page.locator("#export-result").inner_text().removeprefix("Проект: "))
            assert load_project(exported) == first["project"]
            campaign = output / "ui-campaign"
            create_campaign(exported, campaign)
            entry = next(
                e
                for e in load_campaign(campaign)["entries"]
                if e["scenario_id"] == "mixed_dev" and e["policy_id"] == "hold"
            )
            job = run_job(campaign / entry["job"])
            assert job["status"] == "completed"
            native = create_eos_episode(job["spec"]["scenario"]["config"])
            commands = job["spec"]["policy"]["heat_power_w"]
            while not native._done:
                native.step([commands[k] for k in native.config.action_channels])
            assert native.export_episode() == job["episode"]
            page.fill("#horizon", "0.05")
            page.fill("#note", '<img src=x onerror="window.unwanted=1">')
            page.click("#preview-scenario")
            ready(page)
            page.click("#commit")
            ready(page)
            assert page.locator("#history img").count() == 0
            assert store.inspect(folder)["revision"] == 2
            page.select_option("#restore-target", "0")
            page.click("#restore")
            ready(page)
            assert store.inspect(folder)["revision"] == 3 and store.inspect(folder)[
                "project"
            ] == load_project(a.project)
            page.screenshot(path=str(output / "restored-desktop.png"), full_page=True)
            page.select_option("#preset", "uniform")
            page.click("#preview-scenario")
            ready(page)
            assert (
                "error" in page.locator("#status").get_attribute("class")
                and store.inspect(folder)["revision"] == 3
            )
            page.screenshot(path=str(output / "invalid-binding.png"), full_page=True)
            page.set_viewport_size(dict(width=390, height=844))
            page.click("#refresh")
            ready(page)
            assert page.evaluate("document.documentElement.scrollWidth<=window.innerWidth")
            page.screenshot(path=str(output / "mobile.png"), full_page=True)
            assert not errors, errors
            version = browser.version
            browser.close()
        report = dict(
            schema_version=1,
            passed=True,
            chromium_version=version,
            initial_revision=0,
            final_revision=3,
            preview_is_read_only=True,
            provenance_visible_on_demand=True,
            form_change_invalidates_preview=True,
            commit_undo_export_verified=True,
            ui_export_native_episode_exact=True,
            malicious_note_rendered_as_text=True,
            invalid_binding_preserves_revision=True,
            mobile_document_has_no_horizontal_overflow=True,
            page_errors=errors,
            assets={
                p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                for p in (
                    Path(atomic_model_project_ui.__file__).resolve().parent / "assets"
                ).iterdir()
            },
            screenshots=[p.name for p in sorted(output.glob("*.png"))],
            validated=False,
            plant_referenced=False,
        )
        (output / "browser_proof.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n"
        )
        print(json.dumps(report))
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


if __name__ == "__main__":
    main()
