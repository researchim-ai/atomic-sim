"""Real local HTTP editor, source-admitted commits and browser-origin boundaries."""

import http.client
import json
import re
import shutil
import threading
from pathlib import Path

import pytest

from atomic_model_project_ui.server import MAX_BODY, create_server
from atomic_model_projects import store
from atomic_model_workspace.project import load_project


@pytest.fixture
def ui(tmp_path):
    source = Path(__file__).resolve().parents[1] / "examples/eos_workspace"
    shutil.copytree(source, tmp_path / "project")
    folder = tmp_path / "store"
    store.create(tmp_path / "project/project.json", folder)
    server = create_server(folder, port=0, exports=tmp_path / "exports")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    state = dict(server=server, folder=folder, port=server.server_port, token=None)
    status, body, headers = call(state, "GET", "/")
    assert status == 200
    state["token"] = re.search(r'name="project-token" content="([^"]+)"', body.decode()).group(1)
    yield state
    server.shutdown()
    server.server_close()
    thread.join(timeout=5)


def call(ui, method, path, body=None, headers=None):
    conn = http.client.HTTPConnection("127.0.0.1", ui["port"], timeout=20)
    request_headers = {"Host": f"127.0.0.1:{ui['port']}"}
    if ui.get("token"):
        request_headers["X-Project-Token"] = ui["token"]
    if method == "POST":
        request_headers.update(
            {"Origin": f"http://127.0.0.1:{ui['port']}", "Content-Type": "application/json"}
        )
    if headers:
        request_headers.update(headers)
    if isinstance(body, dict):
        body = json.dumps(body)
    conn.request(method, path, body=body, headers=request_headers)
    response = conn.getresponse()
    data = response.read()
    result = (response.status, data, dict(response.getheaders()))
    conn.close()
    return result


def payload(ui):
    status, data, _ = call(ui, "GET", "/api/inspect")
    assert status == 200
    view = json.loads(data)
    return dict(
        schema_version=1,
        expected_revision=view["revision"],
        expected_project_sha256=view["project_sha256"],
        operations=[
            dict(op="set_scenario_config", id="mixed_dev", value=dict(episode_horizon_s=0.04))
        ],
        note="<script>untrusted note</script>",
    )


def test_http_form_preview_commit_history_restore_and_export(ui):
    req = payload(ui)
    status, data, _ = call(ui, "POST", "/api/preview", req)
    assert status == 200 and json.loads(data)["committed"] is False
    assert store.inspect(ui["folder"])["revision"] == 0
    assert call(ui, "POST", "/api/edit", req)[0] == 200
    assert call(ui, "POST", "/api/edit", req)[0] == 409
    current = store.inspect(ui["folder"])
    assert current["revision"] == 1 and current["history"][0]["note"] == req["note"]
    status, data, _ = call(ui, "POST", "/api/export", {})
    assert status == 200 and load_project(json.loads(data)["project"]) == current["project"]
    restore = dict(
        expected_revision=1, expected_project_sha256=current["project_sha256"], target_revision=0
    )
    assert call(ui, "POST", "/api/restore", restore)[0] == 200
    assert store.inspect(ui["folder"])["revision"] == 2


@pytest.mark.parametrize(
    "headers",
    [
        {"Host": "evil.example"},
        {"Origin": "https://evil.example"},
        {"X-Project-Token": "wrong"},
        {"X-Project-Token": "é"},
        {"Sec-Fetch-Site": "cross-site"},
        {"Sec-Fetch-Site": "same-site"},
    ],
)
def test_remote_origin_wrong_host_and_bad_tokens_cannot_change_project(ui, headers):
    req = payload(ui)
    assert call(ui, "POST", "/api/edit", req, headers=headers)[0] == 403
    assert store.inspect(ui["folder"])["revision"] == 0


def test_read_api_requires_token_and_write_requires_origin(ui):
    assert call(ui, "GET", "/api/inspect", headers={"X-Project-Token": ""})[0] == 403
    assert call(ui, "POST", "/api/edit", payload(ui), headers={"Origin": ""})[0] == 403
    assert call(ui, "OPTIONS", "/api/edit")[0] == 405


@pytest.mark.parametrize(
    "path",
    ["/../sdk/experiment.py", "/%2e%2e/sdk/experiment.py", "/api/inspect?file=secret", "/unknown"],
)
def test_no_file_browser_or_query_driven_paths(ui, path):
    assert call(ui, "GET", path)[0] == 404


def test_payload_size_and_content_type_are_bounded(ui):
    assert call(ui, "POST", "/api/edit", "{}", headers={"Content-Type": "text/plain"})[0] == 415
    assert (
        call(ui, "POST", "/api/edit", "{}", headers={"Content-Length": str(MAX_BODY + 1)})[0] == 413
    )
    assert call(ui, "POST", "/api/edit", '{"schema_version":1,"schema_version":2}')[0] == 409
    assert store.inspect(ui["folder"])["revision"] == 0


def test_export_destination_is_not_supplied_by_browser(ui):
    assert call(ui, "POST", "/api/export", {"output": "/tmp/untrusted-output"})[0] == 409


def test_static_assets_csp_and_no_dynamic_html_injection(ui):
    status, body, headers = call(ui, "GET", "/")
    assert status == 200 and b"__PROJECT_TOKEN__" not in body
    assert "frame-ancestors 'none'" in headers["Content-Security-Policy"]
    assert "script-src 'self'" in headers["Content-Security-Policy"]
    assert headers["Cache-Control"] == "no-store" and headers["X-Content-Type-Options"] == "nosniff"
    status, js, headers = call(ui, "GET", "/app.js")
    assert status == 200 and b"innerHTML" not in js and b"textContent" in js and b"commit" in js
    assert call(ui, "GET", "/style.css")[0] == 200


def test_invalid_binding_shows_error_and_preserves_head(ui):
    req = payload(ui)
    req["operations"][0]["value"]["preset"] = "uniform"
    status, data, _ = call(ui, "POST", "/api/preview", req)
    assert status == 409 and json.loads(data)["error"]
    assert store.inspect(ui["folder"])["revision"] == 0


def test_ui_inspection_exposes_actual_source_and_runtime_versions(ui):
    status, data, _ = call(ui, "GET", "/api/inspect")
    assert status == 200
    value = json.loads(data)
    assert value["source"] == store.source_identity()
    native = value["source"]["workspace"]["runtime"]["native_signature"]
    assert (
        len(native["source_sha256"]) == 64
        and native["python"]
        and native["numpy"]
        and native["scipy"]
    )
