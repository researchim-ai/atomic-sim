"""Real loopback result comparison/export extension retains existing origin guards."""

import http.client
import json
import re
import threading

import pytest

from atomic_model_project_ui.server import create_server
from atomic_model_results import bundle
from atomic_model_results.backend import Backend
from tests.test_model_results import base as results_fixture

base = results_fixture


@pytest.fixture
def ui(base, tmp_path):
    backend = Backend(base[0])
    server = create_server(base[0], backend=backend, exports=tmp_path / "exports")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port = server.server_port
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=60)
    conn.request("GET", "/")
    token = re.search(
        r'name="project-token" content="([^"]+)"', conn.getresponse().read().decode()
    ).group(1)
    conn.close()

    def call(operation, body, headers=None):
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=60)
        h = {
            "Origin": f"http://127.0.0.1:{port}",
            "X-Project-Token": token,
            "Content-Type": "application/json",
        }
        h.update(headers or {})
        conn.request("POST", "/api/" + operation, json.dumps(body), h)
        r = conn.getresponse()
        result = r.status, json.loads(r.read())
        conn.close()
        return result

    yield call, backend, base[0], base[1]
    server.shutdown()
    server.server_close()
    thread.join()
    backend.close()


def test_compare_export_and_inventory(ui):
    call, backend, folder, states = ui
    view = backend.inspect(folder)
    assert view["results_enabled"] and len(view["result_evaluations"]) == 4
    req = dict(evaluation_ids=[s["evaluation_id"] for s in states.values()])
    status, report = call("results-compare", req)
    assert status == 200 and len(report["groups"]) == 2
    status, result = call("results-export", req)
    assert status == 200
    assert bundle.verify(__import__("pathlib").Path(result["bundle"]).parent) == report


@pytest.mark.parametrize(
    "headers",
    [
        {"Origin": "https://evil.example"},
        {"Host": "evil.example"},
        {"X-Project-Token": "wrong"},
        {"Sec-Fetch-Site": "cross-site"},
    ],
)
def test_result_writes_keep_security_guards(ui, headers):
    call, _, _, states = ui
    assert (
        call(
            "results-export",
            dict(evaluation_ids=[s["evaluation_id"] for s in states.values()]),
            headers,
        )[0]
        == 403
    )


def test_arbitrary_server_paths_and_bad_ids_refused(ui):
    call, _, _, _ = ui
    assert call("results-export", dict(evaluation_ids=["../outside"]))[0] == 409
    assert call("results-export", dict(evaluation_ids=[], output="/etc/passwd"))[0] == 409
    assert call("results-compare", dict(evaluation_ids=[]))[0] == 409
