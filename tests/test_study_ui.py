"""Real loopback study data import and worker endpoints preserve UI origin guards."""

import base64
import http.client
import json
import re
import threading

import pytest

from atomic_model_project_ui.server import create_server
from atomic_model_study.backend import Backend
from tests.test_model_study import request
from tests.test_model_study import study as study_fixture

study = study_fixture


@pytest.fixture
def ui(study, tmp_path):
    backend = Backend(study)
    server = create_server(study, backend=backend, exports=tmp_path / "exports")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port = server.server_port
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=30)
    conn.request("GET", "/")
    page = conn.getresponse().read().decode()
    token = re.search(r'name="project-token" content="([^"]+)"', page).group(1)
    conn.close()

    def call(operation, payload, headers=None):
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=30)
        h = {
            "Origin": f"http://127.0.0.1:{port}",
            "X-Project-Token": token,
            "Content-Type": "application/json",
        }
        h.update(headers or {})
        conn.request("POST", "/api/" + operation, json.dumps(payload), h)
        response = conn.getresponse()
        result = response.status, json.loads(response.read())
        conn.close()
        return result

    yield call, backend, study
    server.shutdown()
    server.server_close()
    thread.join()
    backend.close()


def upload(study):
    req, raw = request(study)
    return dict(request=req, csv_base64=base64.b64encode(raw).decode())


def test_http_import_evaluate_summary_curve_export(ui):
    call, backend, study = ui
    payload = upload(study)
    assert call("data-preview", payload)[0] == 200
    assert call("data-attach", payload)[0] == 200
    assert call("data-attach", payload)[0] == 409
    view = backend.inspect(study)
    code, state = call(
        "evaluation-start",
        dict(
            binding_id="energy",
            expected_revision=view["revision"],
            expected_checkpoint_sha256=view["checkpoint_sha256"],
        ),
    )
    assert code == 200
    backend.manager.wait(state["evaluation_id"])
    req = dict(evaluation_id=state["evaluation_id"])
    assert call("evaluation-status", req)[1]["status"] == "completed"
    summary = call("evaluation-summary", req)
    assert summary[0] == 200 and summary[1]["passed"] is True
    assert call("evaluation-curve", dict(**req, channel="energy_residual"))[1]["decimated"] is False
    assert call("export", {})[1]["study"].endswith("study.json")


@pytest.mark.parametrize(
    "headers",
    [
        {"Origin": "https://evil.example"},
        {"X-Project-Token": "wrong"},
        {"Host": "evil.example"},
        {"Sec-Fetch-Site": "cross-site"},
    ],
)
def test_data_endpoints_keep_origin_guards(ui, headers):
    call, backend, study = ui
    assert call("data-attach", upload(study), headers)[0] == 403
    assert backend.inspect(study)["revision"] == 0


def test_data_paths_bad_base64_hash_and_unknown_fields_refused(ui):
    call, backend, study = ui
    payload = upload(study)
    payload["csv_base64"] = "!!!"
    assert call("data-attach", payload)[0] == 409
    payload = upload(study)
    payload["request"]["declaration"]["csv"]["path"] = "/etc/passwd"
    assert call("data-attach", payload)[0] == 409
    payload = upload(study)
    payload["server_path"] = "/etc/passwd"
    assert call("data-preview", payload)[0] == 409
    assert backend.inspect(study)["revision"] == 0
    assert call("evaluation-status", dict(evaluation_id="../outside"))[0] == 409
