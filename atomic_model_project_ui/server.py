"""Small loopback HTTP adapter; project admission remains in editor.store."""

import json
import secrets
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

from atomic_model_projects import store
from atomic_model_workspace.common import keys, loads

MAX_BODY = 1024 * 1024


def create_server(folder, *, port=0, exports=None, backend=None):
    service = backend if backend is not None else store
    study_routes = (
        tuple(
            "/api/" + name
            for name in (
                "data-preview",
                "data-attach",
                "data-detach",
                "evaluation-start",
                "evaluation-status",
                "evaluation-cancel",
                "evaluation-resume",
                "evaluation-summary",
                "evaluation-curve",
            )
        )
        if getattr(service, "study_enabled", False)
        else ()
    )
    results_routes = (
        ("/api/results-compare", "/api/results-export")
        if getattr(service, "results_enabled", False)
        else ()
    )
    if type(port) is not int or not 0 <= port <= 65535:
        raise ValueError("Invalid local port")
    folder = Path(folder).resolve()
    service.inspect(folder)
    exports = (
        Path(exports).resolve()
        if exports is not None
        else folder.parent / (folder.name + "-exports")
    )
    token = secrets.token_urlsafe(32)
    assets = Path(__file__).parent / "assets"

    class Handler(BaseHTTPRequestHandler):
        server_version = "AtomicProjectUI/1.0"

        def log_message(self, *args):
            pass

        def reply(self, status, payload, kind="application/json; charset=utf-8"):
            data = (
                json.dumps(payload, ensure_ascii=False, allow_nan=False).encode()
                if isinstance(payload, (dict, list))
                else payload
            )
            self.send_response(status)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header(
                "Content-Security-Policy",
                "default-src 'none'; script-src 'self'; style-src 'self'; connect-src 'self'; img-src 'none'; base-uri 'none'; form-action 'none'; frame-ancestors 'none'",
            )
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header("X-Frame-Options", "DENY")
            self.end_headers()
            self.wfile.write(data)

        def permitted(self, *, api=False, write=False):
            origin = f"http://127.0.0.1:{self.server.server_port}"
            if self.headers.get("Host") != f"127.0.0.1:{self.server.server_port}":
                self.reply(403, dict(error="Host must match loopback listener"))
                return False
            supplied = self.headers.get("Origin")
            if supplied is not None and supplied != origin:
                self.reply(403, dict(error="Cross-origin access denied"))
                return False
            if write and supplied != origin:
                self.reply(403, dict(error="Same-origin header required"))
                return False
            if api and not secrets.compare_digest(
                self.headers.get("X-Project-Token", "").encode(), token.encode()
            ):
                self.reply(403, dict(error="Project token required"))
                return False
            if self.headers.get("Sec-Fetch-Site") in ("cross-site", "same-site"):
                self.reply(403, dict(error="Cross-site request denied"))
                return False
            return True

        def do_GET(self):
            path = urlsplit(self.path)
            if path.query or path.fragment:
                self.reply(404, dict(error="Unknown resource"))
                return
            api = path.path.startswith("/api/")
            if not self.permitted(api=api):
                return
            try:
                if path.path == "/":
                    page = (assets / "index.html").read_text().replace("__PROJECT_TOKEN__", token)
                    self.reply(200, page.encode(), "text/html; charset=utf-8")
                elif path.path in ("/app.js", "/style.css"):
                    name = path.path[1:]
                    mime = (
                        "text/javascript; charset=utf-8"
                        if name.endswith(".js")
                        else "text/css; charset=utf-8"
                    )
                    self.reply(200, (assets / name).read_bytes(), mime)
                elif path.path == "/api/inspect":
                    result = service.inspect(folder)
                    result["source"] = service.source_identity()
                    self.reply(200, result)
                else:
                    self.reply(404, dict(error="Unknown resource"))
            except (ValueError, OSError, KeyError, TypeError):
                self.reply(
                    409,
                    dict(
                        error="Project unavailable or source/history changed; inspect the store with CLI"
                    ),
                )

        def do_POST(self):
            if not self.permitted(api=True, write=True):
                return
            path = urlsplit(self.path)
            if (
                path.query
                or path.fragment
                or path.path
                not in (
                    "/api/preview",
                    "/api/edit",
                    "/api/restore",
                    "/api/export",
                    *study_routes,
                    *results_routes,
                )
            ):
                self.reply(404, dict(error="Unknown operation"))
                return
            if (
                self.headers.get("Transfer-Encoding") is not None
                or self.headers.get("Content-Type", "").split(";")[0] != "application/json"
            ):
                self.reply(415, dict(error="Bounded application/json body required"))
                return
            raw = self.headers.get("Content-Length", "")
            limit = (
                12 * MAX_BODY
                if path.path in ("/api/data-preview", "/api/data-attach")
                else MAX_BODY
            )
            if not raw.isdecimal() or not 0 < int(raw) <= limit:
                self.reply(413, dict(error="Body size limit"))
                return
            try:
                data = self.rfile.read(int(raw))
                if len(data) != int(raw):
                    self.reply(400, dict(error="Incomplete request body"))
                    return
                value = loads(data)
                if path.path == "/api/results-compare" and results_routes:
                    result = service.compare(folder, value)
                elif path.path == "/api/results-export" and results_routes:
                    exports.mkdir(parents=True, exist_ok=True)
                    result = service.export_results(
                        folder, value, exports / ("results-" + secrets.token_hex(8))
                    )
                elif path.path in study_routes:
                    result = service.dispatch(path.path.removeprefix("/api/"), folder, value)
                elif path.path == "/api/preview":
                    result = service.preview(folder, value)
                elif path.path == "/api/edit":
                    result = service.edit(folder, value)
                elif path.path == "/api/restore":
                    keys(value, ("expected_revision", "expected_project_sha256", "target_revision"))
                    result = service.restore(folder, **value)
                else:
                    keys(value, ())
                    exports.mkdir(parents=True, exist_ok=True)
                    result = service.export(folder, exports / ("project-" + secrets.token_hex(8)))
                self.reply(200, result)
            except (ValueError, KeyError, TypeError) as exc:
                self.reply(409, dict(error=str(exc)))
            except OSError:
                self.reply(409, dict(error="Project file operation failed; accepted HEAD retained"))

        def do_OPTIONS(self):
            self.reply(405, dict(error="Cross-origin API is not enabled"))

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    server.daemon_threads = True
    # Read requests are bounded too; slow local clients cannot hold a thread forever.
    old_get = server.get_request

    def get_request():
        sock, address = old_get()
        sock.settimeout(10)
        return sock, address

    server.get_request = get_request
    return server
