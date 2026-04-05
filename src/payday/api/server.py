from __future__ import annotations

import base64
import json
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from payday.config import get_settings
from payday.models import BatchUploadItem
from payday.service import PaydayAppService

from .serialization import to_json_compatible


@dataclass(slots=True)
class BackendApiContext:
    service: PaydayAppService


class BackendRequestHandler(BaseHTTPRequestHandler):
    server: "PaydayHttpServer"

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        params = parse_qs(parsed.query)
        path = parsed.path
        try:
            if path == "/healthz":
                self._write_json(HTTPStatus.OK, {"status": "ok"})
                return
            if path == "/v1/runtime/summary":
                self._write_json(HTTPStatus.OK, to_json_compatible(self.server.context.service.runtime_summary()))
                return
            if path == "/v1/runtime/diagnostics":
                self._write_json(HTTPStatus.OK, to_json_compatible(self.server.context.service.runtime_diagnostics()))
                return
            if path == "/v1/jobs":
                limit = _first_int(params, "limit", default=200)
                self._write_json(HTTPStatus.OK, to_json_compatible(self.server.context.service.list_jobs(limit=limit)))
                return
            if path.startswith("/v1/jobs/"):
                job_id = path.split("/")[-1]
                self._write_json(HTTPStatus.OK, to_json_compatible(self.server.context.service.repository.get_job(job_id)))
                return
            if path == "/v1/interviews":
                limit = _first_int(params, "limit", default=100)
                self._write_json(
                    HTTPStatus.OK,
                    to_json_compatible(self.server.context.service.list_recent_interviews(limit=limit)),
                )
                return
            if path == "/v1/interviews/status-overview":
                self._write_json(HTTPStatus.OK, to_json_compatible(self.server.context.service.get_status_overview()))
                return
            if path == "/v1/interviews/stale":
                limit = _first_int(params, "limit", default=500)
                self._write_json(HTTPStatus.OK, {"ids": self.server.context.service.list_stale_interview_ids(limit=limit)})
                return
            if path == "/v1/interviews/failed-or-malformed":
                limit = _first_int(params, "limit", default=500)
                self._write_json(
                    HTTPStatus.OK,
                    {"ids": self.server.context.service.list_failed_or_malformed_interview_ids(limit=limit)},
                )
                return
            if path == "/v1/interviews/stale-corrupted":
                limit = _first_int(params, "limit", default=500)
                self._write_json(
                    HTTPStatus.OK,
                    {"ids": self.server.context.service.list_stale_corrupted_interview_ids(limit=limit)},
                )
                return
            if path.startswith("/v1/interviews/"):
                interview_id = path.split("/")[-1]
                self._write_json(
                    HTTPStatus.OK,
                    to_json_compatible(self.server.context.service.get_interview_detail(interview_id)),
                )
                return
        except KeyError as exc:
            self._write_json(HTTPStatus.NOT_FOUND, {"detail": str(exc)})
            return
        except ValueError as exc:
            self._write_json(HTTPStatus.BAD_REQUEST, {"detail": str(exc)})
            return
        self._write_json(HTTPStatus.NOT_FOUND, {"detail": "Not found"})

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        params = parse_qs(parsed.query)
        path = parsed.path
        payload = self._read_json_body()
        try:
            if path == "/v1/uploads/enqueue":
                files = payload.get("files", [])
                items = [
                    BatchUploadItem(
                        filename=str(item.get("filename", "upload.bin")),
                        content_type=str(item.get("content_type", "application/octet-stream")),
                        data=base64.b64decode(str(item.get("data_base64", ""))),
                    )
                    for item in files
                ]
                self._write_json(HTTPStatus.OK, to_json_compatible(self.server.context.service.enqueue_batch_uploads(items)))
                return
            if path.startswith("/v1/jobs/") and path.endswith("/cancel"):
                job_id = path.split("/")[-2]
                self._write_json(HTTPStatus.OK, to_json_compatible(self.server.context.service.cancel_job(job_id)))
                return
            if path.startswith("/v1/jobs/") and path.endswith("/retry"):
                job_id = path.split("/")[-2]
                self._write_json(HTTPStatus.OK, to_json_compatible(self.server.context.service.retry_job(job_id)))
                return
            if path.startswith("/v1/batches/") and path.endswith("/cancel"):
                batch_id = path.split("/")[-2]
                self._write_json(HTTPStatus.OK, {"updated": self.server.context.service.cancel_batch_jobs(batch_id)})
                return
            if path.startswith("/v1/batches/") and path.endswith("/retry"):
                batch_id = path.split("/")[-2]
                self._write_json(HTTPStatus.OK, {"updated": self.server.context.service.retry_batch_jobs(batch_id)})
                return
            if path.startswith("/v1/interviews/") and path.endswith("/reprocess"):
                interview_id = path.split("/")[-2]
                self._write_json(HTTPStatus.OK, to_json_compatible(self.server.context.service.reprocess_interview(interview_id)))
                return
            if path == "/v1/interviews/reprocess":
                interview_ids = list(payload.get("interview_ids", []))
                self._write_json(
                    HTTPStatus.OK,
                    to_json_compatible(self.server.context.service.reanalyze_interviews(interview_ids)),
                )
                return
            if path == "/v1/interviews/reprocess/stale":
                limit = _first_int(params, "limit", default=500)
                self._write_json(HTTPStatus.OK, to_json_compatible(self.server.context.service.reprocess_stale_interviews(limit=limit)))
                return
            if path == "/v1/interviews/reprocess/failed-or-malformed":
                limit = _first_int(params, "limit", default=500)
                self._write_json(
                    HTTPStatus.OK,
                    to_json_compatible(self.server.context.service.reprocess_failed_or_malformed_interviews(limit=limit)),
                )
                return
            if path == "/v1/interviews/delete/stale-corrupted":
                limit = _first_int(params, "limit", default=500)
                self._write_json(
                    HTTPStatus.OK,
                    to_json_compatible(self.server.context.service.delete_stale_corrupted_interviews(limit=limit)),
                )
                return
            if path.startswith("/v1/interviews/") and path.endswith("/edits"):
                interview_id = path.split("/")[-2]
                self._write_json(
                    HTTPStatus.OK,
                    to_json_compatible(
                        self.server.context.service.save_interview_edits(
                            interview_id,
                            transcript=str(payload.get("transcript", "")),
                            extracted_json=str(payload.get("extracted_json", "{}")),
                            transcript_changed=bool(payload.get("transcript_changed", False)),
                            structured_json_changed=bool(payload.get("structured_json_changed", False)),
                        )
                    ),
                )
                return
        except KeyError as exc:
            self._write_json(HTTPStatus.NOT_FOUND, {"detail": str(exc)})
            return
        except ValueError as exc:
            self._write_json(HTTPStatus.BAD_REQUEST, {"detail": str(exc)})
            return
        self._write_json(HTTPStatus.NOT_FOUND, {"detail": "Not found"})

    def do_DELETE(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path.startswith("/v1/interviews/"):
            interview_id = path.split("/")[-1]
            self._write_json(HTTPStatus.OK, {"deleted": self.server.context.service.delete_interview(interview_id)})
            return
        self._write_json(HTTPStatus.NOT_FOUND, {"detail": "Not found"})

    def _read_json_body(self) -> dict[str, object]:
        length = int(self.headers.get("Content-Length", "0") or "0")
        if length <= 0:
            return {}
        raw = self.rfile.read(length)
        if not raw:
            return {}
        return json.loads(raw.decode("utf-8"))

    def _write_json(self, status: HTTPStatus, payload: object) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status.value)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, _format: str, *_args: object) -> None:
        return


class PaydayHttpServer(ThreadingHTTPServer):
    def __init__(self, server_address: tuple[str, int], context: BackendApiContext) -> None:
        super().__init__(server_address, BackendRequestHandler)
        self.context = context


def build_server(*, host: str = "127.0.0.1", port: int = 8000) -> PaydayHttpServer:
    settings = get_settings()
    context = BackendApiContext(service=PaydayAppService(settings))
    return PaydayHttpServer((host, port), context)


def _first_int(params: dict[str, list[str]], key: str, *, default: int) -> int:
    raw = params.get(key, [str(default)])[0]
    try:
        return int(raw)
    except ValueError:
        return default
