from __future__ import annotations

import base64
import json
from dataclasses import dataclass
from typing import Any
from urllib import error, parse, request

from payday.models import PipelineResult
from payday.repository import DashboardInterviewRecord, DashboardStatusOverview, JobRecord


@dataclass(frozen=True, slots=True)
class BackendApiSettings:
    base_url: str
    timeout_seconds: float = 30.0


class BackendApiError(RuntimeError):
    pass


class BackendApiClient:
    def __init__(self, settings: BackendApiSettings) -> None:
        self._settings = settings

    def close(self) -> None:
        return None

    def runtime_summary(self) -> dict[str, object]:
        return self._request_json("GET", "/v1/runtime/summary")

    def runtime_diagnostics(self) -> dict[str, object]:
        return self._request_json("GET", "/v1/runtime/diagnostics")

    def list_results(self) -> list[PipelineResult]:
        return []

    def list_recent_interviews(self, *, limit: int = 100) -> list[DashboardInterviewRecord]:
        payload = self._request_json("GET", "/v1/interviews", params={"limit": limit})
        return [DashboardInterviewRecord(**item) for item in payload]

    def get_interview_detail(self, interview_id: str) -> DashboardInterviewRecord:
        payload = self._request_json("GET", f"/v1/interviews/{interview_id}")
        return DashboardInterviewRecord(**payload)

    def get_status_overview(self) -> DashboardStatusOverview:
        payload = self._request_json("GET", "/v1/interviews/status-overview")
        return DashboardStatusOverview(**payload)

    def list_jobs(self, *, limit: int = 200) -> list[JobRecord]:
        payload = self._request_json("GET", "/v1/jobs", params={"limit": limit})
        return [JobRecord(**item) for item in payload]

    def enqueue_batch_uploads(self, items: list[object]) -> dict[str, object]:
        files = [
            {
                "filename": item.filename,
                "content_type": item.content_type,
                "data_base64": base64.b64encode(item.data).decode("ascii"),
            }
            for item in items
        ]
        return self._request_json("POST", "/v1/uploads/enqueue", json_body={"files": files})

    def cancel_job(self, job_id: str) -> JobRecord:
        payload = self._request_json("POST", f"/v1/jobs/{job_id}/cancel")
        return JobRecord(**payload)

    def retry_job(self, job_id: str) -> JobRecord:
        payload = self._request_json("POST", f"/v1/jobs/{job_id}/retry")
        return JobRecord(**payload)

    def cancel_batch_jobs(self, batch_id: str) -> int:
        payload = self._request_json("POST", f"/v1/batches/{batch_id}/cancel")
        return int(payload.get("updated", 0))

    def retry_batch_jobs(self, batch_id: str) -> int:
        payload = self._request_json("POST", f"/v1/batches/{batch_id}/retry")
        return int(payload.get("updated", 0))

    def list_processing_events(self, *, file_ids: list[str] | None = None, limit: int = 200) -> list[object]:
        return []

    def save_interview_edits(
        self,
        interview_id: str,
        *,
        transcript: str,
        extracted_json: str,
        transcript_changed: bool,
        structured_json_changed: bool,
    ) -> DashboardInterviewRecord:
        payload = self._request_json(
            "POST",
            f"/v1/interviews/{interview_id}/edits",
            json_body={
                "transcript": transcript,
                "extracted_json": extracted_json,
                "transcript_changed": transcript_changed,
                "structured_json_changed": structured_json_changed,
            },
        )
        return DashboardInterviewRecord(**payload)

    def reprocess_interview(self, interview_id: str) -> DashboardInterviewRecord:
        payload = self._request_json("POST", f"/v1/interviews/{interview_id}/reprocess")
        return DashboardInterviewRecord(**payload)

    def reanalyze_interviews(self, interview_ids: list[str]) -> list[DashboardInterviewRecord]:
        payload = self._request_json("POST", "/v1/interviews/reprocess", json_body={"interview_ids": interview_ids})
        return [DashboardInterviewRecord(**item) for item in payload]

    def reprocess_stale_interviews(self, *, limit: int = 500) -> dict[str, object]:
        return self._request_json("POST", "/v1/interviews/reprocess/stale", params={"limit": limit})

    def list_stale_interview_ids(self, *, limit: int = 500) -> list[str]:
        payload = self._request_json("GET", "/v1/interviews/stale", params={"limit": limit})
        return list(payload.get("ids", []))

    def reprocess_failed_or_malformed_interviews(self, *, limit: int = 500) -> dict[str, object]:
        return self._request_json("POST", "/v1/interviews/reprocess/failed-or-malformed", params={"limit": limit})

    def list_failed_or_malformed_interview_ids(self, *, limit: int = 500) -> list[str]:
        payload = self._request_json("GET", "/v1/interviews/failed-or-malformed", params={"limit": limit})
        return list(payload.get("ids", []))

    def delete_stale_corrupted_interviews(self, *, limit: int = 500) -> dict[str, object]:
        return self._request_json("POST", "/v1/interviews/delete/stale-corrupted", params={"limit": limit})

    def list_stale_corrupted_interview_ids(self, *, limit: int = 500) -> list[str]:
        payload = self._request_json("GET", "/v1/interviews/stale-corrupted", params={"limit": limit})
        return list(payload.get("ids", []))

    def delete_interview(self, interview_id: str) -> bool:
        payload = self._request_json("DELETE", f"/v1/interviews/{interview_id}")
        return bool(payload.get("deleted", False))

    def _request_json(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        json_body: dict[str, Any] | None = None,
    ) -> Any:
        url = f"{self._settings.base_url.rstrip('/')}{path}"
        if params:
            url = f"{url}?{parse.urlencode(params)}"

        payload = None
        headers = {"Accept": "application/json"}
        if json_body is not None:
            payload = json.dumps(json_body).encode("utf-8")
            headers["Content-Type"] = "application/json"

        req = request.Request(url=url, data=payload, method=method, headers=headers)
        try:
            with request.urlopen(req, timeout=self._settings.timeout_seconds) as response:
                body = response.read()
                return json.loads(body.decode("utf-8")) if body else {}
        except error.HTTPError as exc:
            message = exc.read().decode("utf-8", errors="ignore")
            raise BackendApiError(f"Backend API request failed ({method} {path}): {exc.code} {message}") from exc
        except error.URLError as exc:
            raise BackendApiError(f"Backend API request failed ({method} {path}): {exc}") from exc
