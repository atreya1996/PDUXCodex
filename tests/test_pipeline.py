from __future__ import annotations

from payday.config import DatabaseSettings, FeatureFlags, LLMSettings, Settings, SupabaseSettings, TranscriptionSettings
from payday.models import AnalysisResult, BatchUploadItem, PipelineResult, PipelineStage, ProcessingStatus, Transcript, UploadedAsset
from payday.personas import PersonaService
from payday.repository import PaydayRepository
from payday.service import PaydayAppService
from payday.storage import StorageService
from payday.transcription import TranscriptionService


class FlakyTranscriptionService(TranscriptionService):
    def __init__(self, settings: TranscriptionSettings) -> None:
        super().__init__(settings)
        self.calls: dict[str, int] = {}

    def transcribe(self, asset: UploadedAsset, sample_mode: bool = False) -> Transcript:
        self.calls.setdefault(asset.file_id, 0)
        self.calls[asset.file_id] += 1
        if asset.filename == "retry.wav" and self.calls[asset.file_id] == 1:
            raise RuntimeError("temporary transcription issue")
        if asset.filename == "fail.wav":
            raise RuntimeError("hard transcription failure")
        return super().transcribe(asset, sample_mode=sample_mode)


def build_settings(sqlite_path: str = ":memory:") -> Settings:
    return Settings(
        app_env="test",
        database=DatabaseSettings(sqlite_path=sqlite_path),
        supabase=SupabaseSettings(),
        llm=LLMSettings(),
        transcription=TranscriptionSettings(),
        database=DatabaseSettings(path=":memory:"),
        features=FeatureFlags(use_sample_mode=True),
    )


def build_transcript(text: str = "test transcript") -> Transcript:
    return Transcript(text=text, provider="test", model="test")


def build_analysis(structured_output: dict) -> AnalysisResult:
    return AnalysisResult(summary="summary", structured_output=structured_output)


def test_pipeline_process_upload_returns_completed_result() -> None:
    service = PaydayAppService(build_settings())

    result = service.process_upload(
        "demo.wav",
        "audio/wav",
        b"I use WhatsApp and my bank account is active. I earn \\xe2\\x82\\xb912,000 per month.",
    )

    assert result.asset is not None
    assert result.asset.filename == "demo.wav"
    assert result.persisted is True
    assert result.analysis is not None
    assert result.analysis.metrics["word_count"] > 0
    assert result.status is ProcessingStatus.COMPLETED
    assert result.current_stage is PipelineStage.STORAGE
    assert service.list_results()



def test_pipeline_persists_interview_transcript_structured_response_and_insight() -> None:
    service = PaydayAppService(build_settings())

    result = service.process_upload(
        "demo.wav",
        "audio/wav",
        (
            b"I use WhatsApp on my smartphone. My bank account is active. "
            b"I earn \xe2\x82\xb912,000 per month. I borrow from neighbors sometimes. "
            b"I repay monthly after salary. I am worried about scams."
        ),
    )

    detail = service.repository.get_interview_detail(result.file_id)

    assert detail.interview.id == result.file_id
    assert detail.interview.status == ProcessingStatus.COMPLETED.value
    assert detail.interview.transcript is not None
    assert "I use WhatsApp on my smartphone" in detail.interview.transcript
    assert detail.structured_response is not None
    assert detail.structured_response.smartphone_user is True
    assert detail.structured_response.has_bank_account is True
    assert detail.structured_response.income_range == "₹12,000"
    assert detail.structured_response.borrowing_history == "has_borrowed"
    assert detail.structured_response.repayment_preference == "monthly"
    assert detail.structured_response.loan_interest == "fearful_or_uncertain"
    assert detail.insight is not None
    assert detail.insight.summary == result.analysis.summary
    assert detail.insight.key_quotes == result.analysis.evidence_quotes
    assert detail.insight.persona == result.persona.persona_name
    assert detail.insight.confidence_score == 1.0


def test_pipeline_persists_failed_interview_status_for_failed_runs() -> None:
    settings = build_settings()
    service = PaydayAppService(settings)
    service.pipeline.transcription_service = FlakyTranscriptionService(settings.transcription)

    result = service.process_upload("fail.wav", "audio/wav", b"This upload will fail transcription.")

    detail = service.repository.get_interview_detail(result.file_id)

    assert result.status is ProcessingStatus.FAILED
    assert detail.interview.id == result.file_id
    assert detail.interview.status == ProcessingStatus.FAILED.value
    assert detail.interview.transcript is None
    assert detail.structured_response is None
    assert detail.insight is None
    assert result.errors


def test_repository_list_results_returns_pipeline_results_in_insertion_order() -> None:
    repository = PaydayRepository()
    first = repository.save_result(
        PipelineResult(file_id="file-1", filename="first.wav", status=ProcessingStatus.PENDING)
    )
    repository.save_result(PipelineResult(file_id="file-2", filename="second.wav", status=ProcessingStatus.COMPLETED))
    replacement = repository.save_result(
        PipelineResult(file_id="file-1", filename="first-retry.wav", status=ProcessingStatus.FAILED)
    )

    results = repository.list_results()

    assert [result.file_id for result in results] == ["file-1", "file-2"]
    assert results == [replacement, repository.get_result("file-2")]
    assert results[0] is not first
    assert repository.get_result("file-1") == replacement


def test_repository_crud_supports_interview_related_tables(tmp_path) -> None:
    repository = PaydayRepository(database_path=str(tmp_path / "payday.db"))
    interview = repository.create_interview(audio_url="audio/interview-1/demo.wav", status="uploaded")

    repository.update_interview(interview.id, transcript="A direct quote from the interview.", status="completed")
    repository.upsert_structured_response(
        interview.id,
        smartphone_user=True,
        has_bank_account=True,
        income_range="10k-20k",
        borrowing_history="Borrowed from employer during emergencies.",
        repayment_preference="Weekly",
        loan_interest="Interested if trust is high.",
    )
    repository.upsert_insight(
        interview.id,
        summary="Trust matters more than pricing.",
        key_quotes=["I ask my employer first when money is short."],
        persona="Persona 2",
        confidence_score=0.86,
    )

    listing = repository.list_interviews()
    detail = repository.get_interview_detail(interview.id)
    recent = repository.list_recent_interviews()
    status_overview = repository.get_status_overview()

    assert listing[0].id == interview.id
    assert detail.interview.status == "completed"
    assert detail.structured_response is not None
    assert detail.structured_response.smartphone_user is True
    assert detail.insight is not None
    assert detail.insight.key_quotes == ["I ask my employer first when money is short."]
    assert recent[0].filename == "demo.wav"
    assert status_overview.total_interviews == 1
    assert status_overview.status_counts == {"completed": 1}


def test_app_service_uses_sqlite_for_durable_dashboard_reads(tmp_path) -> None:
    database_path = str(tmp_path / "payday-dashboard.db")
    first_service = PaydayAppService(build_settings(database_path))

    result = first_service.process_upload(
        "durable.wav",
        "audio/wav",
        b"I use WhatsApp, I have a bank account, and I ask my employer when money is short.",
    )

    assert result.status is ProcessingStatus.COMPLETED
    assert first_service.list_recent_interviews()[0].id == result.file_id
    assert first_service.get_status_overview().status_counts["completed"] == 1

    second_service = PaydayAppService(build_settings(database_path))
    recent = second_service.list_recent_interviews()
    detail = second_service.get_interview_detail(result.file_id)

    assert second_service.list_results() == []
    assert recent[0].id == result.file_id
    assert detail.summary is not None
    assert detail.filename == "durable.wav"


def test_storage_service_builds_predictable_audio_paths() -> None:
    storage = StorageService(SupabaseSettings())
    asset = UploadedAsset(
        filename="../demo clip.wav",
        content_type="audio/wav",
        size_bytes=3,
        raw_bytes=b"123",
    )

    assert storage.build_audio_path("interview-123", asset.filename) == "audio/interview-123/demo_clip.wav"
    assert storage.store_asset(asset, sample_mode=True) is True


def test_persona_classifier_uses_bank_account_override() -> None:
    persona = PersonaService().classify(
        build_transcript(),
        build_analysis(
            {
                "participant_profile": {
                    "smartphone_user": {"value": True, "evidence": ["whatsapp"]},
                    "has_bank_account": {"value": False, "evidence": ["no bank account"]},
                },
                "persona_signals": {
                    "digital_readiness": {"value": True, "evidence": ["uses apps"]},
                    "trust_fear_barrier": {"value": True, "evidence": ["worried"]},
                },
            }
        ),
    )

    assert persona.persona_id == "persona_3"
    assert persona.is_non_target is True
    assert persona.explanation_payload["triggered_fields"] == ["participant_profile.has_bank_account"]


def test_persona_classifier_uses_smartphone_override() -> None:
    persona = PersonaService().classify(
        build_transcript(),
        build_analysis(
            {
                "participant_profile": {
                    "smartphone_user": {"value": False, "evidence": ["basic phone"]},
                    "has_bank_account": {"value": True, "evidence": ["bank account"]},
                },
                "persona_signals": {
                    "cyclical_borrowing": {"value": True, "evidence": ["every month"]},
                    "repayment_stress": {"value": True, "evidence": ["repay pressure"]},
                },
            }
        ),
    )

    assert persona.persona_id == "persona_3"
    assert persona.is_non_target is True
    assert persona.explanation_payload["triggered_fields"] == ["participant_profile.smartphone_user"]


def test_persona_classifier_uses_structured_fields_for_persona_one() -> None:
    persona = PersonaService().classify(
        build_transcript(),
        build_analysis(
            {
                "participant_profile": {
                    "smartphone_user": {"value": True, "evidence": ["whatsapp"]},
                    "has_bank_account": {"value": True, "evidence": ["bank account"]},
                },
                "persona_signals": {
                    "employer_dependency": {"value": True, "evidence": ["madam helped"]},
                    "digital_borrowing": {"value": True, "evidence": ["loan app"]},
                },
            }
        ),
    )

    assert persona.persona_id == "persona_1"
    assert persona.explanation_payload["triggered_fields"] == [
        "persona_signals.employer_dependency",
        "persona_signals.digital_borrowing",
    ]
    assert persona.evidence_quotes == ("madam helped", "loan app")


def test_persona_classifier_matches_persona_five_before_lower_priority_rules() -> None:
    persona = PersonaService().classify(
        build_transcript(),
        build_analysis(
            {
                "participant_profile": {
                    "smartphone_user": {"value": True, "evidence": ["whatsapp"]},
                    "has_bank_account": {"value": True, "evidence": ["bank account"]},
                },
                "persona_signals": {
                    "cyclical_borrowing": {"value": True, "evidence": ["every month"]},
                    "repayment_stress": {"value": True, "evidence": ["repayment pressure"]},
                    "self_reliance_non_borrowing": {"value": True, "evidence": ["use my savings"]},
                },
            }
        ),
    )

    assert persona.persona_id == "persona_5"
    assert persona.explanation_payload["triggered_fields"] == [
        "persona_signals.cyclical_borrowing",
        "persona_signals.repayment_stress",
    ]
