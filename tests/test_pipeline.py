from payday.analysis import AnalysisService
from payday.config import FeatureFlags, LLMSettings, Settings, SupabaseSettings, TranscriptionSettings
from payday.models import AnalysisResult, BatchUploadItem, PipelineStage, ProcessingStatus, Transcript
from payday.personas import PersonaService
from payday.pipeline import PaydayPipeline
from payday.repository import PaydayRepository
from payday.service import PaydayAppService
from payday.storage import StorageService


class FlakyTranscriptionService(TranscriptionService):
    def __init__(self, settings: TranscriptionSettings) -> None:
        super().__init__(settings)
        self.calls: dict[str, int] = {}

    def transcribe(self, asset, sample_mode: bool = False):
        self.calls.setdefault(asset.file_id, 0)
        self.calls[asset.file_id] += 1
        if asset.filename == "retry.wav" and self.calls[asset.file_id] == 1:
            raise RuntimeError("temporary transcription issue")
        if asset.filename == "fail.wav":
            raise RuntimeError("hard transcription failure")
        return super().transcribe(asset, sample_mode=sample_mode)


def build_settings() -> Settings:
    return Settings(
        app_env="test",
        supabase=SupabaseSettings(),
        llm=LLMSettings(),
        transcription=TranscriptionSettings(),
        features=FeatureFlags(use_sample_mode=True),
    )


def build_transcript(text: str = "test transcript") -> Transcript:
    return Transcript(text=text, provider="test", model="test")


def build_analysis(structured_output: dict) -> AnalysisResult:
    return AnalysisResult(summary="summary", structured_output=structured_output)


def test_pipeline_process_upload_returns_completed_result() -> None:
    service = PaydayAppService(build_settings())

    result = service.process_upload("demo.wav", "audio/wav", b"fake-bytes")

    assert result.asset is not None
    assert result.asset.filename == "demo.wav"
    assert result.persisted is True
    assert result.analysis is not None
    assert result.analysis.metrics["word_count"] > 0
    assert result.status is ProcessingStatus.COMPLETED
    assert result.current_stage is PipelineStage.STORAGE
    assert service.list_results()


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

    assert result.persona is not None
    assert result.persona.persona_id == "persona_3"
    assert result.persona.is_non_target is True
    assert result.persona.explanation_payload["triggered_fields"] == ["participant_profile.has_bank_account"]


    assert listing[0].id == interview.id
    assert detail.interview.status == "completed"
    assert detail.structured_response is not None
    assert detail.structured_response.smartphone_user is True
    assert detail.insight is not None
    assert detail.insight.key_quotes == ["I ask my employer first when money is short."]


def test_storage_service_builds_predictable_audio_paths() -> None:
    storage = StorageService(SupabaseSettings())
    asset = UploadedAsset(
        filename="../demo clip.wav",
        content_type="audio/wav",
        size_bytes=3,
        raw_bytes=b"123",
    )

    assert result.persona is not None
    assert result.persona.persona_id == "persona_3"
    assert result.persona.is_non_target is True
    assert result.persona.explanation_payload["triggered_fields"] == ["participant_profile.smartphone_user"]


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



def test_persona_classifier_prioritizes_override_before_other_matches() -> None:
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
    assert persona.explanation_payload["decision_type"] == "override"



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
