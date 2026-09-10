"""M1-8 완료 조건. ingest와 ask가 실제로 동작한다.

가짜 없이 CLI를 그대로 부른다. Ollama가 떠 있어야 하므로 integration이다.
색인은 tmp_path에 만든다. 저장소의 index/를 건드리지 않는다.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.domain.policy import DISCLAIMER
from src.interface import cli

SAMPLES = Path(__file__).resolve().parents[2] / "samples" / "documents.json"


@pytest.fixture(scope="module")
def index_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """샘플 문서로 색인을 한 번 만든다. 임베딩 호출을 아끼려고 모듈 단위다."""
    path = tmp_path_factory.mktemp("index") / "derma.npz"

    assert cli.main(["ingest", "--source", str(SAMPLES), "--index", str(path)]) == 0
    assert path.is_file()

    return path


@pytest.mark.integration
def test_ask_prints_answer_evidence_and_disclaimer(
    index_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = cli.main(
        ["ask", "민감성 피부는 메이크업 전에 무엇을 주의해야 하나요?", "--index", str(index_path)]
    )

    out = capsys.readouterr().out
    assert code == 0
    assert "근거" in out
    assert "sample-" in out
    assert DISCLAIMER in out


@pytest.mark.integration
def test_ask_refuses_an_unrelated_question(
    index_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = cli.main(
        ["ask", "자동차 엔진 오일은 몇 km마다 교환해야 하나요?", "--index", str(index_path)]
    )

    out = capsys.readouterr().out
    assert code == 0
    assert "답할 수 없습니다" in out
    assert DISCLAIMER not in out
