"""의존성 조립. 전 레이어를 아는 유일한 파일이다.

DI 프레임워크를 쓰지 않는다. 생성자 주입으로 충분하다. 모델이나 어댑터를 바꿀 때
고칠 곳이 이 파일 하나다 (docs/ARCHITECTURE.md 5절).

레이어 검사(scripts/check_layers.py)는 src/domain, src/application,
src/infrastructure, src/interface만 본다. 이 파일은 그 밖에 있으므로 모든 레이어를
import 해도 규칙에 걸리지 않는다. 이것이 의도한 배치다.
"""

from __future__ import annotations

from pathlib import Path

from src.application.answering import AnswerQuestion
from src.application.evaluation import EvaluateAnswers, EvaluateRetrieval
from src.application.indexing import BuildIndex
from src.application.ingest import IngestDocuments
from src.domain.policy import RetrievalPolicy, SafetyPolicy
from src.infrastructure.chunking import ParagraphChunker, WholeDocumentChunker
from src.infrastructure.embedding import OllamaEmbedder
from src.infrastructure.generation import OllamaGenerator
from src.infrastructure.index import NumpyVectorIndex
from src.infrastructure.aihub_loader import (
    AihubQaLoader,
    AihubQaPairLoader,
    AihubQaSetLoader,
)
from src.infrastructure.knowledge_loader import AihubKnowledgeLoader
from src.infrastructure.loaders import JsonDocumentLoader
from src.infrastructure.qa_loader import JsonQaSetLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# M1은 샘플 문서로 돈다. 실데이터는 CLI의 --aihub로 고른다.
DEFAULT_SOURCE = Path("samples/documents.json")

# .gitignore가 index/와 *.npz를 막고 있다. 색인 산출물은 커밋하지 않는다.
DEFAULT_INDEX = Path("index/derma.npz")

# 평가 입력. 샘플용이다. 실데이터는 AIHUB_SOURCE를 평가셋으로도 쓴다 (D-025).
DEFAULT_QASET = Path("eval/qaset.sample.json")

# 실데이터 경로를 코드 안에 둔다. 명령줄에 원본 경로를 적으면
# scripts/guard_data.py 훅이 막는다 (D-024와 같은 이유).
AIHUB_SOURCE = PROJECT_ROOT / "data" / "raw"
AIHUB_INDEX = PROJECT_ROOT / "index" / "aihub-qa.npz"

# 발췌 색인. 기본은 아니지만 웹 UI에서 비교용으로 함께 띄운다 (D-032).
AIHUB_EXCERPT_INDEX = PROJECT_ROOT / "index" / "aihub.npz"

# 논문 원문 색인 (M3-7). 문서 1건이 평균 25,608자라 청크 수가 QA 색인의 몇 배다.
AIHUB_DOC_INDEX = PROJECT_ROOT / "index" / "aihub-doc.npz"

# 사람 판정 기록. eval/raw/는 .gitignore 대상이다. 질문과 답변은 남기지 않는다
# (SAFETY.md S-7).
HUMAN_CHECK_LOG = PROJECT_ROOT / "eval" / "raw" / "human-check.jsonl"

# 색인 재료. "qa"는 QA쌍(질문+답변), "excerpt"는 근거 발췌, "doc"은 논문 원문이다.
# M3-6에서 qa가 F1 0.238 대 0.315로 앞섰다 (MEMORY.md D-032).
DEFAULT_MATERIAL = "qa"
MATERIALS = ("qa", "excerpt", "doc")

# 청킹 기준값. 여기 한 곳에 두는 이유는 평가 결과에 설정을 함께 저장해야 하기
# 때문이다. 값이 두 곳에 있으면 기록된 설정과 실제가 어긋난다 (eval/README.md).
CHUNK_MAX_CHARS = 512
CHUNK_OVERLAP = 64

# 논문 원문용 청크 (M3-7). 512자로 자르면 문서 하나가 청크 155개가 되고 1,315건
# 전량이 20만 청크를 넘는다. 임베딩만 3시간이다. 2,048자로 키우면 문서당 40개로
# 줄어 전량을 색인해도 5만 개 선이다.
#
# 문서 수를 줄이는 대신 청크를 키운 이유는 비교의 공정성 때문이다. 논문 일부만
# 색인하면 정답 근거가 아예 색인에 없는 질문이 생겨, 재료가 나쁜 것인지 빠진
# 것인지 구분할 수 없다.
DOC_CHUNK_MAX_CHARS = 2048
DOC_CHUNK_OVERLAP = 128

# 평가 표본. 10,043쌍 전량은 생성만으로 여러 시간이 걸린다. seed를 고정해 실험마다
# 같은 표본을 쓴다 (MEMORY.md D-026).
DEFAULT_SAMPLE_SIZE = 200
SAMPLE_SEED = 42

# bge-m3로 교체했다 (2026-09-08, D-005 개정). nomic-embed-text는 한국어 질문과
# 문단의 관계를 잡지 못했다. 같은 40쌍에서 정답 근거가 무관한 근거보다 가까운
# 비율이 nomic 17/40, bge-m3 25/40이었다. 임베딩은 약 1.7배 느리다.
EMBEDDING_MODEL = "bge-m3"

# gemma4:31b는 VRAM 16GB를 넘겨 못 쓴다 (ERRORS.md E-001, MEMORY.md D-005).
#
# qwen2.5:14b도 재 봤다. VRAM에는 들어가지만 F1이 0.2376에서 0.2016으로 떨어지고
# 거부가 2건에서 29건으로 늘었다 (M3-5, D-032). 3b를 유지한다.
GENERATION_MODEL = "qwen2.5:3b"


def build_ingest(
    source: Path = DEFAULT_SOURCE,
    material: str = DEFAULT_MATERIAL,
    limit: int | None = None,
) -> IngestDocuments:
    """디렉터리면 AI Hub 실데이터, 파일이면 샘플 JSON으로 본다.

    material이 "qa"면 QA쌍을, "excerpt"면 근거 발췌를, "doc"이면 논문 원문을
    색인 재료로 쓴다. 기본값이 "qa"인 근거는 D-032다. 나머지를 남겨 둔 이유는
    M3-6과 M3-7 비교를 다시 돌릴 수 있어야 하기 때문이다.

    limit은 "doc"에만 쓴다. 논문 전량은 청크가 30만 개를 넘어 임베딩에 몇 시간이
    걸린다. 표본으로 방향을 먼저 본다.
    """
    if material not in MATERIALS:
        raise ValueError(f"모르는 색인 재료다: {material} (가능한 값 {MATERIALS})")

    paragraphs = ParagraphChunker(max_chars=CHUNK_MAX_CHARS, overlap=CHUNK_OVERLAP)
    if not source.is_dir():
        return IngestDocuments(loader=JsonDocumentLoader(source), chunker=paragraphs)

    if material == "doc":
        # 논문은 길어서 쪼갠다. QA쌍(D-032)과 반대다. 청크는 다른 재료보다 크다.
        return IngestDocuments(
            loader=AihubKnowledgeLoader(source, limit=limit),
            chunker=ParagraphChunker(
                max_chars=DOC_CHUNK_MAX_CHARS, overlap=DOC_CHUNK_OVERLAP
            ),
        )
    if material == DEFAULT_MATERIAL:
        # QA쌍은 쪼개지 않는다. 쪼개면 상위 k를 같은 QA의 조각이 차지한다 (D-032).
        return IngestDocuments(
            loader=AihubQaPairLoader(source), chunker=WholeDocumentChunker()
        )
    return IngestDocuments(loader=AihubQaLoader(source), chunker=paragraphs)


def build_index_builder(resume_from: Path | None = None) -> BuildIndex:
    """resume_from에 색인 파일이 있으면 그 위에 이어서 쌓는다 (M2-4).

    없으면 빈 색인으로 시작한다. 경로를 줬는데 파일이 없는 경우도 정상이다.
    첫 실행과 재개를 같은 명령으로 처리하기 위해서다.
    """
    index = (
        NumpyVectorIndex.from_file(resume_from)
        if resume_from is not None and resume_from.is_file()
        else NumpyVectorIndex()
    )
    return BuildIndex(embedder=OllamaEmbedder(model=EMBEDDING_MODEL), index=index)


def existing_chunk_ids(index_path: Path) -> frozenset[str]:
    """이미 색인된 청크 ID. 파일이 없으면 빈 집합."""
    if not index_path.is_file():
        return frozenset()
    return NumpyVectorIndex.from_file(index_path).chunk_ids


def build_answer_usecase(
    index_path: Path = DEFAULT_INDEX,
    k: int = RetrievalPolicy().k,
    min_score: float = RetrievalPolicy().min_score,
) -> AnswerQuestion:
    """저장된 색인을 읽어 답변 유스케이스를 만든다.

    색인 파일이 없으면 NumpyVectorIndex.from_file이 FileNotFoundError를 낸다.
    빈 색인을 만들어 넘기지 않는다. 그러면 "근거 없음"으로 조용히 거부되어
    ingest를 안 돌린 것이 원인이라는 사실이 가려진다 (MEMORY.md D-013).

    k와 min_score를 인자로 받는 이유는 M3-3 실험 때문이다. 값을 바꾸는 실험을
    코드 수정으로 하면 실험마다 소스가 달라져 비교가 흐려진다.
    """
    return AnswerQuestion(
        embedder=OllamaEmbedder(model=EMBEDDING_MODEL),
        index=NumpyVectorIndex.from_file(index_path),
        generator=OllamaGenerator(model=GENERATION_MODEL),
        retrieval_policy=RetrievalPolicy(k=k, min_score=min_score),
        safety_policy=SafetyPolicy.default(),
    )


def build_qaset_loader(
    path: Path = DEFAULT_QASET,
) -> JsonQaSetLoader | AihubQaSetLoader:
    """디렉터리면 AI Hub QA 라벨링을 직접 읽고, 파일이면 평가셋 JSON을 읽는다.

    중간 변환 파일을 만들지 않는다. 질문과 답변을 저장소에 다시 쓰면 원본
    재배포에 해당한다 (D-006).
    """
    return AihubQaSetLoader(path) if path.is_dir() else JsonQaSetLoader(path)


def build_retrieval_eval(
    index_path: Path = DEFAULT_INDEX, k: int = RetrievalPolicy().k
) -> EvaluateRetrieval:
    """검색만 채점한다. 생성 모델을 부르지 않으므로 빠르다."""
    return EvaluateRetrieval(
        embedder=OllamaEmbedder(model=EMBEDDING_MODEL),
        index=NumpyVectorIndex.from_file(index_path),
        k=k,
    )


def build_answer_eval(
    index_path: Path = DEFAULT_INDEX,
    k: int = RetrievalPolicy().k,
    min_score: float = RetrievalPolicy().min_score,
) -> EvaluateAnswers:
    """답변 경로 전체를 돌린다. 케이스마다 생성이 일어나 느리다.

    embedder를 함께 넣어 보조 지표(임베딩 코사인)까지 낸다. 주 지표는 음절
    2-gram F1이다 (D-026).
    """
    return EvaluateAnswers(
        answerer=build_answer_usecase(index_path, k=k, min_score=min_score),
        embedder=OllamaEmbedder(model=EMBEDDING_MODEL),
    )


def eval_config(
    k: int,
    sample_size: int = DEFAULT_SAMPLE_SIZE,
    seed: int = SAMPLE_SEED,
    min_score: float = RetrievalPolicy().min_score,
    material: str = DEFAULT_MATERIAL,
    index_path: Path | None = None,
) -> dict[str, object]:
    """평가 결과와 함께 저장할 설정.

    나중에 어떤 조건에서 나온 점수인지 모르면 비교가 무의미해진다
    (eval/README.md).

    **색인 파일 이름을 함께 적는다.** M3-7에서 논문 색인을 평가했는데 결과
    파일에는 `material: "qa"`, `chunk_size: 512`가 적혔다. 실제는 doc과
    2,048이었다. 청크 크기가 재료마다 다른데 상수 하나만 읽었기 때문이다.
    기록된 설정과 실제가 어긋나는 것은 E-004와 같은 종류의 실패다.

    material은 여전히 사람이 맞게 넣어야 한다. 그래서 색인 파일 이름을 같이
    남긴다. 둘이 어긋나면 나중에 눈에 띈다.
    """
    doc_material = material == "doc"
    config: dict[str, object] = {
        "chunk_size": DOC_CHUNK_MAX_CHARS if doc_material else CHUNK_MAX_CHARS,
        "chunk_overlap": DOC_CHUNK_OVERLAP if doc_material else CHUNK_OVERLAP,
        "top_k": k,
        "min_score": min_score,
        "embed_model": EMBEDDING_MODEL,
        "gen_model": GENERATION_MODEL,
        "material": material,
        "sample_size": sample_size,
        "seed": seed,
    }
    if index_path is not None:
        config["index"] = index_path.name
    return config
