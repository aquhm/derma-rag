# ARCHITECTURE

## 1. 왜 레이어를 나누는가

이 규모(코드 300줄 남짓)에서 레이어 분리는 과해 보일 수 있다. 그럼에도 나누는 이유는 두 가지다.

**교체 가능성이 실제로 있다. 그리고 실제로 일어났다.** 착수 시점의 생성 모델은 `gemma4:31b`였으나 19GB가 VRAM 16GB를 초과해 RAM으로 스필됐고, 응답 하나에 189초가 걸려 `qwen2.5:3b`로 교체했다(`ERRORS.md` E-001). 이때 고친 곳은 `composition.py` 한 줄이었다. 레이어를 나누지 않았으면 호출부를 전부 찾아 고쳐야 했다.

**학습 목적이다.** "왜 이렇게 나누는가"를 글로 읽는 것과 직접 겪는 것은 다르다. 이 프로젝트는 후자를 노린다.

## 2. 네 레이어

```
┌─────────────────────────────────────────────┐
│  interface     CLI · (선택) Gradio/FastAPI   │
└──────────────────┬──────────────────────────┘
                   │ 호출
┌──────────────────▼──────────────────────────┐
│  application   유스케이스 + 포트(인터페이스)   │
└──────────────────┬──────────────────────────┘
                   │ 사용
┌──────────────────▼──────────────────────────┐
│  domain        순수 규칙 · 값 객체            │
└─────────────────────────────────────────────┘
                   ▲
                   │ 포트 구현
┌──────────────────┴──────────────────────────┐
│  infrastructure  Ollama · 파일 · numpy 색인   │
└─────────────────────────────────────────────┘
```

### 의존 규칙

**의존은 항상 안쪽으로만 향한다.**

| 레이어 | import 해도 되는 것 | 절대 import 하면 안 되는 것 |
|---|---|---|
| `domain` | 표준 라이브러리만 | 다른 모든 레이어, `requests`, `numpy` |
| `application` | `domain`, 표준 라이브러리 | `infrastructure`, `interface`, `requests` |
| `infrastructure` | `domain`, `application`(포트만), 외부 패키지 | `interface` |
| `interface` | 전부 | — |

이 규칙은 `scripts/check_layers.py`가 검사한다. hook으로 자동 실행되므로 사람이 기억할 필요가 없다.

**`domain`이 `numpy`를 import 하지 못하는 이유** — `domain`은 "무엇이 참인가"만 담는다. 벡터를 어떻게 저장하고 계산하는지는 `infrastructure`의 사정이다. `domain`이 numpy에 의존하면 저장 방식을 바꿀 때 도메인 규칙까지 흔들린다.

## 3. 각 레이어에 무엇을 두는가

### domain — 순수 규칙

외부에 아무것도 의존하지 않는다. 테스트에 목이 필요 없다.

| 타입 | 역할 |
|---|---|
| `Document` | 지식데이터 1건. 원문과 메타데이터(피부 문제 유형, 부위) |
| `Chunk` | 분할된 문단. 원본 문서 ID와 위치를 안다 |
| `Query` | 사용자 질문 |
| `Evidence` | 검색된 근거. 청크 + 유사도 점수 |
| `Answer` | 생성된 답 + 근거 목록 + 면책 문구 |
| `Refusal` | 답할 수 없음. 사유를 담는다 |
| `SafetyPolicy` | 금지 표현 판정, 필수 부착 문구 |
| `RetrievalPolicy` | 유사도 임계값, k 값, 근거 충분 여부 판정 |

**`Answer`와 `Refusal`을 별도 타입으로 둔다.** 답을 못 하는 상황을 예외가 아니라 값으로 표현한다. 예외는 잡지 않으면 흘러가지만, 값은 반드시 처리해야 한다. 근거 없이 생성되는 사고를 타입 수준에서 막는 장치다.

### application — 유스케이스와 포트

무엇을 할지 정하되, 어떻게 할지는 포트에 맡긴다.

**유스케이스**

| 이름 | 단계 | 하는 일 |
|---|---|---|
| `IngestDocuments` | 2 | 원본 적재 → 청크 분할 |
| `BuildIndex` | 3–4 | 청크 임베딩 → 색인 저장 |
| `AnswerQuestion` | 5–6 | 검색 → 근거 판정 → 생성 또는 거부 |
| `EvaluateAnswers` | — | QA쌍 채점 |

**포트 (application이 정의, infrastructure가 구현)**

```python
class DocumentLoader(Protocol):
    def load(self) -> Iterable[Document]: ...

class Chunker(Protocol):
    def split(self, doc: Document) -> list[Chunk]: ...

class Embedder(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]: ...

class VectorIndex(Protocol):
    def add(self, chunks: list[Chunk], vectors: list[list[float]]) -> None: ...
    def search(self, vector: list[float], k: int) -> list[Evidence]: ...
    def save(self, path: Path) -> None: ...

class Generator(Protocol):
    def generate(self, prompt: str) -> str: ...
```

**`VectorIndex`에 `load`가 없다.** 파일에서 읽는 것은 어댑터의 classmethod(`NumpyVectorIndex.from_file`)가 맡고, 그 호출은 5절의 조립부에서만 일어난다. 포트에 `load`를 두면 아직 로드되지 않은 빈 색인 상태가 존재하게 되고, 그 상태에서 `search()`가 무엇을 반환해야 하는지가 애매해진다 (`MEMORY.md` D-013).

포트를 `application`에 두는 것이 핵심이다. `application`이 `infrastructure`를 import 하지 않고도 그 기능을 쓸 수 있다. 의존이 역전된다.

### infrastructure — 어댑터

포트를 실제로 구현한다. 외부 패키지는 여기서만 쓴다.

| 어댑터 | 구현하는 포트 | 비고 |
|---|---|---|
| `AihubJsonLoader` | `DocumentLoader` | `.json` 처리 |
| `AihubDocLoader` | `DocumentLoader` | `.doc` 처리. 형식 확인 후 구현 |
| `ParagraphChunker` | `Chunker` | 문단 분리 + 길이 제한 + 오버랩 |
| `OllamaEmbedder` | `Embedder` | `requests`로 `/api/embed` 배치 호출 (`MEMORY.md` D-016) |
| `NumpyVectorIndex` | `VectorIndex` | numpy 배열 + 코사인 유사도 + `.npz` 저장 |
| `OllamaGenerator` | `Generator` | `requests`로 `/api/chat` 호출 |

### interface — 진입점

CLI만 만든다. 웹 UI는 M3 이후 선택.

```
python -m src.cli ingest   → IngestDocuments + BuildIndex
python -m src.cli ask      → AnswerQuestion
python -m src.cli eval     → EvaluateAnswers   (M2-5 이후)
```

`src/cli.py`는 `src/interface/cli.py`의 `main`을 부르는 얇은 진입점이다. 문서에 적힌 명령(`python -m src.cli`)과 레이어 규칙(진입점은 `interface` 아래)을 둘 다 지키기 위한 배치다 (`MEMORY.md` D-023).

## 4. 디렉터리

```
src/
├── domain/
│   ├── models.py         Document, Chunk, Query, Evidence, Answer, Refusal, EvalCase
│   ├── policy.py         SafetyPolicy, RetrievalPolicy
│   └── scoring.py        답변 정확도 채점 (음절 2-gram F1)
├── application/
│   ├── ports.py          Protocol 정의
│   ├── ingest.py         IngestDocuments
│   ├── indexing.py       BuildIndex
│   ├── answering.py      AnswerQuestion
│   └── evaluation.py     EvaluateRetrieval, EvaluateAnswers
├── infrastructure/
│   ├── loaders.py        AihubJsonLoader, AihubDocLoader
│   ├── chunking.py       ParagraphChunker
│   ├── embedding.py      OllamaEmbedder
│   ├── index.py          NumpyVectorIndex
│   ├── generation.py     OllamaGenerator
│   └── qa_loader.py      JsonQaSetLoader (평가셋)
├── interface/
│   └── cli.py            argparse 본체, 출력 형식
├── cli.py                python -m src.cli 진입점 (interface.cli.main 호출)
└── composition.py        의존성 조립 (유일하게 전 레이어를 아는 곳)
```

## 5. 조립은 한 곳에서만

`composition.py`가 구현체를 골라 유스케이스에 주입한다. DI 프레임워크를 쓰지 않는다. 생성자 주입으로 충분하다.

```python
def build_answer_usecase() -> AnswerQuestion:
    return AnswerQuestion(
        embedder=OllamaEmbedder(model="nomic-embed-text"),
        index=NumpyVectorIndex.from_file(INDEX_PATH),
        generator=OllamaGenerator(model="qwen2.5:3b"),
        retrieval_policy=RetrievalPolicy(k=5, min_score=0.35),
        safety_policy=SafetyPolicy.default(),
    )
```

모델을 바꿀 때 고칠 곳이 이 함수 하나다.

## 6. 적용하는 설계 패턴

과하게 쓰지 않는다. 각 패턴에 대해 "왜 여기서 필요한가"가 답이 되는 것만 쓴다.

| 패턴 | 어디에 | 왜 필요한가 |
|---|---|---|
| **Ports & Adapters** | `application/ports.py` ↔ `infrastructure/` | 모델 교체가 실제로 예정돼 있다 (MEMORY.md D-005) |
| **Strategy** | `Chunker` | 청킹 전략이 검색 품질을 좌우한다. 여러 개를 만들어 측정 비교할 것이다 |
| **Repository** | `VectorIndex` | 색인 저장 방식을 도메인에서 감춘다. 나중에 벡터 DB로 갈 여지 (D-004 뒤집을 조건) |
| **Result 타입** | `Answer` \| `Refusal` | 근거 없는 생성을 타입으로 막는다. 예외는 안 잡으면 흘러가지만 유니온 타입은 반드시 분기해야 한다 |
| **Policy** | `SafetyPolicy`, `RetrievalPolicy` | 판정 규칙을 값 객체로 빼서 테스트 가능하게 만든다. 임계값 조정이 코드 수정이 아니라 값 변경이 된다 |
| **Pipeline** | `IngestDocuments` | 적재 → 분할 → 임베딩 → 저장이 순차 단계다. 각 단계를 독립적으로 교체·측정할 수 있어야 한다 |

**쓰지 않는 패턴** — Factory, Observer, Decorator, Singleton. 지금 필요한 이유가 없다. 필요해지면 그때 도입하고 MEMORY.md에 기록한다.

## 7. 답변 경로 상세

5–6단계는 안전 규칙이 걸려 있으므로 흐름을 명시한다.

```
질문
 └→ Embedder.embed(질문)
     └→ VectorIndex.search(벡터, k=5)
         └→ RetrievalPolicy.is_sufficient(evidence)
             ├─ False → Refusal("근거 없음")          ← 여기서 끝. 생성 안 함
             └─ True  → 프롬프트 조립 (근거 문단만 포함)
                 └→ Generator.generate(프롬프트)
                     └→ SafetyPolicy.check(생성 결과)
                         ├─ 위반 → Refusal("안전 규칙 위반")
                         └─ 통과 → Answer(본문 + 근거 목록 + 면책 문구)
```

두 개의 게이트가 있다. 앞의 게이트는 **근거가 없으면 생성하지 않는다**(MEMORY.md D-007). 뒤의 게이트는 **생성 결과가 진단 표현을 담으면 내보내지 않는다**.

둘 다 프롬프트 지시가 아니라 코드 분기다. 프롬프트는 어길 수 있고 분기는 못 어긴다.

## 8. 테스트 전략

| 레이어 | 방식 | 목 필요 여부 |
|---|---|---|
| `domain` | 순수 단위 테스트 | 없음 |
| `application` | 포트를 가짜 구현으로 대체 | 가짜 구현 (직접 작성) |
| `infrastructure` | 실제 Ollama를 띄운 통합 테스트 | 없음. 느리므로 별도 마크 |
| `interface` | 스모크 테스트 | — |

`unittest.mock` 대신 포트를 구현한 작은 가짜 클래스를 직접 만든다. 목 라이브러리는 인터페이스가 바뀌어도 조용히 통과해서 위험하다.

## 9. 참고

- 레이어별 코딩 지침: `docs/guidelines/`
- 검사 스크립트: `scripts/check_layers.py`
- 안전 규칙 전문: `docs/process/SAFETY.md`
