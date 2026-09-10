# 02 · application 레이어

경로: `src/application/**`

## 역할

**무엇을 할지** 정한다. **어떻게 할지**는 포트에 맡긴다.

## 의존 규칙

| import 가능 | import 금지 |
|---|---|
| `domain`, 표준 라이브러리 | `infrastructure`, `interface` |
| — | `requests`, `numpy` |

## 포트를 여기에 두는 이유

포트(`Protocol`)를 `application`에 정의하고 `infrastructure`가 구현한다. 그러면 `application`이 `infrastructure`를 import 하지 않고도 그 기능을 쓸 수 있다. 의존이 역전된다.

포트를 `infrastructure`에 두면 이 방향이 무너진다. 반드시 `application/ports.py`에 둔다.

```python
class Embedder(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]: ...
```

`Protocol`을 쓰는 이유는 상속이 필요 없기 때문이다. 어댑터가 `Embedder`를 상속하지 않아도 시그니처만 맞으면 성립한다.

## 유스케이스 규칙

### 하나의 유스케이스는 하나의 일만 한다

| 유스케이스 | 하는 일 |
|---|---|
| `IngestDocuments` | 적재 → 청크 분할 |
| `BuildIndex` | 임베딩 → 색인 저장 |
| `AnswerQuestion` | 검색 → 판정 → 생성 또는 거부 |
| `EvaluateAnswers` | 채점 |

`IngestDocuments`가 임베딩까지 하면 청킹만 바꿔 실험할 수 없다. 단계를 나누는 이유는 개별 측정을 위해서다.

### 의존은 생성자로 받는다

```python
class AnswerQuestion:
    def __init__(
        self,
        embedder: Embedder,
        index: VectorIndex,
        generator: Generator,
        retrieval_policy: RetrievalPolicy,
        safety_policy: SafetyPolicy,
    ) -> None:
        ...
```

유스케이스 안에서 어댑터를 직접 만들지 않는다. 조립은 `composition.py`에서만 한다.

### 두 게이트를 반드시 통과시킨다

`AnswerQuestion`은 아래 순서를 지킨다. 이건 선택이 아니다 (`MEMORY.md` D-007).

```
검색 → RetrievalPolicy.is_sufficient()
        ├ False → Refusal 반환. 생성 호출 안 함
        └ True  → 프롬프트 조립 → 생성 → SafetyPolicy.check()
                                          ├ 위반 → Refusal
                                          └ 통과 → Answer
```

**게이트를 프롬프트 문구로 대체하지 않는다.** 프롬프트는 어길 수 있고 `if` 문은 못 어긴다.

### 프롬프트 조립

프롬프트 문자열은 `application`에서 만든다. 생성 모델이 무엇인지는 모른 채로 만든다. 모델별 포맷 변환은 어댑터의 일이다.

프롬프트에 넣는 것은 **검색된 근거뿐이다.** 색인 전체나 모델의 사전 지식에 기대는 문구를 넣지 않는다.

## 테스트

`tests/application/` 아래. 포트를 구현한 가짜 클래스를 직접 만든다.

```python
class FakeEmbedder:
    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0] for _ in texts]
```

`unittest.mock`을 쓰지 않는 이유는 포트 시그니처가 바뀌어도 목은 조용히 통과하기 때문이다. 가짜 클래스는 시그니처가 어긋나면 바로 깨진다.

**반드시 테스트할 것**: 근거가 부족할 때 `Generator`가 **호출되지 않는지**. 가짜 생성기에 호출 카운터를 두고 0인지 확인한다.
