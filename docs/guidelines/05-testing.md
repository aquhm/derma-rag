# 05 · 테스트 지침

## 레이어별 방식

| 레이어 | 방식 | 외부 의존 |
|---|---|---|
| `domain` | 순수 단위 테스트 | 없음 |
| `application` | 포트를 가짜 구현으로 대체 | 없음 |
| `infrastructure` | 실제 Ollama 통합 테스트 | 있음. `@pytest.mark.integration` |
| `interface` | 스모크 테스트 | 없음 |

## 목을 쓰지 않는다

`unittest.mock` 대신 포트를 구현한 가짜 클래스를 직접 만든다.

```python
class FakeGenerator:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def generate(self, prompt: str) -> str:
        self.calls.append(prompt)
        return "테스트 답변"
```

이유는 두 가지다. 포트 시그니처가 바뀌면 가짜 클래스는 바로 깨지지만 목은 조용히 통과한다. 그리고 호출 여부를 세는 것이 목의 `assert_called` 계열보다 읽기 쉽다.

## 반드시 있어야 하는 테스트

이 세 가지가 없으면 이 프로젝트는 목적을 잃는다.

### 1. 근거가 없으면 생성기를 호출하지 않는다

```python
def test_no_evidence_skips_generation():
    gen = FakeGenerator()
    usecase = AnswerQuestion(..., generator=gen, retrieval_policy=RetrievalPolicy(min_score=0.9))
    result = usecase.execute(Query("무관한 질문"))
    assert isinstance(result, Refusal)
    assert gen.calls == []          # 호출 자체가 없어야 한다
```

`Refusal`이 반환되는 것만으로 부족하다. **생성기가 호출되지 않았는지**까지 확인한다.

### 2. 진단 표현이 걸러진다

`SafetyPolicy`의 금지 표현 목록에 대해, 생성 결과가 그것을 담으면 `Refusal`이 되는지 확인한다.

### 3. 면책 문구가 항상 붙는다

`Answer`가 반환되는 모든 경로에서 면책 문구가 포함되는지 확인한다.

## 경계값을 테스트한다

정책 객체는 경계에서 애매하면 버그가 된다.

```python
@pytest.mark.parametrize("score,expected", [
    (0.34, False),
    (0.35, True),    # 임계값과 정확히 같을 때
    (0.36, True),
])
def test_retrieval_threshold(score, expected): ...
```

## 테스트에 실데이터를 쓰지 않는다

`data/` 아래 원본을 테스트 픽스처로 쓰지 않는다. 커밋될 위험이 있다 (`MEMORY.md` D-006). 짧은 가짜 문서를 코드 안에 둔다.

## 실행

```bash
pytest -q                        # 전체
pytest -q -m "not integration"   # Ollama 없이
pytest -q tests/domain           # 레이어 하나만
```

## 커버리지 목표를 세우지 않는다

숫자를 맞추려고 의미 없는 테스트를 쓰게 된다. 대신 **위 세 가지 필수 테스트가 살아 있는지**를 기준으로 삼는다.
