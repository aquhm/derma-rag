# 01 · domain 레이어

경로: `src/domain/**`

## 역할

이 프로젝트에서 **무엇이 참인가**만 담는다. 어떻게 저장하고 어떻게 계산하는지는 담지 않는다.

## 의존 규칙

| import 가능 | import 금지 |
|---|---|
| 표준 라이브러리 (`dataclasses`, `typing`, `enum`, `re`) | `requests`, `numpy` |
| — | `application`, `infrastructure`, `interface` |

**`numpy`도 금지다.** 벡터를 어떻게 저장하는지는 `infrastructure`의 사정이다. `domain`에서 벡터를 다뤄야 하면 `list[float]`로 받는다.

이 규칙 덕분에 `domain` 테스트에는 목이 필요 없고, Ollama가 꺼져 있어도 테스트가 돈다.

## 담는 것

### 값 객체

`@dataclass(frozen=True)`로 만든다. 불변이면 어디서 바뀌었는지 추적할 필요가 없다.

```python
@dataclass(frozen=True)
class Chunk:
    id: str
    doc_id: str
    text: str
    skin_type: SkinType | None
    area: str | None
```

### 열거형

데이터셋의 4대 분류처럼 값이 고정된 것은 `Enum`으로 만든다. 문자열을 그대로 쓰면 오타가 런타임까지 간다.

```python
class SkinType(Enum):
    SENSITIVE = "민감성"
    INFLAMMATORY = "염증성"
    PIGMENT = "색소문제"
    TEXTURE = "조직변화"
```

### 정책 객체

판정 규칙을 값 객체로 뺀다. 임계값 조정이 코드 수정이 아니라 값 변경이 된다.

```python
@dataclass(frozen=True)
class RetrievalPolicy:
    k: int = 5
    min_score: float = 0.35

    def is_sufficient(self, evidence: list[Evidence]) -> bool:
        return bool(evidence) and evidence[0].score >= self.min_score
```

`SafetyPolicy`도 같은 형태다. 금지 표현 목록과 필수 부착 문구를 담는다.

### 결과 타입

```python
Result = Answer | Refusal
```

`Answer`와 `Refusal`을 분리하는 이유는 근거 없는 생성을 타입 수준에서 막기 위해서다. 호출부가 반드시 두 경우를 분기해야 한다.

## 담지 않는 것

- 파일 읽기·쓰기
- HTTP 호출
- 임베딩 계산
- 로깅 설정
- CLI 출력 형식

## 테스트

`tests/domain/` 아래. 목 없이 순수 단위 테스트만 쓴다.

정책 객체는 경계값을 반드시 테스트한다. `min_score` 정확히 같은 값에서 통과하는지 아닌지가 애매하면 버그가 된다.
