# 03 · infrastructure 레이어

경로: `src/infrastructure/**`

## 역할

`application/ports.py`에 정의된 포트를 실제로 구현한다. 외부 세계와 닿는 유일한 레이어다.

## 의존 규칙

| import 가능 | import 금지 |
|---|---|
| `domain`, `application`(포트만), `requests`, `numpy` | `interface` |

`application`에서는 포트만 가져온다. 유스케이스를 import 하면 방향이 뒤집힌다.

## 어댑터 규칙

### 외부 실패를 도메인 타입으로 바꾼다

Ollama가 죽어 있거나 타임아웃이 나면 `requests.exceptions`가 위로 새어 나가면 안 된다. 어댑터가 잡아서 도메인이 아는 형태로 바꾼다.

```python
def embed(self, texts: list[str]) -> list[list[float]]:
    try:
        r = requests.post(self._url, json={...}, timeout=self._timeout)
        r.raise_for_status()
    except requests.RequestException as e:
        raise EmbeddingUnavailable(str(e)) from e
```

`application`이 `requests`를 몰라도 되게 만드는 것이 목적이다.

### 타임아웃을 반드시 준다

`requests` 호출에 `timeout`을 빼면 무한 대기한다. 모델을 바꾸면 응답 시간이 크게 달라지므로(E-001에서 189초 대 6초) 생성 타임아웃은 넉넉히 잡는다.

| 호출 | 권장 타임아웃 |
|---|---|
| 임베딩 배치 1회 | 30초 |
| 생성 1건 | 300초 |

### 배치와 중간 저장

전체 임베딩(M2-4)은 9,377건이다. 한 번에 돌리다 끊기면 처음부터 다시 해야 한다.

- 배치 단위로 처리한다. `OllamaEmbedder`는 한 요청에 64건을 보낸다
- N건마다 중간 결과를 디스크에 저장한다
- 재시작 시 저장된 지점부터 이어간다

**중간 저장은 어댑터의 일이 아니다.** `Embedder` 포트는 "텍스트를 벡터로"만 한다. 저장과 재개는 `BuildIndex` 유스케이스가 맡는다. 어댑터에 넣으면 임베딩 모델을 바꿀 때 재개 로직까지 다시 짜게 된다.

이건 선택이 아니다. `ERRORS.md`에 예상 실패로 이미 적혀 있다.

## 어댑터 목록

| 어댑터 | 포트 | 비고 |
|---|---|---|
| `AihubJsonLoader` | `DocumentLoader` | `.json` |
| `AihubDocLoader` | `DocumentLoader` | `.doc`. 형식 확인 후 구현 (M2-1) |
| `ParagraphChunker` | `Chunker` | 문단 분리 + 길이 제한 + 오버랩 |
| `OllamaEmbedder` | `Embedder` | `POST /api/embed`. 배치 요청 (`MEMORY.md` D-016) |
| `NumpyVectorIndex` | `VectorIndex` | numpy 배열 + `.npz` 저장 |
| `OllamaGenerator` | `Generator` | `POST /api/chat` |

## NumpyVectorIndex 구현 메모

벡터를 미리 L2 정규화해 두면 코사인 유사도가 내적만으로 나온다. 9,377 × 768 전수 계산이라 이 차이가 실제로 크다.

```python
# 적재 시 1회
self._matrix = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)

# 검색 시
scores = self._matrix @ query_vec_normalized
```

저장은 `np.savez_compressed`로 한다. **파일은 `.npz` 하나다.** 벡터는 `vectors` 배열에, 청크 메타데이터는 같은 파일 안의 `chunks` 키에 JSON 문자열로 넣는다. 파일을 둘로 나누면 한쪽만 갱신되거나 한쪽만 지워진 상태가 생긴다 (`MEMORY.md` D-018).

읽을 때는 `np.load(path, allow_pickle=False)`를 쓴다. pickle을 허용하면 파일을 읽는 것만으로 코드가 실행된다.

## 청킹 전략

`Chunker`는 Strategy 패턴이다. 여러 개를 만들어 측정 비교할 것이다 (`docs/ROADMAP.md` M3-1).

첫 구현 기준값:

| 항목 | 초기값 | 근거 |
|---|---|---|
| 최대 길이 | 512자 | 임베딩 모델 입력 한계와 문맥 보존의 절충 |
| 오버랩 | 64자 | 문단 경계에서 문맥이 끊기는 것을 완화 |
| 분리 기준 | 빈 줄 → 문장 | 문단이 우선, 너무 길면 문장 단위로 재분할 |

**이 값들은 가설이다.** M3-1에서 측정으로 조정한다.

## 메타데이터 부착

청크마다 피부 문제 유형과 부위를 붙인다. 나중에 필터링 검색(M3-2)의 근거가 된다. 원본에서 못 얻으면 `None`으로 두고 추정하지 않는다.

## 로그 주의

**원본 텍스트를 로그에 남기지 않는다.** 청크 ID와 점수만 남긴다 (`00-common.md` 로그 항목).

## 테스트

`tests/infrastructure/` 아래. 실제 Ollama가 필요한 테스트는 `@pytest.mark.integration`으로 표시하고 기본 실행에서 제외한다.

```bash
pytest -q -m "not integration"   # 빠른 실행
pytest -q -m integration          # Ollama 필요
```
