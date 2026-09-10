# PACKAGES

의존성 대장. **여기에 없는 패키지는 설치하지 않는다.** 추가하려면 아래 형식으로 근거를 적고 승인을 받는다.

## 원칙

의존성 하나마다 학습 비용과 유지 비용이 붙는다. 이 프로젝트는 학습이 목적이므로 (`MEMORY.md` D-001), "직접 짜면 몇 줄인가"를 먼저 따진다. 20줄로 되는 일에 패키지를 넣지 않는다.

## 런타임 의존성

| 패키지 | 버전 | 쓰는 곳 | 왜 필요한가 | 직접 짜면 |
|---|---|---|---|---|
| `requests` | 최신 | `infrastructure/embedding.py`, `generation.py` | Ollama HTTP API 호출 | `urllib`로 가능하지만 에러 처리·타임아웃·재시도가 번거롭다 |
| `numpy` | 최신 | `infrastructure/index.py` | 코사인 유사도 전수 계산 | 순수 Python은 9,377 × 768 계산에서 너무 느리다 |

**이게 전부다.** v1은 두 개로 완주한다.

## 개발 의존성

| 패키지 | 쓰는 곳 | 비고 |
|---|---|---|
| `pytest` | 테스트 | 표준 `unittest`도 되지만 픽스처와 파라미터화가 편하다 |

## 조건부 후보

아직 넣지 않는다. 해당 조건이 실제로 발생하면 그때 검토한다.

| 패키지 | 넣을 조건 | 대안 |
|---|---|---|
| `tqdm` | 전체 임베딩(M2-4)에서 진행률이 필요할 때 | `print`로 충분할 수 있다 |
| `pydantic` | 값 객체 검증이 실제로 문제가 될 때 | `dataclass` + `__post_init__`으로 충분 |

## 도입하지 않기로 확정한 것

| 패키지 | 이유 | 근거 |
|---|---|---|
| `langchain` | 6단계를 함수 뒤로 숨긴다. 학습 목적과 충돌 | `MEMORY.md` D-003 |
| `llama-index` | **M4-2에서 실제로 재 봤다.** 품질은 같고(F1 -0.0028, 노이즈 안) 의존성이 2개에서 70개로 는다. 색인 크기 4.5배 | `MEMORY.md` D-003, D-036 |
| `chromadb`, `faiss-cpu`, `lancedb` | 9,377건 규모에서 불필요. numpy 전수 검색이 밀리초 | `MEMORY.md` D-004 |
| `sentence-transformers` | Ollama가 이미 임베딩을 제공한다. torch 의존성이 무겁다 | `MEMORY.md` D-005 |
| `fastapi`, `gradio` | M4 선택 항목. v1 범위 밖 | `docs/ROADMAP.md` |
| `unittest.mock` | 인터페이스가 바뀌어도 조용히 통과한다. 포트를 구현한 가짜 클래스를 직접 만든다 | `docs/ARCHITECTURE.md` §8 |
| `python-docx` | `.doc` 중 ZIP(docx) 724건은 `zipfile` + `xml.etree`로 읽는다. 필요한 것은 문단 텍스트뿐이라 20줄이면 된다 | `src/infrastructure/doc_reader.py` (M3-7) |
| `olefile`, LibreOffice CLI | OLE2 5,513건도 표준 라이브러리로 읽었다. `olefile`을 넣어도 Word 조각표(piece table)는 직접 파싱해야 해서 절반만 얻는다. 실측 6,237건 전량 추출 성공 | `src/infrastructure/doc_reader.py` (M3-7) |

## 비교 실험은 격리한다

M4-2처럼 "쓰지 않기로 한 것"을 실제로 재 봐야 할 때가 있다. 그때도 메인 환경에는 넣지 않는다.

| 항목 | 어떻게 |
|---|---|
| 가상환경 | `.venv-llamaindex` (메인 `.venv`와 분리, `.gitignore` 대상) |
| 의존성 목록 | `requirements-llamaindex.txt` (`requirements.txt` 무변경) |
| 코드 | `experiments/` 아래에만. `src/`는 읽기만 한다 |

이 문서는 원래 "별도 브랜치"라고 적었지만 이 프로젝트는 git 저장소가 아니다. 폴더와 가상환경으로 대신했다. 절차는 `experiments/m4-2-llamaindex/README.md`에 있다.

## 추가 신청 형식

새 패키지가 필요하면 이 형식으로 제안한다.

```markdown
### 제안: <패키지명>

**쓰는 곳**: <파일 경로>
**해결하는 문제**: <구체적으로>
**직접 짜면**: <몇 줄, 어떤 난점>
**대안 검토**: <표준 라이브러리나 이미 있는 패키지로 되는가>
**의존성 무게**: <설치 크기, 하위 의존성 개수>
```

승인되면 이 문서의 표에 추가하고 `requirements.txt`를 갱신한다.

## requirements.txt

```
requests
numpy
```

## requirements-dev.txt

```
-r requirements.txt
pytest
```

## 런타임 외부 의존

패키지는 아니지만 없으면 동작하지 않는 것.

| 대상 | 버전/모델 | 확인 방법 |
|---|---|---|
| Python | 3.12 | `python --version` |
| Ollama | 설치됨 | `ollama list` |
| `nomic-embed-text` | 274 MB | `ollama list`에 존재 |
| `qwen2.5:3b` | 1.9 GB | 생성 모델. VRAM에 여유롭게 들어감. 6.0초 / 14.1 tok/s |

### 쓰지 않는 보유 모델

| 모델 | 크기 | 왜 쓰지 않는가 |
|---|---|---|
| `gemma4:31b` | 19 GB | VRAM 16GB 초과. 189초 / 응답 내용 없음 (`ERRORS.md` E-001) |
| `qwen3.6:35b` | 23 GB | 더 크다. 같은 이유로 배제 |

**모델 선택 기준은 크기가 아니라 VRAM 16GB 안에 들어가는지다.**

### 다음 후보

`qwen2.5:3b`는 빠르지만 3B라 품질 상한이 낮을 수 있다. VRAM에 들어가는 중간 크기 모델(8~14B, 양자화 시 8~10GB)을 받아 비교하는 것이 다음 검토 대상이다. 단 **기준선(M2-8)을 만든 뒤에** 바꾼다.
