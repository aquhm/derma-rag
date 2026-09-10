# eval/

평가 결과를 두는 곳이다.

## 파일

| 파일 | 내용 |
|---|---|
| `baseline.json` | 기준선 점수. **M2-8에서 한 번 만들고 덮어쓰지 않는다** |
| `<타임스탬프>.json` | 각 실험 결과 |
| `raw/<이름>.cases.json` | 케이스별 결과. `.gitignore` 대상 |

## 규칙

- **기준선을 덮어쓰지 않는다.** 모든 개선은 기준선 대비로 판단한다.
- 집계 결과만 커밋한다. 질문·답변 원문은 `raw/`에 두고 커밋하지 않는다.
- Recall@k와 답변 정확도를 분리해 기록한다. 근거는 `MEMORY.md` D-008.

## 형식

```json
{
  "timestamp": "",
  "change": "마지막 평가 이후 바뀐 것 하나",
  "recall_at_5": 0.0,
  "answer_accuracy": 0.0,
  "refusal_rate": 0.0,
  "sample_count": 0,
  "config": {
    "chunk_size": 512,
    "chunk_overlap": 64,
    "top_k": 5,
    "min_score": 0.35,
    "embed_model": "nomic-embed-text",
    "gen_model": "qwen2.5:3b"
  }
}
```

`config`를 반드시 함께 저장한다. 나중에 어떤 설정에서 나온 점수인지 모르면 비교가 무의미해진다.

실제 출력은 위 형식에 다음 키가 더 붙는다. `python -m src.cli eval --out <경로>`가 만든다.

| 키 | 뜻 |
|---|---|
| `graded_count` | 정답 문서 ID가 있어 Recall을 잴 수 있었던 건수 |
| `hit_count` | 그중 적중 건수 |
| `mean_best_score` | 케이스별 최고 유사도의 평균. 임계값 조정(M3-3)의 근거 |
| `scored_count` | 답변 정확도를 잰 건수. 거부되거나 기대 답변이 없는 건은 빠진다 |
| `answer_similarity` | 보조 지표. 기대 답변과 생성 답변의 임베딩 코사인 |
| `config.sample_size`, `config.seed` | 표본 수와 seed. 기본값 200 / 42 (`MEMORY.md` D-026) |

### 케이스별 결과 파일

`--out eval/x.json`을 주면 `eval/raw/x.cases.json`도 함께 만들어진다. 한 행이 케이스 하나다.

```json
{"case_id": "A000101_06_QA1", "outcome": "answered", "accuracy": 0.31, "similarity": 0.82, "detail": ""}
```

`outcome`은 `answered` / `refused` / `failed` 셋 중 하나다. `detail`에는 거부 사유나 예외 종류만 들어간다. **질문과 답변 원문은 들어 있지 않다** (`SAFETY.md` S-7).

이 파일이 필요한 이유는 실험마다 채점 대상이 달라지기 때문이다. 요약 숫자만 비교하면 "거부가 줄어 평균이 내려간" 것과 "답이 나빠진" 것이 구분되지 않는다. 비교는 다음으로 한다.

```bash
.venv/Scripts/python.exe scripts/compare_eval.py eval/raw/이전.cases.json eval/raw/이후.cases.json
```

`answer_accuracy`는 음절 2-gram F1의 평균이다. 절대 품질이 아니라 **기준선 대비 비교용**이다. 정의와 한계는 `MEMORY.md` D-026.
