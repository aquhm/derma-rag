"""M4-2 · 비교 상대편(우리 구현) 색인을 오늘 코드로 다시 만든다.

기준선 `eval/m3-6-qaindex.json`은 2026-09-09에 잰 값이다. 그 뒤 생성기에
`num_ctx`가 붙었고(M3-7), 안전 규칙이 하나 늘었다(D-034). 오래된 값을 상대로
프레임워크를 비교하면 차이가 프레임워크 때문인지 그동안의 변경 때문인지
구분할 수 없다.

**CLI `ingest`에는 평가 케이스를 빼는 기능이 없다.** 누출 없는 색인이 필요한
것은 실험뿐이라 제품 코드에 넣지 않았다. 그 조립을 여기서 한다.

LlamaIndex 쪽(`build_index.py`)과 같은 것을 제외한다. 같은 표본, 같은 seed다.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src import composition
from src.application.evaluation import sample_cases
from src.infrastructure.aihub_loader import AihubQaSetLoader

OUT = PROJECT_ROOT / "index" / "aihub-qa-holdout.npz"
SAMPLE_SIZE = composition.DEFAULT_SAMPLE_SIZE
SAMPLE_SEED = composition.SAMPLE_SEED


def report(done: int, total: int) -> None:
    if done % 1000 == 0:
        print(f"      {done} / {total}", flush=True)


def main() -> int:
    print("[1/4] 문서 적재 + [2/4] 청크 분할 ...")
    chunks = composition.build_ingest(composition.AIHUB_SOURCE, material="qa").execute()

    cases = AihubQaSetLoader(composition.AIHUB_SOURCE).load()
    skip = {case.id for case in sample_cases(cases, size=SAMPLE_SIZE, seed=SAMPLE_SEED)}
    kept = [chunk for chunk in chunks if chunk.doc_id not in skip]
    print(
        f"    청크 {len(chunks)}개 중 {len(kept)}개 "
        f"(평가용 {len(chunks) - len(kept)}개 제외)"
    )

    print("[3/4] 임베딩 + [4/4] 색인 ...")
    started = time.perf_counter()
    count = composition.build_index_builder().execute(kept, OUT, on_progress=report)
    elapsed = time.perf_counter() - started

    print(f"    저장: {OUT} (새로 {count}개, {elapsed / 60:.1f}분)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
