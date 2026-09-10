"""생성 모델 후보를 같은 프롬프트로 비교한다.

M0-6과 MEMORY.md D-005 재검토용. 응답 시간, 초당 토큰, 한국어 지시 준수를
나란히 본다.

사용:
    python scripts/bench_models.py qwen2.5:3b gemma4:31b
"""

from __future__ import annotations

import json
import sys
import time
import urllib.error
import urllib.request

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    except (AttributeError, OSError):
        pass

URL = "http://localhost:11434/api/chat"
PROMPT = "민감성 피부의 볼 부위 메이크업에서 주의할 점을 두 문장으로만 답하세요."


def run(model: str) -> None:
    body = json.dumps(
        {
            "model": model,
            "stream": False,
            "options": {"num_predict": 200, "temperature": 0.6},
            "messages": [{"role": "user", "content": PROMPT}],
        }
    ).encode("utf-8")
    req = urllib.request.Request(
        URL, data=body, headers={"Content-Type": "application/json"}
    )

    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=900) as resp:
            data = json.load(resp)
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        print(f"{model:18} FAIL: {exc}")
        return

    wall = time.time() - t0
    tokens = data.get("eval_count") or 0
    text = (data.get("message") or {}).get("content", "").strip().replace("\n", " ")
    tps = tokens / wall if wall else 0.0

    print(f"{model:18} {wall:7.1f}s  {tokens:5d} tok  {tps:6.1f} tok/s")
    print(f"{'':18} {text[:180]}")
    print()


if __name__ == "__main__":
    for name in sys.argv[1:]:
        run(name)
