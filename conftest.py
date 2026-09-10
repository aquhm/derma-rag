"""pytest가 프로젝트 루트를 sys.path에 넣게 만드는 파일.

pytest는 conftest.py가 있는 디렉터리를 sys.path 앞에 추가한다. 이 파일이 루트에
있어야 테스트에서 `from src.domain.models import ...`가 동작한다.

내용이 비어 있어도 된다. 존재 자체가 목적이다.
"""
