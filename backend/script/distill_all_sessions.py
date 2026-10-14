from pathlib import Path
import sys

# 确保可以从 backend 目录作为根导入 memory_module_v2
BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from memory_module_v2.ingest.session_reader import list_session_ids
from memory_module_v2.service.api import distill_session

# 列出所有 memory_module_v1/sessions 下的 session_id
session_ids = list_session_ids()

for sid in session_ids:
    result = distill_session(sid)
    print(
        f"{sid}: exchanges_total={result.exchanges_total}, "
        f"new={result.exchanges_new}, objects_created={result.objects_created}, "
        f"errors={len(result.errors)}"
    )