from __future__ import annotations

from pathlib import Path

from langchain_core.tools import BaseTool

from tools.fetch_url_tool import FetchURLTool
from tools.python_repl_tool import PythonReplTool
from tools.read_file_tool import ReadFileTool
from tools.terminal_tool import TerminalTool


def get_all_tools(base_dir: Path) -> list[BaseTool]:
    tools: list[BaseTool] = [
        TerminalTool(root_dir=base_dir),
        PythonReplTool(root_dir=base_dir),
        FetchURLTool(),
        ReadFileTool(root_dir=base_dir),
    ]

    from memory_module_v2.service.config import get_memory_backend, get_memory_v2_inject_mode
    if get_memory_backend() == "v2" and get_memory_v2_inject_mode() == "tool":
        from memory_module_v2.integrations.tools import get_memory_tools
        tools.extend(get_memory_tools())

    return tools
