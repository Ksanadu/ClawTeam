from service.memory_indexer import MemoryIndexer


def test_clean_checkout_can_initialize_memory_document(tmp_path, monkeypatch):
    indexer = MemoryIndexer()
    monkeypatch.setattr(indexer, "_supports_embeddings", lambda: False)
    indexer.configure(tmp_path)
    indexer.rebuild_index()

    memory_file = tmp_path / "memory_module_v1" / "long_term_memory" / "MEMORY.md"
    assert memory_file.read_text(encoding="utf-8").startswith("# Long-term Memory")
    assert indexer.retrieve("test") == []
