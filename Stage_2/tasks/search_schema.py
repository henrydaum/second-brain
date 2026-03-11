"""
Shared schema for the BM25 full-text search index.

Used by both index_text and index_ocr tasks, which write to the same
search_content table with different source values.
"""

SEARCH_SCHEMA = """
	CREATE TABLE IF NOT EXISTS search_content (
		path TEXT,
		source TEXT,
		chunk_index INTEGER,
		content TEXT,
		char_count INTEGER,
		indexed_at REAL,
		PRIMARY KEY (path, source, chunk_index)
	);

	CREATE VIRTUAL TABLE IF NOT EXISTS search_index USING fts5(
		path,
		content,
		source,
		chunk_index,
		content=search_content,
		content_rowid=rowid,
		tokenize='porter unicode61'
	);

	CREATE TRIGGER IF NOT EXISTS search_content_ai AFTER INSERT ON search_content BEGIN
		INSERT INTO search_index(rowid, path, content, source, chunk_index)
		VALUES (new.rowid, new.path, new.content, new.source, new.chunk_index);
	END;

	CREATE TRIGGER IF NOT EXISTS search_content_ad AFTER DELETE ON search_content BEGIN
		INSERT INTO search_index(search_index, rowid, path, content, source, chunk_index)
		VALUES('delete', old.rowid, old.path, old.content, old.source, old.chunk_index);
	END;

	CREATE TRIGGER IF NOT EXISTS search_content_au AFTER UPDATE ON search_content BEGIN
		INSERT INTO search_index(search_index, rowid, path, content, source, chunk_index)
		VALUES('delete', old.rowid, old.path, old.content, old.source, old.chunk_index);
		INSERT INTO search_index(rowid, path, content, source, chunk_index)
		VALUES (new.rowid, new.path, new.content, new.source, new.chunk_index);
	END;
"""
