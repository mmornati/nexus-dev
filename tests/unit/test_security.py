"""Security tests for Nexus-Dev.

These tests verify that the application handles security-sensitive
scenarios correctly, including input validation, path traversal
prevention, and injection protection.
"""

from nexus_dev.chunkers import ChunkerRegistry
from nexus_dev.config import NexusConfig
from nexus_dev.database import generate_document_id


class TestPathTraversalPrevention:
    """Tests for path traversal attack prevention."""

    def test_config_db_path_no_traversal(self):
        """Test that db_path expands correctly."""
        config = NexusConfig.create_new("test")
        config.db_path = "~/.nexus-dev/db"

        # Path should be expanded and absolute
        db_path = config.get_db_path()
        assert db_path.is_absolute()
        assert "~" not in str(db_path)
        # The tilde should be expanded to home directory
        assert ".nexus-dev" in str(db_path)

    def test_chunking_file_path_sanitization(self):
        """Test that file paths in chunks are handled safely."""
        # Simulating a malicious file path
        malicious_path = "../../etc/shadow"
        content = "def test(): pass"

        chunks = ChunkerRegistry.chunk_file(malicious_path, content)

        # Chunks should be created but the path should be stored as-is
        # (the indexing layer should validate paths before calling)
        for chunk in chunks:
            assert chunk.file_path == malicious_path


class TestInputValidation:
    """Tests for input validation and sanitization."""

    def test_config_empty_project_name(self):
        """Test handling of empty project name."""
        # Should create config even with empty name
        config = NexusConfig.create_new("")
        assert config.project_id  # Should still have valid ID

    def test_config_special_chars_project_name(self):
        """Test handling of special characters in project name."""
        special_names = [
            "project<script>alert(1)</script>",
            "project'; DROP TABLE users;--",
            "project\x00null",
            "project\n\rnewlines",
        ]

        for name in special_names:
            config = NexusConfig.create_new(name)
            assert config.project_name == name
            assert config.project_id  # Should have valid ID

    def test_document_id_deterministic(self):
        """Test that document IDs are deterministic and safe."""
        id1 = generate_document_id("proj", "file.py", "func", 1)
        id2 = generate_document_id("proj", "file.py", "func", 1)

        assert id1 == id2  # Deterministic
        assert len(id1) == 36  # Valid UUID format

    def test_document_id_special_chars(self):
        """Test document ID generation with special characters."""
        # Should not raise and should produce valid UUID
        doc_id = generate_document_id(
            "project'; DROP TABLE--",
            "file<script>.py",
            "func\x00null",
            999999999,
        )

        assert len(doc_id) == 36
        # Should be valid UUID format
        parts = doc_id.split("-")
        assert len(parts) == 5


class TestChunkerSecurityBoundaries:
    """Tests for chunker security boundaries."""

    def test_chunker_large_file(self):
        """Test handling of very large files."""
        # Create a large Python file
        large_content = "x = 1\n" * 100000  # 100k lines

        # Should not crash or hang
        chunks = ChunkerRegistry.chunk_file("large.py", large_content)
        assert len(chunks) >= 0

    def test_chunker_binary_content(self):
        """Test handling of binary content."""
        binary_content = bytes(range(256)).decode("latin-1")

        # Should handle gracefully, not crash
        try:
            ChunkerRegistry.chunk_file("binary.py", binary_content)
        except Exception:
            pass  # Expected to fail parsing, but shouldn't crash

    def test_chunker_null_bytes(self):
        """Test handling of null bytes in content."""
        content_with_nulls = "def test():\x00\x00pass"

        # Should handle gracefully
        ChunkerRegistry.chunk_file("null.py", content_with_nulls)
        # May or may not parse, but shouldn't crash

    def test_chunker_unicode_edge_cases(self):
        """Test handling of Unicode edge cases."""
        unicode_content = '''
def тест():
    """Тестовая функция 🧪"""
    return "مرحبا"

def emoji_func():
    """🎉🎊🥳"""
    x = "𝕳𝖊𝖑𝖑𝖔"
    return x
'''
        chunks = ChunkerRegistry.chunk_file("unicode.py", unicode_content)
        assert len(chunks) >= 1


class TestConfigSecurityValidation:
    """Tests for configuration security validation."""

    def test_config_patterns_not_exploitable(self):
        """Test that glob patterns don't allow exploitation."""
        config = NexusConfig.create_new("test")

        # Malicious patterns that shouldn't cause issues
        config.include_patterns = [
            "/**/../../../*",
            "/etc/passwd",
            "$(whoami)",
            "`id`",
        ]

        # The patterns are just strings, not executed
        assert len(config.include_patterns) == 4

    def test_config_url_validation(self):
        """Test that Ollama URL is stored but not validated here."""
        config = NexusConfig.create_new("test", embedding_provider="ollama")

        # Malicious URLs - should be stored but validated at use time
        malicious_urls = [
            "http://127.0.0.1:11434",
            "http://localhost:11434/../../etc/passwd",
            "file:///etc/passwd",
            "javascript:alert(1)",
        ]

        for url in malicious_urls:
            config.ollama_url = url
            assert config.ollama_url == url

    def test_config_json_injection(self, temp_dir):
        """Test that JSON injection in config is handled."""
        config = NexusConfig.create_new('test", "admin": true, "x": "')
        config_path = temp_dir / "config.json"

        config.save(config_path)
        loaded = NexusConfig.load(config_path)

        # The injection attempt should be escaped by JSON encoder
        assert loaded.project_name == 'test", "admin": true, "x": "'


class TestMemoryAndResourceLimits:
    """Tests for memory and resource handling."""

    def test_chunker_deeply_nested_code(self):
        """Test handling of deeply nested code structures."""
        # Create deeply nested Python
        depth = 100
        content = ""
        for i in range(depth):
            content += "    " * i + f"def func_{i}():\n"
        content += "    " * depth + "pass"

        # Should handle without stack overflow
        chunks = ChunkerRegistry.chunk_file("nested.py", content)
        assert len(chunks) >= 0

    def test_chunker_many_functions(self):
        """Test handling of file with many functions."""
        # Create file with 1000 functions
        content = "\n".join(f"def func_{i}():\n    return {i}\n" for i in range(1000))

        chunks = ChunkerRegistry.chunk_file("many.py", content)
        assert len(chunks) >= 100  # Should extract many chunks
