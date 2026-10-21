from contextlib import redirect_stdout
import io
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from rag_lab.cli import main
from rag_lab.config import Settings
from rag_lab.database import create_db_engine


class ConfigurationTests(unittest.TestCase):
    def test_dotenv_password_is_literal_hidden_and_url_safe(self):
        password = "test@pass/word:${UNCHANGED}"
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {}, clear=True):
            env_file = Path(directory) / ".env"
            env_file.write_text(f'POSTGRES_PASSWORD="{password}"\nPOSTGRES_DB=from_file\n')
            settings = Settings.from_env(env_file)
            self.assertEqual(settings.postgres_db, "from_file")
            self.assertNotIn(password, repr(settings))
            engine = create_db_engine(settings)
            try:
                self.assertEqual(engine.url.drivername, "postgresql+psycopg2")
                self.assertEqual(engine.url.password, password)
                self.assertNotIn(password, str(engine.url))
            finally:
                engine.dispose()

    def test_shell_values_override_dotenv(self):
        with tempfile.TemporaryDirectory() as directory:
            env_file = Path(directory) / ".env"
            env_file.write_text("POSTGRES_PORT=55432\nPOSTGRES_DB=file_db\n")
            with patch.dict(os.environ, {"POSTGRES_PORT": "55433", "POSTGRES_DB": "shell_db"}, clear=True):
                settings = Settings.from_env(env_file)
            self.assertEqual((settings.postgres_port, settings.postgres_db), (55433, "shell_db"))

    def test_missing_password_fails_before_connecting(self):
        with self.assertRaisesRegex(ValueError, "POSTGRES_PASSWORD"):
            create_db_engine(Settings())

    def test_invalid_port_has_an_actionable_error(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {"POSTGRES_PORT": "99999"}, clear=True):
            with self.assertRaisesRegex(ValueError, "65535"):
                Settings.from_env(Path(directory) / "missing.env")


class CommandTests(unittest.TestCase):
    def test_help_does_not_read_configuration(self):
        with patch("rag_lab.cli.Settings.from_env", side_effect=AssertionError("Configuration was read")), redirect_stdout(io.StringIO()):
            with self.assertRaises(SystemExit) as result:
                main(["--help"])
        self.assertEqual(result.exception.code, 0)

    def test_importing_every_module_has_no_network_or_docling_side_effects(self):
        code = """
import importlib, socket, sys
from unittest.mock import patch
with patch.object(socket.socket, 'connect', side_effect=AssertionError('Network at import')):
    for name in ('config', 'database', 'corpus', 'extraction', 'ollama_client', 'chunking', 'populate', 'retrieval', 'indexing', 'benchmark_index', 'pull', 'ask', 'evaluate', 'evaluate_answers', 'tune_chunking', 'cli', '__main__'):
        importlib.import_module('rag_lab.' + name)
assert 'docling.document_converter' not in sys.modules
"""
        result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, "")


if __name__ == "__main__":
    unittest.main()
