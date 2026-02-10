import logging
import os
import shutil
import tempfile
from pathlib import Path
import pytest
from rag_helper.logger import setup_logging

def test_setup_logging_basic():
    """Test basic logging setup without file logging."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        # We use force=True to ensure we can re-configure in tests
        setup_logging(log_level="DEBUG", log_base_dir=tmp_dir, force=True)
        
        logger = logging.getLogger("test_logger")
        assert logger.getEffectiveLevel() <= logging.DEBUG
        
        # Check if console handler exists (at least one handler should be StreamHandler)
        root_logger = logging.getLogger()
        has_stream = any(isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler) for h in root_logger.handlers)
        assert has_stream

def test_setup_logging_with_file():
    """Test logging setup with file logging enabled."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        setup_logging(log_level="INFO", log_base_dir=tmp_dir, log_subdirs=False, force=True)
        
        # Check if log file was created
        log_files = list(Path(tmp_dir).glob("app_*.log"))
        assert len(log_files) == 1
        
        # Check if root logger has file handler
        root_logger = logging.getLogger()
        has_file = any(isinstance(h, logging.FileHandler) for h in root_logger.handlers)
        assert has_file

def test_setup_logging_subdirs():
    """Test logging setup with subdirectories."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        setup_logging(log_level="INFO", log_base_dir=tmp_dir, log_subdirs=True, force=True)
        
        # Check if subdirectories were created (e.g., YYYY/MM/DD)
        # We don't know the exact path without re-implementing the logic, 
        # but we can check if there's any .log file deep inside
        log_files = list(Path(tmp_dir).rglob("app_*.log"))
        assert len(log_files) == 1
        
def test_setup_logging_error_handling(capsys):
    """Test error handling when directory creation fails."""
    # Try to use a path that is likely to fail (e.g., a file that exists where a dir should be)
    with tempfile.NamedTemporaryFile() as tmp_file:
        setup_logging(log_base_dir=tmp_file.name, force=True)
        captured = capsys.readouterr()
        # The code prints a warning to stderr on exception
        assert "Warning: Could not setup file logging" in captured.err
