import sys
import os
from unittest.mock import patch, MagicMock
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from rag_helper.cli import main, export_to_file, view_head, search_library, match_queries
from rag_helper.query_library import QueryLibrary


# --- Unit tests for export_to_file ---

def test_export_to_file_csv(tmp_path, capsys):
    """Test export_to_file creates a valid CSV file."""
    df = pd.DataFrame({"A": [1, 2], "B": ["x", "y"]})
    output_file = tmp_path / "test_output.csv"
    
    export_to_file(df, str(output_file))
    
    captured = capsys.readouterr()
    assert "Successfully exported" in captured.out
    assert output_file.exists()
    
    loaded_df = pd.read_csv(output_file, index_col=0)
    assert len(loaded_df) == 2
    assert "A" in loaded_df.columns


def test_export_to_file_json(tmp_path, capsys):
    """Test export_to_file creates a valid JSON file."""
    df = pd.DataFrame({"A": [1, 2], "B": ["x", "y"]})
    output_file = tmp_path / "test_output.json"
    
    export_to_file(df, str(output_file))
    
    captured = capsys.readouterr()
    assert "Successfully exported" in captured.out
    assert output_file.exists()
    
    loaded_df = pd.read_json(output_file)
    assert len(loaded_df) == 2


def test_export_to_file_xlsx(tmp_path, capsys):
    """Test export_to_file creates a valid XLSX file."""
    df = pd.DataFrame({"A": [1, 2], "B": ["x", "y"]})
    output_file = tmp_path / "test_output.xlsx"
    
    export_to_file(df, str(output_file))
    
    captured = capsys.readouterr()
    assert "Successfully exported" in captured.out
    assert output_file.exists()
    
    loaded_df = pd.read_excel(output_file, index_col=0)
    assert len(loaded_df) == 2


def test_export_to_file_unsupported_format(tmp_path, capsys):
    """Test export_to_file handles unsupported formats gracefully."""
    df = pd.DataFrame({"A": [1]})
    output_file = tmp_path / "test_output.txt"
    
    export_to_file(df, str(output_file))
    
    captured = capsys.readouterr()
    assert "Unsupported export format" in captured.out
    assert not output_file.exists()


# --- Integration tests for CLI commands with --output ---

@pytest.fixture
def mock_ql():
    ql = MagicMock()
    ql.querylib_name = "test_lib"
    ql.df_querylib = pd.DataFrame({
        "QUESTION": ["Test Q1", "Test Q2"],
        "QUESTION_TYPE": ["QA", "QA"]
    })
    ql.col_question = "QUESTION"
    ql.col_query_executable = "QUERY"
    return ql


def test_cli_view_with_output(mock_ql, tmp_path, capsys):
    """Integration test for `view --output`."""
    output_file = tmp_path / "view_output.csv"
    
    with patch("rag_helper.cli.load_library", return_value=mock_ql):
        with patch.object(sys, 'argv', ['rag-cli', '-f', 'dummy.db', 'view', '--limit', '2', '-o', str(output_file)]):
            main()
    
    captured = capsys.readouterr()
    assert "Successfully exported" in captured.out
    assert output_file.exists()
    
    loaded_df = pd.read_csv(output_file, index_col=0)
    assert len(loaded_df) == 2


def test_cli_search_with_output(mock_ql, tmp_path, capsys):
    """Integration test for `search --output`."""
    output_file = tmp_path / "search_output.json"
    
    with patch("rag_helper.cli.load_library", return_value=mock_ql):
        with patch.object(sys, 'argv', ['rag-cli', 'search', 'Test', '-o', str(output_file)]):
            main()
    
    captured = capsys.readouterr()
    assert "Successfully exported" in captured.out
    assert output_file.exists()


def test_cli_match_with_output(mock_ql, tmp_path, capsys):
    """Integration test for `match --output`."""
    output_file = tmp_path / "match_output.csv"
    results = pd.DataFrame({
        "Score": [0.9],
        "QUESTION": ["Test Q1"],
        "QUERY": ["SELECT 1"]
    })
    
    with patch("rag_helper.cli.load_library", return_value=mock_ql):
        with patch("rag_helper.rag.RAGProcessor.get_similar_queries", return_value=results):
            with patch.object(sys, 'argv', ['rag-cli', 'match', 'How many?', '-o', str(output_file)]):
                main()
    
    captured = capsys.readouterr()
    assert "Successfully exported" in captured.out
    assert output_file.exists()


# --- Additional coverage gap tests ---

def test_load_library_fails_on_bad_file(capsys):
    """Test that load_library exits gracefully when QueryLibrary.load returns None."""
    from rag_helper.cli import load_library
    
    with pytest.raises(SystemExit):
        with patch("rag_helper.query_library.QueryLibrary.load", return_value=None):
            load_library("/path/to/nonexistent.db")
    
    captured = capsys.readouterr()
    assert "Error" in captured.out


def test_rag_processor_empty_results(capsys):
    """Test RAGProcessor handles empty result set gracefully."""
    mock_ql = MagicMock()
    mock_ql.col_query_executable = "QUERY"
    
    empty_results = pd.DataFrame()
    
    with patch("rag_helper.cli.load_library", return_value=mock_ql):
        with patch("rag_helper.rag.RAGProcessor.get_similar_queries", return_value=empty_results):
            with patch.object(sys, 'argv', ['rag-cli', 'match', 'nonexistent query']):
                main()
    
    captured = capsys.readouterr()
    assert "No similar queries found" in captured.out


def test_view_info_with_output(tmp_path, capsys):
    """Integration test for `info --output`."""
    mock_ql = MagicMock()
    mock_ql.querylib_name = "test_lib"
    mock_ql.col_query_w_placeholders = "QP"
    mock_ql.col_query_executable = "QE"
    
    row = pd.Series({"QUESTION": "Detailed Q", "QP": "SELECT ?", "QE": "SELECT 1"})
    mock_ql.df_querylib = MagicMock()
    mock_ql.df_querylib.loc.__getitem__.return_value = row
    
    output_file = tmp_path / "info_output.json"
    
    with patch("rag_helper.cli.load_library", return_value=mock_ql):
        with patch.object(sys, 'argv', ['rag-cli', 'info', '1', '-o', str(output_file)]):
            main()
    
    captured = capsys.readouterr()
    assert "Successfully exported" in captured.out
    assert output_file.exists()
