import sys
from unittest.mock import patch, MagicMock
import pytest
import pandas as pd
import numpy as np
from rag_helper.cli import main

@pytest.fixture
def mock_ql():
    ql = MagicMock()
    ql.querylib_name = "test_lib"
    ql.df_querylib = pd.DataFrame({
        "QUESTION": ["Test Q1"],
        "QUESTION_TYPE": ["QA"]
    })
    ql.col_question = "QUESTION"
    ql.col_query_executable = "QUERY"
    return ql

def test_cli_help_command(capsys):
    with patch.object(sys, 'argv', ['rag-cli', 'help']):
        main()
    captured = capsys.readouterr()
    assert "Quick Overview" in captured.out

def test_cli_view_command(mock_ql, capsys):
    with patch("rag_helper.cli.load_library", return_value=mock_ql):
        with patch.object(sys, 'argv', ['rag-cli', '-f', 'dummy.db', 'view', '--limit', '1']):
            main()
    captured = capsys.readouterr()
    assert "Query Library: test_lib" in captured.out
    assert "Test Q1" in captured.out

def test_cli_search_command(mock_ql, capsys):
    with patch("rag_helper.cli.load_library", return_value=mock_ql):
        with patch.object(sys, 'argv', ['rag-cli', 'search', 'Test']):
            main()
    captured = capsys.readouterr()
    # Check for the search term and the record in the output
    assert "Test" in captured.out
    assert "Test Q1" in captured.out

def test_cli_match_command(mock_ql, capsys):
    # results from processor is a join of score and the df
    results = pd.DataFrame({
        "Score": [0.9],
        "QUESTION": ["Test Q1"],
        "QUERY": ["SELECT 1"]
    })
    
    with patch("rag_helper.cli.load_library", return_value=mock_ql):
        with patch("rag_helper.rag.RAGProcessor.get_similar_queries", return_value=results):
            with patch.object(sys, 'argv', ['rag-cli', 'match', 'How many?']):
                main()
    captured = capsys.readouterr()
    assert "Query: How many?" in captured.out
    assert "0.9000" in captured.out

def test_cli_info_command(mock_ql, capsys):
    # Properly mock the .loc[id] call
    row = pd.Series({"QUESTION": "Detailed Q"})
    mock_ql.df_querylib = MagicMock()
    mock_ql.df_querylib.loc.__getitem__.return_value = row
    
    with patch("rag_helper.cli.load_library", return_value=mock_ql):
        with patch.object(sys, 'argv', ['rag-cli', 'info', '1']):
            main()
    captured = capsys.readouterr()
    assert "Record 1" in captured.out
    assert "Detailed Q" in captured.out

def test_cli_info_command_multiple_ids(mock_ql, capsys):
    """Test info command with multiple IDs."""
    row1 = pd.Series({"QUESTION": "First Question"})
    row2 = pd.Series({"QUESTION": "Second Question"})
    mock_ql.df_querylib = MagicMock()
    mock_ql.df_querylib.loc.__getitem__.side_effect = [row1, row2]
    
    with patch("rag_helper.cli.load_library", return_value=mock_ql):
        with patch.object(sys, 'argv', ['rag-cli', 'info', '1', '2']):
            main()
    captured = capsys.readouterr()
    assert "Record 1" in captured.out
    assert "First Question" in captured.out
    assert "Record 2" in captured.out
    assert "Second Question" in captured.out
