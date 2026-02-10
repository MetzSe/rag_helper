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
        "QUESTION_MASKED": ["Masked Q1"],
        "QUESTION_TYPE": ["QA"]
    })
    ql.col_question = "QUESTION"
    ql.col_query_executable = "QUERY"
    ql.col_query_w_placeholders = "PLACEHOLDER"
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
    assert "Masked Q1" in captured.out

def test_cli_search_command(mock_ql, capsys):
    with patch("rag_helper.cli.load_library", return_value=mock_ql):
        with patch.object(sys, 'argv', ['rag-cli', 'search', 'Test']):
            main()
    captured = capsys.readouterr()
    # Check for the search term and the masked question in the output
    assert "Test" in captured.out
    assert "Masked Q1" in captured.out

def test_cli_match_command(mock_ql, capsys):
    # results from processor is a join of score and the df
    results = pd.DataFrame({
        "Score": [0.9],
        "QUESTION_MASKED": ["Masked Q1"],
        "QUERY": ["SELECT 1"]
    })
    
    with patch("rag_helper.cli.load_library", return_value=mock_ql):
        with patch("rag_helper.rag.RAGProcessor.get_similar_queries", return_value=results):
            with patch.object(sys, 'argv', ['rag-cli', 'match', 'How many?']):
                main()
    captured = capsys.readouterr()
    assert "Query: How many?" in captured.out
    assert "SourceQuery" in captured.out
    assert "0.9000" in captured.out
    assert "Masked Q1" in captured.out

def test_cli_info_command(mock_ql, capsys):
    # Properly mock the .loc[id] call
    row = pd.Series({"QUESTION_MASKED": "Masked Detailed Q", "QUESTION_TYPE": "QA"})
    mock_ql.df_querylib = MagicMock()
    # Mock return must be a DataFrame with index 1 to match expected output "Record 1"
    mock_ql.df_querylib.loc.__getitem__.return_value = pd.DataFrame([row], index=[1])
    
    with patch("rag_helper.cli.load_library", return_value=mock_ql):
        with patch.object(sys, 'argv', ['rag-cli', 'info', '1']):
            main()
    captured = capsys.readouterr()
    assert "Record 1" in captured.out
    assert "Masked Detailed Q" in captured.out

def test_cli_info_command_multiple_ids(mock_ql, capsys):
    """Test info command with multiple IDs."""
    row1 = pd.Series({"QUESTION_MASKED": "First Masked"})
    row2 = pd.Series({"QUESTION_MASKED": "Second Masked"})
    mock_ql.df_querylib = MagicMock()
    # When view_info calls .loc[ids], it returns a DataFrame with all rows
    # api.info calls loc once with list of IDs
    mock_ql.df_querylib.loc.__getitem__.return_value = pd.DataFrame([row1, row2], index=[1, 2])
    
    with patch("rag_helper.cli.load_library", return_value=mock_ql):
        with patch.object(sys, 'argv', ['rag-cli', 'info', '1', '2']):
            main()
    captured = capsys.readouterr()
    assert "Record 1" in captured.out
    assert "First Masked" in captured.out
    assert "Record 2" in captured.out
    assert "Second Masked" in captured.out

def test_cli_search_with_columns(mock_ql, capsys):
    """Test search command with custom columns."""
    mock_ql.df_querylib["QUESTION_TYPE"] = ["QA"]
    mock_ql.df_querylib["QUESTION_MASKED"] = ["Masked Q1"]
    with patch("rag_helper.cli.load_library", return_value=mock_ql):
        with patch.object(sys, 'argv', ['rag-cli', 'search', 'Test', '-c', 'QUESTION_MASKED', 'QUESTION_TYPE']):
            main()
    captured = capsys.readouterr()
    assert "QUESTION_MASKED" in captured.out
    assert "QUESTION_TYPE" in captured.out
    assert "Masked Q1" in captured.out
    assert "QA" in captured.out
    # INDEX should not be in the output as it wasn't requested
    assert "INDEX" not in captured.out

def test_cli_match_with_columns(mock_ql, capsys):
    """Test match command with custom columns."""
    results = pd.DataFrame({
        "Score": [0.95],
        "QUESTION": ["Matched Q"],
        "QUERY": ["SELECT 1"],
        "QUESTION_MASKED": ["Masked Q"]
    })
    with patch("rag_helper.cli.load_library", return_value=mock_ql):
        with patch("rag_helper.rag.RAGProcessor.get_similar_queries", return_value=results):
            with patch.object(sys, 'argv', ['rag-cli', 'match', 'Query', '-c', 'Score', 'QUESTION_MASKED']):
                main()
    captured = capsys.readouterr()
    assert "Score" in captured.out
    assert "QUESTION_MASKED" in captured.out
    assert "0.9500" in captured.out
    assert "Masked Q" in captured.out
    # SourceQuery should not be in the output as it wasn't requested
    assert "SourceQuery" not in captured.out

def test_cli_search_default_columns(mock_ql, capsys):
    """Test search command fallback to default columns."""
    mock_ql.df_querylib["QUESTION_MASKED"] = ["Masked Q1"]
    with patch("rag_helper.cli.load_library", return_value=mock_ql):
        with patch.object(sys, 'argv', ['rag-cli', 'search', 'Test']):
            main()
    captured = capsys.readouterr()
    assert "SearchTerm" in captured.out
    assert "QUESTION_MASKED" in captured.out

def test_cli_match_default_columns(mock_ql, capsys):
    """Test match command fallback to default columns."""
    results = pd.DataFrame({
        "Score": [0.8],
        "QUESTION": ["Default Q"],
        "QUERY": ["SELECT 1"],
        "QUESTION_MASKED": ["Masked Q"]
    })
    with patch("rag_helper.cli.load_library", return_value=mock_ql):
        with patch("rag_helper.rag.RAGProcessor.get_similar_queries", return_value=results):
            with patch.object(sys, 'argv', ['rag-cli', 'match', 'Query']):
                main()
    captured = capsys.readouterr()
    assert "SourceQuery" in captured.out
    assert "Score" in captured.out
    assert "QUESTION_MASKED" in captured.out

def test_cli_ordered_export(mock_ql, tmp_path):
    """Test that export respects column selection and order."""
    mock_ql.df_querylib["QUESTION_MASKED"] = ["Masked Q1"]
    mock_ql.df_querylib["QUESTION_TYPE"] = ["QA"]
    output_file = tmp_path / "test_export.csv"
    
    with patch("rag_helper.cli.load_library", return_value=mock_ql):
        with patch.object(sys, 'argv', ['rag-cli', 'search', 'Test', '-c', 'QUESTION_TYPE', 'QUESTION_MASKED', '-o', str(output_file)]):
            main()
    
    assert output_file.exists()
    df_exported = pd.read_csv(output_file)
    # Check column order and content
    assert list(df_exported.columns) == ["QUESTION_TYPE", "QUESTION_MASKED"]
    assert df_exported.iloc[0]["QUESTION_TYPE"] == "QA"
    assert df_exported.iloc[0]["QUESTION_MASKED"] == "Masked Q1"

def test_cli_view_with_columns(mock_ql, capsys):
    """Test view command with custom columns."""
    mock_ql.df_querylib["QUESTION_TYPE"] = ["QA"]
    with patch("rag_helper.cli.load_library", return_value=mock_ql):
        with patch.object(sys, 'argv', ['rag-cli', 'view', '-c', 'QUESTION', 'QUESTION_TYPE']):
            main()
    captured = capsys.readouterr()
    assert "QUESTION" in captured.out
    assert "QUESTION_TYPE" in captured.out
    assert "Test Q1" in captured.out
    assert "QA" in captured.out
    assert "INDEX" not in captured.out

def test_cli_info_with_columns(mock_ql, capsys):
    """Test info command with custom columns."""
    row = pd.Series({"QUESTION": "Detailed Q", "QUESTION_TYPE": "QA"})
    mock_ql.df_querylib = MagicMock()
    mock_ql.df_querylib.loc.__getitem__.return_value = pd.DataFrame([row], index=[1])
    
    with patch("rag_helper.cli.load_library", return_value=mock_ql):
        with patch.object(sys, 'argv', ['rag-cli', 'info', '1', '-c', 'QUESTION_TYPE']):
            main()
    captured = capsys.readouterr()
    assert "QUESTION_TYPE" in captured.out
    assert "QA" in captured.out
    # QUESTION should not be the title of ANY panel
    assert "title=QUESTION " not in captured.out # This is a bit specific to rich/mocking but let's try just value check
    assert "Detailed Q" not in captured.out
