import sys
from unittest.mock import patch, MagicMock
import pytest
import pandas as pd
from rag_helper.cli import main

def test_cli_validation_invalid_columns(capsys):
    """Test that the CLI exits with an error when invalid columns are provided."""
    # Mock load_library to avoid trying to load a DB
    with patch("rag_helper.cli.load_library") as mock_load:
        with patch.object(sys, 'argv', ['rag-cli', 'view', '-c', 'WRONG_COL']):
            with pytest.raises(SystemExit) as excinfo:
                main()
            assert excinfo.value.code == 1
            
    captured = capsys.readouterr()
    assert "Error: Invalid column name(s): WRONG_COL" in captured.out
    assert "Valid columns are:" in captured.out
    # load_library should NOT have been called
    mock_load.assert_not_called()

def test_cli_validation_mixed_columns(capsys):
    """Test validation with a mix of valid and invalid columns."""
    with patch("rag_helper.cli.load_library") as mock_load:
        with patch.object(sys, 'argv', ['rag-cli', 'match', 'query', '-c', 'INDEX', 'TYPO', 'QUESTION']):
            with pytest.raises(SystemExit) as excinfo:
                main()
            assert excinfo.value.code == 1
            
    captured = capsys.readouterr()
    assert "Error: Invalid column name(s): TYPO" in captured.out
    mock_load.assert_not_called()

def test_cli_validation_valid_columns(capsys):
    """Test that valid columns pass validation."""
    mock_ql = MagicMock()
    mock_ql.df_querylib = pd.DataFrame({"A": [1]}) # Minimal DF
    
    with patch("rag_helper.cli.load_library", return_value=mock_ql) as mock_load:
        # Match also needs processor mock
        with patch("rag_helper.rag.RAGProcessor.get_similar_queries", return_value=pd.DataFrame()):
             with patch.object(sys, 'argv', ['rag-cli', 'view', '-c', 'INDEX', 'QUESTION_MASKED']):
                main()
    
    captured = capsys.readouterr()
    assert "Error" not in captured.out
    mock_load.assert_called_once()

def test_cli_validation_helper_columns(capsys):
    """Test that helper columns (Score, SourceQuery) are considered valid."""
    mock_ql = MagicMock()
    mock_ql.df_querylib = pd.DataFrame()
    
    # Mock match_queries dependencies
    results = pd.DataFrame({"Score": [0.9], "QUESTION_MASKED": ["T"], "QP": ["Q"]}, index=[1])
    mock_ql.col_query_w_placeholders = "QP"
    mock_ql.querylib_name = "test"
    
    with patch("rag_helper.cli.load_library", return_value=mock_ql):
        with patch("rag_helper.rag.RAGProcessor.get_similar_queries", return_value=results):
            with patch.object(sys, 'argv', ['rag-cli', 'match', 'test', '-c', 'Score', 'SourceQuery']):
                main()
    
    captured = capsys.readouterr()
    assert "Error" not in captured.out
