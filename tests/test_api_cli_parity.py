import pytest
import pandas as pd
from unittest.mock import patch, MagicMock
from rag_helper.cli import main
import rag_helper.api as api
import sys
import io

@pytest.fixture
def mock_ql():
    ql = MagicMock()
    ql.querylib_name = "test_lib"
    ql.df_querylib = pd.DataFrame({
        "QUESTION": ["What is A?", "How many B?"],
        "QUESTION_MASKED": ["What is [A]?", "How many [B]?"],
        "QUESTION_TYPE": ["QA", "COHORT_GENERATOR"],
        "QUERY_MASKED": ["SELECT A", "SELECT B"],
        "QUERY_EXECUTABLE": ["SELECT A", "SELECT B"]
    }, index=[1, 2])
    ql.col_question = "QUESTION"
    ql.col_question_masked = "QUESTION_MASKED"
    ql.col_query_w_placeholders = "QUERY_MASKED"
    ql.col_query_executable = "QUERY_EXECUTABLE"
    ql.embeddings = [{
        "model_name": "test-model",
        "embed_matrix": MagicMock() # Use a mock or zero matrix
    }]
    ql.source = "test.db"
    # Also mock the embedding model on the instance
    ql.embedding_model = MagicMock()
    ql.embedding_model.encode.return_value = MagicMock()
    return ql

def test_view_parity(mock_ql):
    """Verify api.view returns same data as CLI view processing targets."""
    with patch("rag_helper.cli.load_library", return_value=mock_ql):
        # API call
        api_df = api.view(mock_ql, limit=2)
        
        # We don't easily capture the rich table, but we can check the call to api.view inside cli.py
        with patch("rag_helper.api.view", wraps=api.view) as mock_api_view:
            with patch.object(sys, 'argv', ['rag-cli', 'view', '--limit', '2']):
                main()
                mock_api_view.assert_called_once()
                # Ensure it was called with record limit 2
                assert mock_api_view.call_args[1]['limit'] == 2

def test_info_parity(mock_ql):
    """Verify api.info is called correctly by CLI."""
    with patch("rag_helper.cli.load_library", return_value=mock_ql):
        with patch("rag_helper.api.info", wraps=api.info) as mock_api_info:
            with patch.object(sys, 'argv', ['rag-cli', 'info', '1', '2']):
                main()
                mock_api_info.assert_called_once()
                # Check renamed parameter record_id
                args, kwargs = mock_api_info.call_args
                assert kwargs['record_id'] == [1, 2]

def test_match_parity(mock_ql):
    """Verify api.match is called correctly by CLI with renamed type argument."""
    with patch("rag_helper.cli.load_library", return_value=mock_ql), \
         patch("rag_helper.api.QueryLibraryManager.get_instance", return_value=mock_ql), \
         patch("rag_helper.rag.QueryLibraryManager.get_instance", return_value=mock_ql):
        
        # We mock get_similar_questions on the QL instance to avoid embedding logic
        with patch.object(mock_ql, "get_df_recs", return_value=pd.DataFrame()) as mock_get_recs:
             with patch("rag_helper.api.match", wraps=api.match) as mock_api_match:
                with patch.object(sys, 'argv', ['rag-cli', 'match', 'test query', '--type', 'QA']):
                    main()
                    mock_api_match.assert_called_once()
                    # Check renamed parameter question_type
                    args, kwargs = mock_api_match.call_args
                    assert kwargs['question_type'] == 'QA'
