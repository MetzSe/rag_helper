from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from rag_helper.query_library import QueryLibrary


@pytest.fixture
def mock_query_lib():
    # Create a small dummy dataframe
    df = pd.DataFrame({
        "QUESTION": ["Give me all patients", "Show me drugs"],
        "QUESTION_MASKED": ["Give me all [ENTITY]", "Show me [ENTITY]"],
        "QUERY_SNOWFLAKE_WITH_PLACEHOLDERS": [
            "SELECT * FROM patients",
            "SELECT * FROM drugs"
        ]
    })

    with patch(
        "rag_helper.query_library.pd.read_excel",
        return_value=df
    ):
        ql = QueryLibrary(
            querylib_name="test_lib",
            source="test",
            querylib_source_file="dummy.xlsx",
            col_question="QUESTION",
            col_question_masked="QUESTION_MASKED",
            col_query_w_placeholders="QUERY_SNOWFLAKE_WITH_PLACEHOLDERS"
        )
        return ql


def test_query_library_init(mock_query_lib):
    assert len(mock_query_lib.df_querylib) == 2
    assert mock_query_lib.querylib_name == "test_lib"


@patch("rag_helper.query_library.make_sentence_transformer")
def test_calc_embedding(mock_transformer, mock_query_lib):
    # Mock the model's encode method
    mock_model = MagicMock()
    # Mock encode to return a 1D array for each row
    mock_model.encode.return_value = np.random.rand(384).astype(np.float32)
    mock_transformer.return_value = mock_model

    mock_query_lib.calc_embedding(use_masked=False)

    assert len(mock_query_lib.embeddings) == 1
    assert mock_query_lib.embeddings[0]["model_name"] == "BAAI/bge-large-en-v1.5"
    assert mock_query_lib.embeddings[0]["embed_matrix"].shape == (2, 384)
