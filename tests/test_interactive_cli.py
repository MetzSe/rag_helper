from unittest.mock import patch, MagicMock
import pytest
from rag_helper.interactive_cli import InteractiveExplorer

@pytest.fixture
def explorer():
    e = InteractiveExplorer()
    return e

def test_scan_databases(explorer, tmp_path):
    # Set project root to tmp_path
    explorer.project_root = tmp_path
    
    (tmp_path / "test1.db").touch()
    (tmp_path / "test2.db").touch()
    (tmp_path / ".hidden.db").touch()
    
    dbs = explorer.scan_databases()
    assert len(dbs) == 2
    assert any("test1.db" in db for db in dbs)
    assert not any(".hidden.db" in db for db in dbs)

def test_select_db_flow(explorer, tmp_path):
    # Set project root to tmp_path
    explorer.project_root = tmp_path
    
    with patch.object(explorer, 'scan_databases', return_value=["db1.db", "db2.db"]):
        with patch("rag_helper.interactive_cli.Prompt.ask", return_value="1"):
            with patch("rag_helper.query_library.QueryLibrary.load", return_value=MagicMock(querylib_name="LoadedDB")):
                explorer.select_db()
                assert "db2.db" in explorer.selected_db
                assert explorer.ql.querylib_name == "LoadedDB"

def test_interactive_search_call(explorer):
    explorer.ql = MagicMock()
    # Prompt mocks: 1. Search term, 2. Inspection choice (press enter to skip)
    with patch("rag_helper.interactive_cli.Prompt.ask", side_effect=["brain", ""]):
        with patch("rag_helper.interactive_cli.search_library") as mock_search:
            explorer.interactive_search()
            mock_search.assert_called_once_with(explorer.ql, "brain")

def test_interactive_match_call(explorer):
    explorer.ql = MagicMock()
    # Prompt mocks: 1. Question, 2. Similarity, 3. Inspection choice (skip)
    with patch("rag_helper.interactive_cli.Prompt.ask", side_effect=["What?", "0.5", ""]):
        with patch("rag_helper.interactive_cli.IntPrompt.ask", return_value=3): # top_k
            with patch("rag_helper.interactive_cli.match_queries") as mock_match:
                explorer.interactive_match()
                mock_match.assert_called_once_with(explorer.ql, ["What?"], top_k=3, threshold=0.5)

def test_menu_shows_db_status_when_loaded(explorer, capsys):
    """Test that menu shows selected database name."""
    mock_ql = MagicMock()
    mock_ql.querylib_name = "test_library"
    explorer.ql = mock_ql
    
    # Mock the run loop to exit immediately after one iteration
    with patch("rag_helper.interactive_cli.Prompt.ask", return_value="7"):
        explorer.run()
    
    captured = capsys.readouterr()
    assert "test_library" in captured.out

def test_menu_shows_no_db_when_not_loaded(explorer, capsys):
    """Test that menu shows 'No database selected' when none loaded."""
    explorer.ql = None
    
    with patch("rag_helper.interactive_cli.Prompt.ask", return_value="7"):
        explorer.run()
    
    captured = capsys.readouterr()
    assert "No database selected" in captured.out

def test_search_offers_inspection(explorer, capsys):
    """Test that search results offer inspection option."""
    explorer.ql = MagicMock()
    
    with patch("rag_helper.interactive_cli.Prompt.ask", side_effect=["test", ""]):
        with patch("rag_helper.interactive_cli.search_library"):
            explorer.interactive_search()
    
    captured = capsys.readouterr()
    assert "inspect" in captured.out.lower()

def test_match_offers_inspection(explorer, capsys):
    """Test that match results offer inspection option."""
    explorer.ql = MagicMock()
    
    with patch("rag_helper.interactive_cli.Prompt.ask", side_effect=["test", "0.5", ""]):
        with patch("rag_helper.interactive_cli.IntPrompt.ask", return_value=5):
            with patch("rag_helper.interactive_cli.match_queries"):
                explorer.interactive_match()
    
    captured = capsys.readouterr()
    assert "inspect" in captured.out.lower()

def test_back_to_menu_on_empty_input(explorer):
    """Test that empty input returns to menu."""
    explorer.ql = MagicMock()
    
    with patch("rag_helper.interactive_cli.Prompt.ask", return_value=""):
        with patch("rag_helper.interactive_cli.search_library") as mock_search:
            explorer.interactive_search()
            mock_search.assert_not_called()

def test_back_to_menu_on_q_input(explorer):
    """Test that 'q' input returns to menu."""
    explorer.ql = MagicMock()
    
    with patch("rag_helper.interactive_cli.Prompt.ask", return_value="q"):
        with patch("rag_helper.interactive_cli.search_library") as mock_search:
            explorer.interactive_search()
            mock_search.assert_not_called()
