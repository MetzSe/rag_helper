import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path
from rag_helper.interactive_cli import InteractiveExplorer

@pytest.fixture
def explorer():
    return InteractiveExplorer()

def test_scan_databases(explorer, tmp_path):
    """Test scanning for database files."""
    # Setup dummy structure
    (tmp_path / "subdir").mkdir()
    (tmp_path / "test1.db").touch()
    (tmp_path / "subdir" / "test2.db").touch()
    (tmp_path / ".hidden").mkdir()
    (tmp_path / ".hidden" / "test3.db").touch()
    
    explorer.project_root = tmp_path
    dbs = explorer.scan_databases()
    
    assert "test1.db" in dbs
    assert str(Path("subdir/test2.db")) in dbs
    assert not any(".hidden" in db for db in dbs)

@patch("rag_helper.interactive_cli.Prompt.ask")
@patch("rag_helper.interactive_cli.Table")
@patch("rag_helper.interactive_cli.Console.print")
def test_select_db_no_dbs(mock_print, mock_table, mock_ask, explorer):
    """Test selection when no DBs exist."""
    explorer.scan_databases = MagicMock(return_value=[])
    explorer.select_db()
    mock_print.assert_any_call("[yellow]No database files found in the current directory.[/yellow]")

@patch("rag_helper.interactive_cli.Prompt.ask")
@patch("rag_helper.interactive_cli.QueryLibraryManager.load")
@patch("rag_helper.interactive_cli.Table")
@patch("rag_helper.interactive_cli.Console.print")
def test_select_db_success(mock_print, mock_table, mock_load, mock_ask, explorer):
    """Test successful database selection and loading."""
    explorer.scan_databases = MagicMock(return_value=["test.db"])
    mock_ask.return_value = "0"
    mock_load.return_value = MagicMock(querylib_name="TestLib")
    
    explorer.select_db()
    
    assert explorer.selected_db.endswith("test.db")
    assert explorer.ql is not None
    mock_print.assert_any_call("[bold green]Successfully loaded library: TestLib[/bold green]")

@patch("rag_helper.interactive_cli.Prompt.ask")
@patch("rag_helper.interactive_cli.Console.print")
def test_interactive_match_not_loaded(mock_print, mock_ask, explorer):
    """Test match command when no DB is loaded."""
    explorer.ql = None
    explorer.interactive_match()
    mock_print.assert_any_call("[red]Please select and load a database first.[/red]")

@patch("rag_helper.interactive_cli.Prompt.ask")
@patch("rag_helper.interactive_cli.IntPrompt.ask")
@patch("rag_helper.interactive_cli.match_queries")
def test_interactive_match_success(mock_match, mock_int_ask, mock_ask, explorer):
    """Test successful interactive match flow."""
    explorer.ql = MagicMock()
    # Mock inputs for: question, top_k, threshold, then 'q' for inspect record prompt
    mock_ask.side_effect = ["How is the weather?", "0.5", ""]
    mock_int_ask.return_value = 3
    
    explorer.interactive_match()
    
    mock_match.assert_called_once()
    assert mock_match.call_args[0][1] == ["How is the weather?"]
    assert mock_match.call_args[1]["top_k"] == 3

@patch("rag_helper.interactive_cli.Prompt.ask")
def test_interactive_view_not_loaded(mock_ask, explorer, capsys):
    """Test view command when no DB is loaded."""
    explorer.ql = None
    explorer.interactive_view()
    # Using capsys because console.print might be direct or mocked
    # In this case InteractiveExplorer uses global 'console'
    # Let's mock console globally for this test
    with patch("rag_helper.interactive_cli.console.print") as mock_console_print:
        explorer.interactive_view()
        mock_console_print.assert_called_with("[red]Please select and load a database first.[/red]")

@patch("rag_helper.interactive_cli.Prompt.ask")
def test_run_exit(mock_ask, explorer):
    """Test exiting the main loop."""
    mock_ask.return_value = "7"
    with patch("rag_helper.interactive_cli.console.print") as mock_print:
        explorer.run()
        mock_print.assert_any_call("[yellow]Goodbye![/yellow]")

@patch("rag_helper.interactive_cli.InteractiveExplorer.select_db")
@patch("rag_helper.interactive_cli.Prompt.ask")
def test_run_select_db_choice(mock_ask, mock_select, explorer):
    """Test choosing the select db option from menu."""
    mock_ask.side_effect = ["1", "7"] # Choice 1 then Exit
    explorer.run()
    mock_select.assert_called_once()
