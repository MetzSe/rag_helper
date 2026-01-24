import sys
from unittest.mock import patch
import argparse
import inspect
import pytest
from rag_helper.cli import get_parser, generate_cli_docs, view_head, view_info, search_library, match_queries
from rag_helper.interactive_cli import main as interactive_main

def test_get_parser():
    """Verify that the parser has all expected commands and arguments."""
    parser = get_parser()
    subparsers_action = next(
        action for action in parser._actions 
        if isinstance(action, argparse._SubParsersAction)
    )
    
    expected_commands = ["view", "info", "search", "match", "docs"]
    for cmd in expected_commands:
        assert cmd in subparsers_action.choices
        
    # Check specific arguments for "view"
    view_parser = subparsers_action.choices["view"]
    arg_destinations = [action.dest for action in view_parser._actions]
    assert "limit" in arg_destinations

def test_generate_cli_docs_output(capsys):
    """Verify that generate_cli_docs prints a table with expected content."""
    generate_cli_docs()
    captured = capsys.readouterr()
    
    # Check for table title
    assert "CLI Command Documentation" in captured.out
    
    # Check for command names
    assert "view" in captured.out
    assert "match" in captured.out
    assert "docs" in captured.out
    
    # Check for docstring fragments
    assert "View the first N rows" in captured.out
    assert "Perform similarity matching" in captured.out
    
    # Check for arguments
    assert "--limit" in captured.out
    assert "--threshold" in captured.out

def test_cli_docs_command_integration(capsys):
    """Verify that 'rag-cli docs' triggers documentation display."""
    from rag_helper.cli import main
    with patch.object(sys, 'argv', ['rag-cli', 'docs']):
        main()
    captured = capsys.readouterr()
    assert "CLI Command Documentation" in captured.out

def test_interactive_cli_docs_flag(capsys):
    """Verify that '--docs' flag in interactive CLI triggers documentation display."""
    with patch.object(sys, 'argv', ['rag-cli-interactive', '--docs']):
        interactive_main()
    captured = capsys.readouterr()
    assert "CLI Command Documentation" in captured.out

def test_docstring_sync_accuracy():
    """Verify that the documentation generator correctly pulls docstrings."""
    # We can check specific commands
    parser = get_parser()
    subparsers_action = next(
        action for action in parser._actions 
        if isinstance(action, argparse._SubParsersAction)
    )
    
    # Test 'view' command docstring
    view_doc = inspect.getdoc(view_head)
    # The logic in generate_cli_docs uses this exactly
    assert "View the first N rows of the library." in view_doc

    # Test 'search' command docstring
    search_doc = inspect.getdoc(search_library)
    assert "Search questions for a term." in search_doc
