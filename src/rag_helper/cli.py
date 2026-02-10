import argparse
import inspect
import sys
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box
import pandas as pd

from rag_helper.query_library_manager import QueryLibraryManager, get_latest_querylib_file
import rag_helper.api as api
from rag_helper.utils import normalize_library_df, export_to_file as utils_export


def export_to_file(df: pd.DataFrame, output_path: str):
    """Export a DataFrame to a file, handling errors gracefully."""
    try:
        utils_export(df, output_path)
        console.print(f"[green]Successfully exported results to: {output_path}[/green]")
    except Exception as e:
        console.print(f"[red]Error exporting to file:[/red] {e}")

console = Console()

CANONICAL_COLUMNS = ["INDEX", "QUESTION_TYPE", "QUESTION", "QUESTION_MASKED", "QUERY_MASKED", "QUERY_EXECUTABLE"]
HELPER_COLUMNS = ["SourceQuery", "Score", "SearchTerm"]

def validate_columns(requested_columns: Optional[list[str]]):
    """
    Validate that user-provided column names are valid.
    Exits with error message if invalid columns found.
    """
    if not requested_columns:
        return
        
    valid_set = set(CANONICAL_COLUMNS + HELPER_COLUMNS)
    invalid = [c for c in requested_columns if c not in valid_set]
    
    if invalid:
        console.print(f"[red]Error:[/red] Invalid column name(s): [bold]{', '.join(invalid)}[/bold]")
        console.print(f"Valid columns are: {', '.join(sorted(list(valid_set)))}")
        sys.exit(1)

def load_library(file_path: Optional[str]) -> QueryLibraryManager:
    """Load the query library from a file."""
    if not file_path:
        # Try to find the latest one in the current directory
        file_path = get_latest_querylib_file(str(Path.cwd()))
        if not file_path:
            console.print("[red]Error:[/red] No query library file specified and none found in current directory.")
            sys.exit(1)
            
    console.print(f"[dim]Loading library from: {file_path}[/dim]")
    ql = api.load(file_path)
    if not ql:
        console.print(f"[red]Error:[/red] Could not load library from {file_path}")
        sys.exit(1)
    return ql


def view_head(ql: QueryLibraryManager, n: int = 10, output_path: Optional[str] = None, columns: Optional[list[str]] = None):
    """
    View the first N rows of the library.
    
    Default columns: INDEX, QUESTION_MASKED, QUESTION_TYPE
    """
    view_df = api.view(ql, limit=n, output=None)
    
    # Visual display logic
    table = Table(title=f"Query Library: {ql.querylib_name} (First {n} rows)", box=box.ROUNDED)
    
    # Defaults for visual display if no columns specified
    display_cols = columns if columns else ["INDEX", "QUESTION_MASKED", "QUESTION_TYPE"]
    # Re-normalize just to be safe/consistent with visual formatting logic which expects certain columns
    # The API returns a fully normalized DF, so we just check for existence
    display_cols = [c for c in display_cols if c in view_df.columns]
    
    for col in display_cols:
        style = "cyan" if col == "INDEX" else "green" if "QUESTION" in col else "magenta"
        table.add_column(col, style=style)
    
    for _, row in view_df.iterrows():
        row_data = []
        for col in display_cols:
            val = str(row.get(col, "N/A"))
            if col != "INDEX" and len(val) > 80:
                val = val[:77] + "..."
            row_data.append(val)
        table.add_row(*row_data)
        
    console.print(table)
    console.print(f"[dim]Total records: {len(ql.df_querylib)}[/dim]")

    if output_path:
        # We need to re-normalize to ensure columns are respected for export if not already
        # But view_df is already the result of API view which is normalized.
        # Wait, API view returns standard columns?
        # api.view returns normalize_library_df(ql, df_raw). 
        # API doesn't take columns arg, so it returns ALL default columns.
        # CLI filters columns for display.
        # We should respect CLI columns for export too.
        if columns:
            export_df = normalize_library_df(ql, view_df, columns)
        else:
             export_df = view_df
        export_to_file(export_df, output_path)

def view_info(ql: QueryLibraryManager, record_ids: list[int], output_path: Optional[str] = None, columns: Optional[list[str]] = None):
    """
    View full details for one or more record IDs.
    
    Default columns: QUESTION_MASKED, QUESTION_TYPE, QUERY_MASKED, QUERY_EXECUTABLE
    """
    valid_rows = []
    
    # Use API to get the dataframe
    # We pass None for output because CLI handles valid_rows logic for consistent console output
    # But actually, API handles it fine. Let's trust API for data retrieval.
    try:
        # Convert IDs to int if possible, as API expects mixed types might fail/warn
        clean_ids = []
        for rid in record_ids:
            try:
                clean_ids.append(int(rid))
            except ValueError:
                clean_ids.append(rid)
        
        df_norm = api.info(ql, record_id=clean_ids, output=None)
    except Exception as e:
        console.print(f"[red]Error retrieving info:[/red] {e}")
        return

    if df_norm.empty:
         console.print(f"[yellow]No records found for IDs: {record_ids}[/yellow]")
         return
    
    # Visual display
    # Default visual columns if none provided
    if not columns:
        columns = ["QUESTION_MASKED", "QUESTION_TYPE", "QUERY_MASKED", "QUERY_EXECUTABLE"]
    
    for idx, (_, row) in enumerate(df_norm.iterrows()):
        # Try to match back to input ID for display, though row has INDEX
        rec_id = row.get("INDEX", "N/A")
        console.print(f"[bold cyan]--- Record {rec_id} ---[/bold cyan]")
        
        for col in columns:
            if col in row:
                val = row.get(col, "N/A")
                style = "green" if "QUESTION" in col else "yellow" if "QUERY" in col else "blue"
                console.print(Panel(f"[{style}]{val}[/{style}]", title=col, border_style=style))
        
        console.print("")

    if output_path and not df_norm.empty:
        if columns:
             export_df = normalize_library_df(ql, df_norm, columns)
        else:
             export_df = df_norm
        export_to_file(export_df, output_path)

def search_library(ql: QueryLibraryManager, term: str, output_path: Optional[str] = None, columns: Optional[list[str]] = None):
    """
    Search questions for a term.
    
    Default columns: SearchTerm, INDEX, QUESTION, QUESTION_MASKED
    """
    df_norm = api.search(ql, term, output=None)
    
    if df_norm.empty:
        console.print(f"[yellow]No results found for search term: '{term}'[/yellow]")
        return
        
    table = Table(title=f"Search Results for: '{term}'", box=box.ROUNDED)
    
    default_cols = ["SearchTerm", "INDEX", "QUESTION", "QUESTION_MASKED"]
    # df_norm from API already has standard columns
    display_cols = columns if columns else [c for c in default_cols if c in df_norm.columns]
    
    for col in display_cols:
        style = "cyan" if col == "INDEX" or col == "SearchTerm" else "green" if "QUESTION" in col else "magenta"
        table.add_column(col, style=style)
    
    for _, row in df_norm.iterrows():
        row_data = []
        for col in display_cols:
            val = str(row.get(col, "N/A"))
            if col != "INDEX" and col != "SearchTerm" and len(val) > 100:
                val = val[:97] + "..."
            row_data.append(val)
        table.add_row(*row_data)
        
    console.print(table)
    console.print(f"[dim]Found {len(df_norm)} matches.[/dim]")

    if output_path:
        if columns:
            export_df = normalize_library_df(ql, df_norm, columns)
        else:
            export_df = df_norm
        export_to_file(export_df, output_path)

def match_queries(ql: QueryLibraryManager, queries: list[str], top_k: int = 5, threshold: float = 0.0, output_path: Optional[str] = None, columns: Optional[list[str]] = None, embedding_model: Optional[str] = None, question_type: Optional[str] = None):
    """
    Perform similarity matching for one or more queries.
    
    Default columns: SourceQuery, Score, INDEX, QUESTION_MASKED, QUERY_MASKED
    """
    if embedding_model:
        console.print(f"[dim]Using embedding model: {embedding_model}[/dim]")
    
    # We call API match separately for each query to maintain the CLI visual behavior 
    # of printing headers query-by-query, or we call it once and group.
    # The API match returns a combined DF.
    # To keep CLI output identical (streaming/iterative feel), we can call API per query.
    
    all_dfs = []
    
    for query in queries:
        console.print(f"[bold cyan]Query:[/bold cyan] {query}")
        
        # Use API for single query
        df_results = api.match(
            ql, 
            query=query, 
            k=top_k, 
            threshold=threshold, 
            question_type=question_type, 
            embedding_model=embedding_model,
            output=None # We handle export at the end to combine all
        )
        
        if df_results.empty:
            console.print("[yellow]No matches found above threshold.[/yellow]\n")
            continue
            
        all_dfs.append(df_results)

        table = Table(title=f"Top {top_k} Matches (Threshold: {threshold})", box=box.SIMPLE)
        
        default_cols = ["SourceQuery", "Score", "INDEX", "QUESTION_MASKED", "QUERY_MASKED"]
        display_cols = columns if columns else [c for c in default_cols if c in df_results.columns]
        
        for col in display_cols:
            style = "cyan" if col == "SourceQuery" else "magenta" if col == "Score" else "green"
            table.add_column(col, style=style)
        
        for _, row in df_results.iterrows():
            row_data = []
            for col in display_cols:
                val = str(row.get(col, "N/A"))
                if col == "Score":
                    try:
                        val = f"{float(val):.4f}"
                    except (ValueError, TypeError):
                        pass
                elif col != "SourceQuery" and len(val) > 80:
                    val = val[:77] + "..."
                row_data.append(val)
            table.add_row(*row_data)
            
        console.print(table)
        console.print("\n")

    if output_path and all_dfs:
        combined_df = pd.concat(all_dfs, ignore_index=True)
        export_to_file(combined_df, output_path)


def embeddings_list(ql: QueryLibraryManager):
    """
    List all embedding models stored in the library.
    """
    models = api.embeddings(ql, action="list")
    if not models:
        console.print("[yellow]No embeddings found in this library.[/yellow]")
        return

    table = Table(title="Stored Embedding Models", box=box.ROUNDED)
    table.add_column("INDEX", style="cyan")
    table.add_column("Model Name", style="green")
    table.add_column("Matrix Shape", style="magenta")

    for idx, emb in enumerate(models):
        shape = str(emb["embed_matrix"].shape)
        is_default = " (default)" if idx == 0 else ""
        table.add_row(str(idx), emb["model_name"] + is_default, shape)

    console.print(table)


def embeddings_add(ql: QueryLibraryManager, model_name: str, db_path: str):
    """
    Calculate embeddings for a new model and save to the database.
    """
    console.print(f"[bold]Adding embedding for model:[/bold] {model_name}")
    api.embeddings(ql, action="add", model=model_name)
    # The API wrapper handles adding and saving if we passed a path, 
    # but here ql might be loaded from file. 
    # API.embeddings for 'add' does: ql.add_embedding + ql.save_embedding_to_db if possible.
    # In CLI we just call it. API handles the persistence logic.
    console.print(f"[bold green]Successfully requested embedding addition for '{model_name}'[/bold green]")

def print_help_overview():
    """Print a comprehensive usage guide."""
    console.print(Panel.fit(
        "[bold cyan]Query Library Visualizer & RAG Matcher[/bold cyan]\n\n"
        "This tool provides two ways to interact with your query libraries:\n"
        "1. [bold]Pipeline Mode[/bold] (cli.py): Fast, non-interactive commands for scripts.\n"
        "2. [bold]Interactive Mode[/bold] (interactive_cli.py): Guided exploration and database scanning.\n\n"
        "[bold]Common Commands:[/bold]\n"
        "  [magenta]view[/magenta]      - See the first few records\n"
        "  [magenta]search[/magenta]    - Filter by keyword\n"
        "  [magenta]match[/magenta]     - Similarity search for questions\n"
        "  [magenta]info[/magenta]      - Detailed record inspection\n"
        "  [magenta]docs[/magenta]      - Auto-generated documentation from docstrings\n\n"
        "[bold]Examples:[/bold]\n"
        "  rag-cli -f lib.db view --limit 20\n"
        "  rag-cli -f lib.db match \"How many patients?\" -k 3",
        title="Quick Overview",
        border_style="cyan"
    ))

def generate_cli_docs():
    """Generates comprehensive documentation from docstrings and argparse configuration."""
    parser = get_parser()
    
    table = Table(title="CLI Command Documentation", box=box.ROUNDED, show_lines=True)
    table.add_column("Command", style="cyan", no_wrap=True)
    table.add_column("Description (from Docstring)", style="green")
    table.add_column("Arguments", style="yellow")

    # Access subparsers
    subparsers_action = next(
        action for action in parser._actions 
        if isinstance(action, argparse._SubParsersAction)
    )

    command_to_func = {
        "view": view_head,
        "info": view_info,
        "search": search_library,
        "match": match_queries,
        "docs": generate_cli_docs
    }

    for choice, subparser in subparsers_action.choices.items():
        # Get docstring
        func = command_to_func.get(choice)
        doc = inspect.getdoc(func) if func else "No documentation available."
        
        # Get arguments
        args_info = []
        for action in subparser._actions:
            if isinstance(action, argparse._HelpAction):
                continue
            arg_name = "/".join(action.option_strings) if action.option_strings else action.dest
            help_text = action.help if action.help else ""
            default = f" (default: {action.default})" if action.default is not None and action.default != argparse.SUPPRESS else ""
            args_info.append(f"[bold]{arg_name}[/bold]: {help_text}{default}")
        
        table.add_row(
            choice,
            doc,
            "\n".join(args_info)
        )

    console.print(table)

def get_parser():
    """Creates and returns the argparse parser for the CLI."""
    parser = argparse.ArgumentParser(description="Visualize and search QueryLibraryManager databases.")
    parser.add_argument("--file", "-f", help="Path to the .db query library file.")
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Help command
    subparsers.add_parser("help", help="Show overview of the tool and exit")
    
    # View command
    view_parser = subparsers.add_parser("view", help="View the first N records")
    view_parser.add_argument("--limit", "-l", type=int, default=10, help="Number of records to show")
    view_parser.add_argument("--columns", "-c", nargs="+", help="Columns to display in results")
    view_parser.add_argument("--output", "-o", help="Path to export results (csv, json, xlsx)")
    
    # Info command
    info_parser = subparsers.add_parser("info", help="Show full details for one or more IDs")
    info_parser.add_argument("id", type=str, nargs='+', help="Record ID(s) to view")
    info_parser.add_argument("--columns", "-c", nargs="+", help="Columns to display in results")
    info_parser.add_argument("--output", "-o", help="Path to export results (csv, json, xlsx)")
    
    # Search command
    search_parser = subparsers.add_parser("search", help="Search questions for a keyword")
    search_parser.add_argument("term", help="The keyword to search for")
    search_parser.add_argument("--columns", "-c", nargs="+", help="Columns to display in results")
    search_parser.add_argument("--output", "-o", help="Path to export results (csv, json, xlsx)")

    # Match command
    match_parser = subparsers.add_parser("match", help="Find similar queries for a given question")
    match_parser.add_argument("query", nargs="+", help="The question(s) to match")
    match_parser.add_argument("--k", "-k", type=int, default=5, help="Number of results")
    match_parser.add_argument("--threshold", "-t", type=float, default=0.0, help="Min similarity")
    match_parser.add_argument("--embedding-model", "-e", default=None, help="Embedding model name to use for matching (default: first stored model)")
    match_parser.add_argument("--type", "-y", choices=["QA", "COHORT"], help="Filter by question type (QA or COHORT)")
    match_parser.add_argument("--columns", "-c", nargs="+", help="Columns to display in results")
    match_parser.add_argument("--output", "-o", help="Path to export results (csv, json, xlsx)")

    # Embeddings command
    emb_parser = subparsers.add_parser("embeddings", help="Manage embedding models in the library")
    emb_subparsers = emb_parser.add_subparsers(dest="emb_command", help="Embeddings sub-command")
    emb_subparsers.add_parser("list", help="List all stored embedding models")
    emb_add_parser = emb_subparsers.add_parser("add", help="Add a new embedding model")
    emb_add_parser.add_argument("--model", "-m", required=True, help="HuggingFace model identifier (e.g. intfloat/multilingual-e5-large)")

    # Docs command
    subparsers.add_parser("docs", help="Show auto-generated documentation from docstrings")
    
    return parser

def main():
    parser = get_parser()
    args = parser.parse_args()
    
    if args.command == "help":
        print_help_overview()
        return

    if args.command == "docs":
        generate_cli_docs()
        return

    if not args.command:
        parser.print_help()
        return

    # Validate columns if provided
    if hasattr(args, 'columns'):
        validate_columns(args.columns)

    ql = load_library(args.file)
    
    if args.command == "view":
        view_head(ql, args.limit, output_path=args.output, columns=args.columns)
    elif args.command == "info":
        view_info(ql, args.id, output_path=args.output, columns=args.columns)
    elif args.command == "search":
        search_library(ql, args.term, output_path=args.output, columns=args.columns)
    elif args.command == "match":
        match_queries(ql, args.query, top_k=args.k, threshold=args.threshold, output_path=args.output, columns=args.columns, embedding_model=args.embedding_model, question_type=args.type)
    elif args.command == "embeddings":
        if args.emb_command == "list":
            embeddings_list(ql)
        elif args.emb_command == "add":
            embeddings_add(ql, args.model, args.file)
        else:
            console.print("[yellow]Use 'embeddings list' or 'embeddings add --model MODEL'[/yellow]")

if __name__ == "__main__":
    main()
