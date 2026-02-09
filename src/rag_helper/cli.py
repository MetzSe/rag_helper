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

from rag_helper.query_library import QueryLibrary
from rag_helper.rag import get_latest_querylib_file

console = Console()

def load_library(file_path: Optional[str]) -> QueryLibrary:
    """Load the query library from a file."""
    if not file_path:
        # Try to find the latest one in the current directory
        file_path = get_latest_querylib_file(str(Path.cwd()))
        if not file_path:
            console.print("[red]Error:[/red] No query library file specified and none found in current directory.")
            sys.exit(1)
            
    console.print(f"[dim]Loading library from: {file_path}[/dim]")
    ql = QueryLibrary.load(file_path)
    if not ql:
        console.print(f"[red]Error:[/red] Could not load library from {file_path}")
        sys.exit(1)
    return ql

def export_to_file(df: pd.DataFrame, output_path: str):
    """Export a DataFrame to a file based on the extension."""
    path = Path(output_path)
    ext = path.suffix.lower()
    
    try:
        if ext == '.csv':
            df.to_csv(output_path, index=True)
        elif ext == '.json':
            df.to_json(output_path, orient='records', indent=4)
        elif ext == '.xlsx':
            df.to_excel(output_path, index=True)
        else:
            console.print(f"[red]Error:[/red] Unsupported export format: {ext}. Use .csv, .json, or .xlsx")
            return
        console.print(f"[green]Successfully exported results to: {output_path}[/green]")
    except Exception as e:
        console.print(f"[red]Error exporting to file:[/red] {e}")

def view_head(ql: QueryLibrary, n: int = 10, output_path: Optional[str] = None):
    """View the first N rows of the library."""
    df = ql.df_querylib.head(n)
    
    table = Table(title=f"Query Library: {ql.querylib_name} (First {n} rows)", box=box.ROUNDED)
    table.add_column("ID", style="cyan", no_wrap=True)
    table.add_column("Question", style="green")
    table.add_column("Type", style="magenta")
    
    # We use the index as ID since it's common in pandas-based SQLite exports
    for idx, row in df.iterrows():
        q_type = str(row.get("QUESTION_TYPE", "N/A"))
        question = str(row.get("QUESTION", "N/A"))
        # Truncate long questions for the table
        if len(question) > 80:
            question = question[:77] + "..."
        table.add_row(str(idx), question, q_type)
        
    console.print(table)
    console.print(f"[dim]Total records: {len(ql.df_querylib)}[/dim]")

    if output_path:
        export_to_file(df, output_path)

def view_info(ql: QueryLibrary, record_ids: list[int], output_path: Optional[str] = None):
    """View full details for one or more record IDs."""
    all_rows = []
    
    for record_id in record_ids:
        try:
            row = ql.df_querylib.loc[record_id]
        except (KeyError, ValueError, IndexError):
            console.print(f"[red]Error:[/red] Record ID {record_id} not found.")
            continue

        console.print(Panel(f"[bold green]Question:[/bold green]\n{row.get('QUESTION', 'N/A')}", title=f"Record {record_id}", border_style="cyan"))
        
        if "QUESTION_MASKED" in row:
            console.print(Panel(f"{row.get('QUESTION_MASKED', 'N/A')}", title="Masked Question", border_style="dim"))
            
        if ql.col_query_w_placeholders in row:
            console.print(Panel(f"[yellow]{row.get(ql.col_query_w_placeholders, 'N/A')}[/yellow]", title="Query with Placeholders", border_style="yellow"))

        if ql.col_query_executable in row:
            console.print(Panel(f"[blue]{row.get(ql.col_query_executable, 'N/A')}[/blue]", title="Executable Query", border_style="blue"))
        
        console.print("")
        all_rows.append(row)

    if output_path and all_rows:
        df = pd.DataFrame(all_rows)
        export_to_file(df, output_path)

def search_library(ql: QueryLibrary, term: str, output_path: Optional[str] = None):
    """Search questions for a term."""
    df = ql.df_querylib[ql.df_querylib[ql.col_question].str.contains(term, case=False, na=False)]
    
    if df.empty:
        console.print(f"[yellow]No results found for search term: '{term}'[/yellow]")
        return
        
    table = Table(title=f"Search Results for: '{term}'", box=box.ROUNDED)
    table.add_column("ID", style="cyan")
    table.add_column("Question", style="green")
    
    for idx, row in df.iterrows():
        question = str(row.get("QUESTION", "N/A"))
        if len(question) > 100:
            question = question[:97] + "..."
        table.add_row(str(idx), question)
        
    console.print(table)
    console.print(f"[dim]Found {len(df)} matches.[/dim]")

    if output_path:
        export_to_file(df, output_path)

def match_queries(ql: QueryLibrary, queries: list[str], top_k: int = 5, threshold: float = 0.0, output_path: Optional[str] = None):
    """Perform similarity matching for one or more queries."""
    from rag_helper.rag import RAGConfig, RAGProcessor
    
    config = RAGConfig(sim_threshold=threshold)
    processor = RAGProcessor(config)
    # Inject the loaded library into the manager
    from rag_helper.rag import QueryLibraryManager
    manager = QueryLibraryManager.get_instance()
    manager.querylib = ql
    ql.load_embedding_model("BAAI/bge-large-en-v1.5")
    
    all_results = []
    for q in queries:
        console.print(Panel(f"[bold]Query:[/bold] {q}", border_style="cyan"))
        results = processor.get_similar_queries(q, top_k=top_k)
        
        if results.empty:
            console.print("[yellow]No similar queries found matching the criteria.[/yellow]")
            continue
            
        if output_path:
            # Append query string for context in the exported file
            results["SourceQuery"] = q
            all_results.append(results)

        table = Table(title=f"Top {top_k} Matches (Threshold: {threshold})", box=box.SIMPLE)
        table.add_column("Score", style="magenta")
        table.add_column("Matched Question", style="green")
        table.add_column("Query", style="yellow")
        
        # results is a subset of df_querylib joined with similarity scores
        # We need to ensure we're accessing columns correctly based on results structure
        for _, row in results.iterrows():
            score = row.get("Score", 0.0)
            matched_q = row.get("QUESTION", "N/A")
            executable = row.get(ql.col_query_executable, "N/A")
            
            if len(executable) > 50:
                executable = executable[:47] + "..."
            
            table.add_row(f"{score:.4f}", matched_q, executable)
            
        console.print(table)
        console.print("\n")

    if output_path and all_results:
        final_df = pd.concat(all_results)
        export_to_file(final_df, output_path)

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
    parser = argparse.ArgumentParser(description="Visualize and search QueryLibrary databases.")
    parser.add_argument("--file", "-f", help="Path to the .db query library file.")
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Help command
    subparsers.add_parser("help", help="Show overview of the tool and exit")
    
    # View command
    view_parser = subparsers.add_parser("view", help="View the first N records")
    view_parser.add_argument("--limit", "-l", type=int, default=10, help="Number of records to show")
    view_parser.add_argument("--output", "-o", help="Path to export results (csv, json, xlsx)")
    
    # Info command
    info_parser = subparsers.add_parser("info", help="Show full details for one or more IDs")
    info_parser.add_argument("id", type=int, nargs="+", help="The record ID(s) to inspect (row numbers)")
    info_parser.add_argument("--output", "-o", help="Path to export results (csv, json, xlsx)")
    
    # Search command
    search_parser = subparsers.add_parser("search", help="Search questions for a keyword")
    search_parser.add_argument("term", help="The keyword to search for")
    search_parser.add_argument("--output", "-o", help="Path to export results (csv, json, xlsx)")

    # Match command
    match_parser = subparsers.add_parser("match", help="Find similar queries for a given question")
    match_parser.add_argument("query", nargs="+", help="The question(s) to match")
    match_parser.add_argument("--k", "-k", type=int, default=5, help="Number of results")
    match_parser.add_argument("--threshold", "-t", type=float, default=0.0, help="Min similarity")
    match_parser.add_argument("--output", "-o", help="Path to export results (csv, json, xlsx)")

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

    ql = load_library(args.file)
    
    if args.command == "view":
        view_head(ql, args.limit, output_path=args.output)
    elif args.command == "info":
        view_info(ql, args.id, output_path=args.output)
    elif args.command == "search":
        search_library(ql, args.term, output_path=args.output)
    elif args.command == "match":
        match_queries(ql, args.query, top_k=args.k, threshold=args.threshold, output_path=args.output)

if __name__ == "__main__":
    main()
