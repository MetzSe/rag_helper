import os
import sys
from pathlib import Path
from typing import List, Optional

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt, IntPrompt, Confirm
from rich.table import Table
from rich import box

from rag_helper.query_library_manager import QueryLibraryManager
from rag_helper.cli import view_head, view_info, search_library, match_queries, print_help_overview, generate_cli_docs

console = Console()

BACK_HINT = "[dim]Enter 'q' or press Enter to return to Main Menu.[/dim]"

class InteractiveExplorer:
    def __init__(self):
        self.selected_db: Optional[str] = None
        self.ql: Optional[QueryLibraryManager] = None
        self.project_root: Path = Path.cwd()

    def scan_databases(self) -> List[str]:
        """Scan for .db files in the current directory, returning relative paths."""
        db_files = list(self.project_root.glob("**/*.db"))
        # Filter out hidden directories and files, return relative paths
        relative_paths = []
        for f in db_files:
            if not any(part.startswith('.') for part in f.parts):
                try:
                    relative_paths.append(str(f.relative_to(self.project_root)))
                except ValueError:
                    relative_paths.append(str(f))
        return relative_paths

    def select_db(self):
        """Prompt user to select a database from a list."""
        dbs = self.scan_databases()
        if not dbs:
            console.print("[yellow]No database files found in the current directory.[/yellow]")
            return

        table = Table(title="Available Databases (relative to project root)", box=box.SIMPLE)
        table.add_column("Index", style="cyan")
        table.add_column("File Path", style="green")
        
        for i, db in enumerate(dbs):
            table.add_row(str(i), db)
            
        console.print(table)
        console.print(BACK_HINT)
        
        choice_str = Prompt.ask("Select database index", default="0")
        if choice_str.lower() == 'q' or choice_str == '':
            return
        
        try:
            choice = int(choice_str)
            if choice < 0 or choice >= len(dbs):
                console.print("[red]Invalid index.[/red]")
                return
        except ValueError:
            console.print("[red]Invalid input.[/red]")
            return
            
        self.selected_db = str(self.project_root / dbs[choice])
        
        console.print(f"[green]Selected:[/green] {dbs[choice]}")
        try:
            self.ql = QueryLibraryManager.load(self.selected_db)
            console.print(f"[bold green]Successfully loaded library: {self.ql.querylib_name}[/bold green]")
        except Exception as e:
            console.print(f"[red]Error loading database:[/red] {e}")
            self.ql = None
            self.selected_db = None

    def _prompt_inspect_record(self):
        """Prompt user to inspect a specific record ID."""
        console.print("[dim]Enter 'i' to inspect a record, or press Enter to return.[/dim]")
        choice = Prompt.ask("Choice", default="")
        if choice.lower() == 'i':
            record_id = IntPrompt.ask("Enter record ID")
            view_info(self.ql, record_id)
            console.print(BACK_HINT)
            Prompt.ask("Press Enter to continue")

    def interactive_view(self):
        """Interactive paging through the library with entry type filter."""
        if not self.ql:
            console.print("[red]Please select and load a database first.[/red]")
            return
        
        # Entry type filter
        console.print("\n[bold]Filter by Entry Type:[/bold]")
        console.print("1. All entries")
        console.print("2. QA entries only")
        console.print("3. COHORT_GENERATOR entries only")
        console.print(BACK_HINT)
        
        filter_choice = Prompt.ask("Choose filter", choices=["1", "2", "3", "q", ""], default="1")
        if filter_choice.lower() == 'q' or filter_choice == '':
            return
        
        question_type = None
        if filter_choice == "2":
            question_type = "QA"
        elif filter_choice == "3":
            question_type = "COHORT_GENERATOR"
        
        # Filter the dataframe
        df = self.ql.df_querylib
        if question_type and "QUESTION_TYPE" in df.columns:
            df = df[df["QUESTION_TYPE"] == question_type]
        
        if df.empty:
            console.print("[yellow]No entries found matching the filter.[/yellow]")
            return
        
        # Pagination
        page_size = 10
        total_records = len(df)
        total_pages = (total_records + page_size - 1) // page_size
        current_page = 0
        
        while True:
            start_idx = current_page * page_size
            end_idx = min(start_idx + page_size, total_records)
            page_df = df.iloc[start_idx:end_idx]
            
            table = Table(title=f"Query Library: {self.ql.querylib_name} (Page {current_page + 1}/{total_pages})", box=box.ROUNDED)
            table.add_column("ID", style="cyan", no_wrap=True)
            table.add_column("Question", style="green")
            table.add_column("Type", style="magenta")
            
            for idx, row in page_df.iterrows():
                q_type = str(row.get("QUESTION_TYPE", "N/A"))
                question = str(row.get("QUESTION", "N/A"))
                if len(question) > 80:
                    question = question[:77] + "..."
                table.add_row(str(idx), question, q_type)
                
            console.print(table)
            console.print(f"[dim]Showing {start_idx + 1}-{end_idx} of {total_records} records. Full SQL available via 'i' (inspect).[/dim]")
            
            # Navigation options
            nav_options = []
            if current_page > 0:
                nav_options.append("p")
                console.print("[cyan]p[/cyan] - Previous page")
            if current_page < total_pages - 1:
                nav_options.append("n")
                console.print("[cyan]n[/cyan] - Next page")
            nav_options.extend(["i", "q"])
            console.print("[cyan]i[/cyan] - Inspect a specific record ID")
            console.print("[cyan]q[/cyan] - Back to Main Menu")
            
            nav_choice = Prompt.ask("Choose", choices=nav_options, default="q")
            
            if nav_choice == "n":
                current_page += 1
            elif nav_choice == "p":
                current_page -= 1
            elif nav_choice == "i":
                record_id = IntPrompt.ask("Enter record ID")
                view_info(self.ql, record_id)
                console.print(BACK_HINT)
                Prompt.ask("Press Enter to continue")
            elif nav_choice == "q":
                break

    def interactive_search(self):
        """Interactive keyword search."""
        if not self.ql:
            console.print("[red]Please select and load a database first.[/red]")
            return
        
        console.print(BACK_HINT)
        term = Prompt.ask("Enter search term")
        if term.lower() == 'q' or term == '':
            return
        search_library(self.ql, term)
        console.print("[dim]Full SQL available via 'i' (inspect a record ID).[/dim]")
        self._prompt_inspect_record()

    def interactive_match(self):
        """Interactive similarity matching."""
        if not self.ql:
            console.print("[red]Please select and load a database first.[/red]")
            return
        
        console.print(BACK_HINT)
        question = Prompt.ask("Enter question to match")
        if question.lower() == 'q' or question == '':
            return
        
        top_k = IntPrompt.ask("How many results (top-k)?", default=5)
        threshold = Prompt.ask("Similarity threshold (0.0 to 1.0)?", default="0.0")
        
        try:
            match_queries(self.ql, [question], top_k=top_k, threshold=float(threshold))
        except ValueError:
            console.print("[red]Invalid threshold. Please enter a number.[/red]")
            return
        
        console.print("[dim]Full SQL available via 'i' (inspect a record ID).[/dim]")
        self._prompt_inspect_record()

    def run_tests(self):
        """Invoke pytest and display results in the console."""
        console.print("[yellow]Running internal test suite...[/yellow]")
        import subprocess
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pytest", "tests/"],
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                console.print(Panel(result.stdout, title="Test Success", border_style="green"))
            else:
                console.print(Panel(result.stdout + "\n" + result.stderr, title="Test Failure", border_style="red"))
        except Exception as e:
            console.print(f"[red]Failed to run tests:[/red] {e}")
        
        console.print(BACK_HINT)
        Prompt.ask("Press Enter to continue")

    def show_docs(self):
        """Display auto-generated documentation from docstrings."""
        generate_cli_docs()
        console.print(BACK_HINT)
        Prompt.ask("Press Enter to continue")

    def run(self):
        """Main loop for the interactive CLI."""
        console.print(Panel("[bold cyan]RAG CLI: Interactive Explorer[/bold cyan]\n[dim]Enter 'q' at any prompt to return to this menu.[/dim]", border_style="cyan"))
        
        while True:
            # Show selected database status
            if self.ql:
                db_status = f"[bold green]Selected: {self.ql.querylib_name}[/bold green]"
            else:
                db_status = "[bold yellow]No database selected[/bold yellow]"
            
            console.print(f"\n[bold]Main Menu[/bold] ({db_status})")
            
            # Highlight option 1 if no database selected
            if self.ql:
                console.print("1. [magenta]Scan & Select Database[/magenta]")
                console.print("2. [magenta]View Library (Paging & Filter)[/magenta]")
                console.print("3. [magenta]Search by Keyword[/magenta]")
                console.print("4. [magenta]Similarity Match (RAG)[/magenta]")
            else:
                console.print("1. [bold green]Scan & Select Database[/bold green] [dim](required first)[/dim]")
                console.print("2. [dim red]View Library (Paging & Filter)[/dim red]")
                console.print("3. [dim red]Search by Keyword[/dim red]")
                console.print("4. [dim red]Similarity Match (RAG)[/dim red]")
            
            console.print("5. [yellow]Run Stress Tests[/yellow]")
            console.print("6. [cyan]View Full CLI Documentation[/cyan]")
            console.print("7. [red]Exit[/red]")
            
            choice = Prompt.ask("Choose an option", choices=["1", "2", "3", "4", "5", "6", "7"], default="1")
            
            if choice == "1":
                self.select_db()
            elif choice == "2":
                self.interactive_view()
            elif choice == "3":
                self.interactive_search()
            elif choice == "4":
                self.interactive_match()
            elif choice == "5":
                self.run_tests()
            elif choice == "6":
                self.show_docs()
            elif choice == "7":
                console.print("[yellow]Goodbye![/yellow]")
                break

def main():
    """Entry point for the interactive CLI."""
    if "help" in sys.argv:
        print_help_overview()
        return

    if "--docs" in sys.argv:
        generate_cli_docs()
        return

    explorer = InteractiveExplorer()
    explorer.run()

if __name__ == "__main__":
    main()
