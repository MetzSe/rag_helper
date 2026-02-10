import pandas as pd
from pathlib import Path
from typing import Optional
from rag_helper.query_library_manager import QueryLibraryManager

def normalize_library_df(ql: QueryLibraryManager, df: pd.DataFrame, requested_columns: Optional[list[str]] = None) -> pd.DataFrame:
    """
    Translates internal column names to standard CLI names:
    INDEX, QUESTION_TYPE, QUESTION, QUESTION_MASKED, QUERY_MASKED, QUERY_EXECUTABLE
    
    Args:
        ql: The QueryLibraryManager instance containing metadata.
        df: The DataFrame to normalize.
        requested_columns: List of canonical column names to include/order.
    """
    # 1. Map internal columns to canonical names
    mapping = {
        ql.col_question: "QUESTION",
        ql.col_question_masked: "QUESTION_MASKED",
        ql.col_query_w_placeholders: "QUERY_MASKED",
        ql.col_query_executable: "QUERY_EXECUTABLE",
        "question_type": "QUESTION_TYPE"  # Standard naming in DB
    }
    
    # 2. Ensure we have a DataFrame and add INDEX
    if isinstance(df, pd.Series):
        df_norm = df.to_frame().T
    else:
        df_norm = df.copy()
        
    # Use the index as the default INDEX column
    df_norm["INDEX"] = df_norm.index
    
    # Apply mapping (only for columns that exist)
    rename_map = {k: v for k, v in mapping.items() if k in df_norm.columns}
    # If any of the renamed columns is "INDEX" (e.g. if ql.col_id was "INDEX"), 
    # it will overwrite the index one during rename.
    df_norm = df_norm.rename(columns=rename_map)
    
    # 3. Filter to requested columns or return all canonical + helpers
    canonical_cols = ["INDEX", "QUESTION_TYPE", "QUESTION", "QUESTION_MASKED", "QUERY_MASKED", "QUERY_EXECUTABLE"]
    helper_cols = ["SourceQuery", "Score", "SearchTerm"]
    
    if requested_columns:
        # Only include columns that actually end up in the DF
        valid_cols = [c for c in requested_columns if c in df_norm.columns]
        return df_norm[valid_cols]
    
    # Default viewable columns
    return df_norm[[c for c in (canonical_cols + helper_cols) if c in df_norm.columns]]

def export_to_file(df: pd.DataFrame, output_path: str):
    """Export a DataFrame to a file based on the extension."""
    path = Path(output_path)
    ext = path.suffix.lower()
    
    if ext == '.csv':
        df.to_csv(output_path, index=False)
    elif ext == '.json':
        df.to_json(output_path, orient='records', indent=4)
    elif ext == '.xlsx':
        df.to_excel(output_path, index=False)
    else:
        raise ValueError(f"Unsupported export format: {ext}. Use .csv, .json, or .xlsx")
