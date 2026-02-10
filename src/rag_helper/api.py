import logging
from typing import List, Optional, Union
import pandas as pd
from rag_helper.query_library_manager import QueryLibraryManager, get_latest_querylib_file
from rag_helper.rag import RAGProcessor, RAGConfig
from rag_helper.utils import normalize_library_df, export_to_file

logger = logging.getLogger(__name__)

def _resolve_ql(file_or_ql: Union[str, QueryLibraryManager]) -> QueryLibraryManager:
    """Helper to resolve file path or existing QL instance."""
    if isinstance(file_or_ql, QueryLibraryManager) or hasattr(file_or_ql, 'df_querylib'):
        return file_or_ql
    return QueryLibraryManager.load(file_or_ql)

def load(file: Optional[str] = None) -> QueryLibraryManager:
    """
    Load a query library from a file.
    If file is None, attempts to find the latest .db file in current directory.
    """
    if file is None:
        import os
        file = get_latest_querylib_file(os.getcwd())
        if not file:
            raise FileNotFoundError("No query library file specified and none found in current directory.")
            
    return QueryLibraryManager.load(file)

def view(file: Union[str, QueryLibraryManager], limit: int = 10, output: Optional[str] = None) -> pd.DataFrame:
    """
    View the first N rows of the library.
    Equivalent to `rag-cli view`.
    """
    ql = _resolve_ql(file)
    df_raw = ql.df_querylib.head(limit)
    df_norm = normalize_library_df(ql, df_raw)
    
    if output:
        export_to_file(df_norm, output)
        
    return df_norm

def info(file: Union[str, QueryLibraryManager], record_id: Union[int, List[int]], output: Optional[str] = None) -> pd.DataFrame:
    """
    View full details for one or more record IDs.
    Equivalent to `rag-cli info`.
    """
    ql = _resolve_ql(file)
    if isinstance(record_id, int):
        ids = [record_id]
    else:
        ids = record_id
        
    try:
        df_subset = ql.df_querylib.loc[ids]
    except KeyError as e:
        logger.warning(f"Some IDs not found: {e}")
        # Try to return what we found, or empty if strict lookup failed for all
        existing_ids = [i for i in ids if i in ql.df_querylib.index]
        if not existing_ids:
            return pd.DataFrame()
        df_subset = ql.df_querylib.loc[existing_ids]

    df_norm = normalize_library_df(ql, df_subset)
    
    if output:
        export_to_file(df_norm, output)
        
    return df_norm

def search(file: Union[str, QueryLibraryManager], term: str, output: Optional[str] = None) -> pd.DataFrame:
    """
    Search questions for a keyword (case-insensitive).
    Equivalent to `rag-cli search`.
    """
    ql = _resolve_ql(file)
    mask = ql.df_querylib[ql.col_question].str.contains(term, case=False, na=False)
    df_raw = ql.df_querylib[mask].copy()
    
    if df_raw.empty:
        return pd.DataFrame()
        
    df_raw['SearchTerm'] = term
    df_norm = normalize_library_df(ql, df_raw)
    
    if output:
        export_to_file(df_norm, output)
        
    return df_norm

def match(
    file: Union[str, QueryLibraryManager], 
    query: Union[str, List[str]], 
    k: int = 5, 
    threshold: float = 0.0, 
    question_type: Optional[str] = None, 
    embedding_model: Optional[str] = None,
    output: Optional[str] = None
) -> pd.DataFrame:
    """
    Perform similarity matching for one or more queries.
    Equivalent to `rag-cli match`.

    Note: This function modifies the global Singleton state of QueryLibraryManager.
    """
    ql = _resolve_ql(file)
    
    # 1. Configure RAG
    config = RAGConfig(
        sim_threshold=threshold,
        embedding_model_name=embedding_model,
        question_type=question_type
    )
    # 2. Setup Singleton (RAGProcessor relies on this pattern)
    # We must explicitly sync the state because RAGProcessor uses the singleton
    manager = QueryLibraryManager.get_instance()
    manager.df_querylib = ql.df_querylib
    manager.embeddings = ql.embeddings
    manager.embedding_model = ql.embedding_model
    manager.current_embedding_model_name = ql.current_embedding_model_name
    manager.col_question = ql.col_question
    manager.col_question_masked = ql.col_question_masked
    manager.col_query_w_placeholders = ql.col_query_w_placeholders
    
    processor = RAGProcessor(config=config)
    
    # 3. Process Queries
    if isinstance(query, str):
        queries = [query]
    else:
        queries = query
        
    all_results = []
    
    for q in queries:
        # Get raw results (already matches threshold in some logic, but we enforce here too)
        res = processor.get_similar_queries(q, top_k=k)
        
        if res.empty:
            continue
            
        # Apply threshold again to be safe / explicit like CLI
        res = res[res['Score'] >= threshold].copy()
        
        if res.empty:
            continue
            
        res['SourceQuery'] = q
        
        # Normalize columns
        df_norm = normalize_library_df(ql, res)
        all_results.append(df_norm)
        
    if not all_results:
        return pd.DataFrame()
        
    combined_df = pd.concat(all_results, ignore_index=True)
    
    if output:
        export_to_file(combined_df, output)
        
    return combined_df

def embeddings(file: Union[str, QueryLibraryManager], action: str = "list", model: Optional[str] = None) -> Union[List[dict], None]:
    """
    Manage embedding models.
    action="list" -> returns list of model dicts
    action="add" -> adds model (requires model arg)
    """
    ql = _resolve_ql(file)
    
    if action == "list":
        return ql.embeddings
    elif action == "add":
        if not model:
            raise ValueError("Must provide 'model' argument when action is 'add'")
        ql.add_embedding(embedding_model_name=model)
        # Verify persistence if file path is known
        # In this wrapper, we assume if user passed string path, they want persistence.
        # If they passed ql object, we check if it has a path source.
        if isinstance(file, str):
            ql.save_embedding_to_db(file, model)
        elif hasattr(ql, 'source') and str(ql.source).endswith('.db'):
             ql.save_embedding_to_db(ql.source, model)
        return None
    else:
        raise ValueError(f"Unknown action: {action}")
