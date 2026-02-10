import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from rag_helper.query_library_manager import QueryLibraryManager

logger = logging.getLogger(__name__)


@dataclass
class RAGConfig:
    """Configuration for the RAG processor.
    
    Attributes:
        main_path (Optional[Path]): Base path for database discovery.
        querylib_file (Optional[Path]): Specific path to a .db file.
        sim_threshold (float): Similarity threshold for results.
        embedding_model_name (Optional[str]): Model name to use for matching.
            If None, uses the default (first) embedding.
        question_type (Optional[str]): Optional filter for question type (e.g. 'QA', 'COHORT').
    """
    main_path: Optional[Path] = None
    querylib_file: Optional[Path] = None
    sim_threshold: float = 0.0
    embedding_model_name: Optional[str] = None
    question_type: Optional[str] = None


class RAGProcessor:
    """Processor for retrieving similar queries based on questions.
    
    Uses the QueryLibraryManager to access stored queries and embeddings.
    """
    def __init__(self, config: RAGConfig):
        """Initialize the processor with a RAGConfig."""
        self.config = config
        self.querylib_manager = QueryLibraryManager.get_instance()

    @property
    def querylib(self):
        return self.querylib_manager

    def get_similar_queries(self, question: str, top_k: int = 5):
        """
        Get similar queries from the library for a given question.
        """
        # Ensure library is loaded if not already
        if not hasattr(self.querylib, 'df_querylib') or self.querylib.df_querylib.empty:
            self.querylib_manager.load_querylib(
                main_path=str(self.config.main_path) if self.config.main_path else None,
                querylib_file=str(self.config.querylib_file) if self.config.querylib_file else None
            )

        return self.querylib.get_df_recs(
            [[question]],
            top_k=top_k,
            sim_threshold=self.config.sim_threshold,
            question_type=self.config.question_type,
            embedding_model_name=self.config.embedding_model_name,
        )

