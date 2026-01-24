"""
RAG (Retrieval Augmented Generation) processing logic.

Coordinates the loading of query libraries and provides the interface
for similarity-based retrieval of executable queries.
"""

import glob
import logging
import os
import threading
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

from rag_helper.query_library import QueryLibrary

logger = logging.getLogger(__name__)


def get_latest_querylib_file(main_path: str) -> Optional[str]:
    """
    Find the latest query library database file in the given path.
    Files are expected to follow the pattern querylib_YYYYMMDD.db
    """
    querylib_files = glob.glob(os.path.join(main_path, "querylib_*.db"))
    if not querylib_files:
        return None

    try:
        querylib_files.sort(
            key=lambda filename: datetime.strptime(
                filename.split("_")[-1].split(".")[0], "%Y%m%d"
            ),
            reverse=True,
        )
        return querylib_files[0]
    except (ValueError, IndexError) as e:
        logger.warning(f"Error parsing querylib filenames: {e}")
        return querylib_files[0]  # Fallback to simple sort if date parsing fails


class QueryLibraryManager:
    """Singleton manager for loading and caching the QueryLibrary.
    
    Ensures that the library is only loaded once and provides a global
    access point for the RAGProcessor.
    """
    _instance = None
    _lock = threading.Lock()

    def __init__(self):
        """Initialize the manager."""
        self.querylib = None
        self.logger = logging.getLogger(self.__class__.__name__)

    @classmethod
    def get_instance(cls):
        if not cls._instance:
            with cls._lock:
                if not cls._instance:
                    cls._instance = cls()
        return cls._instance

    def load_querylib(
        self,
        main_path: Optional[str] = None,
        querylib_file: Optional[str] = None,
        embedding_model: str = "BAAI/bge-large-en-v1.5"
    ):
        if self.querylib is not None:
            return self.querylib

        main_path = main_path or os.getcwd()
        querylib_file = querylib_file or get_latest_querylib_file(main_path)

        if not querylib_file:
            raise FileNotFoundError(f"No query library file found in {main_path}")

        querylib = QueryLibrary(
            querylib_name="patient_counts",
            source="gold_label_dec_2023",
            querylib_source_file=None,
            col_question="QUESTION",
            col_question_masked="QUESTION_MASKED",
            col_query_w_placeholders="QUERY_WITH_PLACEHOLDERS",
            col_query_executable="QUERY_RUNNABLE",
        )
        querylib = querylib.load(querylib_file=querylib_file)
        querylib.load_embedding_model(embedding_model_name=embedding_model)

        self.logger.info(f"Embedding loaded from {querylib_file}")
        self.querylib = querylib
        return self.querylib


@dataclass
class RAGConfig:
    """Configuration for the RAG processor.
    
    Attributes:
        main_path (Optional[Path]): Base path for database discovery.
        querylib_file (Optional[Path]): Specific path to a .db file.
        sim_threshold (float): Similarity threshold for results.
    """
    main_path: Optional[Path] = None
    querylib_file: Optional[Path] = None
    sim_threshold: float = 0.0


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
        return self.querylib_manager.querylib

    def get_similar_queries(self, question: str, top_k: int = 5):
        """
        Get similar queries from the library for a given question.
        """
        if self.querylib is None:
            raise ValueError("Query library not loaded. Call load_querylib first.")

        return self.querylib.get_df_recs(
            [[question]],
            top_k=top_k,
            sim_threshold=self.config.sim_threshold,
            question_type=None
        )
