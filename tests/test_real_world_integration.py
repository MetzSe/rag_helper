import os
import subprocess
import pytest
import pandas as pd
import numpy as np
import threading
import time
import gc
from pathlib import Path
import rag_helper as rh

# Use the real database file if it exists, otherwise skip integration tests
REAL_DB_PATH = Path("./querylib_20250825.db").absolute()
HAS_REAL_DB = REAL_DB_PATH.exists()

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not HAS_REAL_DB, reason="Real database './querylib_20250825.db' not found for integration testing.")
]

@pytest.fixture
def real_ql():
    return rh.load(str(REAL_DB_PATH))

# ---------------------------------------------------------------------------
# 1. CLI End-to-End Workflows
# ---------------------------------------------------------------------------

def run_cli_command(args):
    """Helper to run rag-cli via subprocess."""
    # Using 'python -m rag_helper.cli' to ensure we use the same environment
    cmd = ["uv", "run", "python", "-m", "rag_helper.cli", "-f", str(REAL_DB_PATH)] + list(args)
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result

def test_cli_view_real_data():
    """rag-cli view -l 50: Verify displays and no crashes."""
    res = run_cli_command(["view", "-l", "50"])
    assert res.returncode == 0
    assert "INDEX" in res.stdout
    assert "QUESTION_MASKED" in res.stdout

def test_cli_info_real_data():
    """rag-cli info 1: Verify retrieval of a specific known record."""
    res = run_cli_command(["info", "1", "--columns", "QUESTION", "QUESTION_TYPE"])
    assert res.returncode == 0
    assert "atopic dermatitis" in res.stdout.lower()

def test_cli_search_real_data():
    """rag-cli search 'cancer': Verify mixed result types."""
    res = run_cli_command(["search", "cancer", "--columns", "QUESTION", "QUESTION_TYPE"])
    assert res.returncode == 0
    assert "QA" in res.stdout
    assert "COHORT_GENERATOR" in res.stdout

@pytest.mark.slow
def test_cli_match_qa():
    """rag-cli match: Verify semantic relevance for QA."""
    res = run_cli_command(["match", "How many patients had a biopsy?"])
    assert res.returncode == 0
    assert "Score" in res.stdout
    assert "biopsy" in res.stdout.lower()

@pytest.mark.slow
def test_cli_match_cohort():
    """rag-cli match: Verify complex inclusion/exclusion matching."""
    query = "Women having at least two diagnoses of endometriosis between 2010 and 2019"
    res = run_cli_command(["match", query, "--type", "COHORT", "-k", "3"])
    assert res.returncode == 0
    assert "endometriosis" in res.stdout.lower()
    assert "QA" not in res.stdout

# ---------------------------------------------------------------------------
# 2. Python API Data Handling
# ---------------------------------------------------------------------------

def test_api_batch_matching(real_ql):
    """Test rh.match with multiple queries."""
    queries = [
        "How many patients with diabetes?",
        "Patients with cancer and anemia"
    ]
    # Small batch to avoid memory spikes in test runner
    df = rh.match(real_ql, query=queries, k=1)
    assert not df.empty
    assert "SourceQuery" in df.columns

def test_api_index_integrity(real_ql):
    """Verify ID columns remain correct after matching real data."""
    query = "Hypertension or anemia"
    df = rh.match(real_ql, query=query, k=5)
    assert not df.empty
    if "INDEX" in df.columns:
        assert all(df["INDEX"].astype(float) >= 0)

# ---------------------------------------------------------------------------
# 3. File Interop (The Round-Trip)
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_file_interop_round_trip(real_ql, tmp_path):
    """Verify data preservation across SQLite -> Memory -> Disk -> Pandas."""
    query = "Atopic dermatitis and cancer"
    df_orig = rh.match(real_ql, query=query, k=5)
    
    csv_path = tmp_path / "test.csv"
    rh.match(real_ql, query=query, k=5, output=str(csv_path))
    
    df_csv = pd.read_csv(csv_path)
    assert len(df_csv) == len(df_orig)
    assert df_csv["Score"].iloc[0] == pytest.approx(df_orig["Score"].iloc[0])

# ---------------------------------------------------------------------------
# 4. Advanced Robustness & Stress
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_concurrency_robustness(real_ql):
    """Simulate concurrent matches on same QL instance."""
    def worker():
        df = rh.match(real_ql, "How many patients?", k=1)
        assert len(df) is not None

    threads = []
    for _ in range(3): # Reduced matching concurrency for test environment
        t = threading.Thread(target=worker)
        threads.append(t)
        t.start()
    
    for t in threads:
        t.join()

@pytest.mark.slow
def test_large_document_matching():
    """Test overflow of token limits with a massive query via CLI."""
    large_query = "patient " * 1500 
    res = run_cli_command(["match", large_query, "-k", "1"])
    assert res.returncode == 0

def test_schema_resilience(tmp_path):
    """Verify failure on DB missing required columns."""
    bad_db = tmp_path / "bad.db"
    import sqlite3
    conn = sqlite3.connect(bad_db)
    conn.execute("CREATE TABLE queries (ID INTEGER, QUESTION TEXT)")
    conn.commit()
    conn.close()
    assert rh.load(str(bad_db)) is None

# ---------------------------------------------------------------------------
# 5. Performance Monitoring
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_matching_performance_cli():
    """Ensure baseline speed floor via CLI."""
    start = time.time()
    res = run_cli_command(["match", "Endometriosis", "-k", "1"])
    elapsed = time.time() - start
    assert res.returncode == 0
    # Startup + Loading + Inference < 10s is safe for real-world CI
    assert elapsed < 10.0
