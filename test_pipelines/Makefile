# Makefile for Advanced RAG Testing
# Usage:
#   make match
#   make match QUERY="How many patients?"

LIB ?= querylib_20250825.db
TOP_K ?= 3
THRESHOLD ?= 0.0

.PHONY: match
# Define multi-line query correctly
define DEFAULT_QUERY
How many patients with condition?
endef

# Use default query if not overridden from command line
QUERY ?= $(DEFAULT_QUERY)

# Basic comparison run
match:
	@echo "Running match with BAAI/bge-large-en-v1.5..."
	uv run rag-cli -f $(LIB) match "$(QUERY)" -k $(TOP_K) -t $(THRESHOLD) -e BAAI/bge-large-en-v1.5
	@echo "Running match with intfloat/multilingual-e5-large..."
	uv run rag-cli -f $(LIB) match "$(QUERY)" -k $(TOP_K) -t $(THRESHOLD) -e intfloat/multilingual-e5-large
	
