# Context Budget Optimizer

The Context Budget Optimizer is a system that intelligently selects document chunks based on semantic relevance and token cost. When a user asks a question, the system retrieves many potentially relevant text chunks using vector search. Each chunk is then evaluated based on semantic relevance, estimated token cost, and optional metadata such as recency or importance. A deterministic optimizer ranks these chunks by "value per token" and selects the best subset that fits within a configurable context budget. Only this optimized context is sent to the LLM, and the final answer is generated along with transparent metadata showing which chunks were included or excluded.

The system architecture consists of four main layers: ingestion, retrieval, optimization, and generation. The ingestion layer handles document processing, splitting text into manageable chunks and generating embeddings using a local SentenceTransformers model. These embeddings are stored in ChromaDB, a local vector database, while metadata about documents and chunks is tracked in SQLite. The retrieval layer performs similarity search against the vector store to fetch the top-N candidate chunks for any given query. This is where the system finds potentially relevant information before making any optimization decisions.

The optimization layer is the core of the project. It estimates token counts for each retrieved chunk using the Mistral tokenizer, computes relevance scores from embedding similarity, and combines these signals into a ranking formula. The system calculates "value per token" by dividing relevance by token count, then uses a greedy algorithm to pack the highest-value chunks until the token budget is exhausted. This ensures maximum information density within the available context window. The generation layer constructs the final prompt using only the selected context chunks and calls the Mistral API to produce the answer. Throughout this process, all intermediate decisions are logged and exposed for inspection, making the system explainable and easy to reason about.

The entire flow works like this: you start by ingesting documents through the CLI, which processes them into chunks and stores embeddings locally. When you ask a question through the web interface or CLI, the system retrieves candidate chunks, scores them, optimizes selection based on your token budget, and generates an answer. The web interface shows you exactly which chunks were selected, why they were chosen, and how the budget was utilized. You can adjust the token budget with a slider to see how it affects chunk selection and answer quality. The system is designed to be transparent about its decisions, so you can understand the trade-offs between budget constraints and information quality.

This project focuses on LLM systems engineering rather than prompt experimentation. It demonstrates real problems faced in production GenAI systems like cost control, explainability, and retrieval quality. Everything runs locally except for the LLM calls, which use the Mistral API. 


## Features

- Intelligent chunk selection based on value per token
- Configurable token budgets with real-time optimization
- Full observability into decision-making process
- Local-first design for embeddings and vector search
- Modern web interface with charts and statistics
- CLI and web UI support
- Caching for improved performance
- Batch processing capabilities

## Technology Stack

- Python  
- SentenceTransformers for embeddings
- ChromaDB for vector storage
- Mistral API for LLM generation
- FastAPI for the backend
- SQLite for metadata tracking

 
