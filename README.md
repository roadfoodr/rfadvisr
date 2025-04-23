# Roadfood Search App

A Streamlit application that allows users to search for restaurants from the Roadfood guide (10th edition) and generate summary articles using OpenAI's language models.

Live demo: https://roadfoodr--roadfood-search-app-run.modal.run/

## Features

- Search for restaurants based on cuisine, location, or other criteria
- Generate summary articles using different prompt types and models
- Compare results from base and fine-tuned models
- Download search results and summaries


## Project Structure

- `roadfood_search_app.py`: The main Streamlit application
- `modal_app.py`: Modal deployment configuration
- `prompts/`: Directory containing prompt templates
  - `basic_summary_prompt.txt`: Basic prompt for generating summaries
  - `advanced_summary_prompt.txt`: Advanced prompt for generating summaries
- `data/`: Directory containing the Chroma vector database and other data files
  - `chroma_rf10th/`: Chroma vector database for the 10th edition

## Notes

- The app uses a fine-tuned GPT-3.5 model for specialized Roadfood summaries
- The Chroma database contains vector embeddings of restaurant descriptions for RAG purposes
- The app is configured to work both locally and in the Modal cloud environment

## Roadmap

```mermaid
graph TD
  A[User Query] --> B[LLM Guardrail Check]
  B --> C[Cache Lookup]
  C -->|Hit| Z[Return Cached Result]
  C -->|Miss| D[LLM Query Analysis]
  D --> E[Tool Calls<br>for specialized search]
  E --> F[Pre-filter corpus<br>~1,000 records]
  F --> G[Semantic Search<br>on Filtered Subset]
  G --> H[Re-rank by<br>Quality Signals]
  H --> I[RAG + LLM for<br>Narrative Generation]
  I --> J[Final Response to User]
```
