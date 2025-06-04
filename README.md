# Roadfood Advisor

A Streamlit application that allows users to search for restaurants from the Roadfood guide (10th edition) and generate summary articles using OpenAI's language models.

Live demo: https://rfadvisr.us

## Features

- Search for restaurants based on cuisine, location, or other criteria
- Generate summary articles using different prompt types and models
- Compare results from base and fine-tuned models
- Download search results and summaries


## Project Structure

- `rfadvisr_app.py`: The main Streamlit application
- `modal_app.py`: Modal deployment configuration
- `prompts/`: Directory containing prompt templates
- `data/`: Directory containing the Chroma vector database and other data files
- `chroma_rf10th/`: Chroma vector database for the 10th edition

## Notes

- The app uses a fine-tuned GPT-3.5 model for specialized Roadfood summaries
- The Chroma database contains vector embeddings of restaurant descriptions for RAG purposes
- The app is configured to work both locally and in the Modal cloud environment

## Application Flow Diagram

```mermaid
graph TD
    A[User Input via Streamlit UI] --> B[LLM Scope Guardrail];
    B --> C[LangGraph Filter Pipeline];
    C --> D[Tool Calling Node];
    D --> E[LLM Selects Filter Tools];
    E --> F[State, Region Extraction];
    F --> G[Format Filter Node];
    G --> H[DB Query using ChromaDB];
    H --> I[Similarity Search w/ Filter];
    I --> J[Result Handling];
    J -- Summarization ON --> K[Generate Summary via LLM];
    J -- Summarization OFF --> L[Format Raw Results];
    K --> M[Post-Process Summary];
    M --> N[Display Results in Streamlit];
    L --> N;
    N --> O[Optional: User Feedback];
    N --> P[Optional: Download Results];

    subgraph "Filter Tools Module - filter_tools.py"
        F
    end

    subgraph "Main Application - roadfood_search_app.py"
        A
        B
        C
        D
        E
        G
        subgraph "ChromaDB Filter + Search"
            H
            I
        end
        J
        K
        L
        M
        N
        O
        P
    end


classDef llm fill:#ffe0b3,stroke:#cc7a00,stroke-width:2px,color:#000;
class B,E,J llm;


``` 
