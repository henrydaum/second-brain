# Second Brain  
# *Local RAG Agent* <mark>by Henry Daum</mark>

---

## Screenshots & GIFs

<img width="1858" height="1343" alt="Screenshot 2025-09-19 123225" src="https://github.com/user-attachments/assets/805f5a0d-520a-496e-8c45-8ef1fb975e61" />

https://github.com/user-attachments/assets/ca803713-4199-4403-91ac-8b64056ae790

---

## Features

- Semantic search: Searches the entire knowledgebase to find information based on deep semantic meaning.  
- Supported files: .txt, .pdf, .docx, .gdoc, .png, .jpg, .jpeg, .gif, .webp  
- Multimodal support: search for both text and images.  
- Find similar images: attach an image to find similar looking ones in your filebase.  
- Attachments of any size are supported.  
- Optional AI Mode:  
  * LLM Integration: Supports local models via LM Studio and cloud models via Gemini.  
  * **Retrieval-Augmented Generation (RAG)**: Uses search results to provide contextually aware AI responses.  
  * **AI-Powered Query Expansion:** Generates multiple related search queries to find more comprehensive results.  
  * **AI Result Filtering**: Reranks and filters search results for relevance.  
  * **AI Summarization**: Synthesizes information from multiple sources into a concise summary.  
  * The AI Mode is optional; a search can be run without AI if there are not enough computational resources available.  
- Security and privacy: local search ensures no data is sent to the internet (only applies if not using Gemini API option).  
- Open source: all the code is right here.

---

## How It Works  
### *Backend Code*  
<mark>SecondBrainBackend.py</mark>

This file handles two things: syncing the directory and performing searches. 

**Syncing:** Scans the target directory, detects new, updated, or deleted files. Extracts text from files and splits it into manageable chunks, while also batching images together for the sake of speed. Converts text chunks and images into vector embeddings using sentence-transformer models. Stores embeddings in a persistent ChromaDB vector database. 

**Retrieval**: Performs semantic search, reranks results using Maximal Marginal Relevance (MMR), and filters for relevance.

### *Frontend Code*  
<mark>SecondBrainFrontend.py</mark>

The frontend code controls the user interface for the application. It controls threading for non-blocking backend operations, prompting patterns for the AI, and logging functions. Finds the most relevant parts of text attachments, ensures smooth user experience, and integrates the backend. Allows the user to send messages in a chat-like format. Performs many different functions simultaneously, and orchestrates blocking and re-enabling different buttons at different times to prevent crashes. *Looks good, too, if I may say so myself.*

### *Settings*  
<mark>config.json</mark>

This file can be edited and changes the features of Second Brain.

### *Google Drive Authentication*  
<mark>credentials.json</mark>

This file must be added if using Google Drive, as it allows the syncing of Google Doc files. 

---

## Getting Started

1. Prerequisites  
2. Installation  
3. Configuration  
4. Running the application

---

## Usage Guide

- Syncing your files:  
  * The sync can be cancelled midway through with no consequences.  
- Searching:  
  *   
- Using AI mode:  
  *   
- Attaching files:

---

## Configuration Details  
config.json

| Parameter Name | Function | Range | Default |
| ----- | ----- | ----- | ----- |
| credentials\_path | Path to credentials.json which is used to authenticate Google Drive downloads. | Any path | "credentials.json" |
| target\_directory | Which directory to sync to. Embeds all valid files in the directory. | Any directory | "C:\\\\Users\\\\user\\\\My Drive" |
| text\_model\_name | SentenceTransformers text embedding model name. "BAAI/bge-large-en-v1.5" and "BAAI/bge-small-en-v1.5" work well. | Any valid name | "BAAI/bge-large-en-v1.5" |
| image\_model\_name | SentenceTransformers image embedding model name. "clip-ViT-B-16" and "clip-ViT-B-32" work well. | Any valid name | "clip-ViT-B-16" |
| batch\_size | How many text chunks/images are processed in parallel with the embedders. Embedding is faster with higher values but laggier. | 1-50 | 20 |
| chunk\_size | Maximum size of text chunks created from the text splitter; 200 and 0 overlap is shown in research to be very good. | 50-1000 | 200 |
| chunk\_overlap | Used to preserve continuity when splitting text. | 0-200 | 0 |
| mmr\_lambda | Prioritize diversity or relevance in search results; 0 \= prioritize diversity only, 1 \= prioritize relevance only. | 0.0-1.0 | 0.7 |
| search\_multiplier | How many results to process for each query. | At least 1 | 20 |
| z\_score | Only search results with an MMR score better than this many standard deviations will be returned. | 1-3 | 1.5 |
| ai\_mode | Whether or not to use AI to aid in searches. | true or false | true |
| llm\_backend | Choose which AI to use. | “LM Studio” or “Gemini” | "LM Studio" |
| lms\_model\_name | Can be any language model from LM Studio, but it must be already downloaded. | Any valid name | "gemma-3-4b-it" |
| gemini\_api\_key | Key to connect to paid, online Gemini service. | \- | "YOUR\_GEMINI\_API\_KEY\_HERE" |
| max\_results | Sets the maximum for both text and image results. | 1-30 | 7 |
| search\_prefix | Phrase prefixed to the start of text search queries; useful with some text embedding models. Set to “” to disable. | \- | "Represent this sentence for searching relevant passages: " |
| query\_multiplier | How many queries the AI is asked to make to augment the search, based on the user’s attachment and user prompt. | At least 1 | 5 |
| n\_attachment\_keywords | Keywords are extracted from text attachments. With AI Mode off, these are used as additional queries. In AI Mode, these are used as additional context for the AI. | 0-10 | 3 |
| n\_attachment\_chunks | Text chunks are extracted from attachments. With AI Mode off, these are used as additional queries. In AI Mode, these are used as additional context for the AI. Decrease if the context window is small. | 0-10 | 3 |

---

## Dependencies
dependencies \= \[  
"flet",  
"PyPDF2",  
"chromadb==1.0.20",  
"torch==2.8.0",  
"numpy",  
"scipy",  
"transformers==4.56.1",  
"langchain-text-splitters==0.3.11",  
"sentence-transformers==5.1.0",  
"yake",  
"lmstudio==1.5.0",  
"python-docx",  
"Pillow",  
"google-api-python-client",  
"google-auth",  
"google-auth-oauthlib",  
"google-generativeai==0.8.5",  
"requests"  
\]  
---

## Further Reading / Sources  
[https://lmstudio.ai/docs/python](https://lmstudio.ai/docs/python)

[https://research.trychroma.com/evaluating-chunking](https://research.trychroma.com/evaluating-chunking)

[https://github.com/ALucek/chunking-strategies/blob/main/chunking.ipynb](https://github.com/ALucek/chunking-strategies/blob/main/chunking.ipynb)

[https://flet.dev/docs/](https://flet.dev/docs/)

[https://huggingface.co/spaces/mteb/leaderboard](https://huggingface.co/spaces/mteb/leaderboard)  
