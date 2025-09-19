#Second Brain

*Local RAG Agent *by Henry Daum


---

Screenshots & GIFs


---

Features



* Semantic search: Searches the entire knowledgebase to find information based on deep semantic meaning.
* Supported files: .txt, .pdf, .docx, .gdoc, .png, .jpg, .jpeg, .gif, .webp
* Multimodal support: search for both text and images.
* Find similar images: attach an image to find similar looking ones in your filebase.
* Attachments of any size are supported.
* <span style="text-decoration:underline;">Optional</span> AI Mode:
    * LLM Integration: Supports local models via LM Studio and cloud models via Gemini.
    * **Retrieval-Augmented Generation (RAG)**: Uses search results to provide contextually aware AI responses.
    * **AI-Powered Query Expansion:** Generates multiple related search queries to find more comprehensive results.
    * **AI Result Filtering**: Reranks and filters search results for relevance.
    * **AI Summarization**: Synthesizes information from multiple sources into a concise summary.
    * The AI Mode is optional; a search can be run without AI if there are not enough computational resources available.
* Security and privacy: local search ensures no data is sent to the internet (only applies if not using Gemini API option).
* Open source: all the code is right here.


---

How It Works

*Backend Code*

SecondBrainBackend.py

This file handles two things: syncing the directory and performing searches. 

**Syncing: **Scans the target directory, detects new, updated, or deleted files. Extracts text from files and splits it into manageable chunks, while also batching images together for the sake of speed. Converts text chunks and images into vector embeddings using sentence-transformer models. Stores embeddings in a persistent ChromaDB vector database. 

**Retrieval**: Performs semantic search, reranks results using Maximal Marginal Relevance (MMR), and filters for relevance.

*Frontend Code*

SecondBrainFrontend.py

The frontend code controls the user interface for the application. It controls threading for non-blocking backend operations, prompting patterns for the AI, and logging functions. Finds the most relevant parts of text attachments, ensures smooth user experience, and integrates the backend. Allows the user to send messages in a chat-like format. Performs many different functions simultaneously, and orchestrates blocking and re-enabling different buttons at different times to prevent crashes. *Looks good, too, if I may say so myself.*

*Settings*

config.json

This file can be edited and changes the features of Second Brain.

*Google Drive Authentication*

credentials.json

This file must be added if using Google Drive, as it allows the syncing of Google Doc files. 


---

Getting Started



1. Prerequisites
2. Installation
3. Configuration
4. Running the application


---

Usage Guide



* Syncing your files:
    * The sync can be cancelled midway through with no consequences.
* Searching:
    * 
* Using AI mode:
    * 
* Attaching files:


---

Configuration Details

config.json


<table>
  <tr>
   <td><strong>Parameter Name</strong>
   </td>
   <td><strong>Function</strong>
   </td>
   <td><strong>Range</strong>
   </td>
   <td><strong>Default</strong>
   </td>
  </tr>
  <tr>
   <td>credentials_path
   </td>
   <td>Path to credentials.json which is used to authenticate Google Drive downloads.
   </td>
   <td>Any path
   </td>
   <td>"credentials.json"
   </td>
  </tr>
  <tr>
   <td>target_directory
   </td>
   <td>Which directory to sync to. Embeds all valid files in the directory.
   </td>
   <td>Any directory
   </td>
   <td>"C:\\Users\\user\\My Drive"
   </td>
  </tr>
  <tr>
   <td>text_model_name
   </td>
   <td>SentenceTransformers text embedding model name. "BAAI/bge-large-en-v1.5" and "BAAI/bge-small-en-v1.5" work well.
   </td>
   <td>Any valid name
   </td>
   <td>"BAAI/bge-large-en-v1.5"
   </td>
  </tr>
  <tr>
   <td>image_model_name
   </td>
   <td>SentenceTransformers image embedding model name. "clip-ViT-B-16" and "clip-ViT-B-32" work well.
   </td>
   <td>Any valid name
   </td>
   <td>"clip-ViT-B-16"
   </td>
  </tr>
  <tr>
   <td>batch_size
   </td>
   <td>How many text chunks/images are processed in parallel with the embedders. Embedding is faster with higher values but laggier.
   </td>
   <td>1-50
   </td>
   <td>20
   </td>
  </tr>
  <tr>
   <td>chunk_size
   </td>
   <td>Maximum size of text chunks created from the text splitter; 200 and 0 overlap is shown in research to be very good.
   </td>
   <td>50-1000
   </td>
   <td>200
   </td>
  </tr>
  <tr>
   <td>chunk_overlap
   </td>
   <td>Used to preserve continuity when splitting text.
   </td>
   <td>0-200
   </td>
   <td>0
   </td>
  </tr>
  <tr>
   <td>mmr_lambda
   </td>
   <td>Prioritize diversity or relevance in search results; 0 = prioritize diversity only, 1 = prioritize relevance only.
   </td>
   <td>0.0-1.0
   </td>
   <td>0.7
   </td>
  </tr>
  <tr>
   <td>search_multiplier
   </td>
   <td>How many results to process for each query.
   </td>
   <td>At least 1
   </td>
   <td>20
   </td>
  </tr>
  <tr>
   <td>z_score
   </td>
   <td>Only search results with an MMR score better than this many standard deviations will be returned.
   </td>
   <td>1-3
   </td>
   <td>1.5
   </td>
  </tr>
  <tr>
   <td>ai_mode
   </td>
   <td>Whether or not to use AI to aid in searches.
   </td>
   <td>true or false
   </td>
   <td>true
   </td>
  </tr>
  <tr>
   <td>llm_backend
   </td>
   <td>Choose which AI to use.
   </td>
   <td>“LM Studio” or “Gemini”
   </td>
   <td>"LM Studio"
   </td>
  </tr>
  <tr>
   <td>lms_model_name
   </td>
   <td>Can be any language model from LM Studio, but it must be already downloaded.
   </td>
   <td>Any valid name
   </td>
   <td>"gemma-3-4b-it"
   </td>
  </tr>
  <tr>
   <td>gemini_api_key
   </td>
   <td>Key to connect to paid, online Gemini service.
   </td>
   <td>-
   </td>
   <td>"YOUR_GEMINI_API_KEY_HERE"
   </td>
  </tr>
  <tr>
   <td>max_results
   </td>
   <td>Sets the maximum for both text and image results.
   </td>
   <td>1-30
   </td>
   <td>7
   </td>
  </tr>
  <tr>
   <td>search_prefix
   </td>
   <td>Phrase prefixed to the start of text search queries; useful with some text embedding models. Set to “” to disable.
   </td>
   <td>-
   </td>
   <td>"Represent this sentence for searching relevant passages: "
   </td>
  </tr>
  <tr>
   <td>query_multiplier
   </td>
   <td>How many queries the AI is asked to make to augment the search, based on the user’s attachment and user prompt.
   </td>
   <td>At least 1
   </td>
   <td>5
   </td>
  </tr>
  <tr>
   <td>n_attachment_keywords
   </td>
   <td>Keywords are extracted from text attachments. With AI Mode off, these are used as additional queries. In AI Mode, these are used as additional context for the AI.
   </td>
   <td>0-10
   </td>
   <td>3
   </td>
  </tr>
  <tr>
   <td>n_attachment_chunks
   </td>
   <td>Text chunks are extracted from attachments. With AI Mode off, these are used as additional queries. In AI Mode, these are used as additional context for the AI. Decrease if the context window is small.
   </td>
   <td>0-10
   </td>
   <td>3
   </td>
  </tr>
</table>



---

Dependencies


    dependencies = [


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


    ]


---

Further Reading / Sources

[https://lmstudio.ai/docs/python](https://lmstudio.ai/docs/python)

[https://research.trychroma.com/evaluating-chunking](https://research.trychroma.com/evaluating-chunking)

[https://github.com/ALucek/chunking-strategies/blob/main/chunking.ipynb](https://github.com/ALucek/chunking-strategies/blob/main/chunking.ipynb)

[https://flet.dev/docs/](https://flet.dev/docs/)

[https://huggingface.co/spaces/mteb/leaderboard](https://huggingface.co/spaces/mteb/leaderboard)
