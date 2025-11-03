# Second Brain  
# *Local RAG Agent* <mark>by Henry Daum</mark>

**If you find any errors, please let me know.**

---

## Screenshots & GIFs

<img width="1134" height="826" alt="Screenshot 2025-09-19 123225" src="https://github.com/user-attachments/assets/805f5a0d-520a-496e-8c45-8ef1fb975e61" />

https://github.com/user-attachments/assets/ca803713-4199-4403-91ac-8b64056ae790

https://github.com/user-attachments/assets/38dd0998-328a-4164-8b5e-9ec1a52dcb84

---

## Features

- Semantic search: Searches the entire knowledgebase to find information based on deep semantic meaning.
- Lexical search: find a needle in a haystack using keywords.
- Supported files: .txt, .pdf, .docx, .gdoc, .png, .jpg, .jpeg, .gif, .webp  
- Multimodal in three ways:
  * Can embed and search for both text and images.
  * Can attach text documents and images, and use them to search.
  * Can use AI language models with vision.
- Optional AI Mode:
  * **LLM Integration**: Supports local models via LM Studio and cloud models, like GPT-5, via OpenAI.  
  * **Retrieval-Augmented Generation (RAG)**: Uses search results to provide high-quality AI responses rooted in the knowledge-base.
  * The AI Mode is entirely optional; a search can be run without AI.
- Sync 100,000+ files.
- Security and privacy: local search ensures no data is sent to the internet.
- Works with Google Drive.
- Open source: all the code is right here.

---

## Infographic of How it Works
<img width="1206" height="1275" alt="Second Brain Search Flowchart Infographic" src="https://github.com/user-attachments/assets/efdb2226-32c0-4b07-aed7-73aef4188ee1" />

---

## How It Works  
### *Backend Code*  
<mark>SecondBrainBackend.py</mark>

This file handles two things: syncing the directory and performing searches. 

**Syncing:** Scans the target directory, detects new, updated, or deleted files. Extracts text from files and splits it into manageable chunks, while also batching images together for the sake of speed. Converts text chunks and images into vector embeddings using sentence-transformer models. Stores embeddings in a persistent ChromaDB vector database. To enable lexical search, image captions are created and stored along with image embeddings. A lexical database of both text documents and image captions is created using BM25.

**Retrieval**: Performs both a semantic and lexical search, then combines the outputs, reranks all results using Maximal Marginal Relevance (MMR), takes the top N outputs, and finally filters for relevance using AI (last step optional).

### *Frontend Code*  
<mark>SecondBrainFrontend.py</mark>

The frontend code controls the user interface for the application. It controls threading for non-blocking backend operations, prompting patterns for the AI, and logging functions. Manages data from text and image attachments to construct complex prompts for the LLM. Allows the user to send messages in a chat-like format. Performs many different functions simultaneously, and orchestrates blocking and re-enabling different buttons at different times to prevent crashes. *Looks good, too, if I may say so myself.*

Uniquely, image and text results can be attached in order to perform continous searches, like web-surfing but through your own filebase. There is no other experience exactly like it. Simply click on an image result or the four squares by a text result and then click "attach result." Other similar amenities are available in the app.

### *Settings*  
<mark>config.json</mark>

This file can be edited by the user and changes the cool features of Second Brain (see below).

### *Google Drive Authentication*  
<mark>credentials.json</mark>

This file must be added if using Google Drive (optional), as it allows the syncing of Google Doc (.gdoc) files.

### *Image Labels*
<mark>image_labels.csv</mark>

This CSV is used as a pool for possible image labels. The image labels are chosen based on how close the image embedding is to each label embedding (a categorization task). It was constructed based on Google's Open Images dataset for object identification.

---

## Getting Started

### 1. Prerequisites
Before running the application, ensure you have the following:
   - Python: Second Brain requires Python 3.9 or higher.
   - LM Studio: To use AI Mode with a local LLM, you must download and install LM Studio with a .gguf model, ideally with vision capabilities (unsloth/gemma-3-4b-it works well, but any model may be used).
   - Open AI API: To use a model from the OpenAI API, make sure you have an API key from platform.openai.com with sufficient funds.
   - Dependencies: Install the required Python libraries (see below - dependencies) using pip.
   - System requirements: The search engine can be used with GPU or CPU. It will automatically detect if a GPU exists and use it if available. Different text and image embedding models use different amounts of memory, and can be configured. For example, the default models use 2GB of VRAM/RAM.
### 2. Installation
Download ```SecondBrainBackend```, ```SecondBrainFrontend```, ```config.json```, and ```image_labels.csv``` from this repository. Only these four files are needed. Not bad, right? Place them in a folder.
### 3. Configuration
Open ```config.json``` and update the <mark>target_directory</mark> to point to the folder you want to use as your knowledge base.

(Optional - skip if irrelevant) To enable Google Doc syncing, follow the [Google Cloud API](https://developers.google.com/workspace/drive/api/guides/about-sdk) instructions to get a ```credentials.json``` file using OAuth. Either place the file in the project directory, or place it somewhere else and update <mark>credentials_path</mark> in ```config.json``` to point to it. The first time you sync, a browser window will open for you to authorize the application. Authentication is very finnicky and it might be necessary to delete the authorization token (```token.json```) and then reauthorize to get Drive syncing to work. You can do this with the "Reauthorize Drive" button.
### 4. Running the application
To start the application, run **SecondBrainFrontend.py** from the project folder. During the first run, it may take a while to download the models.

---

## Usage Guide

#### Syncing your files:  
*Click the Sync Directory button to start embedding the files from your <mark>target_directory</mark>. Depending on the size of the directory, it can take a while (hours), but is necessary to enable search.
The sync can be cancelled midway through by clicking the *Cancel Sync* button with no consequences. If restarted, the sync will continue where it left off.*
#### Searching:  
*Text results are based on relevant chunks of text from the embedding database.
You can click on image results and file paths to open them directly, attach them, see their parent folder, or copy their path.*
#### Using AI mode:  
*Toggle the AI Mode checkbox to enable or disable **LLM augmented search**.
When enabled, the Second Brain loads the selected LLM from LM Studio or the OpenAI API. Searches are (optionally) enhanced with AI-generated queries, results are (optionally) filtered by the AI for relevance, and a final "AI Insights" summary is always streamed in the results container, when AI Mode is active. It is optional but recommended to use vision-enabled models, since it is needed to filter images and give insights on them. When disabled, the LLM is unloaded to save system resources, and the app performs a direct vector/lexical search with no query expansion, filtering, or summary.*
#### Attaching files:  
*If you attach a text document (.pdf, .docx, etc.) with a query, the entire text will be added if it is below a certain size (see config: max_attachment_size). If it is too large, it extracts several relevant chunks to provide focused context for the search.
If you attach an image, you can send it to find visually similar images in your database and related documents.
You can send an attachment without a message.*
#### Saving Insights:  
*If you find that Second Brain has produced a good insight, you can save it for future use. When you click the "Save Insight" button, found after the AI Insights section of a response, the query and response get saved to a .txt file. These text files will be embedded, which gives Second Brain a long-term memory of its own contributions. (You can re-read or delete entries by going to the saved_insights folder, in the Second Brain directory.)*

---

## Configuration Details  
config.json

| Parameter Name | Function | Range | Default |
| ----- | ----- | ----- | ----- |
| credentials\_path | Path to credentials.json which is used to authenticate Google Drive downloads. | Any path | "credentials.json" |
| target\_directory | Which directory to sync to. Embeds all valid files in the directory. | Any directory | "C:\\\\Users\\\\user\\\\My Drive" |
| text\_model\_name | SentenceTransformers text embedding model name. "BAAI/bge-m3", "BAAI/bge-large-en-v1.5", and "BAAI/bge-small-en-v1.5" work well, with the small version using fewer system resources, and m3 using the most. Also, BAAI/bge-m3 is multilingual. If you change the text model name, you need to remake all of your embeddings with that model - delete the chroma_db folder or use the helper function at the bottom of SecondBrainBackend. | Any valid name | "BAAI/bge-large-en-v1.5" |
| image\_model\_name | SentenceTransformers image embedding model name. "clip-ViT-L-14", "clip-ViT-B-16", and "clip-ViT-B-32" work well, with the 32 version using fewer system resources, and L-14 using the most. If you change the image model name, you need to remake all of your embeddings with that model. | Any valid name | "clip-ViT-B-16" |
| embed\_use\_cuda | Turning this to false forces the embedding models to use CPU. If set to true, uses CUDA if possible. | true or false | true |
| batch\_size | How many text chunks/images are processed in parallel with the embedders. Embedding is faster with higher values, depending on hardware. | 1-50 | 20 |
| chunk\_size | Maximum size of text chunks created from the text splitter, in tokens; 200 and 0 overlap is shown in research to be very good. | 50-2000 | 200 |
| chunk\_overlap | Used to preserve continuity when splitting text. | 0-200 | 0 |
| max_seq_length | Maximum input size for the text embedding model, in tokens. Lower values are faster. Must be larger than chunk_size. | Depends on model, 250-8192 | 512 |
| mmr\_lambda | Prioritize diversity or relevance in search results; 0 \= prioritize diversity only, 1 \= prioritize relevance only. | 0.0-1.0 | 0.5 |
| mmr\_alpha | The MMR rerank uses a hybrid semantic-lexical diversity metric. An mmr\_alpha of 0.0 prioritizes lexical diversity, while 1.0 is for using only semantic diversity in choosing how to rerank. | 0.0-1.0 | 0.5 |
| search\_multiplier | How many results to process for each query. | At least 1 | 20 |
| ai\_mode | Whether or not to use AI to aid in searches. | true or false | true |
| llm_filter_results | Filtering results is somewhat slow. This gives the option to turn that part of ai_mode on/off. | true or false | false |
| llm\_backend | Choose which AI backend to use. OpenAI is slower but has smarter models. | “LM Studio” or "OpenAI" | "LM Studio" |
| lms\_model\_name | Can be any language model from LM Studio, but it must be already downloaded. | Any valid name | "unsloth/gemma-3-4b-it" |
| openai_model_name | Can be any OpenAI text model. (Some OpenAI models, like gpt-5, require additional verification to use.) | Any OpenAI model | "gpt-5-mini" |
| openai_api_key | When using OpenAI as a backend, an API key from [platform.openai.com](https://platform.openai.com/) is required. Using it costs money, so the account must have enough funds. If this field is left blank (""), the OpenAI client will look for an *environmental variable* called OPENAI_API_KEY, and use that. Otherwise, the client will use the string found here. | Any API key | "" |
| max\_results | Sets the maximum for both text and image results. | 1-30 | 7 |
| search\_prefix | Phrase prefixed to the start of text search queries; recommended with the text embedding models BAAI/bge-large-en-v1.5 and BAAI/bge-small-en-v1.5. Not needed for BAAI/bge-m3. Set to “” to disable. | \- | "Represent this sentence for searching relevant passages: " |
| query\_multiplier | How many queries the AI is asked to make to augment the search, based on the user’s attachment and user prompt. Set to 0 to turn off the feature. | 0 or more | 5 |
| max\_attachment\_size | Will try to add an entire attachment to an AI as context if it is below this size. Measured in tokens. Decrease if the context window is small. If an attachment is larger than this size, only the most relevant chunks of the attachment will be added. | 256-8192 | 1024 |
| n\_attachment\_chunks | Relevant text chunks are extracted from attachments. These are used as additional queries for lexical and semantic search. | 1-10 | 3 |
| system\_prompt | Special instructions for how the AI should do its job. Feel free to change the special instructions to anything. | Any string | "You are a personal search assistant, made to turn user prompts into accurate and relevant search results, using information from the user's database. Special instruction: Sound casually confident and lightly playful, as if you enjoy the user's company but won't admit it. Not too warm, but still focused on the user. Avoid over-explaining or being too sweet. Be concise and pragmatic." |
| generate\_queries\_prompt | Special instructions for how the AI should do its job. It's best not to change this, but you can. | Any string | "{system_prompt}\n\n{user_request}\n{attachment_context}\nBased on the user's prompt, generate {n} creative search queries that could retrieve relevant {content} to answer the user. These queries will go into a semantic search algorithm to retreive relevant {content} from the user's database. The queries should be broad enough to find a variety of related items. These queries will search a somewhat small and personal database (that is, the user's hard drive). Respond with a plain list with no supporting text or markdown.{reminder_suffix}" |
| evaluate\_text\_relevance\_prompt | Special instructions for how the AI should do its job. It's best not to change this, but you can. | Any string | "{system_prompt}\n\n{attachment_context}\n{user_request}\nDocument excerpt to evaluate:\n\"{chunk}\"\n\nIs this excerpt worth keeping? Respond only with YES or NO.\n\nRelevance is the most important thing. Does the snippet connect to the user's request?\n\nIf the excerpt is gibberish, respond with NO.\n\n(Again: respond only with YES or NO.)" |
| evaluate\_image\_relevance\_prompt | Special instructions for how the AI should do its job. It's best not to change this, but you can. | Any string | "{system_prompt}\n\n{attachment_context}\n{user_request}\nIs the provided image worth keeping? Respond only with YES or NO.\n\nRelevance is the most important thing. Does the photo connect to the user's request?\n\nIf the image is blank, corrupted, or unreadable, respond with NO.\n\nImage file path: {image_path}\n\nIf the user's query has an exact match within the file path, respond with YES.\n\n(Again: respond only with YES or NO.)" |
| synthesize\_results\_prompt | Special instructions for how the AI should do its job. It's best not to change this, but you can. | Any string | "{system_prompt}\n\nIt is {date_time}.\n\n{user_request}\n{attachment_context}\n{database_results}\n**Your Task:**\nBased exclusively on the information provided above, write a concise and helpful response. Your primary goal is to synthesize the information to **guide the user towards what they want**.\n\n**Instructions:**\n- The text search results are **snippets** from larger documents and may be incomplete.\n- Do **not assume or guess** the author of a document unless the source text makes it absolutely clear.\n- The documents don't have timestamps; don't assume the age of a document unless the source text makes it absolutely clear.\n- Cite every piece of information you use from the search results with its source, like so: (source_name).\n- If the provided search results are not relevant to the user's request, state that you could not find any relevant information.\n- Use markdown formatting (e.g., bolding, bullet points) to make the response easy to read.\n- If there are images, make sure to consider them for your response." |

---

## Dependencies

*You can install the Python dependencies with the following command:*

```pip install flet python-docx numpy chromadb torch Pillow requests transformers langchain sentence-transformers scipy fitz langchain_core```

dependencies \= \[  
"fitz",
"flet",  
"PyPDF2",  
"chromadb==1.0.20",  
"torch==2.8.0",  
"numpy",  
"scipy",  
"transformers==4.56.1",
"langchain_core",
"langchain-text-splitters==0.3.11",  
"sentence-transformers==5.1.0",  
"lmstudio==1.5.0",  
"python-docx",  
"Pillow",  
"google-api-python-client",  
"google-auth",  
"google-auth-oauthlib",  
"google-generativeai==0.8.5",
"rank_bm25",
"requests"  
\]

## Hidden Variables
These can be found in the code to change minor features.
```IMG_THUMBNAIL, stop_words, jpeg_quality, min_len, low_compression_threshold, high_compression_threshold, temperature, openai_vision_keywords, image_preview_size```

---

## Further Reading / Sources  
[https://lmstudio.ai/docs/python](https://lmstudio.ai/docs/python)

[https://research.trychroma.com/evaluating-chunking](https://research.trychroma.com/evaluating-chunking)

[https://github.com/ALucek/chunking-strategies/blob/main/chunking.ipynb](https://github.com/ALucek/chunking-strategies/blob/main/chunking.ipynb)

[https://flet.dev/docs/](https://flet.dev/docs/)

[https://huggingface.co/spaces/mteb/leaderboard](https://huggingface.co/spaces/mteb/leaderboard)  
