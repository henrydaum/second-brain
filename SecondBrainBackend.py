# Imports
import os.path
import io
import pathlib
import json
import re
from typing import List, Dict, Any, Optional
import math
import time
import requests

# 3rd party imports
import PyPDF2
import chromadb
import torch
import numpy as np
from docx import Document
from PIL import Image
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaIoBaseDownload
from transformers import AutoTokenizer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cdist

def _log(msg: str, log_callback=None):
    """Send log message to UI if available, else fallback to print."""
    if log_callback:
        log_callback(msg)
    else:
        print(msg)

# Internet
def is_connected():
    try:
        requests.head('http://www.google.com', timeout=1)
        return True
    except (requests.ConnectionError, requests.Timeout):
        return False

# Google Cloud API
def get_drive_service(log_callback, config):
    """Handles the OAuth 2.0 flow and creates a Google Drive API service object. If you get the error 'NoneType' object has no attribute 'files', it means you must delete token.json and try again."""
    # Test if connected to internet
    if not is_connected():
        _log("No internet — skipping Google Drive.", log_callback)
        return None  # If no internet, no drive service.
    
    # See if cred path exists.
    cred_path = pathlib.Path(config.get("credentials_path", "credentials.json")) if config else pathlib.Path("credentials.json")
    if not cred_path.exists():
        _log("No credentials.json found — skipping Google Drive.", log_callback)
        return None
    
    # Define the scopes your application will need.
    SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]

    creds = None
    # The file token.json stores the user's access and refresh tokens.
    # It's created automatically when the authorization flow completes for the first time.
    if os.path.exists("token.json"):
        creds = Credentials.from_authorized_user_file("token.json", SCOPES)
    
    # If there are no (valid) credentials available, let the user log in.    
    try:
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                # This uses your credentials.json file to trigger the browser-based login.
                flow = InstalledAppFlow.from_client_secrets_file(cred_path, SCOPES)
                auth_url, _ = flow.authorization_url(prompt='consent')
                _log(f"Authenticating...", log_callback)
                creds = flow.run_local_server(port=0)
            
            # Save the credentials for the next run
            with open("token.json", "w") as token:
                token.write(creds.to_json())
            service = build("drive", "v3", credentials=creds)
            return service
    except Exception as e:
        _log(f"Authentication failed — skipping Google Drive.", log_callback)
        return None

def download_drive_content(drive_service, doc_id: str, mimeType: str) -> str:
    """Downloads a Google Doc's content as plain text using its file ID."""
    try:
        # Use the 'export_media' method to download the Google Doc as plain text.
        # The mimeType tells the API how to convert the Doc to text.
        request = drive_service.files().export_media(fileId=doc_id, mimeType=mimeType)
        
        # Use an in-memory binary stream to hold the downloaded content.
        fh = io.BytesIO()
        downloader = MediaIoBaseDownload(fh, request)
        
        done = False
        while not done:
            status, done = downloader.next_chunk()
            # print(f"Download progress: {int(status.progress() * 100)}%.")
            
        # After downloading, go to the beginning of the stream and decode it as text.
        fh.seek(0)
        return fh.read().decode('utf-8')

    except HttpError as error:
        print(f"  [ERROR] Could not download Google Drive file: {error}")
        return ""

# Parsers
def parse_gdoc(file_path: pathlib.Path, drive_service, log_callback) -> str:
    """Parses a .gdoc file. These are JSON files containing a URL to the real doc."""
    # print(f"-> Processing .gdoc: {file_path.name}")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            gdoc_data = json.load(f)
        
        doc_url = gdoc_data.get('doc_id')
        if not doc_url:
            print(f"  [Warning] Could not find URL in {file_path.name}")
            return ""
        
        # API Call
        # print(f"  Found URL: {doc_url}")
        content = download_drive_content(drive_service, doc_url, "text/plain")
        return content
    except json.JSONDecodeError:
        print(f"  [Error] Could not decode JSON from {file_path.name}")
        return ""
    except Exception as e:
        print(f"  [Error] Failed to parse {file_path.name}: {e}")
        if os.path.exists("token.json") and drive_service:
            # If the token exists and there is a drive service, it means the token is invalid
            os.remove("token.json")
            _log("Attempting to reauthenticate. Please retry sync.", log_callback)
            # Need to call get_drive_service HERE to reauthenticate
        return ""

def parse_docx(file_path: pathlib.Path) -> str:
    """Parses a .docx file using the python-docx library."""
    # print(f"-> Processing .docx: {file_path.name}")
    try:
        doc = Document(file_path)
        full_text = [para.text for para in doc.paragraphs]
        return '\n'.join(full_text)
    except Exception as e:
        print(f"  [Error] Failed to parse {file_path.name}: {e}")
        return ""

def parse_pdf(file_path: pathlib.Path) -> str:
    """Parses a .pdf file using the PyPDF2 library."""
    # print(f"-> Processing .pdf: {file_path.name}")
    try:
        text = ""
        with open(file_path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                text += (page.extract_text() or "") + " "
        return text
    except Exception as e:
        print(f"  [Error] Failed to parse {file_path.name}: {e}")
        return ""

def parse_txt(file_path: pathlib.Path) -> str:
    """Parses a plain .txt file."""
    # print(f"-> Processing .txt: {file_path.name}")
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            return f.read()
    except Exception as e:
        print(f"  [Error] Failed to parse {file_path.name}: {e}")
        return ""
    
def parse_image(file_path: pathlib.Path) -> str:
    """Returns a placeholder string for image files."""
    # print(f"-> Found image: {file_path.name}")
    return "[IMAGE]" # A special string to identify this as an image

# Text splitter
def create_text_splitter(embedding_model_name: str, chunk_size: int, chunk_overlap: int):
    """Splits the text into chunks by paragraph. According to research by ChromaDB, a chunk size of 200 and no overlap performs very well, despite being simple."""
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(embedding_model_name)

    def get_token_count(text: str) -> int:  # Used for FIXED chunk size
        # Simply returns the token count of a given text string.
        return len(tokenizer.encode(text, add_special_tokens=False))

    text_splitter = RecursiveCharacterTextSplitter(
        separators= ["\n\n", "\n", ".", "?", "!", " ", ""],  # These are the best separators, according to research.
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=get_token_count)
    
    return text_splitter

# Embedding logic
def get_text_embeddings(chunks: List[str], embedding_model, batch_size, log_callback) -> List[List[float]]:
    """Uses the text embedding model to embed a group of chunks in batches until all the chunks are accounted for."""
    try:
        embeddings = embedding_model.encode(
            chunks,
            convert_to_numpy=True,
            batch_size=batch_size,  # keeps memory in check
            normalize_embeddings=True
            )
        return embeddings.tolist()
    except Exception as e:
        _log(f"  [Error] get_text_embeddings failed: {e}", log_callback)
        return []

def get_image_embeddings(file_paths: List[pathlib.Path], embedding_model, batch_size, log_callback) -> List[List[float]]:
    """Uses the image embedding model to embed a batch of images."""
    try:
        images = []
        successful_file_paths = []
        for file_path in file_paths:
            try:
                Image.MAX_IMAGE_PIXELS = None  # Fixes decompression bomb error
                with Image.open(file_path).convert("RGB") as img:
                    img.thumbnail((4096, 4096))  # Still large, but cuts off the too massive ones.
                    images.append(img)
                    successful_file_paths.append(file_path)
            except Exception as e:
                _log(f"  [Error] Failed to load image {file_path.name}: {e}", log_callback)

        image_embeddings = embedding_model.encode(
            images,
            convert_to_numpy=True,
            batch_size=batch_size,
            normalize_embeddings=True
            )
        return image_embeddings.tolist(), successful_file_paths
    except Exception as e:
        _log(f"  [Error] get_image_embeddings failed: {e}", log_callback) # Corrected line
        return []

def store_text_embeddings(embeddings: List[List[float]], chunks: List[str], file_path: str, collections, log_callback):
    """Stores embeddings and their associated metadata in ChromaDB using a single batched call."""
    base_file_name = os.path.basename(file_path)
    # Get the last modified time (as a Unix timestamp)
    last_modified_time = os.path.getmtime(file_path)
    
    # Build the lists for the batched add operation
    ids = [f"{file_path}_chunk_{i}" for i in range(len(chunks))]
    
    metadatas = []
    for _ in range(len(chunks)):
        metadatas.append({"source_file": file_path, 
                    "last_modified": last_modified_time, 
                    "type": "text"})

    # Perform a single, batched add operation. This is much more efficient.
    collections['text'].add(
        embeddings=embeddings,
        documents=chunks,
        metadatas=metadatas,
        ids=ids)
    
    _log(f"➔ Added {len(chunks)} chunks from: {base_file_name}", log_callback)
    # Known bug was fixed by making the id unique using the source file - identical file names in different folders caused problems

def store_image_embeddings(embeddings, file_paths, collections, log_callback):
    """Prepare lists for the single, batched database add."""
    ids = []
    documents = []
    metadatas = []
    
    for i in range(len(embeddings)):
        file_path = file_paths[i]
        base_file_name = os.path.basename(file_path)
        last_modified_time = os.path.getmtime(file_path)
        
        # Create a unique ID, document string, and metadata for each successful embedding
        ids.append(f"{file_path}")
        documents.append(f"({base_file_name} in folder {file_path.parent.name})")
        metadatas.append({
            "source_file": str(file_path),
            "last_modified": last_modified_time,
            "type": "image"})

    # Perform the single, batched add operation
    collections['image'].add(
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas,
        ids=ids)
    
    _log(f"➔ Added {len(embeddings)} images in a single batch", log_callback)

# File handler with file processors
def file_handler(extension: str, is_multimodal: bool):
    """Returns the appropriate parser function based on the file extension."""
    handlers = {
        '.gdoc': parse_gdoc,
        '.docx': parse_docx,
        '.pdf': parse_pdf,
        '.txt': parse_txt,
        # '.xlsx': parse_xlsx,
        # '.csv': parse_csv,
        # '.gsheet': parse_gsheet,
    }
    # If model is multimodal, support images
    if is_multimodal:
        for ext in ['.png', '.jpg', '.jpeg', '.gif', '.webp']:
            handlers[ext] = parse_image

    return handlers.get(extension.lower())

def process_text_file(file_path: pathlib.Path, drive_service, text_splitter, models, is_multimodal, collections, batch_size, log_callback):
    """Central function: handles parsing, chunking, embedding, and routing the correct model and collection."""
    handler = file_handler(file_path.suffix, is_multimodal)
    if not handler:
        _log(f"  - Skipped: {file_path.name}", log_callback)
        return
    
    # If there is no drive service and it is a Google doc, skip it (save it for when there is a handler)
    if not drive_service and handler == parse_gdoc:
        return

    content = handler(file_path, drive_service, log_callback) if handler == parse_gdoc else handler(file_path)

    if not content:
        return

    # Preprocess content
    content = re.sub(r'\s+', ' ', content).strip() # Replace multiple spaces with a single one and strip extra spaces.
    chunks = text_splitter.split_text(content)
    # Remove leading periods (random bug...)
    for i, chunk in enumerate(chunks):
        chunks[i] = chunk.lstrip('. ')
    # For better recall, add this prefix
    prefix = f"<Source: {file_path.name}>"
    prefixed_chunks = [f"{prefix} {chunk}" for chunk in chunks]
    text_embeddings = get_text_embeddings(prefixed_chunks, models['text'], batch_size, log_callback)
    
    if not text_embeddings:
        return
    
    # Store in the text collection
    store_text_embeddings(text_embeddings, prefixed_chunks, str(file_path), collections, log_callback)
        
def process_image_batch(file_paths: List[pathlib.Path], models, collections, batch_size, log_callback, cancel_event):
    """Given a list of all file paths, breaks it up into batches, gets embeddings for the batch, and stores them. Batching improves speed and efficiency."""
    if not file_paths:
        return
    
    # To save memory, split the list of file paths into chunks
    all_image_batches = [file_paths[i:i + batch_size] for i in range(0, len(file_paths), batch_size)]

    for image_batch in all_image_batches:
        if cancel_event and cancel_event.is_set():
            _log("✖ Sync canceled by user.", log_callback)
            return
        # Get all embeddings in a single batched call
        image_embeddings, successful_file_paths = get_image_embeddings(image_batch, models['image'], batch_size, log_callback)

        if not image_embeddings:
            return
            
        # Store in the image collection
        store_image_embeddings(image_embeddings, successful_file_paths, collections, log_callback)

# First major function
def sync_directory(drive_service, text_splitter, models, collections, config, cancel_event=None, log_callback=None):
    """Scans a directory and syncs it with the ChromaDB collection by adding, updating, and deleting files as needed."""
    root_path = pathlib.Path(config['target_directory'])

    if not root_path.is_dir():
        print(f"Error: Path '{root_path}' is not a valid directory.")
        return

    _log(f"Starting sync for directory: {root_path} (This may take a while.)", log_callback)
    start_time = time.perf_counter()

    local_files = {}
    for p in root_path.rglob('*'):  # This processes files sort of at random, but it does get to every one.
        if p.is_file():
            local_files[str(p)] = p.stat().st_mtime

    _log(f"Total number of files in directory: {len(local_files)}", log_callback)

    db_files = {}
    # Iterate over each collection (e.g., 'text', 'image') in the dictionary
    for collection_name, collection_obj in collections.items():
        # print(f"Checking existing files in '{collection_name}' collection...")
        results = collection_obj.get(include=['metadatas'])  # Checking last modified time for updates
        if 'metadatas' in results and results['metadatas']:
            for mdata in results['metadatas']:
                path = mdata['source_file']
                # Only add if not already present. This correctly merges the file lists
                # from both text and image collections into one master list.
                if path not in db_files:
                    db_files[path] = mdata.get('last_modified', 0)

    _log(f"Total number of files in collection: {len(db_files)}", log_callback)

    local_set = set(local_files.keys())
    db_set = set(db_files.keys())
    
    files_to_add = list(local_set - db_set)  # Files to add: in local but not in DB
    files_to_delete = list(db_set - local_set)  # Files to delete: in DB but not in local
    files_to_update = []  # Files to check for updates: in both
    for path in local_set.intersection(db_set):
        # Use math.isclose for safer float comparison of timestamps
        if not math.isclose(local_files[path], db_files[path]):  # Compare last modified times
            files_to_update.append(path)

    # DELETE FILES
    if files_to_delete + files_to_update:  # Delete files to update before re-adding them
        _log(f"Deleting {len(files_to_delete)} files from database...", log_callback)
        for path_str in files_to_delete:
            if cancel_event and cancel_event.is_set():
                _log("✖ Sync canceled by user.", log_callback)
                return
            path_obj = pathlib.Path(path_str)
            for collection in collections.values():
                collection.delete(where={"source_file": path_str})
            _log(f"➔ Deleted: {path_obj.name}", log_callback)

    # PROCESS FILES - ADD AND UPDATE
    text_files_to_process = []
    image_files_to_process = []

    files_to_process = files_to_add + files_to_update

    is_multimodal = models['image']
    unsupported_counter = 0
    for path_str in files_to_process:
        if cancel_event and cancel_event.is_set():
            _log("✖ Sync canceled by user.", log_callback)
            return
        
        path_obj = pathlib.Path(path_str)

        handler = file_handler(path_obj.suffix, is_multimodal)
        if handler == parse_image:
            image_files_to_process.append(path_obj)  # To get images from PDFs, need to pass .pdf files to here *and* text_files_to_process; if handler == parse_pdf...
        elif handler:
            text_files_to_process.append(path_obj)
        else:
            unsupported_counter += 1
    _log(f"Total unsupported files: {unsupported_counter}", log_callback)

    if text_files_to_process:
        _log(f"Processing {len(text_files_to_process)} text files...", log_callback)
        for path_obj in text_files_to_process:
            if cancel_event and cancel_event.is_set():
                _log("✖ Sync canceled by user.", log_callback)
                return
            process_text_file(path_obj, drive_service, text_splitter, models, is_multimodal, collections, config['batch_size'], log_callback)
    
    if image_files_to_process:
        _log(f"Processing {len(image_files_to_process)} image files...", log_callback)
        process_image_batch(image_files_to_process, models, collections, config['batch_size'], log_callback, cancel_event)

    # Done.
    end_time = time.perf_counter()
    _log(f"Sync complete.", log_callback)
    _log(f"Syncing took {(end_time - start_time):.4f} seconds.", log_callback)
    _log(f"{collections['image'].count()} images in the collection", log_callback)
    _log(f"{collections['text'].count()} text chunks in the collection", log_callback)

def mmr_rerank(query_embedding: np.ndarray, result_embeddings: np.ndarray, results: List[Dict[str, Any]], mmr_lambda: float = 0.5, n_results: int = 5,) -> List[Dict[str, Any]]:
    """Maximal Marginal Relevance re-ranking (works for text & images)."""
    # Compute relevance (cosine similarity)
    relevance = 1 - cdist(query_embedding, result_embeddings, metric="cosine")[0]
    # Compute redundancy
    similarity_matrix = 1 - cdist(result_embeddings, result_embeddings, metric="cosine")

    selected = []
    mmr_scores_all = np.full(len(results), float('-inf'))
    # Select the first document (most relevant)
    first_doc_idx = np.argmax(relevance)
    selected.append(first_doc_idx)
    
    # Iteratively select the next best document
    while len(selected) < n_results:
        remaining = [i for i in range(len(results)) if i not in selected]
        if not remaining:
            break
        # Calculate MMR scores for all remaining documents at once
        mmr_scores = mmr_lambda * relevance[remaining] - \
                     (1 - mmr_lambda) * np.max(similarity_matrix[remaining][:, selected], axis=1)
        best_idx_in_remaining = np.argmax(mmr_scores)
        best_idx = remaining[best_idx_in_remaining]
        selected.append(best_idx)

        mmr_scores_all[best_idx] = mmr_scores[best_idx_in_remaining]

    # Attach MMR scores to results
    reranked_results = []
    for idx in selected:
        result = results[idx].copy()
        result["mmr_score"] = mmr_scores_all[idx]
        reranked_results.append(result)

    return reranked_results

def calculate_std_dev_threshold(scores: List[float], z_score: float = 1.5) -> Optional[float]:
    """Calculates a threshold based on the mean and standard deviation of mmr scores."""
    if not scores or len(scores) < 2:
        return None
        
    mean_dist = np.mean(scores)
    std_dev_dist = np.std(scores)
    
    threshold = mean_dist + z_score * std_dev_dist
    return threshold

def format_results(final_results: List[Dict]) -> List[Dict]:
    """Helper function to format search results for output. If given an empty list, returns a dictionary with None for all values."""
    return [{
            "rank": i + 1,
            "file_path": r['metadata'].get('source_file', 'N/A'),
            "documents": r.get('documents'), # Use .get() for safety
            "metadata": r['metadata'],
            "distance": r['distance'],
            "mmr_score": r['mmr_score'],
            "query": r['query']
        } for i, r in enumerate(final_results)]

def perform_search(query_embedding: np.ndarray, queries: List[Any], models, collections, config, search_type: str, max_results: int) -> Optional[Dict[str, str]]:
    """Private helper containing the core search, rerank, and filtering logic."""
    collection = collections[search_type]

    # Fetch a large amount of results for reranking
    fetch_k = max_results * config['search_multiplier']
    chroma_results = collection.query(
        query_embeddings=query_embedding.tolist(),
        n_results=fetch_k,  # Search fetch_k many times for each query
        where={"type": search_type},
        include=["documents", "metadatas", "embeddings", "distances"])
    if not chroma_results or not chroma_results.get('ids') or not any(chroma_results['ids']):
        return []

    # Combine all results for all queries
    all_results = [
        {
            "id": chroma_results['ids'][q][j],
            "distance": chroma_results['distances'][q][j],
            "documents": chroma_results['documents'][q][j],
            "metadata": chroma_results['metadatas'][q][j],
            "embedding": np.array(chroma_results['embeddings'][q][j]),
            "query": queries[q]
        }
        for q in range(len(queries)) for j in range(len(chroma_results['ids'][q]))]

    # De-duplicate
    unique_results_dict = {}
    for res in sorted(all_results, key=lambda x: x['distance']):
        if res['id'] not in unique_results_dict:
            unique_results_dict[res['id']] = res    
    unique_results = list(unique_results_dict.values())
    if not unique_results:
        return []

    # Re-rank the unique result set with MMR
    result_embeddings = np.array([r['embedding'] for r in unique_results])
    mmr_reranked_results = mmr_rerank(query_embedding, result_embeddings, unique_results, config['mmr_lambda'], fetch_k)

    # Find a threshold using statistics
    mmr_scores = [r["mmr_score"] for r in mmr_reranked_results if np.isfinite(r["mmr_score"])]
    score_threshold = calculate_std_dev_threshold(mmr_scores, config['z_score'])
    # print(f"Score Threshold: {score_threshold}")
    if score_threshold is None:
        return []
    
    # Remove items based on the threshold
    final_results = []
    for res in mmr_reranked_results:
        if res['mmr_score'] >= score_threshold:
            final_results.append(res)

    # Finally, cap the results at max_results
    final_results = final_results[:max_results]
    if not final_results:
        return []

    return format_results(final_results)

# Second major function of this code is to make searches
def semantic_search(queries: List[str], models, collections, config, search_type: str) -> Optional[Dict[str, str]]:
    """Performs a semantic search with three steps: 1) MMR rerank, 2) filter using std_dev, 3) cap at maximum result number. Searches with multiple queries will be searched individually, then combined into one output. To do individual searches, only input one string for the query list."""
    if not queries or search_type not in models or search_type not in collections:
        return []

    model = models[search_type]
    query_embedding = model.encode(queries, convert_to_numpy=True, normalize_embeddings=True)

    # Return the final results
    return perform_search(query_embedding, queries, models, collections, config, search_type, config['max_results'])

def find_similar_images(image_paths: List[str], models, collections, config) -> Optional[Dict[str, str]]:
    """Performs semantic search for IMAGE queries to find similar IMAGES. Input must be a list of system paths for IMAGES."""
    if not image_paths or "image" not in models or "image" not in collections:
        return []
    
    model = models["image"]
    images = []
    valid_paths = []
    for file_path in image_paths:
        try:
            with Image.open(file_path).convert("RGB") as img:
                images.append(img)
                valid_paths.append(file_path) # Keep track of which paths were successful
        except Exception as e:
            print(f" [Warning] Failed to load image {file_path}: {e}")
    
    if not images:
        return []

    query_embedding = model.encode(images, convert_to_numpy=True, batch_size=config['batch_size'], normalize_embeddings=True)
    
    return perform_search(query_embedding, image_paths, models, collections, config, "image", config['max_results'])

def load_config(file_path):
    """Loads configuration from a JSON file."""
    with open(file_path, 'r') as config_file:
        config = json.load(config_file)
        return config

# To make setup easier
def machine_setup(config, log_callback):
    """Initializes all the necessary 'machines' needed for sync_directory and semantic_search."""
    # Define model names. Set to None if you don't want to use one.
    text_model_name = config['text_model_name']  # Options: BAAI/bge-small-en-v1.5, BAAI/bge-large-en-v1.5 (in order of increasing power)
    image_model_name = config['image_model_name']  # Options: clip-ViT-B-32, clip-ViT-B-16, clip-ViT-L-14 (in order of increasing power)
    # Find device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _log(f"Using device: {device}", log_callback)
    # Initialize ChromaDB client
    chroma_client = chromadb.PersistentClient(path="./chroma_db")
    # Use these dictionaries to hold our models and collections
    models = {}
    collections = {}
    # --- Load Text Model & Collection ---
    if text_model_name:
        _log(f"Loading text embedder: {config['text_model_name']}", log_callback)
        models['text'] = SentenceTransformer(text_model_name, device=device)
        collections['text'] = chroma_client.get_or_create_collection(name="text_collection", metadata={"hnsw:space": "cosine"})  # cosine is essential
    # --- Load Image Model & Collection ---
    if image_model_name:
        _log(f"Loading image embedder: {config['image_model_name']}", log_callback)
        models['image'] = SentenceTransformer(image_model_name, device=device)
        collections['image'] = chroma_client.get_or_create_collection(name="image_collection", metadata={"hnsw:space": "cosine"})  # cosine is essential
        
    text_splitter = create_text_splitter(text_model_name, config['chunk_size'], config['chunk_overlap'])  # Splits by paragraph; can improve
      # Only try if connected to internet
    drive_service = get_drive_service(log_callback, config)
    
    return drive_service, text_splitter, models, collections

# Example script usage; can import these functions in another file to do the same
# if __name__ == "__main__":
    # --- DO SETUP ---
    # config = load_config("./config.json")
    # drive_service, text_splitter, models, collections = machine_setup(config)

    # --- RUN THE SYNC ---
    # sync_directory(drive_service, text_splitter, models, collections, config)

    # --- TRY A SEARCH ---
    # Pro tip: prefix text searches with "Represent this sentence for searching relevant passages: " if not expanding the query with an LLM.
    # search_results = semantic_search(["Movie", "Find a movie", "Find movies"], models, collections, config, "text", 5)

    # search_results = find_similar_images(["C:\\Users\\henry\\My Drive\\_Photos and Media\\Photos\\Pictures of Me\\4thyearportraits002CROPADJUSTED.jpg"], models, collections, config, 5)

    # if search_results:
    #     for result in search_results:
    #         print(f"\nRank: {result['rank']}")
    #         print(f"File Path: {result['file_path']}")
    #         print(f"Distance: {result['distance']}")
    #         print(f"MMR Score: {result['mmr_score']}")
    #         print(f"Query: {result['query']}")
    # else:
    #     print("No Results")

    # To-do: pull images from PDFs