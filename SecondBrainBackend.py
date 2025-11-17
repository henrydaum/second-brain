# Imports
import os
import io
import pathlib
import json
import re
from typing import List, Dict, Any, Optional
import math
import time
import pickle
import string
import unicodedata
import zlib
import csv
from pathlib import Path

# 3rd party imports
import requests
from pdfminer.high_level import extract_text
import chromadb
import torch
import numpy as np
from docx import Document
from PIL import Image
from transformers import AutoTokenizer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cdist
from rank_bm25 import BM25Okapi

__version__ = "1.0.0.0"
# Local AppData for mutable files, and Program Files for immutable files
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# DATA_DIR = Path("./")
DATA_DIR = Path(os.getenv('LOCALAPPDATA')) / "Second Brain"
DATA_DIR.mkdir(parents=True, exist_ok=True)

stop_words = set(["i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", "while", "of", "at", "by", "for", "with", "about", "against", "between", "into", "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"])

def get_labels_from_csv(filepath: str, column_name: str) -> list:
    """Replaces the pandas version to avoid a massive import."""
    labels = set()
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                label = row.get(column_name, "").strip()
                if label: # Ensure label is not empty
                    labels.add(label)
    except FileNotFoundError:
        # To handle this in case the file doesn't exist yet
        print(f"Warning: '{filepath}' not found. Image labels will be empty.")
        return []
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return []
    return list(labels)

# This becomes the labels for image classification
LABELS_PATH = os.path.join(BASE_DIR, "image_labels.csv")
CANDIDATE_LABELS = get_labels_from_csv(LABELS_PATH, 'label')

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
    if not config.get("use_drive", False):  # If option is missing (and it will be, if is_final_microsoft_store_product == True), set to False.
        # _log("Drive disabled in config.", log_callback)  # Comment out to keep the user in the dark about Drive settings.
        return None
    
    # Test if connected to internet
    if not is_connected():
        _log("No internet — skipping Google Drive.", log_callback)
        return None  # If no internet, no drive service.
    
    # Google APIs
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow
    from googleapiclient.discovery import build
    
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
    token_path = DATA_DIR / "token.json"
    if os.path.exists(token_path):
        creds = Credentials.from_authorized_user_file(str(token_path), SCOPES)
    
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
            with open(token_path, "w") as token:
                token.write(creds.to_json())
            service = build("drive", "v3", credentials=creds)
            _log("Authentication successful.", log_callback)
            return service
    except Exception as e:
        _log(f"Authentication failed — skipping Google Drive.", log_callback)
        return None

def download_drive_content(drive_service, doc_id: str, mimeType: str) -> str:
    """Downloads a Google Doc's content as plain text using its file ID."""
    try:
        from googleapiclient.http import MediaIoBaseDownload
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

    except Exception as e:
        print(f"[ERROR] Could not download Google Drive file: {e}")
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
            print(f"[ERROR] Could not find Drive URL in {file_path.name}")
            return ""
        
        # API Call
        # print(f"  Found URL: {doc_url}")
        content = download_drive_content(drive_service, doc_url, "text/plain")
        return content
    except json.JSONDecodeError:
        print(f"[ERROR] Could not decode JSON from {file_path.name}")
        return ""
    except Exception as e:
        _log(f"[ERROR] Failed to parse {file_path.name}: {e}. Try reauthenticating Drive.", log_callback)
        return ""

def parse_docx(file_path: pathlib.Path) -> str:
    """Parses a .docx file using the python-docx library."""
    # print(f"-> Processing .docx: {file_path.name}")
    try:
        doc = Document(file_path)
        full_text = [para.text for para in doc.paragraphs]
        return '\n'.join(full_text)
    except Exception as e:
        print(f"[ERROR] Failed to parse {file_path.name}: {e}")
        return ""

def parse_pdf(file_path: pathlib.Path) -> str:
    """Parses a .pdf file using the pdfminer library (PyPDF2 is deprecated and Fitz doesn't allow a commercial license). Whatever the library, it needs to not read some PDFs like 't h i s i s a n e x a m p l e', which was very annoying. """
    # print(f"-> Processing .pdf: {file_path.name}")
    try:
        return extract_text(file_path)
    except Exception as e:
        print(f"[ERROR] Failed to parse {file_path.name}: {e}")
        return ""

def parse_txt(file_path: pathlib.Path) -> str:
    """Parses a plain .txt file."""
    # print(f"-> Processing .txt: {file_path.name}")
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            return f.read()
    except Exception as e:
        print(f"[ERROR] Failed to parse {file_path.name}: {e}")
        return ""
    
def parse_image(file_path: pathlib.Path) -> str:
    """Returns a placeholder string for image files."""
    # print(f"-> Found image: {file_path.name}")
    return "[IMAGE]" # A special string to identify this as an image

# Text splitter
def create_text_splitter(embedding_model_name: str, chunk_size: int, chunk_overlap: int, offline_mode: bool):
    """Splits the text into chunks by paragraph. According to research by ChromaDB, a chunk size of 200 and no overlap performs very well, despite being simple."""
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(embedding_model_name, 
        local_files_only=offline_mode)

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
        _log(f"[ERROR] get_text_embeddings failed: {e}", log_callback)
        return []

def get_image_embeddings(file_paths: List[pathlib.Path], embedding_model, batch_size, log_callback) -> List[List[float]]:
    """Uses the image embedding model to embed a batch of images."""
    try:
        images = []
        successful_file_paths = []
        for file_path in file_paths:
            try:
                Image.MAX_IMAGE_PIXELS = None  # Fixes decompression bomb error
                with Image.open(file_path).convert("RGBA").convert("RGB") as img:  # Convert to RBGA first to stop a PIL warning. Not a big deal overall.
                    img.thumbnail((4096, 4096))  # Still large, but cuts off the too massive ones.
                    images.append(img)
                    successful_file_paths.append(file_path)
            except Exception as e:
                _log(f"[ERROR] Failed to load image {file_path.name}: {e}", log_callback)

        image_embeddings = embedding_model.encode(
            images,
            convert_to_numpy=True,
            batch_size=batch_size,
            normalize_embeddings=True
            )
        return image_embeddings.tolist(), successful_file_paths
    except Exception as e:
        _log(f"[ERROR] get_image_embeddings failed: {e}", log_callback)
        return [], []

def store_text_embeddings(embeddings: List[List[float]], chunks: List[str], file_path: str, collections, percentage_completed, log_callback):
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
    
    _log(f"[{percentage_completed:.0f}%] ✓ Added {len(chunks)} chunks from: {base_file_name}", log_callback)
    # Known bug was fixed by making the id unique using the source file - identical file names in different folders caused problems

def store_image_embeddings(embeddings, file_paths, collections, fraction_completed, log_callback):
    """Prepare lists for the single, batched database add."""
    ids = []
    documents = []
    metadatas = []
    
    for i, embedding in enumerate(embeddings):
        file_path = file_paths[i]
        base_file_name = os.path.basename(file_path)
        last_modified_time = os.path.getmtime(file_path)
        # Find a label
        label_results = collections['image'].query(
            query_embeddings=[embedding],
            n_results=3,
            where={"type": "label"},
            include=["documents"]
        )
        labels = [""]
        # Make sure result exists
        if label_results and label_results.get("documents") and label_results["documents"][0]:
            labels = [label.lower() for label in label_results["documents"][0]]
            # print(f"Labels found for this image: {labels}")
        # Insert the label into the document text, along with the file name and folder, so that it can be searched for with BM25
        doc_text = f"<file {base_file_name} in folder {file_path.parent.name}> {', '.join(labels)}".rstrip()
        # print(doc_text)
        # Create a unique ID, document string, and metadata for each successful embedding
        ids.append(f"{file_path}")
        documents.append(doc_text)
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
    
    percentage_completed = (fraction_completed[0] / fraction_completed[1]) * 100
    
    _log(f"[{percentage_completed:.0f}%] ✓ Added {len(embeddings)} images in a single batch ({fraction_completed[0]}/{fraction_completed[1]})", log_callback)

def create_image_classifiers(labels, model, collection, batch_size, log_callback):
    embeddings = model.encode(labels, convert_to_numpy=True, batch_size=batch_size, normalize_embeddings=True)
    # Find labels; these have to be unique
    ids = [f"label_{label.replace(' ', '_').lower()}" for label in labels]
    metadatas = [{"source_file": None, 
                  "last_modified": None, 
                  "type": "label"} for label in labels]  # Different type for labels is important
    # Add to collection
    collection.upsert(embeddings=embeddings, 
                   documents=labels, 
                   metadatas=metadatas, 
                   ids=ids)
    _log(f"✓ Created {len(labels)} labels in image collection", log_callback)

# File handler with file processors
def file_handler(extension: str, is_multimodal: bool, use_drive: bool):
    """Returns the appropriate parser function based on the file extension."""
    handlers = {
        '.docx': parse_docx,
        '.pdf': parse_pdf,
        '.txt': parse_txt,
        # '.xlsx': parse_xlsx,
        # '.csv': parse_csv,
        # '.gsheet': parse_gsheet,
        # '.md': parse_markdown,
    }
    if use_drive:
        handlers['.gdoc'] = parse_gdoc
    # If model is multimodal, support images
    if is_multimodal:
        for ext in ['.png', '.jpg', '.jpeg', '.gif', '.webp']:
            handlers[ext] = parse_image

    return handlers.get(extension.lower())

def is_gibberish(text, min_len=15, non_standard_threshold=0.15, low_compression_threshold=0.1, high_compression_threshold=0.95):
    """
    Returns true if the text is gibberish based on multiple heuristics:
    1. Too short.
    2. Too many non-printable (binary/control) characters.
    3. Too repetitive (low compression ratio).
    4. Too random (high compression ratio, only for longer texts).
    """
    # 1. Too short
    if not text or len(text) < min_len:
        # print(f"Gibberish (too short): {text}...")
        return True
    # 2. Non-standard character check (your original logic)
    # This is still excellent for catching binary/control characters.
    try:
        normalized_text = unicodedata.normalize("NFKC", text)
    except Exception:
        normalized_text = text  # Fallback if normalization fails for some reason
    allowed = set(string.printable)
    total = len(normalized_text)
    if total == 0:
        return True
    non_standard = sum(ch not in allowed for ch in normalized_text)
    if (non_standard / total) > non_standard_threshold:
        # print(f"Gibberish (non-printable): {text}...")
        return True
    # 3. Compression check (for repetition or randomness)
    # We use 'ignore' to drop any weird bytes that slipped through
    try:
        text_bytes = normalized_text.encode('utf-8', 'ignore')
        if not text_bytes:
            return True
        compressed_len = len(zlib.compress(text_bytes, level=9))  # Use max compression
        compression_ratio = compressed_len / len(text_bytes)
        # Too repetitive (e.g., "aaaaa" or "abababab")
        if compression_ratio < low_compression_threshold:
            # print(f"Gibberish (repetitive): {text}...")
            return True
        # Too random (e.g., "fjdksla jfkdls;a jfkdsla")
        # This check is less reliable on short strings, so we add a length gate
        if len(text_bytes) > 100 and compression_ratio > high_compression_threshold:
            # print(f"Gibberish (random): {text}...")
            return True
    except Exception:
        pass  # If compression fails, just trust the other checks
    return False

def process_text_file(file_path: pathlib.Path, drive_service, text_splitter, models, is_multimodal, collections, batch_size, percentage_completed, use_drive, log_callback):
    """Central function: handles parsing, chunking, embedding, and routing the correct model and collection."""
    handler = file_handler(file_path.suffix, is_multimodal, use_drive)
    if not handler:
        _log(f"[{percentage_completed:.0f}%] ⦸ Unsupported file type: {file_path.name}", log_callback)
        return
    
    # If there is no drive service and it is a Google doc, skip it (save it for when there is a handler)
    if not drive_service and handler == parse_gdoc:
        _log(f"[{percentage_completed:.0f}%] ⦸ No Drive service: {file_path.name}", log_callback)
        return

    content = handler(file_path, drive_service, log_callback) if handler == parse_gdoc else handler(file_path)

    if not content:
        _log(f"[{percentage_completed:.0f}%] ⦸ Empty of text: {file_path.name}", log_callback)
        return

    # Preprocess content
    content = re.sub(r'\s+', ' ', content).strip() # Replace multiple spaces with a single one and strip extra spaces.
    if "saved_insights" in str(file_path.parent):
        chunks = [content] # Treat the entire file content as a single chunk
        _log(f"➔ Processing insight file: {file_path.name}", log_callback)
    else:
        chunks = text_splitter.split_text(content)
    # Remove leading periods (random bug...)
    for i, chunk in enumerate(chunks):
        chunks[i] = chunk.lstrip('. ')
    # Remove gibberish chunks
    chunks = [chunk for chunk in chunks if not is_gibberish(chunk)]
    if not chunks:
        _log(f"[{percentage_completed:.0f}%] ⦸ File is gibberish: {file_path.name}", log_callback)
        return
    # For better recall, add this prefix
    prefix = f"<Source: {file_path.name}>"
    prefixed_chunks = [f"{prefix} {chunk}" for chunk in chunks]
    text_embeddings = get_text_embeddings(prefixed_chunks, models['text'], batch_size, log_callback)
    
    if not text_embeddings:
        _log(f"[{percentage_completed:.0f}%] ⦸ Missing embeddings: {file_path.name}", log_callback)
        return
    
    # Store in the text collection
    store_text_embeddings(text_embeddings, prefixed_chunks, str(file_path), collections, percentage_completed, log_callback)
        
def process_image_batch(file_paths: List[pathlib.Path], models, collections, batch_size, log_callback, cancel_event):
    """Given a list of all file paths, breaks it up into batches, gets embeddings for the batch, and stores them. Batching improves speed and efficiency."""
    if not file_paths:
        return
    
    # To save memory, split the list of file paths into chunks
    all_image_batches = [file_paths[i:i + batch_size] for i in range(0, len(file_paths), batch_size)]

    for i, image_batch in enumerate(all_image_batches):
        if cancel_event and cancel_event.is_set():
            _log("✖ Sync canceled by user.", log_callback)
            return
        # Get all embeddings in a single batched call
        image_embeddings, successful_file_paths = get_image_embeddings(image_batch, models['image'], batch_size, log_callback)

        if not image_embeddings:
            return
        
        fraction_completed = ((i + 1), len(all_image_batches))

        # Store in the image collection
        store_image_embeddings(image_embeddings, successful_file_paths, collections, fraction_completed, log_callback)

def normalize_text(s: str) -> str:
    # Lowercase, remove punctuation/underscores, collapse spaces
    s = re.sub(r'[\W_]+', ' ', s.lower()).strip()
    # Split by spaces
    tokens = s.split()
    # Remove stop words (removes junk)
    filtered_tokens = [t for t in tokens if t not in stop_words]
    return filtered_tokens

# BM25 Indexing for lexical search
def create_keyword_index(collections, log_callback=None):
    """
    Fetches all text documents from ChromaDB and builds a BM25 index.
    This is a one-time, memory-intensive operation run after a sync.
    """
    _log("Building new keyword search indexes...", log_callback)
    if collections['text'] is not None:
        try:
            text_collection = collections['text']
            # 1. Fetch all documents from the text collection in ChromaDB
            # This is the memory-intensive part, done only during indexing.
            text_results = text_collection.get(include=["documents"])
            
            if not text_results or not text_results['ids']:
                _log("No documents in the collection to index. Skipping.", log_callback)
                return
            chunk_ids = text_results['ids']
            text_documents = text_results['documents']
            # Lowercase and use regex to tokenize
            tokenized_corpus = [normalize_text(doc) for doc in text_documents]
            # Create the BM25 index object
            bm25 = BM25Okapi(tokenized_corpus)
            # Save the BM25 index and the list of chunk IDs to disk
            # The index object is pre-computed. The chunk_ids list is needed to map BM25's results (which are list indices) back to ChromaDB IDs.
            with open(DATA_DIR / "bm25_text_index.pkl", "wb") as f:
                pickle.dump(bm25, f)
            with open(DATA_DIR / "bm25_chunk_ids.pkl", "wb") as f:
                pickle.dump(chunk_ids, f)
            _log(f"✓ Text keyword index built successfully for {len(chunk_ids)} text chunks.", log_callback)

        except Exception as e:
            _log(f"[ERROR] Failed to build text BM25 index: {e}", log_callback)
    else:
        _log("No text model - skipping text keyword index", log_callback)

    # Now do the same for images (the documents contain the image filename and folder, which can be searched for)
    # All in one code block:
    if collections['image'] is not None:
        try:
            image_collection = collections['image']
            image_results = image_collection.get(include=["documents"], where={"type": "image"})  # Don't build it using info from labels
            if not image_results or not image_results['ids']:
                _log("No images in the collection to index. Skipping.", log_callback)
                return
            image_ids = image_results['ids']
            image_documents = image_results['documents']
            tokenized_corpus = [normalize_text(doc) for doc in image_documents]
            bm25 = BM25Okapi(tokenized_corpus)
            with open(DATA_DIR / "bm25_image_index.pkl", "wb") as f:
                pickle.dump(bm25, f)
            with open(DATA_DIR / "bm25_image_ids.pkl", "wb") as f:
                pickle.dump(image_ids, f)
            _log(f"✓ Image label index built successfully for {len(image_ids)} images.", log_callback)
        except Exception as e:
            _log(f"[ERROR] Failed to build image BM25 index: {e}", log_callback)
    else:
        _log("No image model - skipping image keyword index", log_callback)

# First major function
def sync_directory(drive_service, text_splitter, models, collections, config, cancel_event=None, log_callback=None):
    """Scans a directory and syncs it with the ChromaDB collection by adding, updating, and deleting files as needed."""
    root_path = pathlib.Path(config['target_directory'])
    if not os.path.exists(root_path):
        _log(f"[ERROR] '{root_path}' does not exist. It may be misspelled or unset.", log_callback)
        return
    
    base_dir = BASE_DIR
    insights_dir = DATA_DIR / "saved_insights"

    if not root_path.is_dir():
        print(f"[ERROR] '{root_path}' needs to be a folder.")
        return

    _log(f"Starting sync for directory: {root_path}", log_callback)
    _log("(This may take a while. Don't move any files while syncing.)", log_callback)
    start_time = time.perf_counter()
    time.sleep(0.1)

    local_files = {}
    for p in root_path.rglob('*'):  # This processes files sort of at random, but it does get to every one.
        if p.is_file():
            local_files[str(p)] = p.stat().st_mtime

    if insights_dir.exists():
        for p in insights_dir.rglob('*'):
            if p.is_file():
                local_files[str(p)] = p.stat().st_mtime
        _log(f"Included saved insights from: {insights_dir}", log_callback)
    else:
        _log(f"No saved insights folder found at: {insights_dir}", log_callback)

    _log(f"Total number of files in directory + insights folder: {len(local_files)}", log_callback)

    db_files = {}
    # Iterate over each collection (e.g., 'text', 'image') in the dictionary
    for collection_name, collection_obj in collections.items():
        # print(f"Checking existing files in '{collection_name}' collection...")
        if collection_obj is None:
            _log(f"[ERROR] The '{collection_name}' collection is null. Try loading a model.", log_callback)
            continue
        results = collection_obj.get(include=['metadatas'])  # Checking last modified time for updates
        if 'metadatas' in results and results['metadatas']:
            for mdata in results['metadatas']:
                path = mdata.get('source_file')
                if path is None:
                    continue # Skips this iteration for labels (and anything else with source_file=None)
                # Only add if not already present. This correctly merges the file lists
                # from both text and image collections into one master list.
                if path not in db_files:
                    db_files[path] = mdata.get('last_modified', 0)

    _log(f"Total number of synced files in collection: {len(db_files)}", log_callback)

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
                break
            path_obj = pathlib.Path(path_str)
            for collection in collections.values():
                if collection is not None:
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
            break
        
        path_obj = pathlib.Path(path_str)

        use_drive = config.get("use_drive", False)
        handler = file_handler(path_obj.suffix, is_multimodal, use_drive)
        if handler == parse_image:
            if is_multimodal:
                image_files_to_process.append(path_obj)  # To get images from PDFs, need to pass .pdf files to here *and* text_files_to_process; if handler == parse_pdf...
            else:
                # Skip images if no image model available
                pass
        elif handler:
            text_files_to_process.append(path_obj)
        else:
            unsupported_counter += 1
    _log(f"Total unsupported files: {unsupported_counter}", log_callback)

    if models['text'] is not None:
        if text_files_to_process:
            if models['text']:
                _log(f"Processing {len(text_files_to_process)} text files...", log_callback)
                for i, path_obj in enumerate(text_files_to_process):
                    if cancel_event and cancel_event.is_set():
                        _log("✖ Sync canceled by user.", log_callback)
                        break
                    # To display progress:
                    percentage_completed = ((i + 1) / len(text_files_to_process)) * 100
                    process_text_file(path_obj, drive_service, text_splitter, models, is_multimodal, collections, config['batch_size'], percentage_completed, use_drive, log_callback)
            else:
                _log("No text embedder - cannot sync text.", log_callback)
        else:
            _log("No new or updated text files to process.", log_callback)
    else:
        _log("No text model - skipping text.", log_callback)

    if models['image'] is not None:
        if not collections['image'].get(where={"type":"label"})['ids']:
            if cancel_event and cancel_event.is_set():
                # User probably doesn't want to wait through this if already canceled
                ...
            else:
                if models['image']:
                    _log(f"Creating image classifiers...", log_callback)
                    classifier_labels = [label.lower() for label in CANDIDATE_LABELS]
                    create_image_classifiers(classifier_labels, models['image'], collections['image'], config['batch_size'], log_callback)
                else:
                    _log("No image embedder - cannot create image classifiers.", log_callback)
        else:
            _log(f"Image classifiers already exist, skipping creation.", log_callback)
    else:
        _log("No image model - skipping image classifiers.", log_callback)
    
    if models['image'] is not None:
        if image_files_to_process:
            if models['image']:
                _log(f"Processing {len(image_files_to_process)} image files...", log_callback)
                process_image_batch(image_files_to_process, models, collections, config['batch_size'], log_callback, cancel_event)
            else:
                _log("No image embedder - cannot sync images.", log_callback)
        else:
            _log("No new or updated images to process.", log_callback)
    else:
        _log("No image model - skipping images.", log_callback)

    # BM25 INDEXING FOR LEXICAL SEARCH
    create_keyword_index(collections, log_callback)

    # Done.
    end_time = time.perf_counter()
    _log(f"Syncing took {(end_time - start_time):.4f} seconds.", log_callback)
    if collections['text']:
        _log(f"{collections['text'].count()} text chunks in the collection", log_callback)
    if collections['image']:
        num_images = len(collections['image'].get(where={"type": "image"}, include=[])["ids"])
        _log(f"{num_images} images in the collection", log_callback)
    _log(f"Sync complete.", log_callback)

def mmr_rerank_hybrid(
        results: List[Dict[str, Any]], 
        mmr_lambda: float = 0.5,
        alpha: float = 0.5, # Weight for semantic vs. lexical similarity
        n_results: int = 5
    ) -> List[Dict[str, Any]]:
    """
    Maximal Marginal Relevance re-ranking using a hybrid (semantic + lexical) diversity metric.
    - alpha = 1.0 means purely semantic diversity.
    - alpha = 0.0 means purely lexical diversity.
    """
    if not results:
        return []

    result_embeddings = np.array([r["embedding"] for r in results])
    relevance_scores = np.array([r["score"] for r in results]) # Use the pre-computed hybrid scores
    result_documents = [r["documents"] for r in results]

    # 1. Calculate semantic similarity
    semantic_sim_matrix = 1 - cdist(result_embeddings, result_embeddings, metric="cosine")

    # 2. Calculate lexical similarity using the Jaccard algorithm
    n = len(results)
    token_sets = [set(normalize_text(r["documents"])) for r in results]

    lexical_sim_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):  # Nested for loop
            # Use the pre-computed sets for the calculation.
            tokens1 = token_sets[i]
            tokens2 = token_sets[j]

            if not tokens1 and not tokens2:
                sim = 1.0
            else:
                intersection = len(tokens1.intersection(tokens2))
                union = len(tokens1.union(tokens2))
                sim = intersection / union if union != 0 else 0.0
            
            lexical_sim_matrix[i, j] = sim
            lexical_sim_matrix[j, i] = sim

    # 3. Combine them into a hybrid similarity matrix
    hybrid_sim_matrix = alpha * semantic_sim_matrix + (1 - alpha) * lexical_sim_matrix
    np.fill_diagonal(hybrid_sim_matrix, -1) # Ensure a doc isn't compared to itself

    # --- Standard MMR logic using the new hybrid similarity matrix ---
    selected_indices = []
    remaining_indices = list(range(n))

    first_doc_idx = np.argmax(relevance_scores)
    selected_indices.append(first_doc_idx)
    remaining_indices.remove(first_doc_idx)

    while len(selected_indices) < n_results and remaining_indices:
        mmr_scores = mmr_lambda * relevance_scores[remaining_indices] - \
                     (1 - mmr_lambda) * np.max(hybrid_sim_matrix[remaining_indices][:, selected_indices], axis=1)
        
        best_idx_in_remaining = np.argmax(mmr_scores)
        best_idx = remaining_indices[best_idx_in_remaining]
        
        selected_indices.append(best_idx)
        remaining_indices.remove(best_idx)

    return [results[i] for i in selected_indices]

def format_results(final_results: List[Dict]) -> List[Dict]:
    """Helper function to format search results for output. If given an empty list, returns a dictionary with None for all values."""
    return [{
            "rank": i + 1,
            "file_path": r['metadata'].get('source_file', 'N/A'),
            "documents": r.get('documents'), # Use .get() for safety
            "metadata": r['metadata'],
            "query": r['query'],
            "score": r['score'],
            "result_type": r['result_type'],
        } 
        for i, r in enumerate(final_results)]

def perform_search(query_embedding: np.ndarray, queries: List[Any], searchfacts, models, collections, config, search_type: str, max_results: int) -> Optional[Dict[str, str]]:
    """Private helper containing the core search and rerank logic, which is both semantic and lexical."""
    collection = collections[search_type]

    if collection.count() == 0:
        print("No data in the collection to search.")
        return []  # No data to search

    fetch_k = max_results * config['search_multiplier']

    # --- SEMANTIC SEARCH WITH MMR ---
    # Fetch initial results from ChromaDB
    chroma_results = collection.query(
        query_embeddings=query_embedding.tolist(),
        n_results=fetch_k,
        where={"type": search_type},
        include=["documents", "metadatas", "embeddings", "distances"])

    # Combine all results for all queries in one list
    semantic_results = [
        {
            "id": chroma_results['ids'][q][j],
            "documents": chroma_results['documents'][q][j],
            "metadata": chroma_results['metadatas'][q][j],
            "embedding": np.array(chroma_results['embeddings'][q][j]),
            "query": queries[q],
            "score": float(1 - chroma_results['distances'][q][j]),
            "result_type": "semantic"
        }
        for q in range(len(queries)) for j in range(len(chroma_results['ids'][q]))]

    # --- LEXICAL (KEYWORD) SEARCH ---
    # Get the right keyword index database from BM25
    if search_type == "text":
        bm25_path = DATA_DIR / "bm25_text_index.pkl"
        ids_path = DATA_DIR / "bm25_chunk_ids.pkl"
    elif search_type == "image":
        bm25_path = DATA_DIR / "bm25_image_index.pkl"
        ids_path = DATA_DIR / "bm25_image_ids.pkl"

    # Check for existence
    if not os.path.exists(bm25_path) or not os.path.exists(ids_path):
        print("BM25 files not found.")
        return []

    # Search the database using BM25
    lexical_id_list = []
    try:
        with open(bm25_path, "rb") as f:
            bm25 = pickle.load(f)
        with open(ids_path, "rb") as f:
            chunk_ids = pickle.load(f)
        # Normalize the search term
        tokenized_search_term = normalize_text(searchfacts.lexical_search_term)
        # Get scores for all documents in the pre-computed index
        bm25_scores = bm25.get_scores(tokenized_search_term)
        # Create a sorted list of (doc_id, score) tuples
        scored_docs = sorted(zip(chunk_ids, bm25_scores), key=lambda x: x[1], reverse=True)
        # Get dictionary of scores for easy lookup later
        score_dict = {doc_id: score for doc_id, score in scored_docs}
        # Create the ranked list of IDs. Can filter out results based on score here if desired with "if doc[1] > min_score"
        lexical_id_list = [doc[0] for doc in scored_docs][:fetch_k]
    except Exception as e:
        # _log("BM25 index not found, proceeding with semantic search only.", None)
        lexical_id_list = [] # Ensure the list is empty if index doesn't exist
        print(f"[Warning] BM25 keyword search failed: {e}")

    # Now that the IDs were found, need to do a reverse search on the ChromaDB to find the respective documents and metadata, etc.
    lexical_results = []
    if lexical_id_list:
        # Grab the actual entries from the database along with documents, metadata, and embeddings
        keyword_data = collection.get(
            ids=lexical_id_list, 
            where={"type": search_type}, 
            include=["documents", "metadatas", "embeddings"])
        
        # Build lexical results directly, matching the format of the semantic results
        for i, id_ in enumerate(keyword_data["ids"]):
            lexical_results.append({
                "id": id_,
                "documents": keyword_data["documents"][i],
                "metadata": keyword_data["metadatas"][i],
                "embedding": np.array(keyword_data["embeddings"][i]),
                "query": searchfacts.lexical_search_term,
                "score": float(score_dict.get(id_, 0.0)),
                "result_type": "lexical"
            })
    # Sort the list by score descending (optional; semantic_scores are already in order)

    # Now we have semantic_results and lexical_results

    # Normalize scores per list (min–max) between 0 and 1
    sem_scores = np.array([r["score"] for r in semantic_results])
    lex_scores = np.array([r["score"] for r in lexical_results])

    try:
        sem_norm = (sem_scores - sem_scores.min()) / (sem_scores.max() - sem_scores.min() + 1e-9)
    except:
        # Dividing by zero:
        sem_norm = np.zeros_like(sem_scores)
    try:
        lex_norm = (lex_scores - lex_scores.min()) / (lex_scores.max() - lex_scores.min() + 1e-9)
    except:
        # Dividing by zero:
        lex_norm = np.zeros_like(lex_scores)

    # Assign normalized values back
    for i, r in enumerate(semantic_results):
        r["score"] = sem_norm[i]
    for i, r in enumerate(lexical_results):
        r["score"] = lex_norm[i]

    # Now the scores are normalized.

    # --- Combine both lists ---
    combined_dict = {}
    # Add all semantic results
    for res in semantic_results:
        combined_dict[res["id"]] = res.copy()
    # Merge in lexical results
    for res in lexical_results:
        # If a duplicate is found, average their scores
        if res["id"] in combined_dict:
            existing = combined_dict[res["id"]]
            avg_score = (existing["score"] + res["score"]) / 2.0
            existing["score"] = avg_score
            existing["result_type"] = "both"  # Make a note for when lexical and semantic find the same results
            existing["query"] = f"Semantic query used: {existing['query']} | Lexical query used: {res['query']}"  # Show both queries
        else:
            combined_dict[res["id"]] = res.copy()
    # Convert back to list
    combined_results = list(combined_dict.values())
    # --- Sort by averaged score descending ---
    combined_results = sorted(combined_results, key=lambda r: r["score"], reverse=True)
    # Remove any results that have the same id as an attachment
    attachment_id = str(searchfacts.attachment_path)
    combined_results = [r for r in combined_results if r['id'] != attachment_id]

    # Apply MMR reranking, which trims to desired count. To save compute, only do MMR on half of the combined list of results.
    final_results = mmr_rerank_hybrid(combined_results[:fetch_k], mmr_lambda=config['mmr_lambda'], alpha=config['mmr_alpha'], n_results=max_results)

    if not final_results:  # This should never happen, however
        return []
    
    # Unfortunately, I have discovered it is not possible to filter the results based on score alone. For now, that is left to the LLM.

    return format_results(final_results)

# Second major function of this code is to make searches
def hybrid_search(queries: List[str], searchfacts, models, collections, config, search_type: str) -> Optional[Dict[str, str]]:
    """Performs a hybrid (semantic and lexical) search with hybrid MMR rerank, and cap at maximum result number. Searches with multiple queries will be searched individually, then combined into one output. To do individual searches, only input one string for the query list."""
    if not queries or search_type not in models or search_type not in collections:
        return []

    model = models[search_type]
    if model:
        query_embedding = model.encode(queries, convert_to_numpy=True, normalize_embeddings=True)
    else:
        return []

    # Return the final results
    return perform_search(query_embedding, queries, searchfacts, models, collections, config, search_type, config['max_results'])

def save_insight_to_file(insight_text: str, original_query: str, image_paths: list, text_paths: list, config: dict, log_callback=None):
    """Saves an AI-generated insight to a .txt file in the 'saved_insights' directory."""
    try:
        _log("Saving AI insight to file...", log_callback)
        if not insight_text.strip():
            _log("  [Warning] Insight text is empty, cannot save.", log_callback)
            return
        # 1. Define the path for the saved insights folder
        base_dir = BASE_DIR
        insights_dir = DATA_DIR / "saved_insights"
        os.makedirs(insights_dir, exist_ok=True) # Create the directory if it doesn't exist
        # 2. Create a unique filename using the current timestamp
        file_name = f"insight_{int(time.time())}.txt"
        file_path = os.path.join(insights_dir, file_name)
        # 3. Format the content with metadata at the top
        # Pre-calculate the list comprehensions
        image_names = [Path(p).name for p in image_paths if Path(p).suffix.lower() != ".gif"]
        text_names = [Path(p).name for p in text_paths]
        
        source_images_str = ', '.join(image_names)
        source_docs_str = ', '.join(text_names)

        # 3. Format the content with metadata at the top
        file_content = (
            f"<Source: AI Insight> This is from a previous interaction with the user."
            f"Original User Query: {original_query}\n"
            f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"Source Images Used: {source_images_str}\n"
            f"Source Documents Used: {source_docs_str}\n"
            f"----------------------------------------\n\n"
            f"{insight_text}"
        )
        # 4. Write the content to the new file
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(file_content)
        _log(f"✓ AI insight saved successfully to {file_path}", log_callback)
        _log("The new insight will be available for search after the next sync.", log_callback)

    except Exception as e:
        _log(f"[ERROR] Failed to save AI insight to file: {e}", log_callback)

def load_config(file_path):
    """Loads configuration from a JSON file."""
    with open(file_path, 'r') as config_file:
        config = json.load(config_file)
        return config

def download_embedding_model(model_name, log_callback):
    """Downloads a model to a folder of the same name."""
    local_path = DATA_DIR / model_name.replace('/', '_')

    if local_path.is_dir():
        _log(f"Model weights found for {model_name}.", log_callback)
        return

    try:
        _log(f"Downloading {model_name} from Hugging Face. This will take some time. Don't close the app.", log_callback)
        # Load the model from Hugging Face (this downloads it)
        model = SentenceTransformer(model_name)
        # Save the model directly to your desired local folder
        # This saves all the necessary files (config.json, weights, tokenizer, etc.)
        model.save(str(local_path))        
        _log(f"Successfully downloaded and saved {model_name} weights.", log_callback)
        
    except Exception as e:
        _log(f"[ERROR] Failed to download/save {model_name}: {e}", log_callback)

# To make setup easier
def machine_setup(config, log_callback):
    """Initializes all the necessary 'machines' needed for sync_directory and semantic_search."""
    connected = is_connected()
    
    # Set environment variables based on internet connectivity, needed for offline mode
    if not connected:
        os.environ['HF_HUB_OFFLINE'] = '1'
        os.environ['TRANSFORMERS_OFFLINE'] = '1'
        _log("Offline: Using local models only.", log_callback)
    else:
        # Remove offline flags if they were set previously
        os.environ.pop('HF_HUB_OFFLINE', None)
        os.environ.pop('TRANSFORMERS_OFFLINE', None)
        _log("Online: Will download models if needed.", log_callback)

    # Define model names. Set to None if you don't want to use one.
    text_model_name = config.get('text_model_name', "BAAI/bge-small-en-v1.5")  # Options: BAAI/bge-small-en-v1.5, BAAI/bge-large-en-v1.5 (in order of increasing power)
    image_model_name = config.get('image_model_name', "clip-ViT-B-32") # Options: clip-ViT-B-32, clip-ViT-B-16, clip-ViT-L-14 (in order of increasing power)

    text_model_dir = str(Path(BASE_DIR) / text_model_name.replace("/", "_"))
    image_model_dir = str(Path(BASE_DIR) / image_model_name.replace("/", "_"))
    text_model_bundled = os.path.exists(text_model_dir)
    image_model_bundled = os.path.exists(image_model_dir)
    
    device = "cuda" if torch.cuda.is_available() and config['use_cuda'] else "cpu"
    _log(f"Using device: {device}", log_callback)
    # Initialize ChromaDB client
    chroma_client = chromadb.PersistentClient(path=str(DATA_DIR / "chroma_db"))
    # Use these dictionaries to hold our models and collections
    models = {}
    collections = {}

    # --- Load Text Model & Collection ---
    try:
        if text_model_name:
            _log(f"Loading text embedder: {config['text_model_name']}", log_callback)
            if not text_model_bundled:
                download_embedding_model(text_model_name, log_callback)
                models['text'] = SentenceTransformer(str(DATA_DIR / text_model_name.replace('/', '_')), device=device, local_files_only=not connected)
            else:
                _log(f"Found bundled model weights locally for: {text_model_name}", log_callback)
                models['text'] = SentenceTransformer(text_model_dir, device=device, local_files_only=not connected)
            models['text'].max_seq_length = config.get("chunk_size", 512) + 100  # Limit sequence length to save compute. +100 for safety. Values that are too large get cut off.

            collections['text'] = chroma_client.get_or_create_collection(name=text_model_name.replace("/", "_"), metadata={"hnsw:space": "cosine"})  # cosine is essential
            _log("✓ Text embedder loaded successfully.")
        else:
            _log("No text embedder name set in config.", log_callback)
            models['text'] = None
            collections['text'] = None
    except Exception as e:
        if not connected:
            _log(f"[ERROR] Cannot load text model '{text_model_name}' - not available offline.", log_callback)
        else:
            _log(f"[ERROR] Failed to load text embedder: {e}", log_callback)
        models['text'] = None
        collections['text'] = None

    # --- Load Image Model & Collection ---
    try:
        if image_model_name:
            _log(f"Loading image embedder: {config['image_model_name']}", log_callback)
            if not image_model_bundled:
                download_embedding_model(image_model_name, log_callback)
                models['image'] = SentenceTransformer(str(DATA_DIR / image_model_name.replace('/', '_')), device=device, local_files_only=not connected)
            else:
                _log(f"Found bundled model weights locally for: {image_model_name}", log_callback)
                models['image'] = SentenceTransformer(image_model_dir, device=device, local_files_only=not connected)

            collections['image'] = chroma_client.get_or_create_collection(name=image_model_name.replace("/", "_"), metadata={"hnsw:space": "cosine"})  # cosine is essential
            _log("✓ Image embedder loaded successfully.")
        else:
            _log("No image embedder name set in config.", log_callback)
            models['image'] = None
            collections['image'] = None
    except Exception as e:
        if not connected:
            _log(f"[ERROR] Cannot load image model '{image_model_name}' - not available offline.", log_callback)
        else:
            _log(f"[ERROR] Failed to load image embedder: {e}", log_callback)
        models['image'] = None
        collections['image'] = None
    
    # --- Create Text Splitter ---
    try:
        if text_model_name:
            _log(f"Creating text splitter...", log_callback)
            if text_model_bundled:
                text_splitter = create_text_splitter(text_model_dir, config['chunk_size'], config['chunk_overlap'], offline_mode=not connected)  # Splits by paragraph; can improve
            else:
                text_splitter = create_text_splitter(str(DATA_DIR / text_model_name.replace('/', '_')), config['chunk_size'], config['chunk_overlap'], offline_mode=not connected)  # Splits by paragraph; can improve
        else:
            text_splitter = None
    except Exception as e:
        if not connected:
            _log(f"[ERROR] Cannot create text splitter '{text_model_name}' not available offline.", log_callback)
        else:
            _log(f"[ERROR] Failed to create text splitter: {e}", log_callback)
        text_splitter = None

    # Only try if connected to internet and libaries installed
    try:
        drive_service = get_drive_service(log_callback, config)
    except Exception as e:
        _log(f"[ERROR] Failed to get Drive service: {e}", log_callback)
        drive_service = None
    
    return drive_service, text_splitter, models, collections

def delete_chroma_collection(db_path: str, collection_name: str):
    """Initializes the client and deletes the specified collection."""
    try:
        # Connect to the persistent database folder
        client = chromadb.PersistentClient(path=db_path)
        
        # Delete the collection by name
        client.delete_collection(name=collection_name)
        
        print(f"Successfully deleted the collection: '{collection_name}' from {db_path}")

    except Exception as e:
        print(f"An error occurred during collection deletion: {e}")

# delete_chroma_collection(CHROMA_DB_PATH, IMAGE_COLLECTION_NAME)

"""To-do: 
pull images from PDFs
add a third way to score search results to resolve ties from lexical/semantic search
add a button to delete a collection (settings would be a good place for this)
add a parser for markdown files"""
