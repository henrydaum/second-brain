import os
from pathlib import Path
import gc
import shutil
import socket
import logging
# 3rd Party - also includes torch, sentence_transformers (imported later)
import numpy as np

from Stage_0.BaseService import BaseService

logger = logging.getLogger("EmbedClass")

BASE_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = Path(os.getenv('LOCALAPPDATA')) / "2nd Brain"

# --- BASE CLASS ---
class BaseEmbedder(BaseService):
    """
    Abstract base class for all embedding models.
    Enforces a standard interface for Loading, Unloading, and Encoding.
    """
    def __init__(self, model_name, chunk_size=512, use_cuda=True):
        super().__init__()
        self.model_name = model_name
        self.shared = True  # Embedders are thread-safe (encode() is stateless)
        self.chunk_size = chunk_size
        self.use_cuda = use_cuda

    def load(self):
        """Must implement model loading logic."""
        raise NotImplementedError

    def unload(self):
        """Must implement RAM cleanup logic."""
        raise NotImplementedError

    def encode(self, inputs):
        """Must return a numpy array (or list of arrays) of embeddings."""
        raise NotImplementedError
    
    @staticmethod
    def is_connected():
        """Helper: Checks for internet connectivity."""
        try:
            socket.create_connection(("8.8.8.8", 53), timeout=3)
            return True
        except OSError:
            return False

# --- SUBCLASS: SENTENCE TRANSFORMERS ---
class SentenceTransformerEmbedder(BaseEmbedder):
    def __init__(self, model_name="BAAI/bge-small-en-v1.5", chunk_size=512, use_cuda=True):
        super().__init__(model_name, chunk_size=chunk_size, use_cuda=use_cuda)
        self.model = None
        self.device = None
        
        # Create a safe folder name (e.g., "BAAI_bge-small-en-v1.5")
        self.bundled_path = BASE_DIR / self.model_name.replace('/', '_')
        self.model_is_bundled = os.path.exists(self.bundled_path)

        self.download_path = DATA_DIR / self.model_name.replace('/', '_')
        self.model_is_downloaded = os.path.exists(self.download_path)

    def _set_offline_env(self, offline=True):
        """Toggles HuggingFace offline mode environment variables."""
        if offline:
            os.environ['HF_HUB_OFFLINE'] = '1'
            os.environ['TRANSFORMERS_OFFLINE'] = '1'
            logger.info("Offline: Using local models only.")
        else:
            os.environ.pop('HF_HUB_OFFLINE', None)
            os.environ.pop('TRANSFORMERS_OFFLINE', None)
            logger.info("Online: Will download models if needed.")

    def download(self):
        """Downloads the model and saves it to the custom local folder. Returns True if successful, False otherwise."""
        from sentence_transformers import SentenceTransformer

        logger.info(f"Downloading {self.model_name}... Do not close the app.")
        try:

            # 1. Download to temporary cache
            temp_model = SentenceTransformer(self.model_name)
            
            # 2. Save to our permanent local folder
            # This extracts the weights/config from the cache to your folder
            temp_model.save(str(self.download_path))
            
            logger.info(f"Successfully saved model to {self.download_path}")
            
            # Clean up temp model from RAM
            del temp_model
            gc.collect()
            return True
        except Exception as e:
            logger.error(f"Download failed: {e}")
            # Clean up partial download if it failed
            if self.download_path.exists():
                shutil.rmtree(self.download_path, ignore_errors=True)
            return False

    def load(self):
        """Loads the model into memory. Returns True if successful, False otherwise."""
        logger.info(f"Loading Sentence Transformer model: {self.model_name}")
        if self.loaded and self.model is not None:
            return True

        import torch 
        from sentence_transformers import SentenceTransformer
        
        # Determine Device — uses self.use_cuda from __init__
        self.device = "cuda" if torch.cuda.is_available() and self.use_cuda else "cpu"
        logger.info(f"Loading model on {self.device}...")

        # 1. Check Internet / Environment
        connected = self.is_connected()
        self._set_offline_env(not connected)

        if not self.model_is_bundled:  # If not bundled, check download folder
            if not self.model_is_downloaded:  # If not downloaded, check internet
                if not connected:
                    logger.error(f"Model {self.model_name} not bundled and no internet.")
                    return False
                
                # If internet, attempt download
                success = self.download()
                if not success:
                    # Download failed
                    return False
            # Model is downloaded, load from download path
            logger.info(f"Found downloaded model weights for: {self.model_name}")
            self.model = SentenceTransformer(
                str(self.download_path),
                device=self.device, 
                local_files_only=True 
            )
        else:
            # Model is bundled, load from BASE_DIR
            try:
                logger.info(f"Found bundled model weights for: {self.model_name}")
                self.model = SentenceTransformer(
                    str(self.bundled_path), 
                    device=self.device, 
                    local_files_only=True 
                )
            except Exception as e:
                logger.error(f"Load failed: {e}")
                return False
        
        # Set Context Length
        self.model.max_seq_length = self.chunk_size
        
        self.loaded = True
        logger.info("Sentence Transformer model loaded.")
        return True

    def unload(self):
        if self.model:
            del self.model
            self.model = None
            self.loaded = False
            
            gc.collect()
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass
            logger.info("Sentence Transformer model unloaded.")

    def encode(self, inputs, batch_size=11):
        """
        Accepts string or list of strings.
        For images (image models), the input should be a list of image file paths.
        Returns numpy array.
        """
        if not self.model or not self.loaded:
            logger.warning("Attempted generation while model is unloaded.")
            return None

        try:
            # normalize_embeddings=True is critical for Cosine Similarity
            return self.model.encode(inputs, normalize_embeddings=True, convert_to_numpy=True)
        except Exception as e:
            logger.error(f"Inference failed: {e}")
            return None


def build_services(config: dict) -> dict:
    return {
        "text_embedder": SentenceTransformerEmbedder(
            model_name=config.get("embed_text_model_name", "BAAI/bge-small-en-v1.5"),
            use_cuda=config.get("embed_use_cuda", False),
            chunk_size=config.get("embed_chunk_size", 512),
        ),
        "image_embedder": SentenceTransformerEmbedder(
            model_name=config.get("embed_image_model_name", "clip-ViT-L-14"),
            use_cuda=config.get("embed_use_cuda", False),
            chunk_size=config.get("embed_chunk_size", 512),
        ),
    }