from pathlib import Path
import os
import logging

logger = logging.getLogger("LLMClass")

class BaseLLM:
    """Abstract base class for Large Language Models."""
    def load(self):
        """Loads the model using the given model name."""
        raise NotImplementedError("Subclasses should implement this method.")
    
    def unload(self):
        """Unloads the model and frees up associated resources."""
        raise NotImplementedError("Subclasses should implement this method.")

    def invoke(self, prompt: str, image_paths: list[str] = None, temperature: float = 1.0) -> str:
        """Processes a prompt with optional images and returns the full response."""
        raise NotImplementedError("Subclasses should implement this method.")
    
    def stream(self, prompt: str, image_paths: list[str] = None, temperature: float = 1.0):
        """Processes a prompt with optional images and yields the response stream."""
        raise NotImplementedError("Subclasses should implement this method.")
    
    @staticmethod
    def get_image_bytes(path: str):
        """Returns image bytes for an image path, handles GIFs by taking the first frame."""
        from PIL import Image, ImageFile
        import io

        IMG_THUMBNAIL = (2048, 2048)
        jpeg_quality = 80
        MAX_IMAGE_SIZE = 50_000_000  # 50 megapixels
        Image.MAX_IMAGE_PIXELS = MAX_IMAGE_SIZE
        ImageFile.LOAD_TRUNCATED_IMAGES = True

        img = None
        try:
            if Path(path).suffix.lower() == ".gif":
                with Image.open(path) as gif_img:
                    gif_img.seek(0)
                    img = gif_img.copy()
            else:
                img = Image.open(path)
            
            if img is None: raise ValueError("Image object is None.")
            if img.mode != 'RGB': img = img.convert('RGB')
            
            img.thumbnail(IMG_THUMBNAIL, Image.Resampling.LANCZOS)
            buffer = io.BytesIO()
            img.save(buffer, format="JPEG", quality=jpeg_quality, optimize=True)
            return buffer.getvalue()
        
        except Exception as e:
            return None
        finally:
            if img: img.close()

    @staticmethod
    def _build_image_prompt(prompt: str, valid_file_names: list[str], attached_image_path: str = None) -> str:
        """Helper to append image references to the text prompt."""
        if not valid_file_names:
            return prompt
            
        source_info = ""
        i = 1
        for name in valid_file_names:
            if attached_image_path:
                # Attached image goes last
                if name == Path(attached_image_path).name and i == len(valid_file_names):
                    source_info += f"\n<User Attached Image: {name}>"
                else:
                    source_info += f"\n<Image Result {i}: {name}>"
                    i += 1
            else:
                source_info += f"\n<Image {i}: {name}>"
                i += 1
            
        final_prompt = (
            f"{prompt}\n\n"
            f"The following images are provided:{source_info}\n\n"
        )
        return final_prompt

class LMStudioLLM(BaseLLM):
    def __init__(self, model_name):
        import lmstudio as lms
        self.model_name = model_name
        self.model = None
        self.vision = None
        self.loaded = False

    def load(self):
        logger.info(f"Loading LM Studio model: {self.model_name}")
        try:
            import lmstudio as lms
            self.model = lms.llm(self.model_name)
            self.vision = self.model.get_info().vision
            if self.vision:
                logger.info(f"Model has vision support.")
            else:
                logger.info(f"Model does not have vision support.")
            self.loaded = True
            logger.info("LM Studio model loaded.")
            return True
        except Exception as e:
            logger.error(f"LM Studio Load rror: {e}")
            return False

    def unload(self):
        # LM Studio library might not expose explicit unload, 
        # but releasing the object usually helps.
        if self.model:
            self.model.unload()
        self.loaded = False
        logger.info("LM Studio model unloaded.")

    def prepare_chat(self, prompt: str, image_paths: list[str], attached_image_path: str = None):
        """Helper to create a Chat object if images are provided."""
        if not image_paths and not attached_image_path:
            return prompt, []

        if attached_image_path:
            image_paths.append(attached_image_path)

        import lmstudio as lms
        import tempfile

        image_handles = []
        valid_file_names = []
        temp_files_to_delete = []

        for path in image_paths:
            if not os.path.exists(path):
                continue
                
            image_bytes = self.get_image_bytes(path)
            if not image_bytes:
                continue

            tmp_path = None
            try:
                # LM Studio needs a file on disk
                with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
                    f.write(image_bytes)
                    tmp_path = f.name
                    f.flush()
                
                image_handles.append(lms.prepare_image(tmp_path))
                valid_file_names.append(os.path.basename(path))
                temp_files_to_delete.append(tmp_path)
            except Exception as e:
                logger.error(f"Temp file error for {path}: {e}", logger.info_callback)
                if tmp_path and os.path.exists(tmp_path):
                    os.remove(tmp_path)

        final_prompt = self._build_image_prompt(prompt, valid_file_names, attached_image_path)
        
        chat = lms.Chat()
        chat.add_user_message(final_prompt, images=image_handles)
        return chat, temp_files_to_delete

    def _cleanup_temp_files(self, temp_files: list[str]):
        for f_path in temp_files:
            try:
                if os.path.exists(f_path):
                    os.remove(f_path)
            except: pass

    def invoke(self, prompt, image_paths=[], attached_image_path=None, temperature=1.0):
        try:
            chat_input, temp_files = self.prepare_chat(prompt, image_paths, attached_image_path)
            response = self.model.respond(chat_input, config={"temperature": temperature})
            return response.content
        except Exception as e:
            logger.error(f"LM Studio Invoke Error: {e}")
            return None
        finally:
            if temp_files:
                import time
                time.sleep(0.1)
                self._cleanup_temp_files(temp_files)
    
    def stream(self, prompt, image_paths=[], attached_image_path=None, temperature=1.0):
        try:
            chat_input, temp_files = self.prepare_chat(prompt, image_paths, attached_image_path)
            for fragment in self.model.respond_stream(chat_input, config={"temperature": temperature}):
                yield fragment.content
        except Exception as e:
            logger.error(f"LM Studio Stream Error: {e}")
            return None
        finally:
            if temp_files:
                import time
                time.sleep(0.1)
                self._cleanup_temp_files(temp_files)

class OpenAILLM(BaseLLM):
    def __init__(self, model_name, api_key=None):
        self.model_name = model_name
        self.api_key = api_key
        self.loaded = False

    def load(self):
        """API is lightweight, not much to load or unload."""
        try:
            logger.info(f"Loading OpenAI model: {self.model_name}")
            import openai
            if self.api_key:
                self.client = openai.OpenAI(api_key=self.api_key)
            else:
                self.client = openai.OpenAI() # Uses env var
            # Check for vision; not as straightforward as LM Studio
            model_name_lower = self.model_name.lower()
            openai_vision_keywords = ["vision", "gpt-4o", "gpt-5", "gpt-4.1", "o3", "turbo"]
            self.vision = any(keyword in model_name_lower for keyword in openai_vision_keywords)
            if self.vision:
                logger.info(f"Model has vision support.")
            else:
                logger.info(f"Model does not have vision support.")
            self.loaded = True
            logger.info("OpenAI model loaded.")
            return True
        except Exception as e:
            logger.error(f"OpenAI Load Error: {e}")
            return False
    
    def unload(self):
        self.loaded = False
        logger.info("OpenAI model unloaded.")

    def prepare_chat(self, prompt: str, image_paths: list[str], attached_image_path: str = None):
        if not image_paths and not attached_image_path:
            return [{"role": "user", "content": prompt}]

        if attached_image_path:
            image_paths.append(attached_image_path)
        
        import base64
        content_list = []
        valid_file_names = []
        input_images = []

        for path in image_paths:
            if not os.path.exists(path): continue
            
            image_bytes = self.get_image_bytes(path)
            if not image_bytes: continue

            try:
                base64_image = base64.b64encode(image_bytes).decode("utf-8")
                input_images.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                })
                valid_file_names.append(os.path.basename(path))
            except: pass

        final_prompt = self._build_image_prompt(prompt, valid_file_names)
        content_list.append({"type": "text", "text": final_prompt})
        content_list.extend(input_images)
        
        return [{"role": "user", "content": content_list}]

    def invoke(self, prompt, image_paths=[], attached_image_path=None, temperature=1.0):
        try:
            messages = self.prepare_chat(prompt, image_paths, attached_image_path)
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=temperature
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"OpenAI Invoke Error: {e}")
            return None

    def stream(self, prompt, image_paths=[], attached_image_path=None, temperature=1.0):
        try:
            messages = self.prepare_chat(prompt, image_paths, attached_image_path)
            stream = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=temperature,
                stream=True
            )
            for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except Exception as e:
            logger.error(f"OpenAI Stream Error: {e}")
            return None