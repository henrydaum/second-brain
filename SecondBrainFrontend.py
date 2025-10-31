import flet as ft
import threading
import os
from pathlib import Path

class BaseLLM:
    """Abstract base class for Large Language Models."""
    def invoke(self, prompt: str) -> str:
        """Processes a prompt and returns the full response as a single string."""
        raise NotImplementedError("Subclasses should implement this method.")
    
    def stream(self, prompt: str):
        """Processes a prompt and yields the response as a stream of text chunks."""
        raise NotImplementedError("Subclasses should implement this method.")
    
    def unload(self):
        """Unloads the model and frees up associated resources."""
        raise NotImplementedError("Subclasses should implement this method.")
    
    @staticmethod
    def get_image_bytes(path: str):
        """Returns image bytes for an image path, which can be turned into a temporary path or a base64-encoded string. Processes .gif files by taking the first frame."""
        from PIL import Image, ImageFile
        import io

        IMG_THUMBNAIL = (2048, 2048)
        jpeg_quality = 80

        Image.MAX_IMAGE_PIXELS = None
        ImageFile.LOAD_TRUNCATED_IMAGES = True

        img = None
        try:
            # Process gifs
            if Path(path).suffix.lower() == ".gif":
                with Image.open(path) as gif_img:
                    gif_img.seek(0)
                    img = gif_img.copy() # Get the first frame
            else:
                img = Image.open(path)
            
            if img is None: # Should be redundant, but safe
                raise ValueError("Image object is None.")

            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            img.thumbnail(IMG_THUMBNAIL, Image.Resampling.LANCZOS)
            buffer = io.BytesIO()
            img.save(buffer, format="JPEG", quality=jpeg_quality, optimize=True)
            
            return buffer.getvalue()
        
        except Exception as e:
            print(f"  Warning: Could not process image {path}: {e}")
            return None
        finally:
            if img:
                img.close()

    @staticmethod
    def _build_image_prompt(prompt: str, valid_file_names: list[str], attached_image_path) -> str:
        """Helper to build the text prompt that lists the images."""
        from pathlib import Path
        if not valid_file_names:
            return prompt
        source_info = ""
        i = 1
        for name in valid_file_names:
            if attached_image_path:
                if name == Path(attached_image_path).name and i == len(valid_file_names):
                    source_info += f"\n<Attached Image: {name}>"
                else:
                    source_info += f"\n<Image Result {i}: {name}>"
                    i += 1
            else:
                source_info += f"\n<Image {i}: {name}>"
                i += 1
        # print(source_info)
        final_prompt = f"{prompt}\n\nThe following images are provided:{source_info}\n\nEach <Image n> corresponds to the nth attached image. When you refer to an image in your response, refer to it by its file name, rather than its number. You can only see the first frame of .gif files."
        # print(final_prompt)
        return final_prompt

class LMStudioLLM(BaseLLM):
    # LM STUDIO's great library makes this almost a cakewalk.
    def __init__(self, model_name):
        import lmstudio as lms
        self.model = lms.llm(model_name)
        self.model_name = model_name

    def prepare_chat(self, prompt: str, sd: dict):
        """Helper to create a Chat object if an image is provided, otherwise returns the prompt string."""
        image_paths = sd.get('image_paths', [])
        if sd['attached_image_path']:
            image_paths.append(sd['attached_image_path'])  # Ensure attached image is last in the list.

        # In case of no images:
        if not image_paths:
            return prompt, []

        import lmstudio as lms
        import tempfile
        # Check if an image is provided
        image_handles = []
        valid_file_names = []
        temp_files_to_delete = []
        # print(f"Len image paths: {len(image_paths)}")
        for path in image_paths:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Image not found at path: {path}")
            # Call the static method to get image bytes
            image_bytes = self.get_image_bytes(path)
            if not image_bytes:
                print(f"  [SKIPPED] Could not process image: {path}")
                continue
            tmp_path = None
            try:
                # Create a temporary file path for the image, which needs to be deleted later.
                with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
                    f.write(image_bytes)
                    tmp_path = f.name
                image_handles.append(lms.prepare_image(tmp_path))
                valid_file_names.append(os.path.basename(path))
                temp_files_to_delete.append(tmp_path)
            except Exception as e:
                print(f"  [SKIPPED] Could not write temp file for {path}: {e}")
                if tmp_path and os.path.exists(tmp_path):
                    os.remove(tmp_path) # Clean up failed attempt
        # Build the final prompt so that it includes the image sources.
        final_prompt = self._build_image_prompt(prompt, valid_file_names, sd['attached_image_path'])
        # print(final_prompt)
        chat = lms.Chat()
        chat.add_user_message(final_prompt, images=image_handles)
        return chat, temp_files_to_delete

    def _cleanup_temp_files(self, temp_files: list[str]):
        """Helper to delete temp files."""
        for f_path in temp_files:
            try:
                if os.path.exists(f_path):
                    os.remove(f_path)
            except Exception as e:
                print(f"  Warning: Could not delete temp file {f_path}: {e}")

    def invoke(self, prompt, temperature=1, sd=None):
        chat_input, temp_files = self.prepare_chat(prompt, sd)
        try:
            response = self.model.respond(chat_input, config={"temperature": temperature})
            return response.content
        finally:
            # This GUARANTEES cleanup, even if respond() fails
            if temp_files:
                self._cleanup_temp_files(temp_files)
    
    def stream(self, prompt, temperature=1, sd=None):
        chat_input, temp_files = self.prepare_chat(prompt, sd)
        try:
            for fragment in self.model.respond_stream(chat_input, config={"temperature": temperature}):
                yield fragment.content
        finally:
            # This GUARANTEES cleanup, even if the stream is broken
            if temp_files:
                self._cleanup_temp_files(temp_files)

    def unload(self):
        self.model.unload()

class OpenAILLM(BaseLLM):
    def __init__(self, model_name, api_key):
        import openai
        # To use an API key straight from config:
        if api_key:
            # Use the provided key
            self.client = openai.OpenAI(api_key=api_key)
        else:
            # Use the environment variable (handles both None and "")
            try:
                # Uses the key from os.getenv("OPENAI_API_KEY")
                self.client = openai.OpenAI()
            except openai.OpenAIError as e:
                raise ValueError("API key not found. Pass a key or set the OPENAI_API_KEY environment variable.")
        self.model_name = model_name

    def prepare_chat(self, prompt: str, sd: dict):
        """OpenAI requires images be presented in a certain way. This is how that's done."""
        image_paths = sd.get('image_paths', [])
        if sd['attached_image_path']:
            image_paths.append(sd['attached_image_path'])  # Ensure attached image is last in the list.

        # Handle the simple, text-only case
        if not image_paths:
            return [{"role": "user", "content": prompt}]
        
        import base64
        # Build the multimodal content list
        content_list = []
        valid_file_names = []
        input_images = []
        for path in image_paths:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Image not found at path: {path}")
            # Call static method to get image bytes
            image_bytes = self.get_image_bytes(path)
            if not image_bytes:
                print(f"  [SKIPPED] Could not process image (or GIF frame): {path}")
                continue
            try:
                # Base64-encode bytes
                base64_image = base64.b64encode(image_bytes).decode("utf-8")
                # Images are presented as a list of dicts:
                input_images.append({
                    "type": "input_image",
                    "image_url": f"data:image/jpeg;base64,{base64_image}"})
                valid_file_names.append(os.path.basename(path))
            except Exception as e:
                print(f"  [SKIPPED] Could not base64 encode image {path}: {e}")
        final_prompt = self._build_image_prompt(prompt, valid_file_names, sd['attached_image_path'])
        # print(final_prompt)
        content_list.append({"type": "input_text", "text": final_prompt})
        # Add all the valid image parts
        content_list.extend(input_images)
        # Return the final messages list
        return [{"role": "user", "content": content_list}]

    def invoke(self, prompt, temperature=1, sd=None):
        messages = self.prepare_chat(prompt, sd)
        # print(f"[DEBUG] Payload size: {len(str(messages)) / 1e6:.6f} MB")
        response = self.client.responses.create(model=self.model_name, input=messages)
        return response.output_text

    def stream(self, prompt, temperature=1, sd=None):
        messages = self.prepare_chat(prompt, sd)
        # print(f"[DEBUG] Payload size: {len(str(messages)) / 1e6:.6f} MB")
        with self.client.responses.stream(model=self.model_name, input=messages) as stream:  # Some models lack temperature functionality.
            for event in stream:
                if event.type == "response.output_text.delta":
                    yield event.delta
            stream.close()

    def unload(self):
        # No action needed. The client is a lightweight object.
        pass

class App:
    def __init__(self, page: ft.Page):
        # Resizing, setting icon, and centering the page.
        page.window.resizable = True
        page.window.width = 840
        page.window.height = 600
        page.window.min_width = 780
        page.window.min_height = 460
        try:
            page.window.icon = "icon.ico"
        except:
            print("Failed to get icon.ico")
        page.window.center()
        page.visible = True
        # Page variable and misc. page settings
        self.page = page
        self.page.title = "Second Brain"
        self.page.padding = 0
        self.page.scroll = None
        # Set theme HERE
        # self.page.theme = ft.Theme(color_scheme_seed=ft.Colors.LIGHT_BLUE_50)
        # BACKEND VARIABLES
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        self.config_path = os.path.join(BASE_DIR, "config.json")
        self.config = {}
        self.drive_service = None
        self.text_splitter = None
        self.models = {}
        self.collections = {}
        self.llm = None
        self.llm_vision = None
        # Message avatars
        self.user_avatar = ft.CircleAvatar(content=ft.Icon(ft.Icons.SEARCH), radius=18)
        self.ai_avatar = ft.CircleAvatar(content=ft.Icon(ft.Icons.WB_SUNNY_OUTLINED), radius=18)
        self.attachment_avatar = ft.CircleAvatar(content=ft.Icon(ft.Icons.ATTACH_FILE_ROUNDED), radius=18)
        # Input field
        self.user_input = ft.TextField(expand=True, multiline=True, label="Compose your message...", shift_enter=True, on_submit=self.send_message, min_lines=1, max_lines=7, max_length=4096, border_radius=15, focused_border_width=2, bgcolor=ft.Colors.with_opacity(0.1, ft.Colors.BLACK))
        # Send button
        self.send_button = ft.IconButton(icon=ft.Icons.SEND_ROUNDED, on_click=self.send_message, tooltip="Send")
        # File picker & Attachment logic
        self.file_picker = ft.FilePicker(on_result=self.attach_files)
        self.page.overlay.append(self.file_picker)
        self.attach_button = ft.IconButton(icon=ft.Icons.ATTACH_FILE_ROUNDED, on_click=lambda _: self.file_picker.pick_files(allow_multiple=False, allowed_extensions=["txt", "pdf", "docx", "gdoc", "png", "jpeg", "jpg", "gif", "webp"]), tooltip="Attach file")  # EXTEND EXTENSIONS HERE
        self.attachment_data = None
        self.attachment_size = None
        self.attachment_path = None
        # AI Mode checkbox, plus settings, reload backend, sync, and reauthorize buttons
        self.ai_mode_checkbox = ft.Checkbox(label="AI Mode", value=False, on_change=self.toggle_ai_mode)
        self.open_settings_btn = ft.ElevatedButton("Open Settings", icon=ft.Icons.SETTINGS_ROUNDED, on_click=self.open_config)
        self.reload_backend_btn = ft.ElevatedButton("Reload Backend", icon=ft.Icons.REFRESH_ROUNDED, on_click=self.reload_backend)
        self.sync_directory_btn = ft.ElevatedButton("Sync Directory", icon=ft.Icons.SYNC_ROUNDED, on_click=self.start_sync_directory)
        self.reauthorize_button = ft.ElevatedButton("Reauthorize Drive", icon=ft.Icons.LOCK_ROUNDED, on_click=self.reauthorize_drive)
        # Top row (starts invisible)
        self.buttons_row = ft.Row([self.ai_mode_checkbox, self.reload_backend_btn, self.sync_directory_btn, self.open_settings_btn, self.reauthorize_button], alignment="spaceAround", visible=False)
        # Chat list
        self.chat_list = ft.ListView(expand=True, spacing=10, auto_scroll=False, padding=ft.padding.only(left=10, right=10, top=10))
        # Attachment display (starts invisible)
        self.attachment_display = ft.Text(visible=False, italic=True)
        # Settings view button
        self.show_settings_btn = ft.IconButton(tooltip="Hide Buttons", icon=ft.Icons.TUNE_OUTLINED, on_click=self.toggle_settings_view)
        # Bottom row
        self.bottom_row = ft.Row([self.show_settings_btn, self.user_input, self.attach_button, self.send_button], alignment="spaceBetween")
        # MAIN COLUMN
        main_layout = ft.Column(controls=[self.buttons_row, self.chat_list, self.attachment_display, self.bottom_row], expand=True)
        # We wrap the layout in a Container to give the app some nice spacing.
        padded_layout = ft.Container(content=main_layout, padding=20, expand=True)
        # Animated loading indicator
        self.loading_indicator = ft.CupertinoActivityIndicator(radius=15, color=ft.Colors.WHITE, animating=True)
        # Loading overlay
        self.overlay = ft.TransparentPointer(ft.Container(content=ft.Column([self.loading_indicator], alignment=ft.MainAxisAlignment.CENTER, horizontal_alignment=ft.CrossAxisAlignment.CENTER), alignment=ft.alignment.center, width=page.width, height=page.height), visible=True)
        # Stacking the overlay on top of the page
        page.add(ft.Stack([padded_layout, self.overlay], expand=True))
        # Focus cursor on the chat input field
        self.user_input.focus()
        # After setting up the UI, INITIALIZE BACKEND THREAD
        self.initialize_backend()

    def toggle_settings_view(self, e=None):
        # Whether or not the settings view is visible is the proxy for whether the utility buttons are visible.
        if not self.buttons_row.visible:
            self.buttons_row.visible = True
            self.show_settings_btn.tooltip = "Hide Buttons"
        else:
            self.buttons_row.visible = False
            self.show_settings_btn.tooltip = "Show Buttons"
        self.page.update()
        self.config["show_buttons"] = bool(self.buttons_row.visible)
        # Save config changes.
        self.save_config()

    def log(self, text, avatar=None, key=None):
        """ADD A MESSAGE TO THE PAGE (can specify an avatar)"""
        # Check avatar, then append content to avatar with a certain alignment, then send it
        message_content = ft.Text(value=text, selectable=True)
        if avatar == "user" or avatar == "attachment":
            # Make a chat bubble with white background and rounded corners
            bubble = ft.Container(
                content=message_content,
                padding=ft.padding.symmetric(vertical=8, horizontal=15),
                border_radius=ft.border_radius.all(18),
                bgcolor=ft.Colors.with_opacity(0.1, ft.Colors.WHITE))
            # To fix the strange formatting, need a container:
            message_container = ft.Container(content=bubble, expand=True, alignment=ft.alignment.center_right, padding=ft.padding.only(left=50))
            avatar = self.user_avatar if avatar == "user" else self.attachment_avatar
            message_row = ft.Row([message_container, avatar], alignment=ft.MainAxisAlignment.END, key=key)
            self.chat_list.controls.append(message_row)
        elif avatar == "ai":
            # This is currently unused, but kept for potential future use.
            # To fix the strange formatting, need a container:
            message_container = ft.Container(content=message_content, expand=True, alignment=ft.alignment.center_left, padding=ft.padding.only(right=50))
            message_row = ft.Row([self.ai_avatar, message_container], key=key)
            self.chat_list.controls.append(message_row)
        else:
            # If no avatar is specified, message is sent with no avatar, just a boring grey log statement.
            message_container = ft.Container(content=message_content, expand=True, alignment=ft.alignment.center_left, key=key)
            self.chat_list.controls.append(message_container)
            self.chat_list.scroll_to(offset=-1, duration=300)
        # Update page
        self.page.update()

    def initialize_backend(self):
        self.log("Initializing backend...")
        # Multithreading support, allows backend to load in background so the user can still click around and stuff (no freeze)
        thread = threading.Thread(target=self.backend_worker, daemon=True)
        thread.start()

    def backend_worker(self):
        try:
            # Disable all of the buttons that would break the page if clicked on while initializing the backend
            self.reauthorize_button.disabled = True  # Backend already checks for authorization, so no point to do it during as well
            self.ai_mode_checkbox.disabled = True  # It can be confusing to do this at the same time, so this is disabled
            self.sync_directory_btn.disabled = True  # Sync requires embedders & drive service
            self.reload_backend_btn.disabled = True  # Self explanatory why this is disabled
            self.send_button.disabled = True  # Sending a message requires embedders
            self.attach_button.disabled = True  # Although just .gdoc files require backend, all attachments are disabled
            self.show_settings_btn.disabled = True  # Prevent toggling buttons during init
            self.overlay.visible = True  # Add animated loading indicator
            self.page.update()
            # All imports are done within functions to speed up initial page loading times.
            from SecondBrainBackend import machine_setup, load_config
            # Load config
            self.config = load_config(self.config_path)
            # AI Mode toggle is saved and stored in config.json, then reloaded here.
            self.ai_mode_checkbox.value = self.config.get("ai_mode", True)
            # Same goes for button row visibility
            self.buttons_row.visible = self.config.get("show_buttons", True)
            # Update the visual
            self.page.update()
            # LOAD MAJOR SERVICE, TEXT SPLITTER, MODELS, and COLLECTIONS here
            self.drive_service, self.text_splitter, self.models, self.collections = machine_setup(self.config, self.log)
            self.log("Successfully initialized backend.")
            # Load language model based on ai_mode checkbox value, which was retreived from config.json
            if self.ai_mode_checkbox.value == True:
                # Load the language model
                self.load_llm()
            else:
                # If the toggle value said not to load a language model, then it's safe to enable these buttons again. Otherwise, they need to stay off and will be turned on after the language model has loaded.
                self.ai_mode_checkbox.disabled = False
                self.overlay.visible = False
                self.send_button.disabled = False
        except Exception as e:
            self.log(f"Backend initialization failed: {e}")
        finally:
            # Re-enable disabled buttons, since it is now safe to do so
            self.reauthorize_button.disabled = False
            self.sync_directory_btn.disabled = False
            self.show_settings_btn.disabled = False
            self.reload_backend_btn.disabled = False
            self.attach_button.disabled = False
            self.page.update()

    def load_llm(self):
        self.log("Loading language model...")
        # Again, threading... IN PARALLEL!
        thread = threading.Thread(target=self.llm_worker, daemon=True)
        thread.start()
    
    def llm_worker(self):
        try:
            # If these were already disabled from initializing the backend, it doesn't hurt to disable them again.
            self.ai_mode_checkbox.disabled = True  # Can't click while it itself is not done
            self.overlay.visible = True  # Loading symbol
            self.send_button.disabled = True  # Sending messages during loading would be messy
            self.page.update()
            # Fill the LLM BACKEND VARIABLE based on the user's backend preferences in config
            # LM STUDIO!
            if self.config.get("llm_backend") == "LM Studio":
                self.llm = LMStudioLLM(model_name=self.config['lms_model_name'])
                self.llm_vision = self.llm.model.get_info().vision
                if self.llm_vision:
                    self.log(f"Model has vision support.")
                else:
                    self.log(f"Model does not have vision support.")
                self.log(f"Language model ({self.config['lms_model_name']}) successfully loaded.")
            # OPENAI!
            elif self.config.get("llm_backend") == "OpenAI":
                # Check for API key:
                api_key=self.config.get('openai_api_key', "")
                # If blank, key comes from environmental variable
                if not api_key:
                    api_key = os.environ.get("OPENAI_API_KEY")
                    if not api_key:
                        self.log("Could not find OPENAI_API_KEY (environmental variable)")
                self.llm = OpenAILLM(model_name=self.config['openai_model_name'], api_key=api_key)
                # Check for vision; not as straightforward
                model_name_lower = self.llm.model_name.lower()
                openai_vision_keywords = ["vision", "gpt-4o", "gpt-5", "gpt-4.1", "o3", "turbo"]
                self.llm_vision = any(keyword in model_name_lower for keyword in openai_vision_keywords)
                if self.llm_vision:
                    self.log(f"Model has vision support.")
                else:
                    self.log(f"Model does not have vision support.")
                self.log(f"Language model ({self.config['openai_model_name']}) successfully loaded.")
            else:
                raise ValueError("Unknown LLM backend")
        except Exception as e:
            self.log(f"Loading the LM failed: {e}")
        finally:
            # Reenable disabled buttons & hide loading screen
            self.ai_mode_checkbox.disabled = False
            self.overlay.visible = False
            self.send_button.disabled = False
            self.page.update()

    def start_sync_directory(self, e=None):
        # Ensure the user can cancel even if other stuff is going on
        if not hasattr(self, "cancel_sync_event"):
            # Listen for cancels; will be tossed into the sync function (sorry I don't really understand how it works, but it does.)
            self.cancel_sync_event = threading.Event()

        if not getattr(self, "sync_running", False):
            self.cancel_sync_event.clear()
            # Flip sync indicator
            self.sync_running = True
            # When a sync is running, the user has an option to cancel it. This is done by changing the icon symbol and tooltip, and allowing the button to send a cancel event while the thread is running in the background.
            self.sync_directory_btn.icon = ft.Icons.CLOSE_ROUNDED
            self.sync_directory_btn.text = "Cancel Sync"
            self.reload_backend_btn.disabled = True  # Can't reload backend while syncing
            self.ai_mode_checkbox.disabled = True  # Can't reasonably load/unload LLMs while syncing
            self.reauthorize_button.disabled = True  # You wouldn't want to reauth during a sync
            self.send_button.disabled = True  # Can't send messages while syncing
            # Signify START OF SYNC:
            self.chat_list.controls.append(ft.Divider())
            self.page.update()
            # Start sync background thread
            thread = threading.Thread(target=self.sync_worker, daemon=True)
            thread.start()
        else:
            # Request cancel
            self.sync_directory_btn.disabled = True   # temporarily disable so user can’t spam
            self.cancel_sync_event.set()
            self.page.update()

    def sync_worker(self):
        try:
            # The function itself + helper
            from SecondBrainBackend import sync_directory, is_connected
            # Network test for Google Drive (very finnicky)
            if not os.path.exists("token.json") and is_connected():
                from SecondBrainBackend import get_drive_service
                # Reload drive service if token.json is missing
                self.drive_service = get_drive_service(self.log, self.config)
            # Start the sync itself! Pass cancel_sync_event
            sync_directory(self.drive_service, self.text_splitter, self.models, self.collections, self.config, cancel_event=self.cancel_sync_event, log_callback=self.log)
        except Exception as e:
            self.log(f"Sync failed: {e}")
        finally:
            # After the sync is done, reset to starting positions.
            self.sync_running = False
            self.reload_backend_btn.disabled = False
            self.ai_mode_checkbox.disabled = False
            self.sync_directory_btn.text = "Sync Directory"
            self.sync_directory_btn.icon = ft.Icons.SYNC_ROUNDED
            self.sync_directory_btn.disabled = False
            self.reauthorize_button.disabled = False
            self.send_button.disabled = False
            # Signify end of sync
            self.chat_list.controls.append(ft.Divider())
            self.page.update()

    def save_config(self):
        import json
        # Replaces current config file with the one in memory
        with open(self.config_path, "w") as f:
            json.dump(self.config, f, indent=4)

    def _handle_image_search(self, sd, results_column):
        """Handles the text-to-image and image-to-image search logic."""
        from SecondBrainBackend import hybrid_search
        image_queries = []
        if self.config['ai_mode'] and self.config['query_multiplier'] >= 1:
            image_queries.extend(self.llm_generate_queries(sd, "image", self.config['query_multiplier']))
        if sd['attachment_chunks']:
            image_queries.extend(sd['attachment_chunks'])
        if sd['msg']:
            image_queries.append(sd['msg'])
        if sd['attached_image']:  # For image-to-image searches
            image_queries.append(sd['attached_image'])
        # Perform the search
        sd['image_search_results'] = hybrid_search(image_queries, sd, self.models, self.collections, self.config, "image")
        # Filter out bad results with LLM, image by image
        if sd['image_search_results'] and self.config['ai_mode'] and self.llm_vision and self.config['llm_filter_results']:
            sd['image_search_results'] = [r for r in sd['image_search_results'] if self.llm_evaluate_image_relevance(r['file_path'], sd)]
        # Display results
        sd['image_paths'] = []
        if sd['image_search_results']:
            sd['image_paths'] = [result['file_path'] for result in sd['image_search_results']]
            image_row = self.image_presentation(sd['image_paths'])
            if sd['attached_image'] and not (sd['msg'] or sd['attachment_chunks']):
                results_column.controls.append(ft.Row([ft.Text("SIMILAR IMAGES", size=12, weight=ft.FontWeight.BOLD)], alignment=ft.MainAxisAlignment.CENTER))
            else:
                results_column.controls.append(ft.Row([ft.Text("IMAGE RESULTS", size=12, weight=ft.FontWeight.BOLD)], alignment=ft.MainAxisAlignment.CENTER))
            results_column.controls.append(image_row)
        else:
            # If no image results, display a message to the user
            message_row = ft.Row([self.ai_avatar, ft.Markdown(value="No image results.", selectable=True, extension_set=ft.MarkdownExtensionSet.GITHUB_WEB)])
            results_column.controls.append(message_row)
        self.page.update()
        return sd

    def _handle_text_search(self, sd, results_column):
        """Handles the text-to-text and image-to-text search logic."""
        from SecondBrainBackend import hybrid_search
        text_queries = []
        if self.config['ai_mode'] and self.config['query_multiplier'] >= 1:
            text_queries.extend(self.llm_generate_queries(sd, "text", self.config['query_multiplier']))
        if sd['attachment_chunks']:
            text_queries.extend(sd['attachment_chunks'])
        if sd['msg']:
            text_queries.append(sd['msg'])
        if not text_queries:  # To enable image to text searches in event of no text input
            text_queries.append(sd['lexical_search_term'])
        # Some embedding models want a prefix for text searches, like BAAI/bge-large-en-v1.5
        prefixed_text_queries = [self.config['text_search_prefix'] + q for q in text_queries]
        sd['text_search_results'] = hybrid_search(prefixed_text_queries, sd, self.models, self.collections, self.config, "text")
        # Filter out bad results with LLM, chunk by chunk
        if sd['text_search_results'] and self.config['ai_mode'] and self.config['llm_filter_results']:
            sd['text_search_results'] = [r for r in sd['text_search_results'] if self.llm_evaluate_text_relevance(['documents'], sd)]
        # Display results
        if sd['text_search_results']:
            results_table = self.results_table(sd['text_search_results'])
            if sd['image_search_results']:
                results_column.controls.append(ft.Divider())
            results_column.controls.append(ft.Row([ft.Text("TEXT RESULTS", size=12, weight=ft.FontWeight.BOLD)], alignment=ft.MainAxisAlignment.CENTER))
            results_column.controls.append(results_table)
            self.page.update()
            # self.chat_list.scroll_to(offset=-1, duration=300)
        else:
            # If no text results, display a message to the user
            message_row = ft.Row([self.ai_avatar, ft.Markdown(value="No text results.", selectable=True, extension_set=ft.MarkdownExtensionSet.GITHUB_WEB)])
            results_column.controls.append(message_row)
        self.page.update()
        return sd
    
    def _handle_ai_insights(self, sd, results_column):
        if self.config['ai_mode'] and ((sd['image_search_results'] and self.llm_vision) or sd['text_search_results']):
            results_column.controls.append(ft.Divider())
            results_column.controls.append(ft.Row([ft.Text(f"AI INSIGHTS\n({self.llm.model_name})", size=12, weight=ft.FontWeight.BOLD, text_align=ft.TextAlign.CENTER)], alignment=ft.MainAxisAlignment.CENTER))
            # Ask LLM to synthesize output (hopefully it does a good job)
            self.llm_synthesize_results(sd, target_column=results_column)
    
    def send_message(self, e=None):
        """Handles sending a message, processing attachments, searching, and displaying search results. Coordinates some very complex LLM prompts. Initializes search_dict (sd), which contains:
        msg - user input message string. May be empty ("")
        attachment - text for text attachments, "[IMAGE]" for image attachments, or None for no attachment
        attachment_path - Path object
        attachment_size - size in tokens of the attachment
        attachment_chunks - list of chunks from do_attachment_RAG()
        attachment_context_string - Used for LLM context. Contains either the entire attachment text (if small enough) or the extracted chunks (if too large), a statement that says the user attached an image, or empty if no attachment.
        attached_image_description - string, self explanatory
        attached_image - PIL Image object
        attached_image_path - string path to attached image file (not Path object)
        attachment_name - name of attachment file
        lexical_search_term - used for lexical search
        image_search_results - nested dictionary for image hybrid_search() results
        image_paths - list of image file paths from search results
        text_search_results - nested dictionary for text hybrid_search() results
        """
        sd = {}
        sd['msg'] = self.user_input.value.strip()
        # If no message, do nothing.
        if not sd['msg'] and not self.attachment_data:
            return
        # This is needed to block the enter key way of sending a message
        if self.send_button.disabled:  # block both button click & Enter key
            return

        # --- Initial Setup ---
        sd['attachment'] = self.attachment_data
        sd['attachment_path'] = self.attachment_path
        sd['attachment_size'] = self.attachment_size
        # Clear attachment right at the start, no particular reason why
        self.remove_attachment()
        # Change send button to loading icon, saving the original icon for when it's reset.
        original_icon = self.send_button.icon
        self.send_button.content = ft.ProgressRing(width=16, height=16, stroke_width=2)
        self.send_button.icon = None
        self.send_button.disabled = True
        self.ai_mode_checkbox.disabled = True
        self.sync_directory_btn.disabled = True
        self.reload_backend_btn.disabled = True
        self.reauthorize_button.disabled = True

        import uuid
        scroll_key = str(uuid.uuid4())
        # We'll pass this key to the *first* message we log.
        first_log_key = scroll_key

        if sd['attachment']:
            self.log(f"{sd['attachment_path'].name}", avatar="attachment", key=first_log_key)
            first_log_key = None
        if sd['msg']:
            self.log(f"{sd['msg']}", avatar="user", key=first_log_key)

        self.user_input.value = ""
        self.page.update()

        # --- Results UI Setup ---
        results_column = ft.Column(spacing=10)
        results_container = ft.Container(
            content=results_column,
            bgcolor=ft.Colors.with_opacity(0.03, ft.Colors.WHITE),
            border_radius=10,
            padding=25,
            visible=True)
        self.chat_list.controls.append(results_container)

        # --- Attachment Logic ---
        sd['attachment_chunks'] = []
        sd['attachment_context_string'] = ""  # For LLM context; given as part of prompt
        sd['attached_image_description'] = ""
        sd['attached_image'] = None
        sd['attached_image_path'] = ""  # For LLM context; image is given as input to vision models
        if sd['attachment']:
            # For text attachments:
            if sd['attachment'] != "[IMAGE]":
                # If larger than maximum, use extracted chunks as context, else, use entire attachment
                sd['attachment_chunks'] = self.do_attachment_RAG(sd['attachment'], [sd['msg']])
                if sd['attachment_size'] > self.config['max_attachment_size']:
                    sd['attachment_context_string'] = "\n\n".join(sd['attachment_chunks'])
                else:
                    sd['attachment_context_string'] = sd['attachment']
            # For image attachments:
            elif sd['attachment'] == "[IMAGE]":
                sd['attached_image_path'] = str(sd['attachment_path'])
                # Get the Image object for embedding for labels and during the actual search. This code embeds the same image twice, which is unnecesary but not that much slower.
                from PIL import Image
                try:
                    with Image.open(str(sd['attachment_path'])).convert("RGB") as img:
                        sd['attached_image'] = img
                except Exception as e:
                    print(f" [Warning] Failed to load image {str(sd['attachment_path'])}: {e}")
                # Find description of image for search purposes, here with LLM if possible
                if self.config['ai_mode'] and self.llm_vision:
                    sd['attached_image_description'] = self.llm.invoke(f"Provide a concise description of the content of the attached image for the purpose of searching a personal knowledge base. Do not include the file name in your description.", temperature=0.5, sd=sd)
                # If no LLM with vision support, find labels with image embedding
                else:
                    # Embed the image to find labels
                    image_embedding = self.models['image'].encode(sd['attached_image'], convert_to_numpy=True, batch_size=self.config['batch_size'], normalize_embeddings=True)
                    # Find a label for the image to aid in the lexical search part
                    label_results = self.collections['image'].query(query_embeddings=[image_embedding], n_results=3, where={"type": "label"}, include=["documents"])
                    labels = [""]
                    # Make sure result exists
                    if label_results and label_results.get("documents") and label_results["documents"][0]:
                        labels = [label.lower() for label in label_results["documents"][0]]
                    # Include path and labels in lexical search term
                    sd['attached_image_description'] = ", ".join(labels)
                
        # --- Search Logic ---
        # Assemble search parts (lexical search term)
        sd['attachment_name'] = sd['attachment_path'].name if sd['attachment_path'] else ""
        sd['attachment_folder'] = sd['attachment_path'].parent.name if sd['attachment_path'] else ""
        sd['lexical_search_term'] = f"{sd['msg']} {sd['attached_image_description']} {sd['attachment_context_string']} {sd['attachment_name']} {sd['attachment_folder']}".strip()
        # More context. Doing this part after making the lexical search term so it isn't polluted with irrelevant words:
        if sd['attachment']:
            if sd['attachment'] != "[IMAGE]":
                sd['attachment_context_string'] += f"\nAttachment name: {sd['attachment_path'].name}"  # Extra context for LLM
            elif sd['attachment'] == "[IMAGE]":
                attachment_context_string = f"The user has attached an image: {sd['attachment_path'].name}"  # This copies the formatting found in the LLM class for the image prompt. It essentially lets the LLM know which image is the attachment.
        # parts of image_search_results and image_paths will be used for llm_synthesize_results
        try:
            sd = self._handle_image_search(sd, results_column)
        except Exception as e:
            results_column.controls.append(ft.Text(f"[ERROR] Image search failed: {e}", selectable=True))
            image_search_results, image_paths = [], []
        # parts of text_search_results will be used for llm_synthesize_results
        try:
            sd = self._handle_text_search(sd, results_column)
        except Exception as e:
            results_column.controls.append(ft.Text(f"[ERROR] Text search failed: {e}", selectable=True))
            text_search_results = []
        # llm_synthesize_results - LLM takes results and summarizes and gives insight on them
        try:
            self._handle_ai_insights(sd, results_column)
        except Exception as e:
            results_column.controls.append(ft.Text(f"[ERROR] AI insight generation failed: {e}", selectable=True))

        # --- Final Cleanup ---
        self.chat_list.scroll_to(key=scroll_key, duration=0)  # Scroll to user's message
        self.send_button.icon = original_icon
        self.send_button.content = None
        self.send_button.disabled = False
        self.ai_mode_checkbox.disabled = False
        self.sync_directory_btn.disabled = False
        self.reload_backend_btn.disabled = False
        self.reauthorize_button.disabled = False
        self.page.update()

    def open_config(self, e=None):
        # For the open settings button
        os.startfile(self.config_path)

    def reauthorize_drive(self, e=None):
        # For the reauthorize button
        if os.path.exists("token.json"):
            os.remove("token.json")
        from SecondBrainBackend import get_drive_service
        self.drive_service = get_drive_service(self.log, self.config)

    def toggle_ai_mode(self, e=None):
        # When clicking the AI Mode toggle, a function call is sent here after the value has flipped
        # Set the config value equal to the new value.
        self.config["ai_mode"] = self.ai_mode_checkbox.value
        # Save config changes.
        self.save_config()
        if self.ai_mode_checkbox.value:
            self.log("AI mode turned on")
            self.load_llm()
        else:
            self.log("AI mode turned off")
            self.unload_llm()

    def reload_backend(self, e=None):
        self.log("Reloading backend...")
        # Clear embedding models
        try:
            # Clear LLM
            if self.llm:
                self.unload_llm()
            # Unload embedders
            if self.models:
                for name, model in self.models.items():
                    try:
                        if hasattr(model, "cpu"):
                            model.cpu()
                        del model
                    except Exception as unload_err:
                        self.log(f"Failed to unload model '{name}': {unload_err}")
                import gc, torch
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            self.log("Successfully unloaded models.")
        except Exception as e:
            self.log(f"Cleanup before reload failed: {e}")
        self.config = {}
        self.drive_service = None
        self.text_splitter = None
        self.models = {}
        self.collections = {}
        self.merged_graph = None
        self.graph_collection = None
        self.initialize_backend()

    def open_settings(self, e=None):
        self.settings.visible = True
        self.page.update()

    def close_settings(self, e=None):
        self.settings.visible = False
        self.page.update()

    def stream_llm_response(self, prompt: str, sd: dict, target_column: ft.Column = None):
        """Handles the UI updates for a streaming LLM response.
        If target_column is provided, it adds the output there.
        Otherwise, it creates a new container in the chat list.
        """
        if not self.llm_vision:
            sd['image_paths'] = None
        # Make a markdown box for the AI
        ai_response_text = ft.Markdown(
            value="▌",  # This creates a cool cursor effect, and will move as the AI streams.
            selectable=True, 
            extension_set=ft.MarkdownExtensionSet.GITHUB_WEB,
            on_tap_link=lambda e: self.page.launch_url(e.data))
        # Create the save insight button so that the user can choose to save a response so that the AI can access it later.
        save_button = ft.IconButton(
            icon=ft.Icons.BOOKMARK_ADD_OUTLINED,
            tooltip="Save this insight",
            visible=False, # Initially hidden
            on_click=lambda e: self.save_insight(
                insight_text=ai_response_text.value,
                original_query=sd['msg'],
                image_paths=sd['image_paths'], 
                text_paths=[r['file_path'] for r in sd['text_search_results']],
                button_to_update=e.control))
        # We wrap the Markdown and the button in a Column so they appear vertically.
        response_content_column = ft.Column([ai_response_text])
        # # Message row structure with the content column
        message_row = ft.Row([response_content_column], vertical_alignment=ft.CrossAxisAlignment.START, wrap=True)
        
        # This is the key part: decide WHERE to place the output. Finding the results_column from send_message is the tricky part.
        target_column.controls.append(message_row)
        # Add button below (starts invisible)
        target_column.controls.append(ft.Row([save_button], alignment=ft.MainAxisAlignment.CENTER))
        self.page.update()

        full_response = ""
        try:
            # [STREAMING] The streaming logic itself, using the "yield" thing
            for chunk in self.llm.stream(prompt, 0.6, sd):
                if not chunk:
                    continue
                full_response += chunk
                ai_response_text.value = full_response + " ▌"  # Move the cursor
                self.page.update()
            # Insert full response after streaming is done.
            ai_response_text.value = str(full_response)
        except Exception as e:
            ai_response_text.value = f"Error streaming response: {e}"
        finally:  # Now that the stream is complete, make the save button visible.
            if full_response.strip(): # Only show if there is content to save
                 save_button.visible = True
            self.page.update()
        return full_response

    def save_insight(self, insight_text, original_query, image_paths, text_paths, button_to_update: ft.IconButton):
        """Runs the backend function to save an insight to a .txt file."""
        try:
            # Disable the button to prevent multiple clicks while saving
            button_to_update.disabled = True
            self.page.update()

            from SecondBrainBackend import save_insight_to_file
            safe_image_paths = image_paths if image_paths is not None else []
            safe_text_paths = text_paths if text_paths is not None else []
            save_insight_to_file(
                insight_text=insight_text,
                original_query=original_query,
                image_paths=safe_image_paths,
                text_paths=safe_text_paths,
                config=self.config, # Pass the config to find the directory
                log_callback=self.log
            )
        except Exception as e:
            self.log(f"Failed to save insight in backend: {e}")
        finally:
            button_to_update.icon = ft.Icons.BOOKMARK_ADDED_ROUNDED # Change icon to show it's saved
            self.page.update()

    def unload_llm(self):
        if self.llm:
            # Add loading screen, disable buttons, etc.
            self.overlay.visible = True
            self.send_button.disabled = True
            self.ai_mode_checkbox.disabled = True
            self.page.update()
            import time
            # If the unload is really fast, the button flickers in a way that looks bad.
            time.sleep(0.05)
            try:
                self.log(f"Unloading {self.llm.__class__.__name__}...")
                self.llm.unload()
                self.llm = None
                self.llm_vision = None
                self.log("Language model unloaded successfully.")
            except Exception as e:
                self.log(f"Error unloading model: {e}")
            # Re-enable.
            self.overlay.visible = False
            self.send_button.disabled = False
            self.ai_mode_checkbox.disabled = False
            self.page.update()

    def attach_files(self, e=ft.FilePickerResultEvent):
        # This function is called by the file picker Flet thing
        if not e.files:
            # If no files, do nothing.
            return
        # Disable attach button while loading attachment.
        self.attach_button.disabled = True
        self.page.update()
        try:
            # Get file path from passed variable
            file_path = Path(e.files[0].path)
            # Store file path, then trigger function to get its data
            self.attachment_path = file_path
            self.attachment_data, self.attachment_size = self.process_attachment(file_path)
            # Toggle attachment display and make it say the attachment name
            self.attachment_display.visible = True
            if self.attachment_data != "[IMAGE]":
                self.attachment_display.value = f"Attached file: {file_path.name} ({self.attachment_size} tokens)"
            else:
                self.attachment_display.value = f"Attached file: {file_path.name}"
            # When done, reenable button
            self.attach_button.disabled = False
            # Change button to X and tooltip to filename, and function to remove attachment
            self.attach_button.icon = ft.Icons.CLOSE_ROUNDED
            self.attach_button.tooltip = "Remove attachment"
            self.attach_button.on_click = lambda _: self.remove_attachment()
        except Exception as e:
            self.log(f"Loading attachment failed: {e}")
            # Reset so that the user isn't stuck with a broken attachment
            self.remove_attachment()
            self.attach_button.disabled = False
        self.page.update()

    def remove_attachment(self):
        # Clear attachment data
        self.attachment_data = None
        self.attachment_size = None
        self.attachment_path = None
        self.attachment_display.value = ""
        self.attachment_display.visible = False
        # Reset button
        self.attach_button.icon = ft.Icons.ATTACH_FILE_ROUNDED
        self.attach_button.tooltip = "Attach file"
        self.attach_button.on_click = lambda _: self.file_picker.pick_files(allow_multiple=False)
        self.page.update()

    def process_attachment(self, path: Path):
        """So this part is actually pretty clever."""
        # All we have to do to process the attachments is import the functions that were already built to do it in the backend.
        from SecondBrainBackend import parse_docx, parse_gdoc, parse_pdf, parse_txt, file_handler
        # Use the same logic as before
        handler = file_handler(path.suffix, True)
        if not handler:
            self.log("Invalid attachment type.")
            return
        # Then pull out the data.
        content = handler(path, self.drive_service, self.log) if handler == parse_gdoc else handler(path)
        if not content:
            return
        # Check attachment size in tokens, if the attachment is not an image:
        if content != "[IMAGE]":
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(self.config['text_model_name'])
            attachment_size = len(tokenizer.encode(content, add_special_tokens=False))
        else:
            attachment_size = 0
        
        return content, attachment_size
    
    def set_attachment_from_path(self, path_str: str):
        """Helper function to add a result as an attachment."""
        try:
            # Clear any existing attachment first
            self.remove_attachment()
            file_path = Path(path_str)
            if not file_path.exists():
                self.log(f"Error: File not found at {file_path}")
                return
            # Use your existing processing logic
            self.attachment_path = file_path
            self.attachment_data, self.attachment_size = self.process_attachment(file_path)
            # Update the UI just like attach_files does
            self.attachment_display.visible = True
            if self.attachment_data != "[IMAGE]":
                self.attachment_display.value = f"Attached file: {file_path.name} ({self.attachment_size} tokens)"
            else:
                self.attachment_display.value = f"Attached file: {file_path.name}"
            self.attach_button.icon = ft.Icons.CLOSE_ROUNDED
            self.attach_button.tooltip = "Remove attachment"
            self.attach_button.on_click = lambda _: self.remove_attachment()
        except Exception as e:
            # self.log(f"Failed to attach result: {e}")
            ...
        finally:
            self.page.update()
    
    def copy_path_to_clipboard(self, path_str: str):
        """Copies the given string to the clipboard and logs a confirmation."""
        try:
            self.page.set_clipboard(str(path_str))
            # self.log(f"Copied path to clipboard.")
        except Exception as e:
            self.log(f"Failed to copy to clipboard: {e}")

    def image_presentation(self, image_paths: list[str]):
        # Takes a list of paths and returns a Flet container object with a row of scrollable images.
        image_preview_size = 160
        image_widgets = []
        for path in image_paths:
            # 1. The image itself
            image_preview = ft.Image(
                src=path, 
                width=image_preview_size, 
                height=image_preview_size, 
                fit=ft.ImageFit.CONTAIN,
                border_radius=10
            )
            
            # 2. The PopupMenuButton that *contains* the image
            image_menu_button = ft.PopupMenuButton(
                content=image_preview, # <-- Image is the button content
                tooltip=f"Click for options: {path}",
                items=[
                    ft.PopupMenuItem(
                        text="Open File", 
                        icon=ft.Icons.OPEN_IN_NEW_ROUNDED,
                        on_click=lambda _, p=path: self.page.launch_url(f"file:///{Path(p)}")
                    ),
                    ft.PopupMenuItem(
                        text="Open File Location",
                        icon=ft.Icons.FOLDER_OPEN_ROUNDED,
                        on_click=lambda _, p=path: os.startfile(Path(p).parent)
                    ),
                    ft.PopupMenuItem(
                        text="Attach File", 
                        icon=ft.Icons.ATTACH_FILE_ROUNDED,
                        on_click=lambda _, p=path: self.set_attachment_from_path(p)
                    ),
                    ft.PopupMenuItem(
                        text="Copy Path",
                        icon=ft.Icons.COPY_ROUNDED,
                        on_click=lambda _, p=path: self.copy_path_to_clipboard(p)
                    ),
                ]
            )

            image_widgets.append(image_menu_button)
        image_row = ft.Row(controls=image_widgets, spacing=10, scroll=ft.ScrollMode.AUTO)
        return image_row
    
    def toggle_image_overlay(self, e, overlay_container):
        """Called by on_hover to show or hide the image overlay."""
        overlay_container.visible = e.data == "true"
        self.page.update()

    def results_table(self, text_results):
        # Given a list of text results, returns a Flet "expansion tile" object with the file names as a title that expand to show the text content in question. Also provides a link that opens the file with the file explorer.
        import re
        results_widgets = []
        for r in text_results:            
            # Remove prefix from embedded chunk and add ellipsis as this is part of a larger document...
            cleaned_chunk = re.sub(r'^<Source: .*?>\s*', '', r['documents'])
            # if not cleaned_chunk.endswith(('.', '?', '!', '…')):
            #     cleaned_chunk += '…'
            # Create the tile object
            menu = ft.PopupMenuButton(
                icon=ft.Icons.GRID_VIEW_OUTLINED,
                tooltip="Options",
                items=[
                    ft.PopupMenuItem(
                        text="Open File", 
                        icon=ft.Icons.OPEN_IN_NEW_ROUNDED,
                        on_click=lambda _, p=r['file_path']: self.page.launch_url(f"file:///{Path(p)}")
                    ),
                    ft.PopupMenuItem(
                        text="Open File Location",
                        icon=ft.Icons.FOLDER_OPEN_ROUNDED,
                        on_click=lambda _, p=r['file_path']: os.startfile(Path(p).parent)
                    ),
                    ft.PopupMenuItem(
                        text="Attach File", 
                        icon=ft.Icons.ATTACH_FILE_ROUNDED,
                        on_click=lambda _, p=r['file_path']: self.set_attachment_from_path(p)
                    ),
                    ft.PopupMenuItem(
                        text="Copy Path",
                        icon=ft.Icons.COPY_ROUNDED,
                        on_click=lambda _, p=r['file_path']: self.copy_path_to_clipboard(p)
                    ),
                ]
            )
        
            tile = ft.ExpansionTile(
                # We show the path in the title now since the button is gone
                title=ft.Text(Path(r['file_path']).stem),
                subtitle=ft.Text(r['file_path'], size=10, italic=True), # Optional: show path here
                controls=[
                    ft.ListTile(
                        # The cleaned chunk is the main content, put it in a special box
                        # Could add f" | Result type: {r['result_type']}"
                        subtitle=ft.Container(
                            content=ft.Text(cleaned_chunk, selectable=True),
                            border=ft.border.all(0.5, ft.Colors.WHITE),
                            bgcolor=ft.Colors.with_opacity(0.33, ft.Colors.BLACK),
                            padding=5,          # Optional: gives the text some space
                            border_radius=7   # Optional: rounds the corners
                        ),
                        trailing=menu
                    )
                ]
            )
            results_widgets.append(tile)
        results = ft.Column(controls=results_widgets, spacing=0)
        return results

    def llm_generate_queries(self, sd, query_type, n=3) -> list[str]:
        # --- Context Preparation ---
        attachment_context = f"CONTEXT FROM ATTACHMENT:\n{sd['attachment_context_string']}\n" if sd['attachment_context_string'] else ""
        user_request = f"USER'S REQUEST:\n'{sd['msg']}'\n" if sd['msg'] else "USER'S REQUEST: The user did not provide a specific prompt; focus on their attachment.\n"
        content = "documents" if query_type == "text" else "images"
        reminder_suffix = " Image search is different from text search, so make sure that the queries are optimized for images." if query_type == "image" else ""
        
        # --- Refined Prompt ---
        prompt = f"""{self.config['system_prompt']}

{user_request}
{attachment_context}
Based on the user's prompt, generate {n} creative search queries that could retrieve relevant {content} to answer the user. These queries will go into a semantic search algorithm to retreive relevant {content} from the user's database. The queries should be broad enough to find a variety of related items. These queries will search a somewhat small and personal database (that is, the user's hard drive). Respond with a plain list with no supporting text or markdown.{reminder_suffix}"""
        # image_paths_proxy = [sd['attached_image_path']] if sd['attached_image_path'] else None
        response = self.llm.invoke(prompt=prompt, temperature=0.7, sd=sd).strip()
        
        # --- Cleaning Results ---
        query_list = [q.strip("-• ").strip() for q in response.splitlines() if q.strip()]
        cleaned_text_queries = [item.lstrip('\'\"0123456789. *').rstrip('\'\"') for item in query_list]  # Strips leading numbers, periods, and spaces. The AI tends to number the list.
        print(f"Generated {query_type} queries: {cleaned_text_queries}")
        return cleaned_text_queries

    def llm_evaluate_text_relevance(self, chunk, sd) -> bool:
        # --- Context Preparation ---
        attachment_context = f"CONTEXT FROM ATTACHMENT:\n{sd['attachment_context_string']}\n" if sd['attachment_context_string'] else ""
        user_request = f"USER'S REQUEST:\n'{sd['msg']}'\n" if sd['msg'] else "USER'S REQUEST: The user did not provide a specific prompt; focus on their attachment.\n"
        
        # --- Refined Prompt ---
        prompt = f"""{self.config['system_prompt']}

{attachment_context}
{user_request}
Document excerpt to evaluate:
"{chunk}"

Is this excerpt worth keeping? Respond only with YES or NO.

Relevance is the most important thing. Does the snippet connect to the user's request?

If the excerpt is gibberish, respond with NO.

(Again: respond only with YES or NO.)"""
        # This might be a crude method for analyzing sentiment, but honestly it works really well.
        # image_paths_proxy = [sd['attached_image_path']] if sd['attached_image_path'] else None
        response = self.llm.invoke(prompt=prompt, temperature=0.01, sd=sd).strip().lower()  # Low temp for objectivity
        
        # --- Getting Boolean ---
        if "yes" in response.lower() or "no" in response.lower():
            return "yes" in response
        else:
            return True

    def llm_evaluate_image_relevance(self, image_path, sd) -> bool:
        # --- Context Preparation ---
        attachment_context = f"CONTEXT FROM ATTACHMENT:\n{sd['attachment_context_string']}\n" if sd['attachment_context_string'] else ""
        user_request = f"USER'S REQUEST:\n'{sd['msg']}'\n" if sd['msg'] else "USER'S REQUEST: The user did not provide a specific prompt; focus on their attachment.\n"
        
        # --- Refined Prompt ---
        prompt = f"""{self.config['system_prompt']}

{attachment_context}
{user_request}
Is the provided image worth keeping? Respond only with YES or NO.

Relevance is the most important thing. Does the photo connect to the user's request?

If the image is blank, corrupted, or unreadable, respond with NO.

Image file path: {image_path}

If the user's query has an exact match within the file path, respond with YES.

(Again: respond only with YES or NO.)"""
        # The image handle creation happens transparently inside _prepare_chat
        # if sd['attached_image_path']:
        #     image_paths_proxy = [image_path].append(sd['attached_image_path'])  # Attached image goes last.
        # else:
        #     image_paths_proxy = [image_path]
        response = self.llm.invoke(prompt=prompt, temperature=0.01, sd=sd).strip().lower()

        # --- Getting Boolean ---
        if "yes" in response.lower() or "no" in response.lower():
            return "yes" in response
        else:
            return True

    def llm_synthesize_results(self, sd, target_column=None):
        # --- Context Preparation ---
        from datetime import datetime
        date_time = datetime.now().strftime("%#I:%M %p, %d %B %Y")
        attachment_context = f"CONTEXT FROM ATTACHMENT:\n{sd['attachment_context_string']}\n" if sd['attachment_context_string'] else ""
        user_request = f"USER'S REQUEST:\n'{sd['msg']}'" if sd['msg'] else "USER'S REQUEST: The user did not provide a specific prompt; focus on their attachment.\n"
        
        relevant_chunks = [r['documents'] for r in sd['text_search_results']]
        formatted_chunks = f"{"\n---\n".join(relevant_chunks)}" if relevant_chunks else "No text results found; focus on the images."
        database_results = f"DATABASE SEARCH RESULTS:\n{formatted_chunks}\n"

        # --- Refined Prompt ---
        prompt = f"""{self.config['system_prompt']}

It is {date_time}.

{user_request}
{attachment_context}
{database_results}
**Your Task:**
Based exclusively on the information provided above, write a concise and helpful response. Your primary goal is to synthesize the information to **guide the user towards what they want**.

**Instructions:**
- The text search results are **snippets** from larger documents and may be incomplete.
- Do **not assume or guess** the author of a document unless the source text makes it absolutely clear.
- The documents don't have timestamps; don't assume the age of a document unless the source text makes it absolutely clear.
- Cite every piece of information you use from the search results with its source, like so: (source_name).
- If the provided search results are not relevant to the user's request, state that you could not find any relevant information.
- Use markdown formatting (e.g., bolding, bullet points) to make the response easy to read.
- If there are images, make sure to consider them for your response."""
        threading.Thread(
            target=self.stream_llm_response, 
            args=(prompt, sd, target_column),
            daemon=True).start()

    def do_attachment_RAG(self, attachment, queries):
        """Attachments are often too large to fit into the text embedding model's sequence length, so this function tries to pick the top n most relevant chunks of the attachment based on the queries - to use as additional queries."""
        n_attachment_chunks = self.config['n_attachment_chunks']
        import numpy as np
        # Split into chunks
        attachment_chunks = self.text_splitter.split_text(attachment)
        # Encode chunks
        attachment_embeddings = self.models['text'].encode(
            attachment_chunks,
            convert_to_numpy=True,
            batch_size=self.config['batch_size'],
            normalize_embeddings=True
        )
        # Encode queries (just one query, msg)
        query_embeddings = self.models['text'].encode(
            queries,
            convert_to_numpy=True,
            batch_size=self.config['batch_size'],
            normalize_embeddings=True
        )
        # Compute similarity: (num_queries x num_chunks)
        similarities = np.dot(query_embeddings, attachment_embeddings.T)
        # Flatten all scores across queries → chunks
        avg_scores = similarities.mean(axis=0)
        # Take top-N chunks
        top_idx = np.argsort(-avg_scores)[:n_attachment_chunks]
        top_chunks = [attachment_chunks[i] for i in top_idx]

        return top_chunks

def main(page: ft.Page):
    page.visible = False
    App(page)

ft.app(target=main, assets_dir="assets")