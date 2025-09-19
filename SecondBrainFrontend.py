import flet as ft
import threading
import os
from pathlib import Path

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

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

class LMStudioLLM(BaseLLM):
    # LM STUDIO's great library makes this a cakewalk.
    def __init__(self, model_name, endpoint="http://localhost:1234/v1"):
        import lmstudio as lms
        self.model = lms.llm(model_name)

    def invoke(self, prompt):
        return self.model.respond(prompt).content
    
    def stream(self, prompt):
        for fragment in self.model.respond_stream(prompt):
            yield fragment.content

    def unload(self):
        self.model.unload()

class GeminiLLM(BaseLLM):
    # This does not work; PLACEHOLDER
    def __init__(self, api_key: str, model="gemini-pro"):
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model)

    def invoke(self, prompt: str) -> str:
        response = self.model.generate_content(prompt)
        return response.text.strip()
    
    def stream(self, prompt: str):
        response = self.model.generate_content(prompt, stream=True)
        for chunk in response:
            if chunk.text:
                yield chunk.text

    def unload(self):
        del self.model

class App:
    def __init__(self, page: ft.Page):
        # Resizing, setting icon, and centering the page.
        page.window.resizable = True
        page.window.width = int(600*1.4)
        page.window.height = int(500*1.2)
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
        self.config_path = os.path.join(BASE_DIR, "config.json")
        self.config = {}
        self.drive_service = None
        self.text_splitter = None
        self.models = {}
        self.collections = {}
        self.llm = None
        # Message avatars
        self.user_avatar = ft.CircleAvatar(content=ft.Icon(ft.Icons.SEARCH), radius=18)
        self.ai_avatar = ft.CircleAvatar(content=ft.Icon(ft.Icons.WB_SUNNY_OUTLINED), radius=18)
        self.attachment_avatar = ft.CircleAvatar(content=ft.Icon(ft.Icons.ATTACH_FILE_ROUNDED), radius=18)
        # Input field
        self.user_input = ft.TextField(expand=True, multiline=True, label="Compose your message...", shift_enter=True, on_submit=self.send_message, min_lines=1, max_lines=7, max_length=1200)
        # Send button
        self.send_button = ft.IconButton(icon=ft.Icons.SEND_ROUNDED, on_click=self.send_message, tooltip="Send")
        # File picker & Attachment logic
        self.file_picker = ft.FilePicker(on_result=self.attach_files)
        self.page.overlay.append(self.file_picker)
        self.attach_button = ft.IconButton(icon=ft.Icons.ATTACH_FILE_ROUNDED, on_click=lambda _: self.file_picker.pick_files(allow_multiple=False, allowed_extensions=["txt", "pdf", "docx", "gdoc", "png", "jpeg", "jpg", "gif", "webp"]), tooltip="Attach file")  # EXTEND EXTENSIONS HERE
        self.attachment_data = None
        self.attachment_path = None
        # AI Mode checkbox, plus settings, reload backend, and sync buttons
        self.ai_mode_checkbox = ft.Checkbox(label="AI Mode", value=False, on_change=self.toggle_ai_mode)
        self.open_settings_btn = ft.ElevatedButton("Open Settings", icon=ft.Icons.SETTINGS_ROUNDED, on_click=self.open_config)
        self.reload_backend_btn = ft.ElevatedButton("Reload Backend", icon=ft.Icons.REFRESH_ROUNDED, on_click=self.reload_backend)
        self.sync_directory_btn = ft.ElevatedButton("Sync Directory", icon=ft.Icons.SYNC_ROUNDED, on_click=self.start_sync_directory)
        # Top row
        self.top_row = ft.Row([self.ai_mode_checkbox, self.open_settings_btn, self.reload_backend_btn, self.sync_directory_btn], alignment="spaceAround")
        # Chat list
        self.chat_list = ft.ListView(expand=True, spacing=10, auto_scroll=True)
        # Attachment display (starts invisible)
        self.attachment_display = ft.Text(visible=False, italic=True)
        # Bottom row
        self.bottom_row = ft.Row([self.user_input, self.attach_button, self.send_button], alignment="spaceBetween")
        # MAIN COLUMN
        main_layout = ft.Column(controls=[self.top_row, self.chat_list, self.attachment_display, self.bottom_row], expand=True)
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

    def log(self, text, avatar=None):
        """ADD A MESSAGE TO THE PAGE (can specify an avatar)"""
        # Turn into markdown here for italics, bold, blue links, etc.
        message_content = ft.Markdown(value=text, selectable=True, extension_set=ft.MarkdownExtensionSet.GITHUB_WEB, on_tap_link=lambda link_e: self.page.launch_url(link_e.data))  # code_theme="ayu-dark"
        # Check avatar, then append content to avatar with a certain alignment, then send it
        if avatar == "user":
            message_row = ft.Row(
                [message_content, self.user_avatar], alignment=ft.MainAxisAlignment.END, wrap=True)
            self.chat_list.controls.append(message_row)
        elif avatar == "ai":
            message_row = ft.Row([self.ai_avatar, message_content], wrap=True)
            self.chat_list.controls.append(message_row)
        elif avatar == "attachment":
            message_row = ft.Row([message_content, self.attachment_avatar], alignment=ft.MainAxisAlignment.END, wrap=True)
            self.chat_list.controls.append(message_row)
        else:
            # If no avatar is specified, message is sent with no avatar, just a boring grey log statement.
            self.chat_list.controls.append(message_content)
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
            self.ai_mode_checkbox.disabled = True  # It can be confusing to do this at the same time, so this is disabled
            self.sync_directory_btn.disabled = True  # Sync requires embedders & drive service
            self.reload_backend_btn.disabled = True  # Self explanatory why this is disabled
            self.send_button.disabled = True  # Sending a message requires embedders
            self.attach_button.disabled = True  # Although just .gdoc files require backend, all attachments are disabled
            self.overlay.visible = True  # Add animated loading indicator
            self.page.update()
            # All imports are done within functions to speed up initial page loading times.
            from SecondBrainBackend import machine_setup, load_config
            # Load config
            self.config = load_config(self.config_path)
            # AI Mode toggle is saved and stored in config.json, then reloaded here.
            self.ai_mode_checkbox.value = self.config['ai_mode']
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
            self.sync_directory_btn.disabled = False
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
            if self.config.get("llm_backend") == "LM Studio":
                self.llm = LMStudioLLM(model_name=self.config['lms_model_name'])
            elif self.config.get("llm_backend") == "Gemini":
                self.llm = GeminiLLM(api_key=self.config["gemini_api_key"])
            else:
                raise ValueError("Unknown LLM backend")
            self.log("Language model successfully loaded.")
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
            # Signify end of sync
            self.chat_list.controls.append(ft.Divider())
            self.page.update()

    def send_message(self, e=None):
        """I apologize to future me in advance for making this behemoth of a code block."""
        # Get the message
        msg = self.user_input.value.strip()
        # If no message, do nothing.
        if not msg and not self.attachment_data:
            return
        # This is needed to block the enter key way of sending a message
        if self.send_button.disabled:  # block both button click & Enter key
            return
        # Get attachment info & clear it.
        attachment = self.attachment_data
        attachment_path = self.attachment_path
        self.remove_attachment()
        # Change send button to loading icon, saving the original icon for when it's reset. This is a bit tougher than just replacing the icon.
        original_icon = self.send_button.icon
        self.send_button.content = ft.ProgressRing(width=16, height=16, stroke_width=2)
        self.send_button.icon = None
        self.send_button.disabled = True
        # The function works if the user sends text or attachments, or both.
        # If the user sent text, put a message to the log with the user avatar.
        if msg:
            self.log(f"{msg}", avatar="user")
        # If the user sent an attachment, put a message to the log with the attachment avatar.
        if attachment:
            self.log(f"{attachment_path.name}", avatar="attachment")
        # Empty the input field.
        self.user_input.value = ""
        self.page.update()
        # Create a large Flet column for the response to go into, organized as a column.
        results_column = ft.Column(spacing=10)
        results_container = ft.Container(
            content=results_column,
            bgcolor=ft.Colors.with_opacity(0.03, ft.Colors.WHITE), # Light gray bubble
            border_radius=10,
            padding=15,
            visible=False # Start as invisible, show only when there are results
        )
        self.chat_list.controls.append(results_container)
        # The computer response has four parts (organized vertically on the column): finding similar images, searching for images, searching for text, and streaming AI summary/insights.

        """SIMILAR IMAGES"""
        # Only occurs if the user attaches an image. [Image to text search is not possible with a text-only language model.]
        similar_images = None
        if attachment == "[IMAGE]":
            # Import the relevant backend function
            from SecondBrainBackend import find_similar_images
            # Run it.
            similar_images = find_similar_images([str(attachment_path)], self.models, self.collections, self.config)
            # If results (which there may not be due to z-score config):
            if similar_images:
                # Get the image paths
                image_paths = [result["file_path"] for result in similar_images]
                # Create a Flet display of the images
                image_row = self.image_presentation(image_paths)
                # Also add a heading (SIMILAR IMAGES)
                results_column.controls.append(ft.Row([ft.Text("SIMILAR IMAGES", size=12, weight=ft.FontWeight.BOLD)], alignment=ft.MainAxisAlignment.CENTER))
                # Then add image display to the column
                results_column.controls.append(image_row)
                # Now that there's a result (it take time to load and it's weird to have an empty container), show the container.
                results_container.visible = True
            else:
                # If no results, add an AI avatar message saying that.
                self.log("No similar images.", "ai")
            self.page.update()

        if msg or (attachment and attachment != "[IMAGE]"):  # Only elicit a text to image or text to text response if there is text. Complicated logic.
            # Import the next search function
            from SecondBrainBackend import semantic_search
            # Now is the time to look at the attachment if it is text.
            attachment_keywords = []
            attachment_keywords_string = ""
            attachment_chunks = []
            attachment_chunks_string = ""
            if attachment and attachment != "[IMAGE]":
                # The strategy for this is to get some keywords from the attachment to aid searching, as well as a selection of the most relevant chunks based on the query if there is an AI that can make use of it. These can also be embedded and used to search!
                n_keywords = self.config['n_attachment_keywords']
                if n_keywords:
                    attachment_keywords = self.get_keywords_from_attachment(attachment, n_keywords=n_keywords) + [attachment_path.stem]  # Also search for the attachment's name. It is a keyword.
                    attachment_keywords_string = ", ".join(attachment_keywords)
                # Look through the attachment to find the most relevant chunks based on the query.
                n_chunks = self.config['n_attachment_chunks']
                if n_chunks:
                    attachment_chunks = self.do_attachment_RAG(attachment, [msg], n_chunks=n_chunks)
                    attachment_chunks_string = "\n\n".join(attachment_chunks)
            
            """IMAGE SEARCH"""
            # If the app is in AI Mode, the AI helps to generate additional queries based on what the user says.
            if self.config['ai_mode']:
                # Get extra queries
                image_queries = self.llm_generate_queries(msg, "image", attachment_keywords_string, attachment_chunks_string, self.config['query_multiplier'])
                if msg:
                    # If the user sent a message (not just a text attachment), then the queries are the AI generated queries PLUS the user's query.
                    image_queries.append(msg)
            # This is for no AI mode.
            else:
                image_queries = []
                if msg:
                    image_queries.append(msg)
                # If the user attached something, then keywords and chunks from the attachments are used to search. That way attachments still do something in manual mode.
                if attachment_keywords:
                    image_queries.extend(attachment_keywords).extend(attachment_chunks)
            # Do the search!
            image_search_results = semantic_search(image_queries, self.models, self.collections, self.config, "image")
            if image_search_results:
                # Like before, get the paths
                image_paths = [result['file_path'] for result in image_search_results]
                # Make the presentation
                image_row = self.image_presentation(image_paths)
                # If there were similar images, add a divider to delineate between image rows
                if similar_images:
                    results_column.controls.append(ft.Divider())
                # Add a heading
                results_column.controls.append(ft.Row([ft.Text("IMAGE RESULTS", size=12, weight=ft.FontWeight.BOLD)], alignment=ft.MainAxisAlignment.CENTER))
                # Add the image.
                results_column.controls.append(image_row)
                results_container.visible = True  # Explained above
            self.page.update()

            """TEXT SEARCH"""
            # Same deal as before, except for text queries. Image and text queries are different because you use different terms to search for them.
            if self.config['ai_mode']:
                text_queries = self.llm_generate_queries(msg, "text", attachment_keywords_string, attachment_chunks_string, self.config['query_multiplier'])
                if msg:
                    text_queries.append(msg)  # The cool trick to append original message too
            # Same deal, extending queries using keywords from attachment in manual mode.
            else:
                text_queries = []
                if msg:
                    text_queries.append(msg)
                if attachment_keywords:
                    text_queries.extend(attachment_keywords).extend(attachment_chunks)
            # Some text embedders (like BAAI/bge-large-en-v1.5) work better with instructional prefixes. Can change in config.
            prefixed_text_queries = [self.config['search_prefix'] + q for q in text_queries]
            # Do the search!
            text_search_results = semantic_search(prefixed_text_queries, self.models, self.collections, self.config, "text")
            # Filter out poor results using AI here (doesn't work for images since the AI can't see images):
            if text_search_results and self.config['ai_mode']:
                # Evaluate relevance by asking AI and keep the relevant ones.
                text_search_results = [r for r in text_search_results if self.llm_evaluate_relevance(msg, r['documents'], attachment_keywords_string, attachment_chunks_string)]
            if text_search_results:
                # Make the results table
                results_table = self.results_table(text_search_results)
                if image_search_results or similar_images:
                    # Add a divider if there were image results... clearly
                    results_column.controls.append(ft.Divider())
                # Add header!
                results_column.controls.append(ft.Row([ft.Text("TEXT RESULTS", size=12, weight=ft.FontWeight.BOLD)], alignment=ft.MainAxisAlignment.CENTER))
                # Add table.
                results_column.controls.append(results_table)
                results_container.visible = True  # Explained above
                self.page.update()

                """AI INSIGHTS"""
                # This part occurs within the text results logic because you can only have AI Insights if you have text results.
                if self.config['ai_mode']:
                    # Add the divider (always needed)
                    results_column.controls.append(ft.Divider())
                    # Header
                    results_column.controls.append(ft.Row([ft.Text("AI INSIGHTS", size=12, weight=ft.FontWeight.BOLD)], alignment=ft.MainAxisAlignment.CENTER))
                    # Add the summary which streams live from the AI for cool effect.
                    self.llm_summarize_results(msg, [r['documents'] for r in text_search_results], attachment_keywords_string, attachment_chunks_string, target_column=results_column)
            # If no results, say no results.
            else:
                self.log("No text results.", "ai")
            if not image_search_results:
                self.log("No image results.", "ai")  # This has to be added here to prevent a glitch.
        # After sending message, reset stuff to start
        self.send_button.icon = original_icon
        self.send_button.content = None
        self.send_button.disabled = False
        self.page.update()

    def save_config(self):
        import json
        # Replaces current config file with the one in memory
        with open(self.config_path, "w") as f:
            json.dump(self.config, f, indent=4)

    def open_config(self, e=None):
        # For the open settings button
        os.startfile(self.config_path)

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
        self.config = {}
        self.drive_service = None
        self.text_splitter = None
        self.models = {}
        self.collections = {}
        self.initialize_backend()

    def stream_llm_response(self, prompt: str, target_column: ft.Column = None):
        """Handles the UI updates for a streaming LLM response.
        If target_column is provided, it adds the output there.
        Otherwise, it creates a new container in the chat list.
        """
        # Make a markdown box for the AI
        ai_response_text = ft.Markdown(
            value="▌",  # This creates a cool cursor effect, and will move as the AI streams.
            selectable=True, 
            extension_set=ft.MarkdownExtensionSet.GITHUB_WEB)
        # This is the key part: decide WHERE to place the output. Finding the results_column from send_message is the tricky part.
        if target_column is not None:
            # If column is found, add the response box.
            target_column.controls.append(ai_response_text)
        else:
            # Fallback to original behavior if no target is given (just paste below results container)
            ai_response_container = ft.Container(
                content=ai_response_text, 
                border_radius=10, 
                padding=10, 
                margin=5, 
                border=ft.border.all(1, ft.Colors.GREY))
            self.chat_list.controls.append(ai_response_container)
        self.page.update()

        full_response = ""
        try:
            # The streaming logic itself, using the "yield" thing
            for chunk in self.llm.stream(prompt):
                if not chunk:
                    continue
                full_response += chunk
                ai_response_text.value = full_response + " ▌"  # Move the cursor
                self.chat_list.scroll_to(offset=-1, duration=100)  # Add scroll
                self.page.update()
            # Insert full response after streaming is done.
            ai_response_text.value = str(full_response)
        except Exception as e:
            ai_response_text.value = f"Error streaming response: {e}"
        finally:
            self.send_button.disabled = False  # Not sure why this is here, but whatever.
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
            self.attachment_data = self.process_attachment(file_path)
            # Toggle attachment display and make it say the attachment name
            self.attachment_display.visible = True
            self.attachment_display.value = "Attached file: " + file_path.name
            # When done, reenable button
            self.attach_button.disabled = False
            # Change button to X and tooltip to filename, and function to remove attachment
            self.attach_button.icon = ft.Icons.CLOSE_ROUNDED
            self.attach_button.tooltip = "Remove attachment"
            self.attach_button.on_click = lambda _: self.remove_attachment()
        except Exception as e:
            self.log(f"Loading attachment failed: {e}")
        self.page.update()

    def remove_attachment(self):
        # Clear attachment data
        self.attachment_data = None
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
            self.log("Invalid attachment type.", avatar="ai")
            return
        # Then pull out the data.
        content = handler(path, self.drive_service, self.log) if handler == parse_gdoc else handler(path)
        if not content:
            return
        return content
    
    def image_presentation(self, image_paths: list[str]):
        # Takes a list of paths and returns a Flet container object with a row of scrollable images.
        image_widgets = []
        for path in image_paths:
            image_preview = ft.Container(
                content=ft.Image(src=path, width=160, height=160, fit=ft.ImageFit.CONTAIN), on_click=lambda _, p=path: self.page.launch_url(f"file:///{Path(p)}"), tooltip=f"Open image: {path}")  # Can adjust image size HERE
                # Clicking the image opens the image location.
            image_widgets.append(image_preview)
        image_row = ft.Row(controls=image_widgets, spacing=10, scroll=ft.ScrollMode.AUTO)
        return image_row

    def results_table(self, text_results):
        # Given a list of text results, returns a Flet "expansion tile" object with the file names as a title that expand to show the text content in question. Also provides a link that opens the file with the file explorer.
        import re
        results_widgets = []
        for r in text_results:            
            # Remove prefix from embedded chunk and add ellipsis as this is part of a larger document...
            cleaned_chunk = re.sub(r'^<Source: .*?>\s*', '', r['documents'])
            if not cleaned_chunk.endswith(('.', '?', '!', '…')):
                cleaned_chunk += '…'
            # 
            tile = ft.ExpansionTile(title=ft.Text(Path(r['file_path']).stem),
                    controls=
                    [ft.ListTile(title=ft.TextButton(text=r['file_path'], on_click=lambda e, p=r['file_path']: self.page.launch_url(f"file:///{Path(p)}")), 
                                subtitle=ft.Text(cleaned_chunk, selectable=True))])
            # Every result gets a tile
            results_widgets.append(tile)
        results = ft.Column(controls=results_widgets, spacing=0)
        return results

    def llm_evaluate_relevance(self, user_query, chunk, attachment_keywords, attachment_chunks) -> bool:
        # Limiting keywords so that responses are focused on the user's prompt
        keyword_prefix = f"Relevant keywords from user's attachment:\n{attachment_keywords}\n\n" if attachment_keywords else ""
        chunks_prefix = f"Relevant passages from user's attachment:\n{attachment_chunks}\n\n" if attachment_chunks else ""
        prompt_prefix = f"User's prompt: '{user_query}'\n\n" if user_query else ""
        
        prompt = f"""{keyword_prefix}{chunks_prefix}{prompt_prefix}
Document snippet: "{chunk}"

Is this snippet relevant for answering the user? Respond only with YES or NO."""
        # Needs a more robust method for analyzing sentiment, but honestly this works really well.
        response = self.llm.invoke(prompt).strip().lower()
        if "yes" in response.lower() or "no" in response.lower():
            return "yes" in response
        else:
            return True
        
    def llm_generate_queries(self, user_query: str, type: str, attachment_keywords, attachment_chunks, n: int = 5) -> list[str]:
        content = "documents" if type == "text" else "images"
        keyword_prefix = f"Relevant keywords from user's attachment:\n{attachment_keywords}.\n\n" if attachment_keywords else ""
        chunks_prefix = f"Relevant passages from user's attachment:\n{attachment_chunks}\n\n" if attachment_chunks else ""
        prompt_prefix = f"User's prompt: '{user_query}'\n\n" if user_query else ""
        reminder_suffix = " Make sure that the search is optimized for images." if type == "images" else ""
        
        prompt = f"""{keyword_prefix}{chunks_prefix}{prompt_prefix}
Based on the user's prompt, generate {n} similar search queries that could retrieve relevant {content}. Keep them clear, creative, and distinct. Respond with a plain list with no supporting text or markdown (just the queries).{reminder_suffix}"""

        response = self.llm.invoke(prompt)
        query_list = [q.strip("-• ").strip() for q in response.splitlines() if q.strip()]
        cleaned_text_queries = [item.lstrip('0123456789. ') for item in query_list]  # Strips leading (and trailing) numbers, periods, and spaces.
        return cleaned_text_queries

    def llm_summarize_results(self, user_query, relevant_chunks, attachment_keywords, attachment_chunks, target_column=None):
        keyword_prefix = f"Relevant keywords from user's attachment: {attachment_keywords}.\n\n" if attachment_keywords else ""
        chunks_prefix = f"Relevant passages from user's attachment:\n{attachment_chunks}\n\n" if attachment_chunks else ""
        prompt_prefix = f"User's prompt: '{user_query}'\n\n" if user_query else ""
        
        prompt = f"""{keyword_prefix}{chunks_prefix}{prompt_prefix}
Your search results:
{"\n".join(relevant_chunks)}

First summarize the results in one paragraph, making sure to cite your sources. In the next paragraph, synthesize them into a final piece of insight for the user, keeping in mind the user's original prompt above all else. Keep your answer to two short paragraphs. Be concise and to the point.

Use stylish markdown and code blocks if there is code. Use in-line citations, like this: (source_name)."""
        
        threading.Thread(target=self.stream_llm_response, args=(prompt, target_column), daemon=True).start()

    def get_keywords_from_attachment(self, attachment, n_keywords=3):
        import yake
        keyword_extractor = yake.KeywordExtractor(lan="en", n=3, top=n_keywords, dedupLim=0.8)
        keywords_result = keyword_extractor.extract_keywords(attachment)
        keywords = [keyword for keyword, score in keywords_result]
        print(f"Attachment keywords: {keywords}")
        return keywords

    def do_attachment_RAG(self, attachment, queries, n_chunks=3):
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
        # Encode queries
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
        top_idx = np.argsort(-avg_scores)[:n_chunks]
        top_chunks = [attachment_chunks[i] for i in top_idx]

        return top_chunks

def main(page: ft.Page):
    page.visible = False
    App(page)

ft.app(target=main, assets_dir="assets")