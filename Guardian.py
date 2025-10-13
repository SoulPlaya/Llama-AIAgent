import json
import logging
import threading
import queue
import time

import ollama
import speech_recognition as sr
import pyttsx3
import tools

logging.basicConfig(level=logging.INFO)


class GuardianAssistant:
    """
    Voice-enabled AI agent using Ollama LLMs and optional tools.
    """

    def __init__(
        self,
        fast_model="llama3.1:8b-instruct-q4_K_M",
        smart_model="qwen2.5:32b-instruct-q4_K_M",
        image_model="llama3.2-vision:11b",
        wake_word="Guardian",
        max_history=10,
    ):
        # Models
        self.fast_model = fast_model
        self.smart_model = smart_model
        self.image_model = image_model

        # Audio setup
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        self.tts_engine = pyttsx3.init()
        self.tts_engine.setProperty("rate", 180)

        # Conversation state
        self.history = []
        self.max_history = max_history
        self.wake_word = wake_word.lower()
        self.is_listening = True

        # Tools
        self.tools = {
            "search_web": tools.search_web,
            "take_screenshot": tools.take_screenshot,
        }

        # TTS worker
        self.tts_queue = queue.Queue()
        threading.Thread(target=self._tts_worker, daemon=True).start()

        logging.info("Guardian Assistant initializing...")
        self.say("Guardian online.")

    # -------------------------------------------------------------------------
    # --- Text-to-Speech ------------------------------------------------------
    # -------------------------------------------------------------------------

    def _tts_worker(self):
        """Background TTS thread."""
        while True:
            text = self.tts_queue.get()
            try:
                self.tts_engine.say(text)
                self.tts_engine.runAndWait()
            except Exception:
                logging.exception("TTS error")
            finally:
                self.tts_queue.task_done()

    def say(self, text):
        """Non-blocking speech output."""
        if text:
            print(f"Guardian: {text}")
            self.tts_queue.put(str(text))

    # -------------------------------------------------------------------------
    # --- Speech Recognition --------------------------------------------------
    # -------------------------------------------------------------------------

    def listen(self, timeout=10, phrase_time_limit=10):
        """Capture speech and return transcribed text."""
        with self.microphone as source:
            try:
                self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
                audio = self.recognizer.listen(source, timeout=timeout, phrase_time_limit=phrase_time_limit)
                text = self.recognizer.recognize_google(audio)
                print(f"You: {text}")
                return text.lower()
            except (sr.WaitTimeoutError, sr.UnknownValueError):
                return None
            except Exception:
                logging.exception("Listen error")
                return None

    # -------------------------------------------------------------------------
    # --- Utility: Extract Text from Ollama Responses -------------------------
    # -------------------------------------------------------------------------

    @staticmethod
    def _extract_text(response):
        """Extract plain text content from an Ollama API response."""
        try:
            if isinstance(response, dict):
                msg = response.get("message")
                if isinstance(msg, dict):
                    return msg.get("content", "")

                choices = response.get("choices") or response.get("outputs")
                if choices and isinstance(choices, list):
                    msg = choices[0].get("message", {})
                    return msg.get("content") or choices[0].get("text", "")
        except Exception:
            logging.exception("Failed to extract text")
        return str(response)

    # -------------------------------------------------------------------------
    # --- Classification ------------------------------------------------------
    # -------------------------------------------------------------------------

    def classify(self, query):
        """Return classification: TOOL | SIMPLE | COMPLEX"""
        system_prompt = (
            "Classify the user query as one of:\n"
            "TOOL: if it requests a web search or screenshot.\n"
            "SIMPLE: if it's a short or factual question.\n"
            "COMPLEX: if it needs reasoning or analysis.\n"
            "Respond ONLY with TOOL, SIMPLE, or COMPLEX."
        )

        try:
            resp = ollama.chat(
                model=self.fast_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": query},
                ],
                options={"temperature": 0.1},
            )
            result = self._extract_text(resp).strip().upper()
            if result in ("TOOL", "SIMPLE", "COMPLEX"):
                return result
        except Exception:
            logging.exception("Classification failed")
        return "SIMPLE"

    # -------------------------------------------------------------------------
    # --- Tool Handling -------------------------------------------------------
    # -------------------------------------------------------------------------

    def _select_tool(self, query):
        """Ask model which tool to use."""
        prompt = (
            "You are a precise AI that selects tools.\n"
            "Available tools: search_web, take_screenshot.\n"
            "Respond ONLY with 'search_web' or 'take_screenshot'."
        )

        try:
            resp = ollama.chat(
                model=self.fast_model,
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": query},
                ],
                options={"temperature": 0.1},
            )
            tool = self._extract_text(resp).strip().lower()
            if tool in self.tools:
                return tool
        except Exception:
            logging.exception("Tool selection failed")

        return None
    
    def _select_tool_args(self, query, func_name):
        """Ask model for tool arguments."""
        prompt = (
            "You are a precise AI that provides function arguments \n"
            "For example a user prompt of hotdog photos using 'search_web', provide the arguement hotdog.\n"
            "For example if the function does not need arguements like 'take_screenshot', provide {}.\n"
            "Respond ONLY with a valid arguements as responses.\n"
            f"The function you are providing an arguement for is {func_name}.\n"
        )

        try:
            resp = ollama.chat(
                model=self.fast_model,
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": query},
                ],
                options={"temperature": 0.1},
            )
            args_text = self._extract_text(resp).strip()
            return json.loads(args_text)
        except Exception:
            logging.exception("Tool argument selection failed")

        return {}

    def _execute_tool(self, tool_name, args=None):
        """Run the specified tool."""
        if tool_name not in self.tools:
            return "Unknown tool requested."
        try:
            func = self.tools[tool_name]
            result = func(**(args or {}))
            if tool_name == "take_screenshot" and result:
                return self._describe_image(result)
            return result or "Tool executed successfully."
        except Exception:
            logging.exception("Tool execution error")
            return "Tool execution failed."

    def _describe_image(self, image_path):
        """Send an image to the vision model for description."""
        try:
            with open(image_path, "rb") as f:
                data = f.read()
            resp = ollama.chat(
                model=self.image_model,
                messages=[
                    {"role": "system", "content": "Describe the image."},
                    {"role": "user", "content": "What's in this image?"},
                ],
                files={"image": data},
            )
            return self._extract_text(resp).strip()
        except Exception:
            logging.exception("Image description failed")
            return "Couldn't describe the image."

    # -------------------------------------------------------------------------
    # --- Conversation + LLM Response ----------------------------------------
    # -------------------------------------------------------------------------

    def chat(self, query, model=None):
        """Query a model and manage conversation history."""
        model = model or self.fast_model
        try:
            self.history.append({"role": "user", "content": query})
            history_slice = self.history[-self.max_history :]
            system_prompt = {"role": "system", "content": "You are Guardian, a concise helpful AI."}

            resp = ollama.chat(model=model, messages=[system_prompt] + history_slice)
            reply = self._extract_text(resp).strip()
            self.history.append({"role": "assistant", "content": reply})
            return reply or "No response."
        except Exception:
            logging.exception("Chat error")
            return "I encountered an error while thinking."

    # -------------------------------------------------------------------------
    # --- Main Query Routing --------------------------------------------------
    # -------------------------------------------------------------------------

    def handle_query(self, query):
        """Route user query to proper handler."""
        classification = self.classify(query)

        if classification == "TOOL":
            tool = self._select_tool(query)
            if not tool:
                return "I couldn't decide which tool to use."
            return self._execute_tool(tool)

        if classification == "COMPLEX":
            self.say("This might take a moment.")
            return self.chat(query, model=self.smart_model)

        # SIMPLE fallback
        return self.chat(query, model=self.fast_model)

    def _process_command(self, command):
        """Run query in background thread."""
        try:
            response = self.handle_query(command)
            if response:
                self.say(response)
        except Exception:
            logging.exception("Command processing failed")

    # -------------------------------------------------------------------------
    # --- Main Loop -----------------------------------------------------------
    # -------------------------------------------------------------------------

    def run(self):
        """Continuously listen for wake word and commands."""
        logging.info("Ready! Say '%s' followed by your command.", self.wake_word)
        while self.is_listening:
            text = self.listen()
            if text and self.wake_word in text:
                command = text.replace(self.wake_word, "").strip()

                if not command:
                    self.say("Yes?")
                    command = self.listen()
                    if not command:
                        continue

                if any(word in command for word in ("exit", "quit", "goodbye", "shutdown")):
                    self.say("Goodbye!")
                    self.is_listening = False
                    break

                threading.Thread(target=self._process_command, args=(command,), daemon=True).start()

            time.sleep(0.1)


# -----------------------------------------------------------------------------
# --- Entry Point -------------------------------------------------------------
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    assistant = GuardianAssistant()
    assistant.run()
