import json
import threading
import queue
import time
import logging
import traceback

import ollama
import speech_recognition as sr
import pyttsx3

import tools

logging.basicConfig(level=logging.INFO)


class LlamaAssistant:
    def __init__(self,
                 fast_model='llama3.1:8b-instruct-q4_K_M',
                 smart_model='qwen2.5:32b-instruct-q4_K_M',
                 image_model='llama3.2-vision:11b',
                 wake_word='llama',
                 max_history=10):
        self.fast_model = fast_model
        self.smart_model = smart_model
        self.image_model = image_model

        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        self.conversation_history = []
        self.max_history = max_history

        self.wake_word = wake_word.lower()
        self.is_listening = True

        # Tools mapping (centralized)
        self.tools = {
            "search_web": {
                "function": tools.search_web,
                "arguments": {"query": str}
            },
            "take_screenshot": {
                "function": tools.take_screenshot,
                "arguments": {}
            }
        }

        # TTS worker queue & engine (single engine reused by worker thread)
        self._tts_queue = queue.Queue()
        self._tts_engine = pyttsx3.init()
        self._tts_engine.setProperty('rate', 180)
        self._tts_thread = threading.Thread(target=self._tts_worker, daemon=True)
        self._tts_thread.start()

        logging.info("llama initializing...")
        self.text_to_speech("llama online.")

    # ----------------- TTS -----------------
    def _tts_worker(self):
        while True:
            text = self._tts_queue.get()
            try:
                # engine.runAndWait() is blocking but this is in worker thread
                self._tts_engine.say(text)
                self._tts_engine.runAndWait()
            except Exception:
                logging.exception("TTS worker error")
            finally:
                self._tts_queue.task_done()

    def text_to_speech(self, text):
        """Queue text for speech (non-blocking)."""
        if not text:
            return
        print(f"llama: {text}")
        self._tts_queue.put(str(text))

    # ----------------- Listening -----------------
    def listen(self, timeout=10, phrase_time_limit=10):
        """Listen and return recognized text (lowercased) or None."""
        with self.microphone as source:
            try:
                self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
                audio = self.recognizer.listen(source, timeout=timeout, phrase_time_limit=phrase_time_limit)
                text = self.recognizer.recognize_google(audio)  # note: uses Google cloud
                logging.debug("Recognized: %s", text)
                return text.lower()
            except sr.WaitTimeoutError:
                return None
            except sr.UnknownValueError:
                return None
            except Exception:
                logging.exception("listen() error")
                return None

    # ----------------- LLM helpers -----------------
    def _extract_content(self, response):
        """Robustly extract textual content from various response shapes."""
        try:
            # dict with 'message' -> {'content': ...}
            if isinstance(response, dict):
                msg = response.get('message')
                if isinstance(msg, dict):
                    return msg.get('content', '')

                # choice-style: {'choices': [{'message': {'content': ...}}]}
                choices = response.get('choices') or response.get('outputs') or None
                if choices and isinstance(choices, (list, tuple)) and len(choices) > 0:
                    first = choices[0]
                    if isinstance(first, dict):
                        if 'message' in first and isinstance(first['message'], dict):
                            return first['message'].get('content', '')
                        if 'text' in first:
                            return first.get('text', '')
                        if 'content' in first:
                            return first.get('content', '')
            # fallback
            return str(response)
        except Exception:
            logging.exception("Error extracting content")
            return str(response)

    # ----------------- Classification -----------------
    def classify_user_input(self, user_input):
        """Return 'TOOL', 'SIMPLE', or 'COMPLEX'."""
        sys_prompt = (
            "Classify this query:\n"
            "- TOOL: ONLY use if user asks for tool or websearch\n"
            "   -The tools you have are: search_web and take_screenshot\n"
            "- SIMPLE: quick answer, converstation or basic factual information\n"
            "- COMPLEX: deep reasoning, analysis, thoughtful dialog.\n\n"
            "Respond with ONLY TOOL, SIMPLE, or COMPLEX."
        )
        try:
            resp = ollama.chat(model=self.fast_model,
                               messages=[{'role': 'system', 'content': sys_prompt},
                                         {'role': 'user', 'content': user_input}],
                               options={'temperature': 0.1})
            content = self._extract_content(resp).strip().upper()
            if 'TOOL' in content:
                return 'TOOL'
            if 'COMPLEX' in content:
                return 'COMPLEX'
            return 'SIMPLE'
        except Exception:
            logging.exception("classify_user_input failed, defaulting SIMPLE")
            return 'SIMPLE'

    # ----------------- Tool selection -----------------
    def choose_tool(self, user_input):
        """Ask the fast model to choose a tool. Returns dict {'tool': name, 'arguments': {...}} or None."""
        system_prompt = (
            "You are a precise AI that selects the best tool for a user query.\n\n"
            "Available tools:\n" + json.dumps(list(self.tools.keys()), indent=2) + "\n\n"
            "Respond with only JSON in this format:\n"
            '{"tool": "tool_name_here", "arguments": {"arg_name": "value"}}\n'
            'If none fit respond: {"tool": null, "arguments": {}}'
        )
        try:
            resp = ollama.chat(model=self.fast_model,
                               messages=[{'role': 'system', 'content': system_prompt},
                                         {'role': 'user', 'content': user_input}],
                               options={'temperature': 0.1})
            content = self._extract_content(resp).strip()
            try:
                parsed = json.loads(content)
            except json.JSONDecodeError:
                # fallback: simple heuristic - look for known tool names in the text
                lowered = content.lower()
                for name in self.tools.keys():
                    if name.lower() in lowered:
                        return {'tool': name, 'arguments': {}}
                logging.warning("choose_tool: failed to parse JSON and no tool keyword found. response=%s", content)
                return None

            tool = parsed.get('tool')
            arguments = parsed.get('arguments', {}) or {}
            if not tool or tool not in self.tools:
                logging.warning("choose_tool: model selected unknown or null tool: %s", tool)
                return None
            # Basic argument validation can happen here
            return {'tool': tool, 'arguments': arguments}
        except Exception:
            logging.exception("choose_tool failed")
            return None

    def safe_execute_tool(self, func, args, user_input):
        """Ensure tool has all required args, and fill missing ones if possible."""
        import inspect

        sig = inspect.signature(func)
        missing_args = [
            p.name for p in sig.parameters.values()
            if p.default == inspect._empty and p.name not in args
        ]

        # If missing args, ask the model to fill them
        if missing_args and user_input is not None:
            logging.info(f"Missing args {missing_args}, asking model to infer...")
            try:
                fill_prompt = (
                    f"User said: '{user_input}'. The tool '{func.__name__}' requires arguments: {missing_args}. "
                    "Respond ONLY with a valid JSON object containing those argument values. No explanations."
                )

                resp = ollama.chat(
                    model=self.fast_model,
                    messages=[
                        {"role": "system", "content": "You are a json generator fill in a missing value. Output ONLY valid JSON."},
                        {"role": "user", "content": fill_prompt}
                    ],
                    options={"temperature": 0.1}
                )

                content = self._extract_content(resp).strip()
                logging.debug(f"Raw argument fill model output: {content}")

                # --- Try to isolate the JSON part robustly ---
                import re
                json_match = re.search(r'\{.*\}', content, re.DOTALL)
                if not json_match:
                    logging.warning(f"Argument fill: no JSON found in model output: {content}")
                    return "Tool invocation failed: missing arguments."

                json_part = json_match.group(0)

                # --- Try to fix minor formatting issues (single quotes → double) ---
                json_part = json_part.replace("'", '"')

                arg_fill = json.loads(json_part)
                args.update(arg_fill)

            except Exception:
                logging.exception("Argument fill failed")
                return "Tool invocation failed: bad arguments."

        # Execute safely
        try:
            result = func(**args) if callable(func) else None
            return result or "Tool executed successfully but returned no output."
        except Exception as e:
            logging.exception("Tool execution error")
            return "Tool invocation failed: bad arguments."


    def execute_tool(self, tool_choice):
        """Execute chosen tool. Special-case screenshot->describe flow."""
        if not tool_choice:
            return None
        name = tool_choice.get('tool')
        args = tool_choice.get('arguments', {}) or {}
        if name not in self.tools:
            logging.warning("execute_tool: unknown tool %s", name)
            return None

        func = self.tools[name]['function']
        try:
            if name == 'take_screenshot':
                screenshot = func(**args)  # could return path, bytes, file-like
                # normalize to bytes
                if isinstance(screenshot, str):
                    with open(screenshot, 'rb') as f:
                        file_bytes = f.read()
                elif hasattr(screenshot, 'read'):
                    file_bytes = screenshot.read()
                else:
                    file_bytes = screenshot

                # send to vision model
                resp = ollama.chat(
                    model=self.image_model,
                    messages=[
                        {'role': 'system', 'content': 'You are an AI that describes images.'},
                        {'role': 'user', 'content': 'Describe the contents of this screenshot.'}
                    ],
                    files={'image': file_bytes}
                )
                return self._extract_content(resp).strip()
            else:
                return self.safe_execute_tool(func, args)
        except Exception:
            logging.exception("execute_tool() error")
            return None

    # ----------------- LLM response / conversation -----------------
    def get_response(self, user_input, model):
        """Query the given model with conversation history; append assistant response to history."""
        try:
            self.conversation_history.append({'role': 'user', 'content': user_input})
            # keep last N user/assistant exchanges
            messages = self.conversation_history[-self.max_history:]
            system_message = {'role': 'system', 'content': 'You are Llama, a helpful AI assistant. Be concise.'}
            resp = ollama.chat(model=model, messages=[system_message] + messages)
            assistant_message = self._extract_content(resp).strip()
            self.conversation_history.append({'role': 'assistant', 'content': assistant_message})
            return assistant_message or "I'm sorry, I didn't get a response."
        except Exception:
            logging.exception("get_response error")
            return "I encountered an error while fetching a response."

    # ----------------- Orchestration -----------------
    def process_query(self, user_input):
        """Main router: choose MODEL or TOOL based on classification."""
        classification = self.classify_user_input(user_input)
        if classification == 'TOOL':
            tool_choice = self.choose_tool(user_input)
            if not tool_choice:
                return "I couldn't determine which tool to use."
            func = self.tools[tool_choice["tool"]]["function"]
            args = tool_choice.get("arguments", {}) or {}
            return self.safe_execute_tool(func, args, user_input) or "Tool returned no result."
        elif classification == 'COMPLEX':
            # optionally notify user
            self.text_to_speech("This requires deeper analysis, one moment...")
            return self.get_response(user_input, self.smart_model)
        else:
            return self.get_response(user_input, self.fast_model)

    def _handle_command(self, command):
        """Run process_query and TTS the result. Runs in a background thread."""
        try:
            result = self.process_query(command)
            if result:
                self.text_to_speech(result)
        except Exception:
            logging.exception("_handle_command failed")

    # ----------------- Main loop -----------------
    def run(self):
        logging.info("llama ready! Say '%s' followed by your command.", self.wake_word)
        while self.is_listening:
            text = self.listen()
            if text and self.wake_word in text:
                command = text.replace(self.wake_word, "").strip()
                if not command:
                    self.text_to_speech("Yes?")
                    command = self.listen()
                    if not command:
                        continue

                if any(w in command.lower() for w in ['exit', 'quit', 'goodbye', 'shut down']):
                    self.text_to_speech("Shutting down. Goodbye!")
                    self.is_listening = False
                    break

                # Process the command asynchronously so we can keep listening
                threading.Thread(target=self._handle_command, args=(command,), daemon=True).start()

            time.sleep(0.1)

    
if __name__ == "__main__":
    assistant = LlamaAssistant()
    assistant.run()