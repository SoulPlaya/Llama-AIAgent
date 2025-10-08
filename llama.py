"""
llama - Voice-Activated AI Assistant with Intelligent Model Routing
"""

import ollama
import speech_recognition as sr
import pyttsx3
import threading
import time
import tools

def text_to_speech(text):
    """Convert text to speech using pyttsx3"""
    print(f"llama: {text}")
    engine = pyttsx3.init()
    engine.setProperty('rate', 180)  # Speed
    engine.say(text)
    engine.runAndWait()



class llama:
    def __init__(self):
        # Model configuration
        self.fast_model = 'llama3.1:8b-instruct-q4_K_M'
        self.smart_model = 'qwen2.5:32b-instruct-q4_K_M'
        self.image_model = 'llama3.2-vision:11b'
        
        # Speech setup
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        
        # Conversation history
        self.conversation_history = []
        
        # Wake word
        self.wake_word = "llama"
        self.is_listening = True
        
        print("llama initializing...")
        text_to_speech("llama online.")
        
    def listen(self):
        """Listen to microphone and convert speech to text"""
        with self.microphone as source:
            print("Listening...")
            self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
            try:
                audio = self.recognizer.listen(source, timeout=10, phrase_time_limit=10)
                text = self.recognizer.recognize_google(audio)
                print(f"You: {text}")
                return text.lower()
            except sr.WaitTimeoutError:
                return None
            except sr.UnknownValueError:
                return None
            except Exception as e:
                print(f"Error: {e}")
                return None
    
    def classify_user_input(self, user_input):
        """Ask 8B model to classify query complexity"""
        try:
            response = ollama.chat(
                model=self.fast_model,
                messages=[{
                    'role': 'system',
                    'content': '''Classify if this query needs:
                    - Tool: web search, volume, or computer function
                    - SIMPLE: Quick answer, command, basic question, or factual lookup
                    - COMPLEX: Deep reasoning, analysis, difficult problems, or detailed explanations
                    
                    Consider Tool: if it needs external data or actions or is an explicit command to use a tool
                    Consider SIMPLE: greetings, commands (open/play/set), simple math, basic facts
                    Consider COMPLEX: "analyze", "compare", "explain why", "best approach", complex coding
                    
                    Respond with ONLY the word TOOL, SIMPLE, or COMPLEX, nothing else.'''
                }, {
                    'role': 'user',
                    'content': user_input
                }],
                options={'temperature': 0.1}  # Low temp for consistent classification
            )
            
            classification = response['message']['content'].strip().upper()
            
            # Fallback if model doesn't follow instructions
            if 'COMPLEX' in classification:
                return 'COMPLEX'
            elif 'TOOL' in classification:
                return 'TOOL'
            else:
                return 'SIMPLE'
                
        except Exception as e:
            print(f"Classification error: {e}")
            return 'SIMPLE'  # Default to fast model on error
    
    def get_response(self, user_input, model):
        """Get response from specified model (handles Ollama's current object API)"""
        try:
            # Add user message to history
            self.conversation_history.append({
                'role': 'user',
                'content': user_input
            })

            # Keep only last 10 messages
            messages = self.conversation_history[-10:]

            # Add system prompt
            system_message = {
                'role': 'system',
                'content': 'You are Llama, a helpful AI assistant. Be concise but informative. For simple queries, keep responses brief.'
            }

            # Call Ollama
            response = ollama.chat(
                model=model,
                messages=[system_message] + messages
            )

            # --- NEW: Handle object API ---
            if hasattr(response, "message") and hasattr(response.message, "content"):
                assistant_message = response.message.content
            elif isinstance(response, dict):  # fallback for old API
                assistant_message = response.get("message", {}).get("content", "")
            else:
                assistant_message = str(response)

            # Add assistant response to history
            self.conversation_history.append({
                'role': 'assistant',
                'content': assistant_message
            })

            return assistant_message.strip() or "I'm sorry, I didn't get a response."

        except Exception as e:
            print(f"get_response error: {e}")
            return f"I encountered an error: {str(e)}"


    
    def process_query(self, user_input):
        """Main routing logic - decides which model to use"""
        
        # Classify complexity
        print("Analyzing query complexity...")
        complexity = self.classify_user_input(user_input)
        
        if complexity == 'COMPLEX':
            print("Routing to 70B model for complex reasoning...")
            text_to_speech("This requires deeper analysis, one moment...")
            response = self.get_response(user_input, self.smart_model)
            return response
        else:
            print("Using 8B model for quick response...")
            response = self.get_response(user_input, self.fast_model)
            return response
    
    def run(self):
        """Main loop - listen for wake word and process commands"""
        print(f"\n llama is ready! Say '{self.wake_word}' followed by your command.\n")
        
        while self.is_listening:
            # Listen for wake word
            text = self.listen()
            
            if text and self.wake_word in text:
                # Remove wake word from command
                command = text.replace(self.wake_word, "").strip()
                
                if not command:
                    text_to_speech("Yes?")
                    command = self.listen()
                    if not command:
                        continue
                
                # Handle exit commands
                if any(word in command for word in ['exit', 'quit', 'goodbye', 'shut down']):
                    self.speak("Shutting down. Goodbye!")
                    self.is_listening = False
                    break
                
                # Process the query
                response = self.process_query(command)
                text_to_speech(response)
                
            time.sleep(0.1)  # Small delay to prevent CPU spinning

def main():
    
    try:
        Llama = llama()
        Llama.run()
    except KeyboardInterrupt:
        print("LLama shutting down...")
    except Exception as e:
        print(f"Fatal error: {e}")

if __name__ == "__main__":
    main()