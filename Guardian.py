from interpreter import interpreter

# Set up Ollama as the model provider" 

# Optional: adjust system behavior or temperature
interpreter.offline = True # Disables online features like Open Procedures
interpreter.llm.model = "ollama_chat/qwen2.5:32b-instruct-q4_K_M"
interpreter.llm.api_base = "http://localhost:11434"
interpreter.llm.api_key = "ollama" 


# Or use it programmatically
if __name__ == "__main__":

    response = interpreter.chat("If I feed you a csv file, can you analyze it and generate an insight as to the important collumns and useful information?")
    print(response)

