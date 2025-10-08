## PREFACE ##
This is a personal project meant to run 24/7 on my local machine for tasks and various things. 
I run windows and am still learning a lot about programing so please excuse me if any code shows issue or this is unoptimized for various machines.

## System Requirements ##
Windows (for now)
12gb or more of VRam - I have a nvidia 4070
42gb of ram 
~55gb of storage for models

## Commands for Neat Execution ##
(dependecies) pip install ollama speechrecognition pyaudio pyttsx3
(fetch models) ollama pull llama3.1:8b-instruct-q4_K_M && ollama pull llama3.1:70b-instruct-q4_K_M && ollama pull llama3.2-vision:11b
(ensure ollama is running) ollama serve