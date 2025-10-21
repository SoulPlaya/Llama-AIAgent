import json 
import urllib.parse
import webbrowser
import mss
import nmap


functions = {
    "search_web": {
        "description": "Opens a web browser to search for a given query.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query to look up on the web."
                }
            },
            "required": ["query"]
        }
    },
    "take_screenshot": {
        "description": "Takes a screenshot of the current screen.",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": []
        }
    },
    "scan_ports": {
        "description": "Scans a target for open ports.",
        "parameters": {
            "type": "object",
            "properties": {
                "target": {
                    "type": "string",
                    "description": "The target IP address or hostname to scan."
                },
                "ports": {
                    "type": "string",
                    "description": "The range of ports to scan (e.g., '1-1024', '80,443').",
                    "default": "1-1024"  # Add default value to schema
                }
            },
            "required": ["target"]
        }
    }
}


# Opens web browser to search query
def search_web(query: str):
    encoded = urllib.parse.quote(query)
    url = f"https://duckduckgo.com/?q={encoded}"
    webbrowser.open(url)

#Returns path to screenshot
def take_screenshot():
    sct = mss.mss()
    shot = sct.shot(output='screenshot.png')
    print(f"Screenshot taken")
    return shot

# Scans a target for open ports
def scan_ports(target: str, ports: str = "1-1024"):
    nm = nmap.PortScanner()
    nm.scan(target, ports)
    result = nm.csv()
    print(f"Port scan completed for {target}")
    return result
