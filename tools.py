import json 
import urllib.parse
import webbrowser
import mss
import nmap

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
