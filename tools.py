import json 
import urllib.parse
import webbrowser
import mss
import nmap
import platform
import subprocess
import time

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

# Monitors screen for changes
# Usable in two modes:
# 1. till_changed=True: runs indefinitely until a significant change is detected, then returns the changed image
# 2. iterations=n: runs for n iterations, checking for changes each time waiting three seconds between checks
def watch_screen_for_changes(till_changed: bool | None, iterations: int | None, threshold: int = 100000):
    sct = mss.mss()
    last_image = None

    while iterations is not None and iterations > 0:
        current_image = sct.grab(sct.monitors[1])
        if last_image is not None:
            diff = sum(abs(a - b) for a, b in zip(current_image.rgb, last_image.rgb))
            if diff > threshold:
                print("Significant screen change detected")
                if till_changed:
                    break
        last_image = current_image
        time.sleep(3)
        iterations -= 1
    
    while till_changed:
        current_image = sct.grab(sct.monitors[1])
        if last_image is not None:
            diff = sum(abs(a - b) for a, b in zip(current_image.rgb, last_image.rgb))
            if diff > threshold:
                return current_image
        last_image = current_image
        time.sleep(3)
    
# Scans a target for open ports
def scan_ports(target: str, ports: str = "1-1024"):
    nm = nmap.PortScanner()
    nm.scan(target, ports)
    result = nm.csv()
    print(f"Port scan completed for {target}")
    return result

# Executes a command in the terminal Windows only not a Linux chad on the home pc just laptop lol
def execute_command_in_terminal(command: str):

    system = platform.system()
    if system == "Windows":
        subprocess.run(["cmd", "/c", command], check=True)