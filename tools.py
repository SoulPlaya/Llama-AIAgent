import json 
import urllib.parse
import webbrowser
import mss

json_schema = {
    "tool_name": str,
    "description": str,
    "arguments": dict,   # Mapping of argument_name → argument schema
    "function_name": str,
    "enabled": bool
}

# Define what each argument should look like
argument_schema = {
    "type": (str, bool, int, float, list, dict),
    "description": str,
    "required": bool
}

def validate_json(model_json_response):
  # Check top-level keys
    for key, expected_type in json_schema.items():
        if key not in model_json_response:
            print(f"Missing key: {key}")
            return False
        if not isinstance(model_json_response[key], expected_type):
            print(f"Key '{key}' has wrong type: expected {expected_type}, got {type(model_json_response[key])}")
            return False

    # Validate arguments if they exist
    args = model_json_response.get("arguments", {})
    for arg_name, arg_data in args.items():
        for akey, atype in argument_schema.items():
            if akey not in arg_data:
                print(f"Argument '{arg_name}' missing field '{akey}'")
                return False
            if not isinstance(arg_data[akey], atype):
                print(f"Argument '{arg_name}' field '{akey}' wrong type: expected {atype}, got {type(arg_data[akey])}")
                return False

    return True

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

