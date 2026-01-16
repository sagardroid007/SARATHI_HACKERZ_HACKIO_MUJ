import os
import google.generativeai as genai

# Try to get API key from environment, but don't crash if missing
GEMINI_API_KEY = "AIzaSyAOA45oChaaQFbeuDma7OPfANSdQE9_JQg"

_configured = False

def configure_gemini(api_key=None):
    """Configure the Gemini API with the provided key or from environment."""
    global _configured
    key_to_use = api_key or GEMINI_API_KEY
    
    if not key_to_use:
        print("[WARN] No Gemini API Key provided. LLM features will be disabled.")
        return False
        
    try:
        genai.configure(api_key=key_to_use)
        _configured = True
        print("[INFO] Gemini API Configured successfully.")
        return True
    except Exception as e:
        print(f"[ERROR] Failed to configure Gemini API: {e}")
        return False

def query_gemini(text_input):
    """
    Send text to Gemini 1.5 Flash and get a concise response.
    Returns: The response text, or None if failed/not configured.
    """
    if not _configured:
        # Try auto-configuring if environment variable exists
        if not configure_gemini():
            return None

    try:
        # Use the correct model name found in listing
        model = genai.GenerativeModel('gemini-3-flash-preview')
        
        # System prompt equivalent (concatenated)
        prompt = (
            "You are Sarathi, a smart driving assistant. "
            "Keep your response short, helpful, and safe for a driver to hear. "
            "Do not use markdown or emojis. "
            f"User input: {text_input}"
        )
        
        response = model.generate_content(prompt)
        return response.text.strip()
        
    except Exception as e:
        print(f"[ERROR] Gemini query failed: {e}")
        return None

if __name__ == "__main__":
    # Simple test
    print("Testing Gemini Client...")
    # You can manually set key here for testing: configure_gemini("YOUR_KEY")
    resp = query_gemini("Tell me a funny road trip fact.")
    print(f"Response: {resp}")
