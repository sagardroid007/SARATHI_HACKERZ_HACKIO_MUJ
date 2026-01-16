import webbrowser

def open_navigation(destination):
    """
    Opens Google Maps Directions in the default browser.
    No API key required.
    """
    if not destination:
        return "I need a destination to start navigation."
        
    print(f"[MAPS] Navigating to: {destination}")
    # Construct URL for directions
    # https://www.google.com/maps/dir/?api=1&destination=New+York
    encoded_dest = destination.replace(" ", "+")
    url = f"https://www.google.com/maps/dir/?api=1&destination={encoded_dest}"
    
    try:
        webbrowser.open(url)
        return f"Opening navigation to {destination}."
    except Exception as e:
        print(f"[ERROR] Failed to open browser: {e}")
        return "I couldn't open the map."
