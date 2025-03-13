import requests
import modal
import socket

# Create a Modal image with our dependencies
image = modal.Image.debian_slim().pip_install("requests")

app = modal.App("execution-info-app")

# Define a Modal function for public IP
@app.function(image=image)  # Use the image with requests installed
def get_public_ip():
    try:
        # Get the public IP using ipify
        response = requests.get('https://api.ipify.org?format=json')
        ip_info = response.json()
        return ip_info.get("ip")
    except requests.RequestException:
        return "Unable to fetch public IP."

# Define a Modal function for location based on IP
@app.function(image=image)  # Use the image with requests installed
def get_ip_location(ip):
    try:
        # Use ipinfo.io to get location information for the IP
        response = requests.get(f'https://ipinfo.io/{ip}/json')
        location_info = response.json()
        return location_info
    except requests.RequestException:
        return "Unable to fetch location data."

def get_local_ip():
    """Get the local IP address"""
    try:
        # Create a socket to get local IP
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        # We don't actually connect, just use it to get local IP
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
        return local_ip
    except Exception:
        return "Unable to fetch local IP."

@app.local_entrypoint()
def main():
    print("\n=== Local Machine Details ===")
    local_ip = get_local_ip()
    print(f"Local IP: {local_ip}")
    
    print("\n=== Your Home Network Details ===")
    # This will get your actual public IP when run locally
    try:
        response = requests.get('https://api.ipify.org?format=json')
        home_ip = response.json().get("ip")
        print(f"Home Public IP: {home_ip}")
        home_location = get_ip_location.remote(home_ip)
        print(f"Home Location: {home_location}")
    except Exception as e:
        print(f"Unable to fetch home network details: {str(e)}")
    
    print("\n=== Modal Server Details ===")
    public_ip = get_public_ip.remote()
    print(f"Modal Server IP: {public_ip}")
    location_data = get_ip_location.remote(public_ip)
    print(f"Modal Server Location: {location_data}")

if __name__ == "__main__":
    import sys
    if "modal" not in sys.modules:
        main()
    else:
        app.run()

