from stealthkit import StealthSession

# Create a stealth session
sr = StealthSession()

# Fetch cookies from a base URL
sr.fetch_cookies("https://www.aoblab.com/urunler/tuz-korozyon-test-kabini-cevrimsel-korozyon-test-kabini")

# Make a GET request with the correct URL
response = sr.get("https://www.aoblab.com")

# Check if the request was successful and print the response
if response:
    try:
        print(response.json())
    except ValueError: # eğer json olarak ayrıştırılamaz ise.
        print(response.text) # text olarak yazdır.
else:
    print("Failed to fetch data")