import requests
from bs4 import BeautifulSoup
import json

def main():
    print("Starting simple test...")
    
    # Fetch the page
    url = "https://quotes.toscrape.com/"
    print(f"Fetching {url}...")
    response = requests.get(url)
    
    if response.status_code != 200:
        print(f"Error: Failed to fetch {url}, status code: {response.status_code}")
        return
    
    # Parse the HTML
    soup = BeautifulSoup(response.text, "html.parser")
    
    # Extract quotes
    quotes = []
    for quote in soup.select(".quote"):
        quotes.append({
            "text": quote.select_one(".text").get_text(),
            "author": quote.select_one(".author").get_text(),
            "tags": [tag.get_text() for tag in quote.select(".tag")]
        })
    
    # Print results
    print(f"Extracted {len(quotes)} quotes")
    print(json.dumps(quotes[:3], indent=2))
    
    # Save to file
    with open("output/quotes.json", "w") as f:
        json.dump(quotes, f, indent=2)
    
    print("Results saved to output/quotes.json")

if __name__ == "__main__":
    main()
