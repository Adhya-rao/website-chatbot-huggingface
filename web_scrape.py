import requests
from bs4 import BeautifulSoup
import pickle
import sys
import os

def scrape_and_save(url):
    """
    Scrape website content and save to data.pkl as {'context': text_content, 'label': title}
    """
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')

        # Extract title
        title = soup.title.string.strip() if soup.title else 'No title found'

        # Prefer main content to reduce nav/footer noise (better answers for pricing, etc.)
        root = soup.find('main') or soup.find('article') or soup.find(
            attrs={'role': 'main'}
        ) or soup.body
        if root is None:
            root = soup

        skip_tags = {
            'style', 'script', 'head', 'title', 'meta', 'noscript',
            '[document]', 'nav', 'footer', 'header', 'aside', 'form',
        }

        def is_skipped_element(el) -> bool:
            for ancestor in el.parents:
                if ancestor is root:
                    break
                name = getattr(ancestor, 'name', None)
                if name in skip_tags or name in ('nav', 'footer', 'header', 'aside'):
                    return True
            return False

        text_elements = root.find_all(string=True)
        filtered_texts = []
        for t in text_elements:
            s = t.strip()
            if not s or t.parent.name in skip_tags:
                continue
            if is_skipped_element(t):
                continue
            filtered_texts.append(s)

        full_text = ' '.join(filtered_texts)
        words = full_text.split()
        # Enough text for ChatGPT-style context; model.py / local QA can still slice smaller
        context = ' '.join(words[:4000])
        
        data = {
            'context': context,
            'label': title
        }
        
        with open('data.pkl', 'wb') as f:
            pickle.dump(data, f)
        
        print(f"Saved scraped data from {url}:")
        print(f"Title: {title}")
        print(f"Context length: {len(context)} chars")
        return True
        
    except Exception as e:
        print(f"Error scraping {url}: {e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) > 1:
        url = sys.argv[1]
    else:
        url = "https://botpenguin.com/"
    
    print(f"Scraping {url}...")
    scrape_and_save(url)

