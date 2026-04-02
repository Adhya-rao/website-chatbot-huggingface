import pickle
import os

def check_data():
    """
    Load and print info from data.pkl
    """
    if not os.path.exists('data.pkl'):
        print("data.pkl not found. Run web_scrape.py first.")
        return
    
    with open('data.pkl', 'rb') as f:
        data = pickle.load(f)
    
    print("Data loaded successfully:")
    print(f"Keys: {list(data.keys())}")
    print(f"Label: {data.get('label', 'N/A')}")
    print(f"Context length: {len(data.get('context', ''))} characters")
    print("\nSample context (first 500 chars):")
    print(data.get('context', '')[:500] + '...' if len(data.get('context', '')) > 500 else data.get('context', ''))

if __name__ == "__main__":
    check_data()

