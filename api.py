import os
from dotenv import load_dotenv
import pickle
from huggingface_hub import InferenceClient

# Load environment variables
load_dotenv()

def demo_api():
    """
    Demonstrate HF API with data.pkl using InferenceClient.post.
    """
    if not os.path.exists('data.pkl'):
        print("data.pkl not found. Run web_scrape.py first.")
        return
    
    with open('data.pkl', 'rb') as f:
        data = pickle.load(f)
    
    context = data.get('context', '')
    label = data.get('label', 'N/A')
    
    print(f"Using context from: {label}")
    print(f"Context length: {len(context)} chars")
    print(f"Preview: {context[:200]}...")
    
    api_key = os.getenv('HUGGINGFACE_API_KEY')
    if not api_key:
        print("HUGGINGFACE_API_KEY not found in .env")
        return
    
    client = InferenceClient(token=api_key)
    
    model = "deepset/roberta-base-squad2"  # Hosted QA model
    question = "What is this website about?"
    payload = {
        "question": question,
        "context": context[:4000]
    }
    
    result = client.post(json=payload, model=model)
    
    if isinstance(result, list) and len(result) > 0:
        res = result[0]
        print(f"\nQuestion: {question}")
        print(f"Answer: {res.get('answer', '')}")
        print(f"Score: {res.get('score', 0.0):.4f}")
    else:
        print("Unexpected API response:", result)

if __name__ == "__main__":
    demo_api()

