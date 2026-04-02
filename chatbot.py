"""
================================================================================
WEBSITE CHATBOT — Hugging Face Inference API (chat completion) — STEP-BY-STEP
================================================================================

1) Environment setup
   - Python 3.10+, pip install -r requirements.txt
   - Create .env with:
       HUGGINGFACE_API_KEY=hf_...
     Optional:
       HF_CHAT_MODEL=<exact model id>  — if unset, the script tries several popular
       chat models until one works with your enabled Inference Providers.
   - Token: Inference permissions; enable providers at huggingface.co/settings/inference
   - Browse models: huggingface.co/inference/models

2) Extract website data (Beautiful Soup)
   - python web_scrape.py [URL]   → writes data.pkl

3) Process data
   - This script loads data.pkl, uses title + scraped text as grounding for the assistant.

4) Chatbot (Hugging Face API)
   - huggingface_hub.InferenceClient.chat_completion with a chat instruct model.
   - System prompt: answer only from scraped content.

5) Console
   - python chatbot.py
   - quit / exit / bye to exit.

Files: web_scrape.py (scrape), chatbot.py (this file)
================================================================================
"""

from __future__ import annotations

import os
import pickle
import sys

from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from huggingface_hub.errors import HfHubHTTPError

load_dotenv()

# Tried in order until one works with your HF account / enabled providers (if HF_CHAT_MODEL unset)
CHAT_MODEL_FALLBACKS = [
    "google/gemma-2-2b-it",
    "meta-llama/Llama-3.2-3B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.3",
    "Qwen/Qwen2.5-7B-Instruct",
    "HuggingFaceTB/SmolLM2-1.7B-Instruct",
]
MAX_CONTEXT_CHARS = int(os.getenv("MAX_WEBSITE_CONTEXT_CHARS", "12000"))


def _chat_model_candidates() -> list[str]:
    explicit = os.getenv("HF_CHAT_MODEL", "").strip()
    if explicit:
        return [explicit]
    return list(CHAT_MODEL_FALLBACKS)


def _is_provider_model_error(exc: BaseException) -> bool:
    text = str(exc).lower()
    return (
        "model_not_supported" in text
        or "not supported by any provider" in text
        or "invalid_request_error" in text and "model" in text
    )


def load_scraped_data(path: str = "data.pkl"):
    if not os.path.exists(path):
        print("data.pkl not found. Run: python web_scrape.py [URL]")
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def build_messages(site_title: str, website_text: str, history: list[dict]) -> list[dict]:
    system = (
        "You are a helpful assistant for a specific website. "
        "Answer the user's questions using ONLY the website content provided below. "
        "If the answer is not contained or inferable from that content, say you don't "
        "have that information from the site. Be concise and clear.\n\n"
        f"Website title: {site_title}\n\n"
        f"Website content:\n{website_text}"
    )
    return [{"role": "system", "content": system}, *history]


def _reply_text(completion) -> str:
    choice = completion.choices[0]
    msg = choice.message
    content = getattr(msg, "content", None) if not isinstance(msg, dict) else msg.get("content")
    return (content or "").strip()


def run_console_chatbot() -> None:
    data = load_scraped_data()
    if not data:
        sys.exit(1)

    raw = data.get("context", "")
    label = data.get("label", "Unknown page")
    website_text = raw[:MAX_CONTEXT_CHARS]
    if len(raw) > MAX_CONTEXT_CHARS:
        website_text += "\n\n[... content truncated for model context limit ...]"

    api_key = os.getenv("HUGGINGFACE_API_KEY")
    if not api_key:
        print("HUGGINGFACE_API_KEY is not set. Add it to your .env file.")
        sys.exit(1)

    client = InferenceClient(token=api_key)
    candidates = _chat_model_candidates()
    active_model: str | None = None
    announced_model = False

    print(f"Chatbot ready — site: {label}")
    if os.getenv("HF_CHAT_MODEL", "").strip():
        print(f"Model (from HF_CHAT_MODEL): {candidates[0]}")
    else:
        print("Model: auto (tries common chat models until one matches your Inference Providers)")
    print(f"Grounding context: {len(website_text)} characters (from scraped data)")
    print("Ask about the website. Commands: quit, exit, bye\n")

    history: list[dict] = []

    while True:
        try:
            user_text = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if user_text.lower() in {"quit", "exit", "bye"}:
            print("Goodbye!")
            break
        if not user_text:
            continue

        history.append({"role": "user", "content": user_text})
        trimmed = history[-8:]
        messages = build_messages(label, website_text, trimmed)

        try:
            completion = None
            if active_model:
                completion = client.chat_completion(
                    model=active_model,
                    messages=messages,
                    temperature=0.3,
                    max_tokens=600,
                )
            else:
                last_err: BaseException | None = None
                for m in candidates:
                    try:
                        completion = client.chat_completion(
                            model=m,
                            messages=messages,
                            temperature=0.3,
                            max_tokens=600,
                        )
                        active_model = m
                        if not announced_model:
                            print(f"(Using inference model: {m})\n")
                            announced_model = True
                        break
                    except (HfHubHTTPError, Exception) as err:
                        last_err = err
                        if _is_provider_model_error(err):
                            continue
                        raise
                if completion is None and last_err is not None:
                    raise last_err

            assert completion is not None
            reply = _reply_text(completion)
        except HfHubHTTPError as e:
            history.pop()
            code = getattr(getattr(e, "response", None), "status_code", None)
            if code == 403:
                print(
                    "Hugging Face returned 403: your token cannot call Inference Providers.\n"
                    "Create a token at https://huggingface.co/settings/tokens with "
                    "Inference permissions, and enable providers at "
                    "https://huggingface.co/settings/inference\n"
                )
            elif _is_provider_model_error(e):
                print(
                    "No chat model in the fallback list worked with your enabled providers.\n"
                    "Set HF_CHAT_MODEL in .env to an ID from https://huggingface.co/inference/models "
                    "(pick one your account can run), or enable more providers in HF settings.\n"
                    f"Details: {e}\n"
                )
            else:
                print(f"Hugging Face API error: {e}\n")
            continue
        except Exception as e:
            print(f"Error calling API: {e}\n")
            history.pop()
            continue

        history.append({"role": "assistant", "content": reply})
        print(f"Bot: {reply}\n")


if __name__ == "__main__":
    run_console_chatbot()
