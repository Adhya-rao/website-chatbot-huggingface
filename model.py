import os
from dotenv import load_dotenv
import pickle

import torch
from huggingface_hub import InferenceClient
from huggingface_hub.errors import HfHubHTTPError
from transformers import AutoModelForQuestionAnswering, AutoTokenizer

# Load environment variables
load_dotenv()

MODEL_ID = "deepset/roberta-base-squad2"
_local_tokenizer = None
_local_model = None


def load_data():
    if not os.path.exists("data.pkl"):
        print("data.pkl not found. Run web_scrape.py first.")
        return None
    with open("data.pkl", "rb") as f:
        return pickle.load(f)


def _use_hf_inference_api() -> bool:
    return os.getenv("USE_HF_INFERENCE_API", "").lower() in ("1", "true", "yes")


def _load_local_qa():
    global _local_tokenizer, _local_model
    if _local_model is None:
        print("Loading local QA model (first run may download model weights)...", flush=True)
        _local_tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        _local_model = AutoModelForQuestionAnswering.from_pretrained(MODEL_ID)
        _local_model.eval()


def _best_span_answer(
    tokenizer,
    model,
    question: str,
    context: str,
    max_answer_tokens: int = 100,
):
    """Extractive QA without the removed transformers v5 `pipeline('question-answering')`."""
    inputs = tokenizer(
        question,
        context,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    )
    seq_ids = inputs.sequence_ids(0)
    with torch.inference_mode():
        outputs = model(**inputs)

    start_logits = outputs.start_logits[0]
    end_logits = outputs.end_logits[0]
    device = start_logits.device
    n = start_logits.shape[0]

    # score[i,j] = start_logit[i] + end_logit[j], only valid spans in context with i <= j
    ctx = torch.tensor(
        [1.0 if sid == 1 else 0.0 for sid in seq_ids],
        device=device,
        dtype=start_logits.dtype,
    )
    scores = start_logits.unsqueeze(1) + end_logits.unsqueeze(0)
    scores = scores.masked_fill(~torch.triu(torch.ones((n, n), device=device, dtype=torch.bool)), float("-inf"))
    scores = scores.masked_fill((ctx.unsqueeze(0) * ctx.unsqueeze(1)) < 1, float("-inf"))
    ii = torch.arange(n, device=device).unsqueeze(1)
    jj = torch.arange(n, device=device).unsqueeze(0)
    span_len = jj - ii + 1
    scores = scores.masked_fill((span_len > max_answer_tokens) | (span_len < 1), float("-inf"))

    flat = scores.view(-1)
    best_flat = int(flat.argmax().item())
    best_start = best_flat // n
    best_end = best_flat % n

    if not torch.isfinite(scores[best_start, best_end]):
        return "", 0.0

    input_ids = inputs["input_ids"][0]
    answer_ids = input_ids[best_start : best_end + 1]
    answer = tokenizer.decode(answer_ids, skip_special_tokens=True).strip()

    p_start = torch.softmax(start_logits, dim=0)[best_start].item()
    p_end = torch.softmax(end_logits, dim=0)[best_end].item()
    score = p_start * p_end
    return answer, float(score)


def _answer_local(question: str, context: str):
    _load_local_qa()
    return _best_span_answer(_local_tokenizer, _local_model, question, context)


def _answer_api(client: InferenceClient, question: str, context: str):
    result = client.question_answering(question=question, context=context)
    items = result if isinstance(result, list) else [result]
    if not items:
        return None, None
    res = items[0]
    answer = res.get("answer", "") if isinstance(res, dict) else getattr(res, "answer", "")
    score = res.get("score", 0.0) if isinstance(res, dict) else getattr(res, "score", 0.0)
    return answer, float(score)


def chatbot():
    """
    Console chatbot: QA via local model by default (no Inference API token permissions needed).

    Set USE_HF_INFERENCE_API=1 and use a HUGGINGFACE_API_KEY with Inference Provider access
    to use the hosted API instead.
    """
    data = load_data()
    if not data:
        return

    context = data.get("context", "")
    label = data.get("label", "N/A")

    print(f"Chatbot ready using data from: {label}")
    print(f"Context length: {len(context)}")
    print("Ask questions about the website (type 'quit' to exit).\n")

    use_api = _use_hf_inference_api()
    api_key = os.getenv("HUGGINGFACE_API_KEY")
    client = None

    if use_api:
        if not api_key:
            print("USE_HF_INFERENCE_API is set but HUGGINGFACE_API_KEY is missing.")
            return
        client = InferenceClient(token=api_key, model=MODEL_ID)
        print("Using Hugging Face Inference API.\n")
    else:
        print(
            "Using local QA model (no cloud Inference API). "
            "To use the API instead, set USE_HF_INFERENCE_API=1 and a token with Inference Provider access.\n"
        )

    ctx = context[:4000]

    while True:
        question = input("You: ").strip()

        if question.lower() in ["quit", "exit", "bye"]:
            print("Goodbye!")
            break

        if not question:
            continue

        try:
            if use_api:
                answer, score = _answer_api(client, question, ctx)
                if answer is None:
                    print("Unexpected API response.\n")
                    continue
            else:
                answer, score = _answer_local(question, ctx)
        except HfHubHTTPError as e:
            if getattr(e, "response", None) is not None and e.response.status_code == 403:
                print(
                    "Inference API returned 403: your token needs permission to call Inference Providers.\n"
                    "Create or edit a token at https://huggingface.co/settings/tokens or unset USE_HF_INFERENCE_API "
                    "to use the local model.\n"
                )
            else:
                print(f"API error: {e}\n")
            continue
        except Exception as e:
            print(f"Error: {e}\n")
            continue

        print(f"Bot: {answer} (score: {score:.4f})\n")


if __name__ == "__main__":
    chatbot()
