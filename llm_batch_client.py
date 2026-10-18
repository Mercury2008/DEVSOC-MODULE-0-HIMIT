"""Send prompts from a text file to an LLM API (Groq) and save the responses as JSON.

Usage:
    export GROQ_API_KEY="your_key_here"
    python llm_batch_client.py --input prompts.txt --output responses.json

Input file format: one prompt per line. Blank lines and lines starting with '#' are ignored.
Get a free API key at https://console.groq.com/keys
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

API_URL = "https://api.groq.com/openai/v1/chat/completions"
DEFAULT_MODEL = "llama-3.1-8b-instant"  # check console.groq.com/docs/models for current models
MAX_RETRIES = 4
REQUEST_TIMEOUT_SECONDS = 60


def read_prompts(input_path: Path) -> list[str]:
    """Read prompts from a text file, skipping blank lines and '#' comments."""
    with input_path.open(encoding="utf-8") as file:
        lines = [line.strip() for line in file]
    return [line for line in lines if line and not line.startswith("#")]


def query_llm(prompt: str, model: str, api_key: str, temperature: float) -> str:
    """Send one prompt to the API and return the model's reply text.

    Retries with exponential backoff on rate limits (429) and server errors (5xx).
    Raises requests.HTTPError / requests.RequestException if all attempts fail.
    """
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
    }

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = requests.post(
                API_URL, headers=headers, json=payload, timeout=REQUEST_TIMEOUT_SECONDS
            )
            if response.status_code == 429 or response.status_code >= 500:
                raise requests.HTTPError(f"HTTP {response.status_code}: {response.text[:200]}")
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
        except requests.RequestException as error:
            if attempt == MAX_RETRIES:
                raise
            wait_seconds = 2 ** attempt
            print(f"  attempt {attempt} failed ({error}); retrying in {wait_seconds}s", file=sys.stderr)
            time.sleep(wait_seconds)
    raise RuntimeError("unreachable")


def save_results(results: list[dict], output_path: Path) -> None:
    """Write results to a JSON file (pretty-printed, UTF-8)."""
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=2, ensure_ascii=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch-query an LLM API from a text file.")
    parser.add_argument("--input", "-i", type=Path, default=Path("prompts.txt"))
    parser.add_argument("--output", "-o", type=Path, default=Path("responses.json"))
    parser.add_argument("--model", "-m", default=DEFAULT_MODEL)
    parser.add_argument("--temperature", "-t", type=float, default=0.7)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        sys.exit("Error: set the GROQ_API_KEY environment variable first.")
    if not args.input.exists():
        sys.exit(f"Error: input file not found: {args.input}")

    prompts = read_prompts(args.input)
    if not prompts:
        sys.exit("Error: no prompts found in the input file.")

    results = []
    for index, prompt in enumerate(prompts, start=1):
        print(f"[{index}/{len(prompts)}] {prompt[:60]}")
        record = {
            "id": index,
            "prompt": prompt,
            "model": args.model,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        try:
            record["response"] = query_llm(prompt, args.model, api_key, args.temperature)
            record["error"] = None
        except (requests.RequestException, KeyError, ValueError) as error:
            record["response"] = None
            record["error"] = str(error)
        results.append(record)
        save_results(results, args.output)  # save after every call so progress isn't lost

    failed = sum(1 for record in results if record["error"])
    print(f"Done. {len(results) - failed} succeeded, {failed} failed. Saved to {args.output}")


if __name__ == "__main__":
    main()
