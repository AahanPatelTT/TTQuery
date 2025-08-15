#!/usr/bin/env python3
import os
import sys
import argparse
from typing import Optional


def query_via_litellm(prompt: str, model: str, api_key: str, base_url: str, timeout: Optional[int] = 60) -> str:
    try:
        from litellm import completion  # type: ignore
    except Exception as exc:
        raise RuntimeError("Please install 'litellm' (pip install litellm)") from exc

    try:
        resp = completion(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            api_key=api_key,
            base_url=base_url,
            # Force OpenAI-compatible routing to LiteLLM proxy
            custom_llm_provider="openai",
            timeout=timeout,
        )
        # OpenAI-style response object
        choice = resp.choices[0]
        # Support both dict and pydantic-like object
        message = getattr(choice, "message", None) or choice.get("message")
        content = getattr(message, "content", None) or (message.get("content") if isinstance(message, dict) else None)
        return str(content) if content is not None else str(resp)
    except Exception as e:
        return f"Error: {e}"


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(description="LiteLLM proxy query helper for Gemini 2.5 Pro")
    parser.add_argument("--prompt", type=str, default="Hello from LiteLLM proxy!", help="Prompt to send")
    parser.add_argument("--model", type=str, default="gemini/gemini-2.5-pro", help="Model name")
    parser.add_argument("--timeout", type=int, default=60, help="Request timeout (seconds)")
    args = parser.parse_args(argv)

    api_key = os.getenv("LITELLM_API_KEY")
    base_url = os.getenv("LITELLM_BASE_URL")

    if not api_key or not base_url:
        print("LITELLM_API_KEY and LITELLM_BASE_URL must be set in the environment.", file=sys.stderr)
        return 2

    output = query_via_litellm(args.prompt, args.model, api_key, base_url, timeout=int(args.timeout))
    print(output)
    return 0


if __name__ == "__main__":
    sys.exit(main())


