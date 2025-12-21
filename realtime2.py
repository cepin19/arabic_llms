import json
import asyncio
import aiohttp
from pathlib import Path
from typing import List
from openai import AsyncOpenAI

# -------- CONFIGURATION --------
#MODEL = "gpt-4.1-mini"  # Default model
MODEL="CohereLabs/aya-expanse-8b"
SYSTEM_PROMPT = (
    "You are a professional, very precise translator and a native speaker. "
    "Translate inputs based on the instructions and always print out only the text of "
    "the best possible translation, no explanations. Keep the same formatting "
    "(e.g. markup, lines, spacing) as the original. Do not translate untranslatable "
    "parts of the input (URLs, code and similar)."
)
MAX_CONCURRENT_REQUESTS = 256  # Limit concurrency to stay under rate limits


# -------- ASYNC TRANSLATION FUNCTION --------
async def translate_line(client: AsyncOpenAI, model:str, line: str, index: int, lang: str):
    """Send one translation request and return (index, translation)."""
    try:
        print(f"{index}: Translate the following text into {lang}: {line.strip()}")
        response = await client.chat.completions.create(
            model=model,
            temperature=0,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Translate the following text into {lang}, only print out the translation, not add any explanations: " + line.strip()},
            ],
        )
        translated = response.choices[0].message.content.strip()
        translated = " ".join(translated.split())
        return index, translated
    except Exception as e:
        print(f"⚠️  Error translating line {index}: {e}")
        return index, ""


# -------- PROCESSING PIPELINE --------
async def translate_file(client: AsyncOpenAI, model: str, input_file: Path, output_file: Path, lang: str):
    """Translate an input file line-by-line using the real-time Chat API."""
    lines = input_file.read_text(encoding="utf-8").split('\n')
    print(f"📘 Loaded {len(lines)} lines from {input_file}")

    results = ["" for _ in range(len(lines))]
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

    async def worker(i, line):
        async with semaphore:
            _, translation = await translate_line(client, model, line, i, lang)
            results[i] = translation

    tasks = [worker(i, line) for i, line in enumerate(lines)]

    print(f"🚀 Starting translation with up to {MAX_CONCURRENT_REQUESTS} concurrent requests...")
    await asyncio.gather(*tasks)
    print("✅ All translations completed.")

    # Save in original order
    output_file.write_text("\n".join(results), encoding="utf-8")
    print(f"🧩 Translations written to {output_file}")


# -------- ENTRY POINT --------
def main():
    import argparse

    parser = argparse.ArgumentParser(description="Translate text in real time using OpenAI-compatible Chat API.")
    parser.add_argument("input", type=Path, help="Path to input text file (one line per segment).")
    parser.add_argument("--output", type=Path, default=Path("translated_realtime.txt"), help="Path to output file.")
    parser.add_argument("--model", type=str, default=MODEL, help="")
    parser.add_argument("--lang", type=str, default="English", help="")

    parser.add_argument(

        "--base-url",
        type=str,
        default="https://api.openai.com/v1",
        help="Base URL for the API (default: OpenAI API). For example: http://localhost:8000/v1",
    )
    args = parser.parse_args()

    # Initialize the client with the user-defined base_url
    client = AsyncOpenAI(base_url=args.base_url)

    asyncio.run(translate_file(client, args.model, args.input, args.output, args.lang))


if __name__ == "__main__":
    main()

