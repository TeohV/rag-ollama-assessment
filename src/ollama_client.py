import requests


def get_embedding(text: str, config):
    response = requests.post(
        f"{config.ollama_url}/api/embed",
        json={
            "model": config.embed_model,
            "input": text,
        },
        timeout=120,
    )

    if response.status_code != 200:
        raise RuntimeError(
            "Ollama embedding error\n"
            f"Status code: {response.status_code}\n"
            f"Response text: {response.text}\n"
            f"Text preview: {text[:500]}"
        )

    data = response.json()
    return data["embeddings"][0]


def chat(prompt: str, config):
    response = requests.post(
        f"{config.ollama_url}/api/chat",
        json={
            "model": config.llm_model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            "stream": False,
        },
        timeout=300,
    )

    response.raise_for_status()

    data = response.json()
    return data["message"]["content"]