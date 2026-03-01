from typing import Iterator, List, Dict, Optional
from openai import OpenAI


class LMStudioClient:
    def __init__(self, base_url: str = "http://localhost:1234/v1", model: str = "local-model", temperature: float = 0.7):
        self._client = OpenAI(base_url=base_url, api_key="not-needed")
        self._model = model
        self._temperature = temperature

    def stream(self, messages: List[Dict], system_message: str) -> Iterator[str]:
        full_messages = [{"role": "system", "content": system_message}] + messages
        response = self._client.chat.completions.create(
            model=self._model,
            messages=full_messages,
            stream=True,
            temperature=self._temperature,
        )
        for chunk in response:
            content = chunk.choices[0].delta.content
            if content:
                yield content

    def respond(self, messages: List[Dict], system_message: str) -> str:
        return "".join(self.stream(messages, system_message))

    def update(self, base_url: str, model: str, temperature: float) -> None:
        self._client = OpenAI(base_url=base_url, api_key="not-needed")
        self._model = model
        self._temperature = temperature


class OllamaClient:
    def __init__(self, model: str = "llama2", temperature: float = 0.7):
        self._model = model
        self._temperature = temperature

    def _build_prompt(self, messages: List[Dict], system_message: str) -> str:
        prompt = f"{system_message}\n\n"
        for msg in messages:
            role = "Human" if msg["role"] == "user" else "Assistant"
            prompt += f"{role}: {msg['content']}\n"
        prompt += "Assistant:"
        return prompt

    def respond(self, messages: List[Dict], system_message: str) -> str:
        from langchain_community.llms import Ollama
        llm = Ollama(model=self._model, temperature=self._temperature)
        return llm(self._build_prompt(messages, system_message))

    def stream(self, messages: List[Dict], system_message: str) -> Iterator[str]:
        yield self.respond(messages, system_message)

    def update(self, model: str, temperature: float) -> None:
        self._model = model
        self._temperature = temperature
