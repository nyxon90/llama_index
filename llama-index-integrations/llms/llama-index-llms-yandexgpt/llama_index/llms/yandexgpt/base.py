from typing import Optional, List, Mapping, Any
import httpx
import json
from llama_index.core.llms import (
    CustomLLM,
    CompletionResponse,
    CompletionResponseGen,
    LLMMetadata,
)
from llama_index.core.base.llms.generic_utils import get_from_param_or_env
from llama_index.core.llms.callbacks import llm_completion_callback
from llama_index.core.bridge.pydantic import Field
import os

DEFAULT_YANDEXGPT_MODEL = "yandexgpt-lite"
DEFAULT_YANDEXGPT_TEMPERATURE = 0.6
DEFAULT_YANDEXGPT_MAXTOKENS = 2000
DEFAULT_YANDEXGPT_ENDPOINT = "https://llm.api.cloud.yandex.net/foundationModels/v1/"

os.environ['YANDEXGPT_FOLDER_ID'] = "b1ggtbfmbkjumd7ledf0"
os.environ['YANDEXGPT_API_KEY'] = "AQVNwyqW1-BhppvknJdHYr2omck3mGxUZyPH3U4k"


class YandexGPT(CustomLLM):
    """
    YandexGPT LLM
    """
    base_url: str = Field(
        default=DEFAULT_YANDEXGPT_ENDPOINT,
        description="Base url the model is hosted under.",
    )


    model: str = Field(
        default=DEFAULT_YANDEXGPT_MODEL,
        description="The YandexGPT model to use.",
    )
    temperature: float = Field(
        default=DEFAULT_YANDEXGPT_TEMPERATURE,
        description="The temperature to use during generation.",
        gte=0.0,
        lte=1.0,
    )
    max_tokens: int = Field(
        default=DEFAULT_YANDEXGPT_MAXTOKENS,
        description="The maximum number of tokens to generate.",
        gt=0,
    )

    api_key: str = Field(
        description="YandexGPT api key."
    )

    folder_id: str = Field(
        description="YandexGPT folder id."
    )

    def __init__(
            self,
            model: str = DEFAULT_YANDEXGPT_MODEL,
            base_url: str = DEFAULT_YANDEXGPT_ENDPOINT,
            temperature: float = DEFAULT_YANDEXGPT_TEMPERATURE,
            max_tokens: int = DEFAULT_YANDEXGPT_MAXTOKENS,
            api_key: Optional[str] = None,
            folder_id: Optional[str] = None,
    ):
        api_key = get_from_param_or_env("api_key", api_key, "YANDEXGPT_API_KEY", "")

        if not api_key:
            raise ValueError(
                "You must provide an API key to use YandexGPT. "
                "You can either pass it in as an argument or set it `YANDEXGPT_API_KEY."
            )

        folder_id = get_from_param_or_env("folder_id", folder_id, "YANDEXGPT_FOLDER_ID", "")

        if not folder_id:
            raise ValueError(
                "You must provide an folder id to use YandexGPT. "
                "You can either pass it in as an argument or set it `YANDEXGPT_FOLDER_ID."
            )

        super().__init__(
            model=model,
            base_url=base_url,
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=api_key,
            folder_id=folder_id,
        )

    @property
    def metadata(self) -> LLMMetadata:
        """Get LLM metadata."""
        return LLMMetadata(
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

    @llm_completion_callback()
    def complete(self, prompt: str, formatted: bool = False, **kwargs: Any) -> CompletionResponse:
        payload = {
            "modelUri": f"gpt://{self.folder_id}/{self.model}",
            "completionOptions": {
                "stream": False,
                "temperature": self.temperature,
                "maxTokens": self.max_tokens
            },
            "messages": [
                {
                    "role": "user",
                    "text": prompt,
                }
            ]
        }
        with httpx.Client() as client:
            response = client.post(
                url=self.base_url + 'completion',
                json=payload,
                headers=
                {
                    "Content-Type": "application/json",
                    "Authorization": f"Api-Key {self.api_key}"
                }
            )
            response.raise_for_status()
            raw = response.json()
            text = raw["result"]["alternatives"][0]["message"]["text"]
            return CompletionResponse(
                text=text,
                raw=raw,
            )

    @llm_completion_callback()
    def stream_complete(
            self, prompt: str, **kwargs: Any
    ) -> CompletionResponseGen:
        payload = {
            "modelUri": f"gpt://{self.folder_id}/{self.model}",
            "completionOptions": {
                "stream": True,
                "temperature": self.temperature,
                "maxTokens": self.max_tokens
            },
            "messages": [
                {
                    "role": "user",
                    "text": prompt,
                }
            ]
        }

        with httpx.Client() as client:
            with client.stream(
                    method="POST",
                    url=self.base_url + 'completion',
                    json=payload,
                    headers=
                    {
                        "Content-Type": "application/json",
                        "Authorization": f"Api-Key {self.api_key}"
                    }
            ) as response:
                response.raise_for_status()
                text = ""
                for line in response.iter_lines():
                    if line:
                        chunk = json.loads(line)
                        new_text = chunk["result"]["alternatives"][0]["message"]["text"]
                        delta = chunk["result"]["alternatives"][0]["message"]["text"]
                        text += delta
                        yield CompletionResponse(
                            delta=delta,
                            text=text,
                            raw=chunk,

                        )


llm = YandexGPT()

# response = llm.stream_complete("Напиши статью о Москве")
# for chunk in response:
#     print("--------------------------------------------------")
#     print(chunk.delta)
#     print("--------------------------------------------------")

