# A collection of APIs to use for generating model responses
# OpenAI/Anthropic models are supported via their respective APIs below
# Other models are routed to LocalAPI, which assumes a VLLM instance running on localhost:8000
# Additional APIs (e.g., Google, Together, etc.) may need to be added
# Only asyncronous apis are supported
# Non-asnyc requests can be handled, but chat() should still be async, so include something like await asyncio.sleep(0)

from abc import ABC, abstractmethod
from typing import List, Dict
import os


import backoff
import openai
import anthropic
import time
import httpx


class ChatAPI(ABC):

    @abstractmethod
    def __init__(self, model):
        pass

    @abstractmethod
    async def chat(self, messages):
        pass


class OpenAIAPI(ChatAPI):

    def __init__(self, model: str):
        self.model = model
        self.client = openai.AsyncClient(
            api_key=os.environ.get("OPENAI_API_KEY"))

    @backoff.on_exception(backoff.fibo, (openai.OpenAIError), max_tries=5, max_value=30)
    async def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        
        if self.model.startswith("o1"):
            if messages[0]["role"] == "system":
                system_message = messages.pop(0)["content"]
                user_message = messages[0]["content"]
                messages[0] = {
                    "role": "user",
                    "content": f"<|BEGIN_SYSTEM_MESSAGE|>\n{system_message.strip()}\n<|END_SYSTEM_MESSAGE|>\n\n{user_message}"
                }

            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
            )
        
        else:   
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                **kwargs,
            )
        return response.choices[0].message.content


class AnthropicAPI(ChatAPI):

    def __init__(self, model: str):
        self.model = model
        self.client = anthropic.AsyncClient(
            api_key=os.environ.get("ANTHROPIC_API_KEY"))

    @backoff.on_exception(backoff.fibo, anthropic.AnthropicError, max_tries=5, max_value=30)
    async def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        
        if messages[0]["role"] == "system":
            system_message = messages[0]["content"]
            messages = messages[1:]
        else:
            system_message = ""

        response = await self.client.messages.create(
            model=self.model,
            messages=messages,
            system=system_message,
            # max_tokens=8192, # 4096 for claude-3-*
            **kwargs,
        )
        
        return response.content[0].text

class TogetherAPI(ChatAPI):

    def __init__(self, model: str):
        self.model = model
        self.client = openai.AsyncClient(
            api_key=os.environ.get("TOGETHER_API_KEY"),
            base_url="https://api.together.xyz/v1"
        )

    @backoff.on_exception(backoff.fibo, (openai.OpenAIError), max_tries=5, max_value=30)
    async def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            **kwargs,
        )
        return response.choices[0].message.content
    

class LocalAPI(ChatAPI):

    def __init__(self, model: str):
        self.model = model
        self.client = openai.AsyncClient(base_url="http://localhost:8000/v1", api_key="EMPTY")

    @backoff.on_exception(backoff.fibo, (openai.OpenAIError), max_tries=5, max_value=30)
    async def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            **kwargs,
        )
        return response.choices[0].message.content
    
    @backoff.on_exception(backoff.fibo, (openai.OpenAIError), max_tries=5, max_value=30)
    async def complete(self, prompt: str, **kwargs) -> str:
        response = await self.client.completions.create(
            model=self.model,
            prompt=prompt,
            **kwargs,
        )
        return response.choices[0].text
    

class MaritacaAPI(ChatAPI):
    """
    Cliente mínimo para a API da Maritaca (OpenAI-compatible).
    Usa a env var MARITACA_API_KEY.
    Ex.: model="sabia-3.1" (ou "sabia-3", "sabiazinho-3" etc.)
    """
    def __init__(self, model: str):
        self.model = model
        self.client = openai.AsyncClient(
            api_key=os.environ.get("MARITACA_API_KEY"),
            base_url="https://chat.maritaca.ai/api"
        )
        if not os.environ.get("MARITACA_API_KEY"):
            raise RuntimeError("Defina MARITACA_API_KEY no ambiente.")

    @backoff.on_exception(backoff.fibo, (openai.OpenAIError), max_tries=5, max_value=30)
    async def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        # ---- normalização de kwargs ----
        # Aceita aliases comuns e converte para o OpenAI-compat da Maritaca
        norm = dict(kwargs) if kwargs else {}
        # converte variantes para max_tokens
        if "max_completion_tokens" in norm and "max_tokens" not in norm:
            norm["max_tokens"] = norm.pop("max_completion_tokens")
        if "max_output_tokens" in norm and "max_tokens" not in norm:
            norm["max_tokens"] = norm.pop("max_output_tokens")
        # alguns servidores rejeitam campos desconhecidos; remova o que sobrou:
        for k in list(norm.keys()):
            if k in {"top_k", "presence_penalty", "frequency_penalty"}:
                # mantenha só se você souber que a API suporta
                norm.pop(k, None)

        resp = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            **norm,
        )
        return resp.choices[0].message.content
    
class OpenRouterAPI(ChatAPI):
    """
    Cliente mínimo para OpenRouter (OpenAI-compatible).
    Usa OPENROUTER_API_KEY e base_url https://openrouter.ai/api/v1
    Use --judge_model "openrouter:<provider/model_id>", ex.:
      openrouter:anthropic/claude-3.5-sonnet
      openrouter:google/gemini-2.5-flash
      openrouter:meta-llama/llama-3.1-70b-instruct
    """
    def __init__(self, model: str):
        # modelo vem como "openrouter:<id_do_modelo>", ex: "openrouter:anthropic/claude-3.5-sonnet"
        if not model.startswith("openrouter:"):
            raise ValueError("Para OpenRouter, use --judge_model 'openrouter:<provider/model_id>'")
        self.model = model.split("openrouter:", 1)[1].strip()

        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise RuntimeError("Defina OPENROUTER_API_KEY no ambiente.")

        # headers opcionais (recomendado pelo OpenRouter p/ roteamento/telemetria)
        referer = os.environ.get("OPENROUTER_REFERER")  # ex.: https://seu-site-ou-repo
        site_title = os.environ.get("OPENROUTER_SITE_TITLE", "JudgeBench-BR")

        default_headers = {}
        if referer:
            default_headers["HTTP-Referer"] = referer
        if site_title:
            default_headers["X-Title"] = site_title

        # Cliente OpenAI-compat apontando para OpenRouter
        self.client = openai.AsyncClient(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
            default_headers=default_headers or None,
        )

    @backoff.on_exception(backoff.fibo, (openai.OpenAIError), max_tries=5, max_value=30)
    async def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        # normalização leve: vários servidores esperam max_tokens e não aceitam nomes alternativos
        norm = dict(kwargs) if kwargs else {}
        if "max_completion_tokens" in norm and "max_tokens" not in norm:
            norm["max_tokens"] = norm.pop("max_completion_tokens")
        if "max_output_tokens" in norm and "max_tokens" not in norm:
            norm["max_tokens"] = norm.pop("max_output_tokens")

        resp = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            **norm,
        )
        return resp.choices[0].message.content


# --- IBM Granite (watsonx.ai) ---
# --- IBM Granite (watsonx.ai) - /ml/v1/text/chat ---
class IBMGraniteAPI(ChatAPI):
    """
    Cliente para IBM watsonx.ai no endpoint de chat:
      POST {IBM_WATSONX_URL}/ml/v1/text/chat?version=YYYY-MM-DD

    Variáveis de ambiente necessárias:
      - IBM_CLOUD_API_KEY        (obrigatória)
      - IBM_WATSONX_URL          (ex.: https://us-south.ml.cloud.ibm.com)  [obrigatória]
      - IBM_WATSONX_PROJECT_ID   (obrigatória)
      - IBM_WATSONX_API_VERSION  (opcional, default interno abaixo; formato YYYY-MM-DD)

    Dica: export IBM_WATSONX_DEBUG=1 para logar payload e resposta raw.
    """

    def __init__(self, model: str):
        import os, re, time, httpx

        self._dbg = os.environ.get("IBM_WATSONX_DEBUG", "0") == "1"

        # --- normaliza model_id ---
        # aceita "granite-3.3-8b-instruct" e converte p/ "ibm/granite-3-3-8b-instruct"
        m = (model or "").strip()
        if not m:
            raise ValueError("Modelo não especificado.")
        # converte '3.3' -> '3-3'
        m = m.replace("3.3", "3-3")
        if not m.startswith("ibm/"):
            m = f"ibm/{m}"
        self.model = m

        # --- env vars ---
        self.api_key    = os.environ.get("IBM_CLOUD_API_KEY")
        self.base_url   = os.environ.get("IBM_WATSONX_URL", "")
        self.project_id = os.environ.get("IBM_WATSONX_PROJECT_ID")
        # escolha uma data estável/recente; pode sobrescrever via env
        self.api_ver    = os.environ.get("IBM_WATSONX_API_VERSION", "2025-04-23")

        if not self.api_key:
            raise RuntimeError("Defina IBM_CLOUD_API_KEY.")
        if not self.base_url.startswith("https://"):
            raise RuntimeError("Defina IBM_WATSONX_URL com https:// (ex.: https://us-south.ml.cloud.ibm.com).")
        if not self.project_id:
            raise RuntimeError("Defina IBM_WATSONX_PROJECT_ID.")
        if not re.match(r"^\d{4}-\d{2}-\d{2}$", self.api_ver):
            raise RuntimeError("IBM_WATSONX_API_VERSION deve ser YYYY-MM-DD (ex.: 2025-04-23).")

        # httpx Timeout: defina TODOS os 4 campos para evitar ValueError
        timeout = httpx.Timeout(connect=30.0, read=180.0, write=180.0, pool=30.0)

        # HTTP/1.1 é suficiente; se quiser HTTP/2, use http2=True
        self._http = httpx.AsyncClient(timeout=timeout)
        self._token = None
        self._token_exp = 0.0
        self._now = time.time

    async def _get_iam_token(self) -> str:
        import httpx
        if self._token and self._now() < self._token_exp - 60:
            return self._token
        r = await self._http.post(
            "https://iam.cloud.ibm.com/identity/token",
            data={
                "grant_type": "urn:ibm:params:oauth:grant-type:apikey",
                "apikey": self.api_key,
            },
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        r.raise_for_status()
        j = r.json()
        self._token = j["access_token"]
        self._token_exp = self._now() + int(j.get("expires_in", 3600))
        return self._token

    def _normalize_messages(self, messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """
        Converte mensagens estilo OpenAI para o formato aceito pelo watsonx.
        Garante role em minúsculas e ignora conteúdos vazios.
        """
        norm = []
        for m in messages:
            role = (m.get("role", "user") or "").lower().strip()
            content = (m.get("content") or "").strip()
            if content:
                norm.append({"role": role, "content": content})
        return norm

    @backoff.on_exception(backoff.fibo, Exception, max_tries=5, max_value=30)
    async def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        import json

        token = await self._get_iam_token()
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

        # parâmetros de geração → contrato do watsonx
        temperature = kwargs.get("temperature", 0.0)
        top_p = kwargs.get("top_p", 1.0)
        max_new = (
            kwargs.get("max_tokens")
            or kwargs.get("max_output_tokens")
            or kwargs.get("max_completion_tokens")
            or 1024  # default razoável p/ juiz
        )
        params = {
            "decoding_method": "greedy" if (temperature is None or float(temperature) <= 0.0) else "sample",
            "max_new_tokens": int(max_new),
            "temperature": float(temperature or 0.0),
            "top_p": float(top_p or 1.0),
        }

        payload = {
            "model_id": self.model,                       # ex.: "ibm/granite-3-3-8b-instruct"
            "messages": self._normalize_messages(messages),
            "parameters": params,
            "project_id": self.project_id,
        }

        url = f"{self.base_url.rstrip('/')}/ml/v1/text/chat?version={self.api_ver}"

        if self._dbg:
            try:
                print("=== [IBM] Payload ===")
                print(json.dumps(payload, ensure_ascii=False)[:2000])
            except Exception:
                print("=== [IBM] Payload (repr) ===")
                print(repr(payload)[:2000])

        r = await self._http.post(url, headers=headers, json=payload)

        # token pode ter expirado entre criar e enviar → tenta 1x com refresh
        if r.status_code == 401:
            token = await self._get_iam_token()
            headers["Authorization"] = f"Bearer {token}"
            r = await self._http.post(url, headers=headers, json=payload)

        try:
            r.raise_for_status()
        except Exception:
            body = None
            try:
                body = r.json()
            except Exception:
                body = r.text
            raise RuntimeError(f"IBM watsonx error {r.status_code}: {body}")

        # -------------------------------
        # Extração robusta do conteúdo
        #   - OpenAI-compatible: choices[].message.content (seu caso)
        #   - Esquema antigo:    results[].generated_text
        # -------------------------------
        data = r.json()

        if self._dbg:
            print("=== [IBM] RAW RESPONSE ===")
            try:
                print(json.dumps(data, ensure_ascii=False)[:2000])
            except Exception:
                print(str(r.text)[:2000])

        text = ""

        # 1) OpenAI-compatible
        if isinstance(data.get("choices"), list):
            collected = []
            for ch in data["choices"]:
                msg = (ch.get("message") or {})
                content = msg.get("content")
                if isinstance(content, str):
                    collected.append(content)
                elif isinstance(content, list):
                    # alguns provedores podem retornar lista de partes
                    parts = []
                    for p in content:
                        if isinstance(p, dict):
                            t = p.get("text")
                            if isinstance(t, str):
                                parts.append(t)
                    if parts:
                        collected.append("".join(parts))
            text = "\n".join(s for s in collected if s).strip()

        # 2) Formato antigo
        if not text and isinstance(data.get("results"), list):
            texts = [res.get("generated_text", "") for res in data["results"] if res.get("generated_text")]
            text = "\n".join(t for t in texts if t).strip()

        if not text:
            # mantém corpo bruto para inspeção rápida
            raise RuntimeError(f"Resposta vazia ou inesperada do IBM watsonx. Corpo bruto: {str(r.text)[:800]}")

        return text






def get_chat_api_from_model(model: str) -> ChatAPI:
    if model.startswith("openrouter:"):
        return OpenRouterAPI(model)
    if model.startswith("sabia") or model.startswith("maritaca"):
        return MaritacaAPI(model)
    if model.startswith("gpt") or model.startswith("o1"):
        return OpenAIAPI(model)
    if model.startswith("claude"):
        return AnthropicAPI(model)
    if model.startswith("ibm-granite") or model.startswith("granite"):
        print("Usando IBM Granite API para modelo:", model)
        return IBMGraniteAPI(model)
    if model == "meta-llama/Meta-Llama-3.1-405B-Instruct":
        return TogetherAPI(model + "-turbo")
    return LocalAPI(model)
