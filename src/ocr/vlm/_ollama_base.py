from pathlib import Path

from src.ocr.base_ocr import VLMOCR


class OllamaVLM(VLMOCR):
    """Base para VLMs executados localmente via Ollama.

    A interação ocorre pela API REST local do Ollama (http://localhost:11434 por
    padrão), enviando a imagem da página e o prompt padronizado de VLMOCR,
    conforme Seção 5.1.2 do TCC.
    """

    #: Tag do modelo no Ollama (ex.: "qwen2.5vl:3b"); definida pelas subclasses.
    model_tag: str = ""

    #: Janela de contexto em tokens. Sem GPU, o Ollama usa 4096 por padrão, mas
    #: uma página A5 a 300 DPI já gera ~4.150 tokens de imagem no Qwen2.5-VL,
    #: antes mesmo do prompt e da resposta. 6144 cobre imagem + ~2.000 tokens
    #: de transcrição sem reservar memória à toa para o cache.
    num_ctx: int = 6144

    #: Tokens processados por vez no prompt. O padrão do Ollama (512) faz o
    #: Qwen2.5-VL 3B passar de 5,8 GB e ser encerrado por falta de memória em
    #: um equipamento de 7,5 GB; com 128 o pico cabe. Só muda o agrupamento do
    #: processamento do prompt, não o modelo nem a saída esperada.
    num_batch: int = 128

    #: Modo de raciocínio ("thinking") dos modelos que o suportam. None mantém o
    #: padrão do modelo; False desliga, para o modelo responder direto.
    think: bool | None = None

    def __init__(self, model_tag: str | None = None, num_ctx: int | None = None,
                 num_batch: int | None = None):
        import ollama

        self._client = ollama
        if model_tag:
            self.model_tag = model_tag
        if num_ctx:
            self.num_ctx = num_ctx
        if num_batch:
            self.num_batch = num_batch

    def recognize(self, image_path: Path) -> str:
        extra = {} if self.think is None else {"think": self.think}
        response = self._client.chat(
            **extra,
            model=self.model_tag,
            messages=[
                {
                    "role": "user",
                    "content": self.PROMPT,
                    "images": [str(image_path)],
                }
            ],
            options={"num_ctx": self.num_ctx, "num_batch": self.num_batch},
        )

        try:
            self.last_tokens_used = response["prompt_eval_count"] + response["eval_count"]
        except (KeyError, TypeError):
            self.last_tokens_used = None

        return response["message"]["content"]
