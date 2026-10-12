import base64
from pathlib import Path

from src.ocr.base_ocr import VLMOCR


class ClaudeOCR(VLMOCR):
    """Claude Opus 5.5 (Anthropic), via SDK oficial `anthropic`.

    A imagem é enviada em base64 junto ao prompt padronizado de VLMOCR.
    Autenticação por ANTHROPIC_API_KEY (ver .env.example) ou por um perfil do
    `ant auth login`.

    Atenção: a transcrição inicial da regra ouro foi gerada por este mesmo
    modelo e depois revisada pelo autor. Os resultados dele no benchmark
    devem ser lidos com essa ressalva (ver Cap. 4 do TCC).

    Sem fallback para outro modelo em caso de recusa: uma página respondida
    por outro modelo seria atribuída ao Claude nos resultados. A recusa vira
    erro e fica registrada como falha da página.
    """

    name = "claude"

    def __init__(self, model_name: str = "claude-opus-5-5", effort: str = "high"):
        import anthropic

        self._client = anthropic.Anthropic()
        self._model_name = model_name
        self._effort = effort

    def recognize(self, image_path: Path) -> str:
        dados = base64.standard_b64encode(Path(image_path).read_bytes()).decode("utf-8")

        response = self._client.messages.create(
            model=self._model_name,
            max_tokens=16000,
            output_config={"effort": self._effort},
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {"type": "base64", "media_type": "image/png", "data": dados},
                    },
                    {"type": "text", "text": self.PROMPT},
                ],
            }],
        )

        self.last_tokens_used = response.usage.input_tokens + response.usage.output_tokens

        if response.stop_reason == "refusal":
            raise RuntimeError(f"Claude recusou a página {image_path}: {response.stop_details}")
        if response.stop_reason == "max_tokens":
            raise RuntimeError(f"Transcrição cortada por max_tokens em {image_path}")

        return "".join(b.text for b in response.content if b.type == "text")
