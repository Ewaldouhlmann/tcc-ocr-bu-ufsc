import os
from pathlib import Path

from src.ocr.base_ocr import VLMOCR


class LlamaParseOCR(VLMOCR):
    """Base comum para as variantes do LlamaParse (LlamaIndex), via API.

    Autenticação por LLAMA_CLOUD_API_KEY (ver .env.example). O SDK aceita tanto
    PDFs quanto imagens (PNG/JPG) como entrada; aqui a página é enviada
    diretamente como PNG, mantendo a interface comum de BaseOCR.recognize().
    """

    #: Modo de parsing do LlamaParse; definido pelas subclasses.
    #: Verifique os valores atuais na documentação, pois o SDK evolui com frequência.
    parse_mode: str = ""

    def __init__(self, api_key: str | None = None):
        from llama_parse import LlamaParse

        self._parser = LlamaParse(
            api_key=api_key or os.getenv("LLAMA_CLOUD_API_KEY"),
            result_type="text",
            parse_mode=self.parse_mode,
            language="pt",
        )

    def recognize(self, image_path: Path) -> str:
        documents = self._parser.load_data(str(image_path))
        return "\n".join(doc.text for doc in documents)


class LlamaParseFreeOCR(LlamaParseOCR):
    """Modo básico (Cost Effective) — 1 crédito/página (~US$ 0,00125/pág.)."""

    name = "llamaparse-free"
    parse_mode = "parse_page_without_llm"


class LlamaParsePremiumOCR(LlamaParseOCR):
    """Modo agêntico (Agentic) — 10 créditos/página (~US$ 0,0125/pág.).

    Eliminado na Fase 2 do TCC por custo financeiro (Seção 5.2.1), mas mantido
    aqui para compor o benchmark comparativo da Fase 1.
    """

    name = "llamaparse-premium"
    parse_mode = "parse_page_with_agent"
