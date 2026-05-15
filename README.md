# tcc-ocr-nupill

Benchmark e pipeline de OCR para digitalização de obras literárias históricas em português, desenvolvido como Trabalho de Conclusão de Curso no Departamento de Informática e Estatística da UFSC.

O projeto avalia 13 ferramentas de OCR — desde motores clássicos (Tesseract, EasyOCR, PaddleOCR, DocTR) até modelos baseados em Transformer (TrOCR) e VLMs (Gemma 4, DeepSeek-VL2, Qwen2.5-VL, Mistral OCR, LlamaParse, Gemini 2.0 Flash) — sobre imagens escaneadas de obras literárias do acervo NUPILL/UFSC. O objetivo é identificar a ferramenta mais adequada para o contexto de textos históricos em português com características de degradação física e ortografia arcaica.

---

## Estrutura do repositório

```
tcc-ocr-nupill/
│
├── src/                        # Código-fonte principal (pacote Python)
│   ├── __init__.py
│   │
│   ├── ocr/                    # Módulo de modelos OCR
│   │   ├── __init__.py
│   │   ├── base_ocr.py         # Classe abstrata BaseOCR
│   │   ├── classical/          # Motores OCR clássicos
│   │   │   ├── __init__.py
│   │   │   ├── tesseract_ocr.py
│   │   │   ├── easyocr_ocr.py
│   │   │   ├── paddleocr_ocr.py
│   │   │   └── doctr_ocr.py
│   │   ├── transformer/        # Modelos baseados em Transformer
│   │   │   ├── __init__.py
│   │   │   ├── trocr_base.py
│   │   │   └── trocr_large.py
│   │   └── vlm/                # Modelos de visão-linguagem
│   │       ├── __init__.py
│   │       ├── gemma_ocr.py
│   │       ├── deepseek_ocr.py
│   │       ├── qwen_ocr.py
│   │       ├── llamaparse_ocr.py
│   │       ├── mistral_ocr.py
│   │       └── gemini_ocr.py
│   │
│   ├── pipeline/               # Pipeline de extração completa
│   │   ├── __init__.py
│   │   ├── converter.py        # Conversão PDF → PNG
│   │   └── extractor.py        # Orquestração do pipeline completo
│   │
│   └── postprocessing/         # Pós-processamento do texto extraído
│       ├── __init__.py
│       └── cleaner.py          # Normalização e limpeza de texto
│
├── benchmark/                  # Avaliação comparativa dos modelos
│   ├── run_benchmark.py        # Ponto de entrada (CLI)
│   ├── metrics.py              # Cálculo de WER, CER e relatórios
│   └── ground_truth/           # Transcrições manuais de referência (.txt)
│
├── data/
│   ├── corpus/                 # PDFs e PNGs das obras literárias
│   │   ├── pdfs/               # Arquivos originais (não versionados)
│   │   └── images/             # Páginas convertidas em PNG
│   └── results/                # Saídas OCR organizadas por modelo/página
│
├── docs/                       # Análises, notebooks e documentação extra
│
├── .env.example                # Variáveis de ambiente necessárias (chaves de API)
├── requirements.txt            # Dependências Python
└── README.md
```

---

## Hierarquia de classes

Todos os modelos herdam de `BaseOCR`, que define a interface comum:

```python
# src/ocr/base_ocr.py
from abc import ABC, abstractmethod
from pathlib import Path

class BaseOCR(ABC):

    @abstractmethod
    def recognize(self, image_path: Path) -> str:
        """Recebe o caminho de uma imagem PNG e retorna o texto extraído."""
        ...

    def batch_recognize(self, image_paths: list[Path]) -> list[str]:
        """Processa uma lista de imagens. Pode ser sobrescrito para otimização."""
        return [self.recognize(p) for p in image_paths]
```

As subclasses são organizadas em três categorias:

| Categoria | Classe base intermediária | Modelos |
|---|---|---|
| OCR clássico | `ClassicalOCR(BaseOCR)` | Tesseract, EasyOCR, PaddleOCR, DocTR |
| Transformer | `TransformerOCR(BaseOCR)` | TrOCR base-handwritten, TrOCR large-handwritten |
| VLM | `VLMOCR(BaseOCR)` | Gemma 4, DeepSeek-VL2, Qwen2.5-VL, LlamaParse Free, LlamaParse Premium, Mistral OCR, Gemini 2.0 Flash |

---

## Requisitos

- Python 3.10+
- GPU NVIDIA com CUDA (recomendado para VLMs locais)
- [Ollama](https://ollama.com/) instalado localmente (para Gemma 4, DeepSeek-VL2 e Qwen2.5-VL)
- Chaves de API configuradas no `.env` (para LlamaParse, Mistral OCR e Gemini)

Instalar dependências:

```bash
pip install -r requirements.txt
```

Copiar e preencher variáveis de ambiente:

```bash
cp .env.example .env
# editar .env com suas chaves de API
```

---

## Como usar

### Converter PDFs em imagens

```bash
python -m src.pipeline.converter \
    --input data/corpus/pdfs/ \
    --output data/corpus/images/ \
    --dpi 300
```

### Rodar o benchmark

```bash
python benchmark/run_benchmark.py \
    --images data/corpus/images/amostra/ \
    --ground-truth benchmark/ground_truth/ \
    --models tesseract easyocr trocr-large gemma4 mistral \
    --output data/results/
```

### Rodar o pipeline completo no corpus

```bash
python -m src.pipeline.extractor \
    --corpus data/corpus/pdfs/ \
    --model mistral \
    --output data/results/
```

---

## Métricas de avaliação

A métrica principal é o **WER** (Word Error Rate), calculado por comparação com as transcrições manuais de referência (*ground truth*). O **CER** (Character Error Rate) é reportado complementarmente para análise de erros em nível de caractere.

```
WER = (Substituições + Deleções + Inserções) / Total de palavras na referência
```

---

## Corpus

As obras utilizadas pertencem ao acervo do [NUPILL](https://nupill.ufsc.br/) (Núcleo de Pesquisas em Informática, Literatura e Linguística — UFSC). Os arquivos PDF originais **não são distribuídos neste repositório** por questões de direitos e tamanho. Consulte o NUPILL para acesso.

---

## Referências principais

- LI, M. et al. TrOCR: Transformer-based OCR with Pre-trained Models. *arXiv:2109.10282*, 2021.
- VESALAINEN, A. et al. Error patterns in historical OCR. *arXiv:2602.14524*, 2026.
- DATA UNBOXED. OCR vs VLM-OCR: accuracy benchmark for scanned documents in 2025. Disponível em: https://www.dataunboxed.io/blog/ocr-vs-vlm-ocr-naive-benchmarking-accuracy-for-scanned-documents

---

## Autor

**Ewaldo Uhlmann** — Ciências da Computação, UFSC  
Orientador: Prof. Dr. Renato Fileto  
Colaboração: Prof. Dr. Alckmar Luiz dos Santos (NUPILL)