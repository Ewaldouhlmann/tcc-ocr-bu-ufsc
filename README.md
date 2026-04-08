# OCR — Digitalização de Obras Históricas

Extração de texto de PDFs escaneados usando múltiplos motores de OCR.  
Projeto desenvolvido em parceria com o NUPILL/UFSC.

## Estrutura

```
ocr_project/
├── src/
│   └── extrair_ocr.py   # script principal
├── input/               # coloque os PDFs aqui (ignorado pelo git)
├── output/              # resultados gerados (ignorado pelo git)
├── requirements.txt
└── README.md
```

## Modelos disponíveis

| Modelo | Característica | Indicado para |
|---|---|---|
| `tesseract` | Clássico, leve, fácil de instalar | Ponto de partida |
| `easyocr` | Sem dependências externas, boa precisão | Uso geral |
| `kraken` | Especializado em documentos históricos | Textos do séc. XIX/XX |
| `paddleocr` | Estado da arte em benchmarks | Comparação acadêmica |

## Instalação

### 1. Tesseract

**Windows:**  
Baixe o instalador: https://github.com/UB-Mannheim/tesseract/wiki  
Durante a instalação, selecione o idioma **Portuguese**.  
Adicione `C:\Program Files\Tesseract-OCR` ao PATH.

**Linux:**
```bash
sudo apt install tesseract-ocr tesseract-ocr-por poppler-utils
```

### 2. Poppler (somente Windows)

Baixe em: https://github.com/oschwartz10612/poppler-windows/releases  
Extraia e adicione a pasta `bin/` ao PATH.

### 3. Dependências Python

```bash
pip install -r requirements.txt
```

> PaddleOCR requer instalação separada:
> ```bash
> pip install paddlepaddle paddleocr
> ```

## Uso

Coloque o PDF na pasta `input/` e rode:

```bash
# Tesseract (padrão)
python src/extrair_ocr.py input/arquivo.pdf

# Escolher outro modelo
python src/extrair_ocr.py input/arquivo.pdf --modelo easyocr
python src/extrair_ocr.py input/arquivo.pdf --modelo kraken
python src/extrair_ocr.py input/arquivo.pdf --modelo paddleocr

# Ajustar DPI (padrão 300)
python src/extrair_ocr.py input/arquivo.pdf --dpi 400
```

## Saída

Os resultados ficam em `output/nome_do_arquivo/modelo/`:

| Arquivo | Conteúdo |
|---|---|
| `pagina_001.txt` | Texto extraído da página |
| `pagina_001.json` | Palavras com posição `(x, y, largura, altura)` e confiança |
| `resultado_completo.txt` | Texto de todas as páginas junto |

A pasta de saída inclui o nome do modelo, então rodar dois modelos no mesmo PDF não sobrescreve os resultados:
```
output/
└── culto_mulher_01/
    ├── tesseract/
    │   ├── pagina_001.txt
    │   └── resultado_completo.txt
    └── easyocr/
        ├── pagina_001.txt
        └── resultado_completo.txt
```
