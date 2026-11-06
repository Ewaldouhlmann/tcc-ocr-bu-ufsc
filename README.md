# tcc-ocr-bu-ufsc

Código, dados e texto do Trabalho de Conclusão de Curso **"Reconhecimento de Texto em
Imagens de Obras Literárias Antigas em Português"** (Ciências da Computação, UFSC, 2026).

O trabalho compara dez ferramentas de OCR sobre 20 páginas de cinco obras dos séculos
XVIII a XX do acervo do [NUPILL](https://nupill.ufsc.br/), com uma regra ouro transcrita
manualmente. Mede CER, WER, robustez, fidelidade às grafias antigas e custo de execução, e
seleciona o **Qwen2.5-VL 7B** para a digitalização do acervo, com o **GLM-OCR** como
alternativa sem GPU.

| Categoria | Ferramentas avaliadas |
|---|---|
| Clássico / CRNN | Tesseract, EasyOCR, PaddleOCR, DocTR, ABBYY FineReader (camada de texto dos PDFs) |
| VLM local (Ollama) | GLM-OCR (CPU), Qwen2.5-VL 7B, Gemma 4 E4B, DeepSeek-OCR (GPU no Colab) |
| Serviço via API | LlamaParse (modo básico) |

O texto completo está em [`escrita tcc/main.pdf`](escrita%20tcc/main.pdf).

## Estrutura

```
benchmark/            benchmark em duas etapas
  extrair.py          1. roda os modelos sobre as páginas e salva o texto, o tempo e a memória
  avaliar.py          2. compara as saídas com a regra ouro (CER/WER) e gera a tabela do TCC
  amostrador.py       mede a RAM do processo e do Ollama durante a extração
  metrics.py          CER, WER e planilha de resultados
  rodar_extracao.sh   deixa a extração rodando sem supervisão no equipamento local
src/
  ocr/                adaptadores dos modelos (BaseOCR -> ClassicalOCR / TransformerOCR / VLMOCR)
                      e o registro MODEL_REGISTRY
  pipeline/
    converter.py      PDF -> PNG (livro inteiro ou só as páginas do manifesto)
    extractor.py      pipeline de extração de um livro inteiro (Apêndice A do TCC)
  catalog.py          leitura de data/catalog.yaml (obras e modelos)
eda/
  analise_resultados.ipynb   figuras e tabelas do Cap. 6 do TCC
  estilo.py                  paleta e formatação das figuras
notebooks/
  benchmark_gpu.ipynb        extração dos VLMs em GPU no Google Colab
  pipeline_colab.ipynb       pipeline de extração no Colab, para quem não tem GPU
data/
  catalog.yaml               metadados das obras e dos modelos
  corpus/regra_ouro/         transcrições de referência das 20 páginas + manifest.csv
  results/extracao/          saída de cada modelo por página, metricas*.csv e falhas
  results/avaliacao/         CER/WER por página e por modelo
escrita tcc/                 LaTeX do TCC (classe ufsc-thesis-rn46-2019), figuras e tabelas
```

Os PDFs e as imagens das obras (`data/corpus/pdfs/` e `data/corpus/images/`) pertencem ao
acervo do NUPILL e **não são versionados**. Para obtê-los, consulte o NUPILL.

## Instalação

```bash
pip install -r requirements.txt
sudo apt install tesseract-ocr tesseract-ocr-por poppler-utils
```

Os VLMs locais usam o [Ollama](https://ollama.com/) (`ollama pull glm-ocr:q8_0`, etc.).
Para os serviços via API, crie um `.env` a partir do [`.env.example`](.env.example).

## Reproduzir os resultados

A regra ouro e as saídas dos modelos estão no repositório, então a avaliação pode ser
refeita sem rodar nenhum modelo:

```bash
python benchmark/avaliar.py                                                        # CER/WER
jupyter nbconvert --to notebook --execute --inplace eda/analise_resultados.ipynb   # figuras e tabelas
```

Para rodar os modelos de novo, coloque os PDFs em `data/corpus/pdfs/{obra_slug}.pdf` e:

```bash
python -m src.pipeline.converter --input data/corpus/pdfs/ --output data/corpus/images/ --manifest
python benchmark/extrair.py                 # todos os modelos locais (retoma de onde parou)
python benchmark/extrair.py --models glm-ocr
```

Os VLMs em GPU (Qwen2.5-VL 7B, Gemma 4 E4B, DeepSeek-OCR) rodam pelo
`notebooks/benchmark_gpu.ipynb`, no Colab; as saídas vão para `data/results/extracao/`.

## Extrair um livro inteiro

```bash
python -m src.pipeline.extractor --pdf livro.pdf --model qwen-7b --output saida/ --paginas 1-50
```

Sem GPU, use o `notebooks/pipeline_colab.ipynb` no Google Colab.

## Compilar o TCC

```bash
sudo apt install texlive-publishers latexmk   # abnTeX2
cd "escrita tcc" && latexmk -pdf main.tex
```

## Autor

**Ewaldo Uhlmann**, Ciências da Computação, UFSC.
Orientador: Prof. Renato Fileto, Dr. Coorientador: Prof. Alckmar Luiz dos Santos, Dr. (NUPILL).
