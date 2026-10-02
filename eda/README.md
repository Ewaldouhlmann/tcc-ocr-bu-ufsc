# eda/ — análise exploratória

Scripts que medem o comportamento dos modelos OCR e geram as figuras e tabelas
usadas no TCC (Cap. 6). Tudo roda a partir da raiz do repositório.

## Fluxo

```bash
python eda/pagina_sintetica.py   # 1. gera as páginas de teste (limpa e degradada)
python eda/perfil_recursos.py    # 2. roda os modelos medindo tempo, CPU e RAM
python eda/analise.py            # 3. consolida CSVs, calcula CER/WER e ortografia
python eda/graficos.py           # 4. gera as figuras em escrita tcc/imagens/eda_*
```

O passo 2 aceita `--models` para rodar só alguns modelos (os demais resultados
são preservados) e `--paginas` para apontar outra pasta de imagens, por exemplo
as páginas reais do corpus quando estiverem disponíveis.

## Arquivos

| Arquivo | O que faz |
|---|---|
| `pagina_sintetica.py` | Página A5/300 DPI com texto próprio em ortografia pré-1911 e 15 formas arcaicas conhecidas |
| `perfil_recursos.py` | Roda cada modelo em processo isolado e amostra CPU/RAM a cada 0,25 s, incluindo os processos do Ollama |
| `_executar_modelo.py` | Processo filho do perfil: carrega o modelo e emite marcadores de fase em JSON |
| `analise.py` | Gera `resumo_modelos.csv`, `ortografia.csv` e `escrita tcc/tabelas/eda_resumo.tex` |
| `graficos.py` | Figuras PDF (tempo, RAM, linha do tempo, CER) e o mosaico das páginas |

Saídas em `dados/`: `perfil_linha_tempo.csv` (uma linha por amostra),
`perfil_eventos.csv` (fases e erros), `saidas/{modelo}/{pagina}.txt` (texto
reconhecido) e os consolidados acima.

## Decisões de medição

- **RAM dos VLMs locais inclui o Ollama.** A inferência acontece no servidor do
  Ollama, não no processo Python; medir só o Python (como o
  `resource.getrusage` do `benchmark/run_benchmark.py`) subestima esses modelos.
- **CPU em % da máquina inteira** (100% = todos os núcleos).
- **Modelos via API:** CPU/RAM medidos são só do cliente local. Quando a API
  devolve erro 429, o tempo é descartado, porque inclui as esperas de novas
  tentativas do cliente.
- **Normalização antes do CER/WER:** espaços e quebras de linha colapsados e
  marcação Markdown removida; maiúsculas, acentos e pontuação mantidos. O
  `benchmark/metrics.py` ainda compara texto bruto — convém adotar a mesma
  normalização lá antes da rodada com a regra ouro.
- **Cores dos gráficos** seguem a categoria do modelo (Clássico `#2a78d6`, VLM
  local `#eb6834`, VLM via API `#1baf7a`), paleta validada para daltonismo.

## Limitações

A página sintética usa fonte digital moderna e não reproduz tipos do século
XVIII nem o *s* longo: serve para testar viabilidade e o processo de avaliação,
não para ranquear modelos. Cada configuração foi executada uma vez, com outros
programas abertos (≈3,5 GB de RAM livres).
