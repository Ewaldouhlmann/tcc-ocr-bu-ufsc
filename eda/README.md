# eda/ — análise dos resultados

Notebook que gera **todas as figuras e tabelas de resultados do TCC** (Cap. 6) a
partir das saídas do benchmark. Nada aqui roda modelo OCR.

## Fluxo (a partir da raiz do repositório)

```bash
python benchmark/extrair.py      # 1. roda os modelos e salva os textos (horas; uma vez só)
python benchmark/avaliar.py      # 2. CER/WER contra a regra ouro (segundos)
jupyter nbconvert --to notebook --execute --inplace eda/analise_resultados.ipynb   # 3. figuras e tabelas
```

Os modelos rodados em GPU na nuvem (`notebooks/benchmark_gpu.ipynb`) e o Claude
(`benchmark/sessao_claude.py`) entram no passo 2 desde que as saídas estejam em
`data/results/extracao/{modelo}/`.

## Arquivos

| Arquivo | O que faz |
|---|---|
| `analise_resultados.ipynb` | Execução (tempo, RAM), qualidade (CER/WER por página, obra e dificuldade), padrões de erro e fidelidade às grafias antigas, resumo para a seleção do modelo |
| `estilo.py` | Cores, fontes e formato numérico comuns a todas as figuras |

## Saídas

- Figuras em `escrita tcc/imagens/` (PNG, 300 dpi): `eda_real_tempo`, `eda_real_resolucao`,
  `eda_real_ram`, `resultados_cer_paginas`, `resultados_heatmap`, `resultados_dificuldade`,
  `resultados_cer_custo`, `resultados_fidelidade`.
- Tabelas em `escrita tcc/tabelas/`: `extracao_resumo.tex`, `erros_resumo.tex`
  (`avaliacao_resumo.tex` vem de `benchmark/avaliar.py`).

## Decisões de medição

- **RAM dos VLMs locais inclui o Ollama**: a inferência acontece no servidor do
  Ollama, não no processo Python (`benchmark/amostrador.py`).
- **Tempos da GPU na nuvem não são comparáveis** aos do equipamento do trabalho;
  aparecem com marcador vazado e não entram no gráfico de memória.
- **Normalização antes do CER/WER** (`benchmark/avaliar.py`): espaços e quebras de
  linha colapsados, hifenização de fim de linha desfeita, Markdown removido, `ſ`→`s`;
  maiúsculas, acentos e pontuação mantidos.
- **Grafias antigas** são identificadas por regras (`REGRAS_GRAFIA_ANTIGA` no
  notebook), uma aproximação que cobre as formas mais frequentes do corpus.
- **Cores** seguem a categoria do modelo (Clássico `#2a78d6`, VLM local `#eb6834`,
  Serviço via API `#1baf7a`), paleta validada para daltonismo.
