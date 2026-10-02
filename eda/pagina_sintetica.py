"""Gera páginas sintéticas com texto de referência conhecido.

Enquanto os PDFs do corpus e a regra ouro não estão disponíveis, estas páginas
permitem exercitar o benchmark de ponta a ponta: medir tempo, CPU e memória de
cada modelo no hardware do projeto e verificar, com um texto de referência
exato, se os modelos preservam ou modernizam grafias históricas.

O texto é original (não pertence a nenhuma obra do corpus) e foi escrito com
ortografia anterior à reforma de 1911 de propósito: cada forma arcaica listada
em FORMAS_ARCAICAS aparece nele, junto com sua grafia moderna equivalente.

São geradas duas versões da mesma página em A5 a 300 DPI (as dimensões do
corpus digitalizado):

- limpa: texto preto sobre fundo branco;
- degradada: papel amarelado, manchas, ruído, leve desfoque e inclinação,
  aproximando a classe "média/difícil" do Cap. 4.

Uso:
    python eda/pagina_sintetica.py            # gera em eda/dados/paginas/
"""

from pathlib import Path
import glob
import textwrap

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

SAIDA = Path(__file__).resolve().parent / "dados" / "paginas"

# A5 a 300 DPI
LARGURA, ALTURA = 1748, 2480
MARGEM_X, MARGEM_Y = 170, 200
TAMANHO_FONTE = 40  # ~9,6 pt a 300 DPI, próximo ao corpo de texto das obras
ENTRELINHA = 1.45

#: forma arcaica -> forma moderna. Usado em explorar_textos.py para medir
#: preservação vs. modernização ortográfica.
FORMAS_ARCAICAS = {
    "aquella": "aquela",
    "hum": "um",
    "he": "é",
    "caza": "casa",
    "pharmacia": "farmácia",
    "sciencia": "ciência",
    "theatro": "teatro",
    "commercio": "comércio",
    "estylo": "estilo",
    "elle": "ele",
    "villa": "vila",
    "annos": "anos",
    "ella": "ela",
    "orthographia": "ortografia",
    "prompto": "pronto",
}

TEXTO = """\
CAPITULO III

Da chegada á villa

Era aquella huma tarde de outono, e a caza do boticario estava cheia de gente. \
Havia annos que elle não recebia tantas visitas; desde que abrira a pharmacia \
junto ao largo do commercio, nunca se vira tamanho movimento. Ella, a filha \
mais velha, servia o café com hum estylo que a todos encantava.

— Isto he obra da sciencia moderna, dizia o boticario, mostrando hum frasco \
de vidro azulado. Ninguem na villa possuia cousa semelhante.

O padre, homem de poucas palavras e muita leitura, observava tudo com \
attenção. Dizia-se que escrevera hum tratado sobre a orthographia da lingua, \
e que o levaria prompto ao prelo assim que chegassem as ferias. Os moços, \
porém, só fallavam do theatro que se ia abrir na praça, e da companhia \
lyrica que viria da côrte.

Quando a noite cahiu, accenderam-se os lampeões da rua, e a conversa \
tornou-se mais intima. Ninguem reparou que aquella visita, tão esperada, \
não havia ainda apparecido."""


def _fonte() -> ImageFont.FreeTypeFont:
    candidatos = glob.glob("/usr/share/fonts/**/DejaVuSerif.ttf", recursive=True)
    if not candidatos:
        raise FileNotFoundError("Fonte DejaVu Serif não encontrada (apt install fonts-dejavu)")
    return ImageFont.truetype(candidatos[0], TAMANHO_FONTE)


def _quebrar_linhas(texto: str, fonte, largura_util: int) -> list[str]:
    """Quebra parágrafos em linhas que cabem na largura útil da página."""
    medida = fonte.getlength("m")
    colunas = max(20, int(largura_util / medida * 1.55))
    linhas = []
    for paragrafo in texto.split("\n"):
        if not paragrafo.strip():
            linhas.append("")
            continue
        # Ajusta a largura de quebra até todas as linhas caberem.
        cols = colunas
        while True:
            partes = textwrap.wrap(paragrafo, cols)
            if all(fonte.getlength(p) <= largura_util for p in partes):
                break
            cols -= 1
        linhas.extend(partes)
    return linhas


def gerar_limpa() -> tuple[Image.Image, str]:
    fonte = _fonte()
    img = Image.new("L", (LARGURA, ALTURA), 255)
    desenho = ImageDraw.Draw(img)
    linhas = _quebrar_linhas(TEXTO, fonte, LARGURA - 2 * MARGEM_X)
    passo = int(TAMANHO_FONTE * ENTRELINHA)
    y = MARGEM_Y
    for i, linha in enumerate(linhas):
        # Título e subtítulo centralizados, como nas aberturas de capítulo.
        if i < 3 and linha:
            x = (LARGURA - fonte.getlength(linha)) / 2
        else:
            x = MARGEM_X
        desenho.text((x, y), linha, fill=0, font=fonte)
        y += passo
    # A referência é o texto na mesma quebra de linhas da imagem.
    referencia = "\n".join(linhas).strip() + "\n"
    return img, referencia


def degradar(img: Image.Image, semente: int = 42) -> Image.Image:
    """Aplica degradações típicas de digitalizações antigas."""
    rng = np.random.default_rng(semente)
    arr = np.asarray(img, dtype=np.float32) / 255.0

    # Papel amarelado e irregular: gradiente suave + manchas gaussianas grandes.
    yy, xx = np.mgrid[0:ALTURA, 0:LARGURA]
    fundo = 0.90 - 0.06 * (xx / LARGURA) - 0.04 * (yy / ALTURA)
    for _ in range(6):
        cx, cy = rng.uniform(0, LARGURA), rng.uniform(0, ALTURA)
        raio = rng.uniform(120, 380)
        intensidade = rng.uniform(0.08, 0.22)
        fundo -= intensidade * np.exp(-(((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * raio**2)))
    # Tinta "desbotada" (contraste reduzido) sobre o papel.
    tinta = 0.18
    pagina = fundo * arr + tinta * (1 - arr)
    pagina += rng.normal(0, 0.035, pagina.shape)
    pagina = np.clip(pagina, 0, 1)

    cinza = Image.fromarray((pagina * 255).astype(np.uint8), "L")
    cinza = cinza.filter(ImageFilter.GaussianBlur(0.9))
    cinza = cinza.rotate(0.8, resample=Image.BICUBIC, fillcolor=int(0.85 * 255))

    # Tom sépia para lembrar o papel envelhecido.
    rgb = Image.merge("RGB", (cinza, cinza.point(lambda v: int(v * 0.96)), cinza.point(lambda v: int(v * 0.84))))
    return rgb


def main():
    destino = SAIDA / "sintetica"
    destino.mkdir(parents=True, exist_ok=True)
    limpa, referencia = gerar_limpa()
    limpa.convert("RGB").save(destino / "001.png", dpi=(300, 300))
    (destino / "001.txt").write_text(referencia, encoding="utf-8")
    degradar(limpa).save(destino / "002.png", dpi=(300, 300))
    (destino / "002.txt").write_text(referencia, encoding="utf-8")
    print(f"Páginas geradas em {destino} (001 = limpa, 002 = degradada)")


if __name__ == "__main__":
    main()
