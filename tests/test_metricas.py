"""Testes das métricas (CER/WER), da normalização e da reprodutibilidade dos resultados.

Rodar a partir da raiz do repositório:
    python -m unittest discover tests -v
"""

import csv
import sys
import unittest
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

from benchmark.avaliar import MANIFESTO, MODELOS, REGRA_OURO, SAIDAS, DESTINO, avaliar, normalizar  # noqa: E402
from benchmark.metrics import compute_cer, compute_wer  # noqa: E402


class TestMetricas(unittest.TestCase):
    """CER e WER em casos cujo valor pode ser conferido à mão."""

    def test_textos_iguais_tem_erro_zero(self):
        self.assertEqual(compute_cer("casa velha", "casa velha"), 0.0)
        self.assertEqual(compute_wer("casa velha", "casa velha"), 0.0)

    def test_substituicao_de_um_caractere(self):
        # "caza" x "casa": 1 substituição em 4 caracteres.
        self.assertAlmostEqual(compute_cer("caza", "casa"), 0.25)
        # Uma palavra errada em uma.
        self.assertAlmostEqual(compute_wer("caza", "casa"), 1.0)

    def test_insercao_e_remocao(self):
        # "pharmacia" tem 9 caracteres; "farmacia" exige remover 1 e trocar 1 = 2 edições.
        self.assertAlmostEqual(compute_cer("farmacia", "pharmacia"), 2 / 9)
        # Palavra a mais na hipótese: 1 inserção sobre 2 palavras da referência.
        self.assertAlmostEqual(compute_wer("o velho livro", "o livro"), 0.5)

    def test_ordem_dos_argumentos(self):
        # A referência é o denominador: trocar a ordem muda o resultado.
        self.assertAlmostEqual(compute_cer("ab", "abcd"), 0.5)
        self.assertAlmostEqual(compute_cer("abcd", "ab"), 1.0)

    def test_cer_pode_passar_de_um(self):
        # Hipótese muito maior que a referência (ex.: laço de repetição do VLM).
        self.assertGreater(compute_cer("aaaaaaaa", "ab"), 1.0)


class TestNormalizacao(unittest.TestCase):
    """A normalização não deve apagar diferenças ortográficas legítimas."""

    def test_preserva_grafias_antigas(self):
        for palavra in ["pharmacia", "theatro", "commercio", "estylo", "nympha", "orthographia"]:
            self.assertEqual(normalizar(palavra), palavra)

    def test_preserva_acentos(self):
        self.assertEqual(normalizar("José, então, Maceió"), "José, então, Maceió")

    def test_s_longo_vira_s(self):
        self.assertEqual(normalizar("paſſado"), "passado")

    def test_junta_palavra_hifenizada_na_quebra_de_linha(self):
        self.assertEqual(normalizar("compa-\nnheiro"), "companheiro")
        # Hífen invisível do ABBYY.
        self.assertEqual(normalizar("compa­\nnheiro"), "companheiro")

    def test_remove_marcacao_markdown_e_html(self):
        self.assertEqual(normalizar("# Capitulo I\n**Era** uma vez<br>"), "Capitulo I Era uma vez")

    def test_unifica_aspas_e_espacos(self):
        self.assertEqual(normalizar("«Sim» ,  disse   ele"), '"Sim", disse ele')

    def test_formas_unicode_equivalentes(self):
        # "é" pré-composto (NFC) e "e" + acento combinante (NFD) devem ficar iguais.
        self.assertEqual(normalizar("café"), normalizar("café"))


class TestRegraOuro(unittest.TestCase):
    """Integridade do corpus de referência."""

    def setUp(self):
        with open(MANIFESTO, encoding="utf-8") as f:
            self.manifesto = list(csv.DictReader(f))

    def test_vinte_paginas(self):
        self.assertEqual(len(self.manifesto), 20)

    def test_toda_pagina_tem_transcricao_nao_vazia(self):
        for linha in self.manifesto:
            arquivo = REGRA_OURO / linha["obra_slug"] / f"{linha['pagina']}.txt"
            with self.subTest(arquivo=arquivo.relative_to(RAIZ)):
                self.assertTrue(arquivo.exists())
                self.assertTrue(arquivo.read_text(encoding="utf-8").strip())

    def test_dificuldades_validas(self):
        for linha in self.manifesto:
            self.assertIn(linha["dificuldade"], {"Fácil", "Médio", "Difícil"})


class TestReprodutibilidade(unittest.TestCase):
    """Recalcula CER/WER a partir das saídas versionadas e compara com o CSV do TCC."""

    def test_resultados_batem_com_por_pagina_csv(self):
        recalculado = avaliar(SAIDAS)
        with open(DESTINO / "por_pagina.csv", encoding="utf-8") as f:
            salvo = {(l["modelo"], l["obra"], l["pagina"]): l for l in csv.DictReader(f)}

        self.assertEqual(set(recalculado["modelo"]), set(MODELOS))
        self.assertEqual(len(recalculado), len(salvo))
        for _, r in recalculado.iterrows():
            chave = (r["modelo"], r["obra"], r["pagina"])
            with self.subTest(chave=chave):
                self.assertAlmostEqual(r["cer"], float(salvo[chave]["cer"]), places=9)
                self.assertAlmostEqual(r["wer"], float(salvo[chave]["wer"]), places=9)


if __name__ == "__main__":
    unittest.main()
