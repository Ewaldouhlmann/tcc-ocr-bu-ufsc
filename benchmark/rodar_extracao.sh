#!/usr/bin/env bash
# Deixa a extração rodando sem supervisão.
#
#   bash benchmark/rodar_extracao.sh
#
# - fecha programas desnecessários (libera memória para os VLMs locais);
# - impede suspensão/desligamento enquanto a extração roda (liberado no fim);
# - roda em paralelo, desacoplado do terminal/VS Code:
#     local: qwen (8 páginas que faltam) e qwen3-vl  -> extracao_local.log
#     API:   modos pagos do LlamaParse, Gemini, Mistral -> extracao_api.log
# Páginas já salvas são puladas; pode rodar de novo quantas vezes quiser.

set -u
cd "$(dirname "$0")/.."
SAIDA=data/results/extracao
mkdir -p "$SAIDA"

MODELOS_LOCAIS="qwen qwen3-vl"
MODELOS_API="llamaparse-premium llamaparse-lvm llamaparse-layout gemini mistral"

echo "Fechando programas desnecessários..."
for padrao in '[t]racker-miner-fs' '[u]pdate-notifier' '[g]nome-calendar' '[s]eahorse' \
              '[a]ppimagelauncherd' '[e]volution-alarm-notify' '[d]ing@rastersoft.com' \
              '[m]s-python.vscode-pylance' '[c]opilot-language-server' '[j]son-language-features/server'; do
    pids=$(ps -u "$USER" -o pid,args | grep "$padrao" | awk '{print $1}')
    [ -n "$pids" ] && kill $pids 2>/dev/null && echo "  encerrado: ${padrao//[\[\]]/}"
done
pkill -x chrome 2>/dev/null && echo "  encerrado: chrome"

# Na bateria, não suspender por inatividade (restaurado no fim).
BATERIA_ANTES=$(gsettings get org.gnome.settings-daemon.plugins.power sleep-inactive-battery-type)
gsettings set org.gnome.settings-daemon.plugins.power sleep-inactive-battery-type 'nothing'

INIBIR="systemd-inhibit --what=shutdown:sleep:idle:handle-lid-switch:handle-power-key:handle-suspend-key --mode=block --who=TCC-OCR --why=Extracao-do-benchmark"

setsid nohup $INIBIR python3 benchmark/extrair.py --models $MODELOS_LOCAIS \
    > "$SAIDA/extracao_local.log" 2>&1 < /dev/null &
PID_LOCAL=$!
setsid nohup $INIBIR python3 benchmark/extrair.py --models $MODELOS_API \
    > "$SAIDA/extracao_api.log" 2>&1 < /dev/null &
PID_API=$!

# Restaura a configuração de bateria quando as duas extrações terminarem.
setsid nohup bash -c "while kill -0 $PID_LOCAL 2>/dev/null || kill -0 $PID_API 2>/dev/null; do sleep 30; done; \
    gsettings set org.gnome.settings-daemon.plugins.power sleep-inactive-battery-type $BATERIA_ANTES" \
    > /dev/null 2>&1 < /dev/null &

sleep 5
echo
echo "Rodando (PIDs $PID_LOCAL local, $PID_API API). Suspensão/desligamento bloqueados até o fim."
echo "Acompanhar:  tail -f $SAIDA/extracao_local.log $SAIDA/extracao_api.log"
echo "Resultado:   cat $SAIDA/_estado/resumo.txt"
echo
echo "Pode fechar o VS Code (Arquivo > Sair) para liberar memória: a extração continua."
