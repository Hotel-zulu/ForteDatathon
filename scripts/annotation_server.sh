#!/usr/bin/env bash
# Запуск разметки для всей команды одной командой:  bash scripts/annotation_server.sh
# Поднимает Label Studio + публичную ссылку ngrok. Закрыть окно = остановить.
set -e
cd "$(dirname "$0")/.."
PUBLIC_URL="${PUBLIC_URL:-https://craving-suffrage-grub.ngrok-free.dev}"
source .venv/bin/activate
pkill -f "ngrok http 8080" 2>/dev/null || true
kill $(lsof -tiTCP:8080 -sTCP:LISTEN) 2>/dev/null || true
sleep 2
ngrok http 8080 --url="$PUBLIC_URL" --log=stdout > annotation/ngrok.log 2>&1 &
NGROK=$!
trap 'kill $NGROK 2>/dev/null' EXIT
echo "Команда открывает: $PUBLIC_URL"
caffeinate -i -w $$ &        # Mac не уснёт, пока работает сервер
LS_PUBLIC_URL="$PUBLIC_URL" python scripts/label_studio.py start
