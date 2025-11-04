#!/bin/bash
# Skript pro zabití všech běžících instancí IsaacLab (bash ./isaaclab.sh ...)

echo "🔍 Hledám běžící procesy IsaacLab..."
pids=$(ps aux | grep "[i]saaclab" | awk '{print $2}')

if [ -z "$pids" ]; then
  echo "✅ Žádné běžící IsaacLab procesy nenalezeny."
  exit 0
fi

echo "⚠️  Nalezeny procesy: $pids"
for pid in $pids; do
  echo "🔪 Ukončuji PID $pid..."
  kill -9 "$pid" 2>/dev/null
done

echo "✅ Všechny procesy IsaacLab byly ukončeny."
