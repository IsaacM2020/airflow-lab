#!/bin/bash
# Double-click to start the wind tunnel: launches the dev server if needed and opens Chrome (WebGPU).
cd "$(dirname "$0")"
if ! lsof -i :5180 >/dev/null 2>&1; then
  nohup npm run dev > /tmp/airflow-lab.log 2>&1 &
  for i in $(seq 1 30); do lsof -i :5180 >/dev/null 2>&1 && break; sleep 0.5; done
fi
open -a "Google Chrome" "http://127.0.0.1:5180" 2>/dev/null || open "http://127.0.0.1:5180"
