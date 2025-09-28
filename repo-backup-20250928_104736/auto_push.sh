#!/bin/bash

while true; do
  cd /home/nabil/big-turkish-llm-project
  git add .
  if ! git diff --cached --quiet; then
    git commit -m "Auto: $(date '+%Y-%m-%d %H:%M:%S')"
    git push
  fi
  sleep 300
done
#!/bin/bash

while true; do
    git add .
    git commit -m "Auto: $(date '+%Y-%m-%d %H:%M:%S')" 2>/dev/null
    git push
    sleep 300  # every 5 minutes
done
#!/bin/bash
while true; do
    git add .
    git commit -m "Auto commit on $(date)"
    git push origin main
    sleep 300  # every 5 minutes
done

