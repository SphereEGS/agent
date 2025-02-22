#!/bin/bash
python main.py > /dev/null 2>&1 &
echo $! > app.pid
tail -f app.log | awk '{lines[NR%10] = $0} NR>=10 {system("clear"); for (i=NR%10+1; i<=NR%10+10; i++) print lines[i%10]}'
