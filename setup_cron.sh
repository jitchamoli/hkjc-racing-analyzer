#!/bin/bash
# HKJC Racing Scheduler Setup
# Run this to set up automatic daily race analysis

SCRIPT_DIR="$HOME/Documents/project_hub/hkjc_racing_analyzer"
SCHEDULER="$SCRIPT_DIR/racing_scheduler.py"

echo "🏇 HKJC Racing Scheduler Setup"
echo "================================"

# Check if Python3 is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 not found. Please install Python 3.8+"
    exit 1
fi

# Create cron jobs
echo ""
echo "Setting up cron jobs..."

# Wednesday races (Happy Valley) - runs at 5:30 PM HKT
(crontab -l 2>/dev/null | grep -v "racing_scheduler"; echo "30 17 * * 3 cd $SCRIPT_DIR && python3 racing_scheduler.py >> $SCRIPT_DIR/logs/scheduler.log 2>&1") | crontab -

# Saturday races (Sha Tin) - runs at 12:30 PM HKT
(crontab -l 2>/dev/null | grep -v "racing_scheduler"; echo "30 12 * * 6 cd $SCRIPT_DIR && python3 racing_scheduler.py >> $SCRIPT_DIR/logs/scheduler.log 2>&1") | crontab -

# Sunday races (Sha Tin) - runs at 12:30 PM HKT  
(crontab -l 2>/dev/null | grep -v "racing_scheduler"; echo "30 12 * * 0 cd $SCRIPT_DIR && python3 racing_scheduler.py >> $SCRIPT_DIR/logs/scheduler.log 2>&1") | crontab -

# Create logs directory
mkdir -p "$SCRIPT_DIR/logs"

echo "✅ Cron jobs created!"
echo ""
echo "Current cron jobs:"
crontab -l | grep racing_scheduler || echo "(no jobs found)"
echo ""
echo "Schedule:"
echo "  Wednesday 5:30 PM - Happy Valley races"
echo "  Saturday 12:30 PM - Sha Tin races"
echo "  Sunday 12:30 PM - Sha Tin races"
echo ""
echo "Results will be saved to: $SCRIPT_DIR/results/"
echo "Logs will be saved to: $SCRIPT_DIR/logs/scheduler.log"
