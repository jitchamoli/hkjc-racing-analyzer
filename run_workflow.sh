#!/bin/bash
# HKJC Racing - Complete Workflow Script
# Opens HKJC, takes screenshot, runs analysis

SCRIPT_DIR="$HOME/Documents/project_hub/hkjc_racing_analyzer"

echo ""
echo "🏇 HKJC RACING - COMPLETE WORKFLOW"
echo "=================================="
echo ""

# Check if Python packages are installed
python3 -c "import pyautogui, PIL" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "Installing required packages..."
    pip3 install Pillow pyautogui --quiet
fi

# Step 1: Open HKJC
echo "📺 Step 1: Opening HKJC Race Card..."
open -a "Google Chrome" "https://racing.hkjc.com/racing/information/English/racing/RaceCard.aspx"

echo ""
echo "⏳ Waiting 5 seconds for page to load..."
sleep 5

# Step 2: Take screenshot
echo ""
echo "📸 Step 2: Taking screenshot..."
python3 "$SCRIPT_DIR/screenshot_capture.py" quick

echo ""
echo "✅ Screenshot captured and saved!"
echo ""

# Step 3: Show screenshot
SCREENSHOT=$(ls -t "$SCRIPT_DIR/screenshots/"*.png 2>/dev/null | head -1)
if [ -n "$SCREENSHOT" ]; then
    echo "🖼️  Opening screenshot..."
    open "$SCREENSHOT"
fi

echo ""
echo "=================================="
echo "NEXT STEPS:"
echo "=================================="
echo ""
echo "1. Look at the screenshot to see the real odds"
echo "2. Run the interactive analyzer:"
echo "   python3 $SCRIPT_DIR/interactive_analyzer.py"
echo ""
echo "3. Enter the odds you see on the HKJC website"
echo ""
echo "4. After the race, update results:"
echo "   python3 $SCRIPT_DIR/update_results.py"
echo ""
echo "=================================="
