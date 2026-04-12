#!/usr/bin/env python3
"""
HKJC Screenshot Capture - Capture race card from your screen
Helps you see and manually enter odds from HKJC platform
"""

import os
import sys
from datetime import datetime
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from PIL import Image, ImageGrab
    import pyautogui
    PYAUTOGUI_AVAILABLE = True
except ImportError:
    PYAUTOGUI_AVAILABLE = False
    logger.warning("pyautogui/Pillow not available")

class ScreenshotCapture:
    def __init__(self):
        self.capture_dir = Path(__file__).parent / 'screenshots'
        self.capture_dir.mkdir(exist_ok=True)
    
    def capture_full_screen(self) -> str:
        if not PYAUTOGUI_AVAILABLE:
            print("❌ Screenshot capture requires Pillow and pyautogui")
            print("   Install with: pip install Pillow pyautogui")
            return None
        
        try:
            screenshot = ImageGrab.grab()
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"hkjc_capture_{timestamp}.png"
            filepath = self.capture_dir / filename
            
            screenshot.save(filepath)
            logger.info(f"Screenshot saved to: {filepath}")
            
            return str(filepath)
        except Exception as e:
            logger.error(f"Error capturing screenshot: {e}")
            return None
    
    def capture_region(self, region: tuple = None) -> str:
        if not PYAUTOGUI_AVAILABLE:
            print("❌ Screenshot capture requires Pillow and pyautogui")
            return None
        
        try:
            if region is None:
                print("\n📸 Click and drag to select region...")
                print("   Press Ctrl+C to cancel")
                
                import time
                time.sleep(2)
                
                region = pyautogui.selectRegion()
                
                if not region:
                    print("❌ No region selected")
                    return None
            
            screenshot = ImageGrab.grab(bbox=region)
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"hkjc_region_{timestamp}.png"
            filepath = self.capture_dir / filename
            
            screenshot.save(filepath)
            logger.info(f"Region screenshot saved to: {filepath}")
            
            return str(filepath)
        except Exception as e:
            logger.error(f"Error capturing region: {e}")
            return None
    
    def show_on_screen(self, filepath: str):
        if filepath and os.path.exists(filepath):
            try:
                img = Image.open(filepath)
                img.show()
                return True
            except Exception as e:
                logger.error(f"Error showing image: {e}")
        return False

def interactive_screenshot():
    if not PYAUTOGUI_AVAILABLE:
        print("\n" + "="*60)
        print("❌ Screenshot tools not installed")
        print("="*60)
        print("\nInstall with:")
        print("  pip install Pillow pyautogui")
        print("\nThen run this script again.")
        return
    
    capture = ScreenshotCapture()
    
    print("\n" + "="*60)
    print("📸 HKJC SCREENSHOT CAPTURE")
    print("="*60)
    print(f"\nScreenshots will be saved to: {capture.capture_dir}")
    
    print("\nOptions:")
    print("  1. Full screen capture")
    print("  2. Region capture (select area)")
    print("  3. View last capture")
    print("  q. Quit")
    
    while True:
        choice = input("\nSelect option: ").strip().lower()
        
        if choice == '1':
            print("\n📸 Taking full screen capture in 3 seconds...")
            print("   Make sure HKJC is visible on screen!")
            import time
            time.sleep(3)
            
            filepath = capture.capture_full_screen()
            if filepath:
                print(f"✅ Saved: {filepath}")
                show = input("Open screenshot? (y/n): ").strip().lower()
                if show == 'y':
                    capture.show_on_screen(filepath)
        
        elif choice == '2':
            print("\n🖱️  Drag to select region on screen")
            print("   (Opens pyautogui selection tool)")
            filepath = capture.capture_region()
            if filepath:
                print(f"✅ Saved: {filepath}")
        
        elif choice == '3':
            files = sorted(capture.capture_dir.glob('*.png'), key=lambda x: x.stat().st_mtime, reverse=True)
            if files:
                print(f"\n📁 Last capture: {files[0]}")
                show = input("Open it? (y/n): ").strip().lower()
                if show == 'y':
                    capture.show_on_screen(str(files[0]))
            else:
                print("No screenshots found")
        
        elif choice == 'q':
            break
        
        else:
            print("Invalid option")

def quick_capture():
    capture = ScreenshotCapture()
    print("\n📸 Quick capture - saving screenshot...")
    filepath = capture.capture_full_screen()
    if filepath:
        print(f"\n✅ Screenshot saved!")
        print(f"   {filepath}")
        print("\n   Open HKJC in your browser, then run this again")
    return filepath

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == 'quick':
        quick_capture()
    else:
        interactive_screenshot()
