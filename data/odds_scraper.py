#!/usr/bin/env python3
"""
HKJC Odds Scraper - Tries multiple methods to get real betting odds
"""

import asyncio
import re
import json
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)

class OddsScraper:
    ODDS_API_URL = "https://racing.hkjc.com/racing/content/json/DisplayOdds.aspx"
    
    def __init__(self):
        self.session = None
        self.cached_odds = {}
    
    async def init_session(self):
        from playwright.async_api import async_playwright
        
        pw = await async_playwright().start()
        self.browser = await pw.chromium.launch(headless=True)
        self.context = await self.browser.new_context(
            user_agent='Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        )
    
    async def close(self):
        if self.browser:
            await self.browser.close()
    
    async def scrape_odds_page(self, race_date: str, race_number: int) -> Optional[Dict[int, float]]:
        page = None
        try:
            page = await self.context.new_page()

            url = f"https://racing.hkjc.com/racing/information/English/racing/DisplayOdds.aspx?RaceDate={race_date}&RaceNo={race_number}"

            await page.goto(url, wait_until='domcontentloaded', timeout=60000)
            await page.wait_for_timeout(5000)

            html = await page.content()

            odds = self._parse_odds_html(html)

            if odds:
                logger.info(f"Found {len(odds)} odds for Race {race_number}")
                return odds

            odds = await self._try_json_endpoint(race_date, race_number)
            if odds:
                return odds

            return None

        except Exception as e:
            logger.error(f"Error scraping odds: {e}")
            return None
        finally:
            if page:
                await page.close()
    
    def _parse_odds_html(self, html: str) -> Dict[int, float]:
        odds = {}
        
        patterns = [
            r'<td[^>]*>\s*(\d+)\s*</td>\s*<td[^>]*>\s*([\d.]+)\s*</td>',
            r'horseNo["\']?\s*:\s*(\d+)[^}]*winOdds["\']?\s*:\s*([\d.]+)',
            r'"no"\s*:\s*(\d+)[^}]*?"odds"\s*:\s*([\d.]+)',
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, html)
            for match in matches:
                try:
                    horse_no = int(match[0])
                    odds_val = float(match[1])
                    if odds_val > 0:
                        odds[horse_no] = odds_val
                except:
                    continue
        
        return odds
    
    async def _try_json_endpoint(self, race_date: str, race_number: int) -> Optional[Dict[int, float]]:
        page = None
        try:
            page = await self.context.new_page()

            url = f"https://racing.hkjc.com/racing/content/json/racecard?date={race_date}&venue=ST&race={race_number}"

            response = await page.goto(url, timeout=30000)

            if response and response.ok:
                content = await response.text()
                data = json.loads(content)
                odds = self._extract_odds_from_json(data)
                if odds:
                    return odds

            return None

        except Exception as e:
            logger.warning(f"JSON endpoint failed: {e}")
            return None
        finally:
            if page:
                await page.close()
    
    def _extract_odds_from_json(self, data) -> Dict[int, float]:
        odds = {}
        
        try:
            if isinstance(data, dict):
                for key, value in data.items():
                    if isinstance(value, list):
                        for item in value:
                            odds.update(self._extract_odds_from_json(item))
                    elif isinstance(value, dict):
                        if 'no' in value and 'odds' in value:
                            try:
                                horse_no = int(value['no'])
                                odds_val = float(value['odds'])
                                if odds_val > 0:
                                    odds[horse_no] = odds_val
                            except:
                                pass
                        else:
                            odds.update(self._extract_odds_from_json(value))
        except:
            pass
        
        return odds
    
    async def get_combined_odds(self, race_date: str, race_number: int) -> Dict[int, Dict[str, float]]:
        combined = {}

        page = await self.context.new_page()

        try:
            url = f"https://racing.hkjc.com/racing/information/English/racing/RaceCard.aspx?RaceDate={race_date}&RaceNo={race_number}"
            await page.goto(url, wait_until='domcontentloaded', timeout=60000)
            await page.wait_for_timeout(8000)

            html = await page.content()

            win_odds = self._parse_odds_html(html)

            place_patterns = [
                r'(\d+).*?place.*?([\d.]+)',
                r'place.*?(\d+).*?([\d.]+)',
            ]

            place_odds = {}
            for pattern in place_patterns:
                matches = re.findall(pattern, html, re.IGNORECASE)
                for match in matches:
                    try:
                        horse_no = int(match[0])
                        odds_val = float(match[1])
                        if 1.0 <= odds_val <= 10.0:
                            place_odds[horse_no] = odds_val
                    except Exception:
                        continue

            for horse_no, win_odd in win_odds.items():
                combined[horse_no] = {
                    'win': win_odd,
                    'place': place_odds.get(horse_no, win_odd / 3)
                }

            if combined:
                logger.info(f"Combined odds: {combined}")
        finally:
            await page.close()

        return combined

async def test_odds_scraper():
    scraper = OddsScraper()
    await scraper.init_session()
    
    try:
        print("Testing HKJC Odds Scraper...")
        
        odds = await scraper.scrape_odds_page("2026/03/22", 1)
        
        if odds:
            print(f"\n✅ Found {len(odds)} odds:")
            for horse_no, odd in sorted(odds.items()):
                print(f"   Horse #{horse_no}: {odd:.1f}")
        else:
            print("\n❌ Could not scrape odds directly")
            print("   Will use manual input or estimation")
    
    finally:
        await scraper.close()

if __name__ == "__main__":
    asyncio.run(test_odds_scraper())
