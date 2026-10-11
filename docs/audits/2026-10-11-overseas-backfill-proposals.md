# Overseas Horse-History Backfill Proposals

**Proposal only. No database, model, predictions or tickets changed.**

Scope: 37 rows across 22 dates. This is separate from the 221 incomplete local races.

## Missing Data

| Field | Missing/unavailable rows |
|---|---:|
| venue | 37 |
| horse_number | 37 |
| course | 37 |
| horse_weight | 37 |
| running_position | 37 |
| sectional_times | 37 |
| win_odds | 4 |
| finish_time | 10 |
| age | 2 |
| class_type | 1 |

Missing includes NULL, blank, `--` and `-`; zero is unavailable for odds/body weight. All race numbers are 1 and require identity reconciliation.

## Row-Level Proposals

Track and Overseas category are source-backed identity proposals. Numbers below are HKJC overseas saddlecloth numbers, NOT barrier draws; racecard evidence must be checked against final results before applying. Blank numbers remain unresolved.

| Row ID | Date | Horse | Foreign track | Proposed number | Evidence |
|---|---|---|---|---:|---|
| 37972 | 2022/09/04 | COMPUTER PATCH | Seoul | unresolved | [Identity](https://www.hkjc.com/english/corporate/racing_news_item.asp?in_file=%2Fenglish%2Fnews%2F2022-09%2Fnews_2022090401830.html) (results) |
| 37973 | 2022/09/04 | KINGS SHIELD | Seoul | unresolved | [Identity](https://www.hkjc.com/english/corporate/racing_news_item.asp?in_file=%2Fenglish%2Fnews%2F2022-09%2Fnews_2022090401830.html) (results) |
| 32857 | 2023/02/18 | RUSSIAN EMPEROR | Al Rayyan | unresolved | [Identity](https://campaign.hkjc.com/media-guide/pdf/media-guide-2023-24-eng.pdf) (results) |
| 32858 | 2023/02/18 | SENOR TOBA | Al Rayyan | unresolved | [Identity](https://campaign.hkjc.com/media-guide/pdf/media-guide-2023-24-eng.pdf) (results) |
| 32412 | 2023/03/04 | RUSSIAN EMPEROR | Meydan | unresolved | [Identity](https://res.hkjc.com/campaigns/wp-content/uploads/sites/714/2023HKIR_Formguide-Eng.pdf) (results) |
| 32413 | 2023/03/04 | SENOR TOBA | Meydan | unresolved | [Identity](https://res.hkjc.com/campaigns/wp-content/uploads/sites/714/2023HKIR_Formguide-Eng.pdf) (results) |
| 31712 | 2023/03/25 | DUKE WAI | Meydan | 4 | [Identity](https://campaign.hkjc.com/media-guide/pdf/media-guide-2024-25-eng.pdf) (results); [number](https://www.hkjc.com/english/corporate/racing_news_item.asp?in_file=%2Fenglish%2Fnews%2F2023-03%2Fnews_2023032401623.html) |
| 31715 | 2023/03/25 | GLORIOUS DRAGON | Meydan | 5 | [Identity](https://campaign.hkjc.com/media-guide/pdf/media-guide-2024-25-eng.pdf) (results); [number](https://www.hkjc.com/english/corporate/racing_news_item.asp?in_file=%2Fenglish%2Fnews%2F2023-03%2Fnews_2023032401623.html) |
| 31713 | 2023/03/25 | RUSSIAN EMPEROR | Meydan | 4 | [Identity](https://campaign.hkjc.com/media-guide/pdf/media-guide-2024-25-eng.pdf) (results); [number](https://www.hkjc.com/english/corporate/racing_news_item.asp?in_file=%2Fenglish%2Fnews%2F2023-03%2Fnews_2023032401623.html) |
| 31714 | 2023/03/25 | SENOR TOBA | Meydan | 5 | [Identity](https://campaign.hkjc.com/media-guide/pdf/media-guide-2024-25-eng.pdf) (results); [number](https://www.hkjc.com/english/corporate/racing_news_item.asp?in_file=%2Fenglish%2Fnews%2F2023-03%2Fnews_2023032401623.html) |
| 31711 | 2023/03/25 | SIGHT SUCCESS | Meydan | 11 | [Identity](https://campaign.hkjc.com/media-guide/pdf/media-guide-2024-25-eng.pdf) (results); [number](https://www.hkjc.com/english/corporate/racing_news_item.asp?in_file=%2Fenglish%2Fnews%2F2023-03%2Fnews_2023032401623.html) |
| 28844 | 2023/06/24 | WELLINGTON | Ascot | unresolved | [Identity](https://racingnews.hkjc.com/english/?p=69341) (results) |
| 27815 | 2023/09/10 | APACHE PASS | Seoul | unresolved | [Identity](https://www.hkjc.com/english/corporate/racing_news_item.asp?in_file=%2Fenglish%2Fnews%2F2023-09%2Fnews_2023091001825.html) (results) |
| 27810 | 2023/09/10 | DUKE WAI | Seoul | unresolved | [Identity](https://www.hkjc.com/english/corporate/racing_news_item.asp?in_file=%2Fenglish%2Fnews%2F2023-09%2Fnews_2023091001825.html) (results) |
| 27146 | 2023/10/07 | ROMANTIC WARRIOR | Flemington | unresolved | [Identity](https://www.hkjc.com/english/corporate/racing_news_item.asp?in_file=%2Fenglish%2Fnews%2F2023-10%2Fnews_2023100701640.html) (results) |
| 26596 | 2023/10/25 | ROMANTIC CHARM | Geelong | unresolved | [Identity](https://racing.hkjc.com/en-us/local/information/otherhorse?horseid=HK_2022_H227) (results) |
| 26592 | 2023/10/28 | ROMANTIC WARRIOR | Moonee Valley | unresolved | [Identity](https://racingnews.hkjc.com/english/2023/10/28/romantic-warrior-creates-history-with-epic-g1-w-s-cox-plate-victory/) (results) |
| 22919 | 2024/02/17 | RUSSIAN EMPEROR | Al Rayyan | unresolved | [Identity](https://racing.hkjc.com/en-us/local/information/otherhorse?horseid=HK_2020_E198) (results) |
| 21669 | 2024/03/24 | VICTOR THE WINNER | Chukyo | unresolved | [Identity](https://racingnews.hkjc.com/english/2024/03/24/victor-the-winner-finishes-third-in-g1-takamatsunomiya-kinen/) (results) |
| 21556 | 2024/03/30 | CALIFORNIA SPANGLE | Meydan | unresolved | [Identity](https://racingnews.hkjc.com/english/2024/03/30/california-spangle-wins-g1-al-quoz-sprint-for-cruz-avdulla/) (results) |
| 21557 | 2024/03/30 | SIGHT SUCCESS | Meydan | unresolved | [Identity](https://racingnews.hkjc.com/english/2024/03/30/california-spangle-wins-g1-al-quoz-sprint-for-cruz-avdulla/) (results) |
| 21558 | 2024/03/30 | STRAIGHT ARRON | Meydan | unresolved | [Identity](https://racingnews.hkjc.com/english/2024/03/30/california-spangle-wins-g1-al-quoz-sprint-for-cruz-avdulla/) (results) |
| 21559 | 2024/03/30 | VOYAGE BUBBLE | Meydan | unresolved | [Identity](https://racingnews.hkjc.com/english/2024/03/30/california-spangle-wins-g1-al-quoz-sprint-for-cruz-avdulla/) (results) |
| 19396 | 2024/06/02 | ROMANTIC WARRIOR | Tokyo | unresolved | [Identity](https://racingnews.hkjc.com/english/2024/06/02/extraordinary-romantic-warrior-wins-yasuda-kinen/) (results) |
| 19409 | 2024/06/02 | VOYAGE BUBBLE | Tokyo | unresolved | [Identity](https://racingnews.hkjc.com/english/2024/06/02/extraordinary-romantic-warrior-wins-yasuda-kinen/) (results) |
| 17377 | 2024/09/29 | MUGEN | Nakayama | unresolved | [Identity](https://www.hkjc.com/english/corporate/racing_news_item.asp?in_file=%2Fenglish%2Fnews%2F2024-09%2Fnews_2024092801708.html) (results) |
| 17376 | 2024/09/29 | VICTOR THE WINNER | Nakayama | unresolved | [Identity](https://www.hkjc.com/english/corporate/racing_news_item.asp?in_file=%2Fenglish%2Fnews%2F2024-09%2Fnews_2024092801708.html) (results) |
| 13588 | 2025/01/24 | ROMANTIC WARRIOR | Meydan | unresolved | [Identity](https://member.hkjc.com/member/english/horse-owner/glorious-moments/heroes-winning-overseas.aspx) (results) |
| 12708 | 2025/02/22 | ROMANTIC WARRIOR | Riyadh | unresolved | [Identity](https://worldpool.hkjc.com/en-US/moment-of-the-year-2025/moment-detail/2025/20250222-saudi-cup-day/) (results) |
| 11347 | 2025/04/05 | ROMANTIC WARRIOR | Meydan | 8 | [Identity](https://racingnews.hkjc.com/english/2025/04/03/simulcast-and-betting-arrangements-for-doncaster-mile-day-and-dubai-world-cup-day/) (entry_only); [number](https://racingnews.hkjc.com/english/2025/04/04/dubai-world-cup-day-reference-odds/) |
| 11348 | 2025/04/05 | SWORD POINT | Meydan | 16 | [Identity](https://racingnews.hkjc.com/english/2025/04/03/simulcast-and-betting-arrangements-for-doncaster-mile-day-and-dubai-world-cup-day/) (entry_only); [number](https://racingnews.hkjc.com/english/2025/04/04/dubai-world-cup-day-reference-odds/) |
| 7765 | 2025/09/07 | CHANCHENG GLORY | Seoul | unresolved | [Identity](https://racingnews.hkjc.com/english/2025/09/07/sha-tin-shines-despite-challenging-weather-with-perfect-start-to-2025-26-racing-season/) (results) |
| 7762 | 2025/09/07 | SELF IMPROVEMENT | Seoul | unresolved | [Identity](https://racingnews.hkjc.com/english/2025/09/07/sha-tin-shines-despite-challenging-weather-with-perfect-start-to-2025-26-racing-season/) (results) |
| 7235 | 2025/09/28 | LUCKY SWEYNESSE | Nakayama | unresolved | [Identity](https://racingnews.hkjc.com/english/2025/09/28/win-carnelian-shocks-in-sprinters-stakes-as-lucky-sweynesse-finishes-down-the-order/) (results) |
| 6626 | 2025/10/18 | KA YING RISING | Randwick | 1 | [Identity](https://racingnews.hkjc.com/english/2025/10/18/ka-ying-rising-surges-into-racing-history-with-g1-the-everest-triumph/) (results); [number](https://campaigns.hkjc.com/simulcast-everest-2025/en/) |
| 2345 | 2026/02/14 | SELF IMPROVEMENT | Riyadh | unresolved | [Identity](https://www.hkjc.com/english/pressrelease/mcs01_showhtml.asp?SelType=NEWS&filename=20260215_004935_E_NEWS.htm) (results) |
| 1944 | 2026/02/28 | SING DRAGON | Meydan | unresolved | [Identity](https://consvc.hkjc.com/-/media/Sites/JCRW/FormGuide/Simulcast/25_26/20260228/OSE20260228_starter_all.pdf) (entry_only) |

## Application Constraints

- Do not apply this JSON directly as an SQL migration; it is an evidence ledger, not an executable patch.
- Local-only code must explicitly accept Sha Tin/Happy Valley before changing NULL venues to Overseas. Truthy venue checks are not sufficient.
- Preserve actual foreign track separately; do not put it into the HK rail-course field.
- Resolve race identity before correcting race_number: HKJC S-session numbers and host-country race numbers are different namespaces.
- Preserve foreign Group/class labels, including Qatar-specific classification; no blanket conversion to HK classes.
- Reference odds are not final odds. No odds replacements are proposed.
- No replacement finish times, body weights, ages or sectionals are proposed without field-specific evidence.
- Stored finish_time="--" is unavailable, not a valid measured time. Never fill a nonwinner with the winning race time.
- These remain partial horse histories even after backfill; they must not join local race fields or become complete overseas training fields.
- Original row snapshot and field-level proposals are in the companion JSON; unresolved fields are explicit.

Verification: 37 records; 22 dates; 8 racecard-number proposals; source rows unchanged (SHA256 `9c8dbe1c741fbee2362c83257a9bb60ac9f2e9defbe00b612117a430bf40e4ac`).
