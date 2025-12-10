#!/usr/bin/env python3
"""
Fast GMB Checker using Playwright for browser automation
Parallel processing for bulk domain checking
"""

import asyncio
import re
from dataclasses import dataclass, field
from typing import Optional, List, Tuple
from urllib.parse import urlparse
import csv
import argparse

try:
    from playwright.async_api import async_playwright, Page, Browser
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False
    print("Warning: Playwright not installed. Run: pip install playwright && playwright install chromium")


@dataclass
class GMBResult:
    """Successful GMB result"""
    domain: str
    has_gmb: bool = False
    business_name: Optional[str] = None
    address: Optional[str] = None
    phone: Optional[str] = None
    rating: Optional[float] = None
    reviews_count: Optional[int] = None
    website_url: Optional[str] = None
    gmb_url: Optional[str] = None
    category: Optional[str] = None
    confidence: str = "none"
    captcha_detected: bool = False


@dataclass 
class FailedGMB:
    """Failed GMB check"""
    domain: str
    error: str
    has_gmb: bool = False
    captcha_detected: bool = False


def normalize_domain(domain: str) -> str:
    """Normalize domain for comparison"""
    domain = domain.lower().strip()
    domain = re.sub(r'^https?://', '', domain)
    domain = re.sub(r'^www\.', '', domain)
    domain = domain.rstrip('/')
    domain = domain.split('/')[0]  # Remove path
    return domain


def domains_match(url1: str, url2: str) -> bool:
    """Check if two URLs have the same domain"""
    if not url1 or not url2:
        return False
    return normalize_domain(url1) == normalize_domain(url2)


async def check_for_captcha(page: Page) -> bool:
    """Check if page has a CAPTCHA"""
    try:
        content = await page.content()
        captcha_indicators = [
            'unusual traffic',
            'not a robot',
            'captcha',
            'recaptcha',
            'verify you are human',
            'automated queries',
            'please verify',
            'g-recaptcha'
        ]
        content_lower = content.lower()
        return any(indicator in content_lower for indicator in captcha_indicators)
    except:
        return False


async def check_gmb_listing(
    page: Page, 
    domain: str, 
    progress_callback=None
) -> GMBResult | FailedGMB:
    """Check if a domain has a GMB listing using Google Maps"""
    
    normalized_domain = normalize_domain(domain)
    search_query = normalized_domain
    maps_url = f"https://www.google.com/maps/search/{search_query}"
    
    try:
        # Navigate to Google Maps search
        await page.goto(maps_url, wait_until='domcontentloaded', timeout=20000)
        await asyncio.sleep(1.5)
        
        # Check for captcha
        if await check_for_captcha(page):
            return FailedGMB(
                domain=domain,
                error="CAPTCHA detected - Google is rate limiting",
                captcha_detected=True
            )
        
        # Wait for results to load
        try:
            await page.wait_for_selector('[role="feed"], [role="main"]', timeout=8000)
        except:
            pass
        
        await asyncio.sleep(1)
        
        # Check if we landed directly on a business page
        current_url = page.url
        if '/place/' in current_url:
            # We're on a business page, extract info
            return await extract_business_info(page, domain, normalized_domain)
        
        # Look for business results in the feed
        results = await page.query_selector_all('[role="feed"] > div')
        
        if not results or len(results) == 0:
            # Try alternative selector
            results = await page.query_selector_all('div[jsaction*="click"] a[href*="/maps/place/"]')
        
        # Iterate through results to find matching business
        for i, result in enumerate(results[:10]):  # Check first 10 results
            try:
                # Try to click on the result
                await result.click()
                await asyncio.sleep(1.5)
                
                # Check for captcha after click
                if await check_for_captcha(page):
                    return FailedGMB(
                        domain=domain,
                        error="CAPTCHA detected - Google is rate limiting",
                        captcha_detected=True
                    )
                
                # Extract business info and check domain match
                result_data = await extract_business_info(page, domain, normalized_domain)
                
                if result_data.has_gmb:
                    return result_data
                    
            except Exception as e:
                continue
        
        # No matching GMB found
        return GMBResult(
            domain=domain,
            has_gmb=False,
            confidence="none"
        )
        
    except Exception as e:
        error_msg = str(e)
        if 'timeout' in error_msg.lower():
            error_msg = "Timeout - page took too long to load"
        return FailedGMB(domain=domain, error=error_msg)


async def extract_business_info(page: Page, original_domain: str, normalized_domain: str) -> GMBResult:
    """Extract business information from a Google Maps business page"""
    
    result = GMBResult(domain=original_domain)
    
    try:
        # Get page content
        content = await page.content()
        
        # Extract business name
        try:
            name_elem = await page.query_selector('h1')
            if name_elem:
                result.business_name = await name_elem.inner_text()
        except:
            pass
        
        # Extract website URL - this is critical for domain matching
        website_url = None
        try:
            # Look for website link
            website_selectors = [
                'a[data-item-id="authority"]',
                'a[href*="http"][data-tooltip*="website"]',
                'a[aria-label*="Website"]',
                'a[data-value*="http"]'
            ]
            
            for selector in website_selectors:
                elem = await page.query_selector(selector)
                if elem:
                    href = await elem.get_attribute('href')
                    if href and 'google.com' not in href:
                        # Google Maps wraps URLs, extract actual URL
                        if 'url=' in href:
                            import urllib.parse
                            parsed = urllib.parse.parse_qs(urllib.parse.urlparse(href).query)
                            if 'url' in parsed:
                                website_url = parsed['url'][0]
                        else:
                            website_url = href
                        break
            
            # Also try to find website in the text
            if not website_url:
                website_pattern = r'(?:Website|web)[:\s]*([a-zA-Z0-9][a-zA-Z0-9-]*\.[a-zA-Z]{2,})'
                match = re.search(website_pattern, content, re.IGNORECASE)
                if match:
                    website_url = match.group(1)
                    
        except Exception as e:
            pass
        
        result.website_url = website_url
        
        # Check if domains match
        if website_url and domains_match(website_url, normalized_domain):
            result.has_gmb = True
            result.confidence = "high"
            result.gmb_url = page.url
        else:
            result.has_gmb = False
            result.confidence = "none"
            return result
        
        # Extract rating
        try:
            rating_elem = await page.query_selector('span[role="img"][aria-label*="star"]')
            if rating_elem:
                rating_text = await rating_elem.get_attribute('aria-label')
                rating_match = re.search(r'([\d.]+)\s*star', rating_text)
                if rating_match:
                    result.rating = float(rating_match.group(1))
        except:
            pass
        
        # Extract reviews count
        try:
            reviews_elem = await page.query_selector('span[aria-label*="review"]')
            if reviews_elem:
                reviews_text = await reviews_elem.get_attribute('aria-label')
                reviews_match = re.search(r'([\d,]+)\s*review', reviews_text)
                if reviews_match:
                    result.reviews_count = int(reviews_match.group(1).replace(',', ''))
        except:
            pass
        
        # Extract address
        try:
            address_elem = await page.query_selector('button[data-item-id="address"]')
            if address_elem:
                result.address = await address_elem.get_attribute('aria-label')
                if result.address:
                    result.address = result.address.replace('Address: ', '')
        except:
            pass
        
        # Extract phone
        try:
            phone_elem = await page.query_selector('button[data-item-id*="phone"]')
            if phone_elem:
                result.phone = await phone_elem.get_attribute('aria-label')
                if result.phone:
                    result.phone = result.phone.replace('Phone: ', '')
        except:
            pass
        
        # Extract category
        try:
            category_elem = await page.query_selector('button[jsaction*="category"]')
            if category_elem:
                result.category = await category_elem.inner_text()
        except:
            pass
            
    except Exception as e:
        pass
    
    return result


async def bulk_check_gmb(
    domains: List[str],
    max_workers: int = 5,
    headless: bool = True,
    progress_callback=None
) -> Tuple[List[GMBResult], List[FailedGMB]]:
    """Check multiple domains for GMB listings in parallel"""
    
    if not PLAYWRIGHT_AVAILABLE:
        return [], [FailedGMB(d, "Playwright not installed") for d in domains]
    
    results = []
    failed = []
    semaphore = asyncio.Semaphore(max_workers)
    
    async def check_with_semaphore(domain: str, browser: Browser):
        async with semaphore:
            context = await browser.new_context(
                viewport={'width': 1280, 'height': 800},
                user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
            )
            page = await context.new_page()
            
            try:
                result = await check_gmb_listing(page, domain, progress_callback)
                return result
            finally:
                await context.close()
    
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=headless)
        
        try:
            tasks = [check_with_semaphore(domain, browser) for domain in domains]
            all_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for i, result in enumerate(all_results):
                if isinstance(result, Exception):
                    failed.append(FailedGMB(domains[i], str(result)))
                elif isinstance(result, FailedGMB):
                    failed.append(result)
                else:
                    results.append(result)
                    
        finally:
            await browser.close()
    
    return results, failed


def check_gmb_sync(domains: List[str], max_workers: int = 5, headless: bool = True) -> Tuple[List[GMBResult], List[FailedGMB]]:
    """Synchronous wrapper for bulk_check_gmb"""
    return asyncio.run(bulk_check_gmb(domains, max_workers, headless))


def save_csv(results: List[GMBResult], failed: List[FailedGMB], filename: str):
    """Save results to CSV"""
    with open(filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            'domain', 'has_gmb', 'business_name', 'rating', 'reviews', 
            'address', 'phone', 'website_url', 'gmb_url', 'category', 
            'confidence', 'captcha', 'error'
        ])
        
        for r in results:
            writer.writerow([
                r.domain, 'Yes' if r.has_gmb else 'No', r.business_name or '',
                r.rating or '', r.reviews_count or '', r.address or '',
                r.phone or '', r.website_url or '', r.gmb_url or '',
                r.category or '', r.confidence, 
                'Yes' if r.captcha_detected else 'No', ''
            ])
        
        for f_item in failed:
            writer.writerow([
                f_item.domain, 'No', '', '', '', '', '', '', '', '', 'none',
                'Yes' if f_item.captcha_detected else 'No', f_item.error
            ])


def main():
    parser = argparse.ArgumentParser(description='Fast GMB Checker')
    parser.add_argument('domains', nargs='*', help='Domains to check')
    parser.add_argument('-f', '--file', help='File with domains (one per line)')
    parser.add_argument('-o', '--output', default='gmb_results.csv', help='Output CSV file')
    parser.add_argument('-w', '--workers', type=int, default=5, help='Parallel workers (default: 5)')
    parser.add_argument('--visible', action='store_true', help='Show browser window')
    args = parser.parse_args()
    
    # Get domains
    domains = list(args.domains) if args.domains else []
    
    if args.file:
        with open(args.file, 'r') as f:
            domains.extend([line.strip() for line in f if line.strip()])
    
    if not domains:
        print("No domains provided. Use: python gmb_checker_fast.py domain1.com domain2.com")
        print("Or: python gmb_checker_fast.py -f domains.txt")
        return
    
    print(f"Checking {len(domains)} domains with {args.workers} workers...")
    
    results, failed = check_gmb_sync(domains, args.workers, headless=not args.visible)
    
    # Print summary
    gmb_count = sum(1 for r in results if r.has_gmb)
    captcha_count = sum(1 for r in results if r.captcha_detected) + sum(1 for f in failed if f.captcha_detected)
    
    print(f"\n{'='*60}")
    print(f"Results: {gmb_count} with GMB, {len(results) - gmb_count} without GMB, {len(failed)} failed")
    if captcha_count > 0:
        print(f"⚠️  CAPTCHA detected {captcha_count} times - reduce workers or wait")
    print(f"{'='*60}")
    
    for r in results:
        status = "✅ GMB" if r.has_gmb else "❌ No GMB"
        extra = f" ({r.business_name})" if r.business_name and r.has_gmb else ""
        print(f"{status}: {r.domain}{extra}")
    
    for f_item in failed:
        captcha = " [CAPTCHA]" if f_item.captcha_detected else ""
        print(f"⚠️  Failed: {f_item.domain} - {f_item.error}{captcha}")
    
    # Save CSV
    save_csv(results, failed, args.output)
    print(f"\nResults saved to {args.output}")


if __name__ == '__main__':
    main()

