#!/usr/bin/env python3
"""
Domain Tools Web UI
Wayback history analysis & Google My Business detection
"""

from flask import Flask, render_template, request, jsonify
import threading
import uuid
import time
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
from wayback_checker import WaybackChecker, DomainReport, FailedDomain, Snapshot
from typing import List, Optional
from datetime import datetime

# Try to import GMB checker
try:
    from gmb_checker_fast import (
        bulk_check_gmb, GMBResult, FailedGMB, 
        PLAYWRIGHT_AVAILABLE, check_gmb_sync
    )
    GMB_AVAILABLE = PLAYWRIGHT_AVAILABLE
except ImportError:
    GMB_AVAILABLE = False
    print("Warning: GMB checker not available. Install playwright: pip install playwright && playwright install chromium")

app = Flask(__name__)
app.config['SECRET_KEY'] = 'wayback-checker-secret'

# Store analysis sessions
sessions = {}

# Max GMB workers
GMB_MAX_WORKERS = 100


class ProgressTrackingChecker(WaybackChecker):
    """Extended checker that tracks progress per domain"""
    
    def __init__(self, session_id: str, domain_id: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.session_id = session_id
        self.domain_id = domain_id
    
    def update_domain_progress(self, snapshots_checked: int, total_snapshots: int, step: str):
        """Update progress for this domain in the session"""
        if self.session_id in sessions:
            session = sessions[self.session_id]
            session['domain_progress'][self.domain_id] = {
                'snapshots_checked': snapshots_checked,
                'total_snapshots': total_snapshots,
                'step': step,
                'percent': int((snapshots_checked / total_snapshots) * 100) if total_snapshots > 0 else 0
            }
    
    def analyze_domain(self, domain: str) -> DomainReport | FailedDomain:
        """Override to track progress"""
        original_domain = domain
        
        # Normalize domain
        if not domain.startswith(('http://', 'https://')):
            domain = f"https://{domain}"
        
        # Update: fetching snapshots
        self.update_domain_progress(0, 1, 'fetching_list')
        
        try:
            all_snapshots = self.get_available_snapshots(domain)
        except Exception as e:
            error_msg = f"Failed to fetch snapshots: {str(e)}"
            return FailedDomain(domain=original_domain, error=error_msg)
        
        if not all_snapshots:
            error_msg = "No archived snapshots found"
            return FailedDomain(domain=original_domain, error=error_msg)
        
        # Select snapshots by interval
        selected = self.select_snapshots_by_interval(all_snapshots)
        total_to_fetch = len(selected)
        
        self.update_domain_progress(0, total_to_fetch, 'fetching_content')
        
        # Fetch content for selected snapshots
        snapshots_data: List[Snapshot] = []
        
        for i, snap in enumerate(selected):
            self.update_domain_progress(i, total_to_fetch, 'fetching_content')
            snapshot = self.fetch_snapshot_content(domain, snap['timestamp'])
            if snapshot:
                snapshots_data.append(snapshot)
            time.sleep(0.3)  # Be nice to Wayback Machine
        
        self.update_domain_progress(total_to_fetch, total_to_fetch, 'analyzing')
        
        if not snapshots_data:
            error_msg = "Could not fetch any snapshot content"
            return FailedDomain(domain=original_domain, error=error_msg)
        
        # Calculate changes between consecutive snapshots
        change_timeline = []
        similarities = []
        change_percents = []
        
        for i in range(1, len(snapshots_data)):
            prev = snapshots_data[i-1]
            curr = snapshots_data[i]
            
            similarity = self.calculate_text_similarity(prev.text_content, curr.text_content)
            change_pct = round((1 - similarity) * 100, 1)
            similarities.append(similarity)
            change_percents.append(change_pct)
            
            change_timeline.append({
                'from': prev.formatted_date,
                'to': curr.formatted_date,
                'similarity': round(similarity * 100, 1),
                'change_percent': change_pct,
                'title_from': prev.title[:50] if prev.title else "N/A",
                'title_to': curr.title[:50] if curr.title else "N/A"
            })
        
        # Detect niche changes
        niche_changed, _ = self.detect_niche_change(snapshots_data)
        
        # Calculate overall consistency score
        if similarities:
            avg_similarity = sum(similarities) / len(similarities)
            consistency_score = round(avg_similarity * 100, 1)
            avg_change = round(sum(change_percents) / len(change_percents), 1)
            max_change = round(max(change_percents), 1)
        else:
            consistency_score = 100.0
            avg_change = 0.0
            max_change = 0.0
        
        first_date = snapshots_data[0].formatted_date
        last_date = snapshots_data[-1].formatted_date
        
        if consistency_score >= 80:
            stability_rating = "Very Stable"
        elif consistency_score >= 60:
            stability_rating = "Moderately Stable"
        elif consistency_score >= 40:
            stability_rating = "Significant Changes"
        else:
            stability_rating = "Major Overhauls"
        
        summary = f"Domain: {domain}\nPeriod: {first_date} → {last_date}\nConsistency: {consistency_score}%"
        
        first_keywords = self.extract_keywords(snapshots_data[0].text_content, top_n=10)
        last_keywords = self.extract_keywords(snapshots_data[-1].text_content, top_n=10)
        
        self.update_domain_progress(total_to_fetch, total_to_fetch, 'complete')
        
        return DomainReport(
            domain=domain,
            total_snapshots=len(all_snapshots),
            analyzed_snapshots=len(snapshots_data),
            first_snapshot=first_date,
            last_snapshot=last_date,
            consistency_score=consistency_score,
            niche_changed=niche_changed,
            change_timeline=change_timeline,
            summary=summary,
            avg_change_percent=avg_change,
            max_change_percent=max_change,
            stability_rating=stability_rating,
            top_keywords_first=", ".join(first_keywords[:5]),
            top_keywords_last=", ".join(last_keywords[:5])
        )


def generate_story(report: DomainReport) -> str:
    """Generate an interesting narrative about the domain's history"""
    domain_clean = report.domain.replace('https://', '').replace('http://', '')
    
    stories = []
    stories.append(f"📖 **The Story of {domain_clean}**")
    stories.append(f"First archived in **{report.first_snapshot}**, this website has been captured **{report.total_snapshots}** times by the Wayback Machine.")
    
    if report.consistency_score >= 80:
        stories.append(f"🏛️ This is a **remarkably stable** website with a {report.consistency_score}% consistency score.")
    elif report.consistency_score >= 60:
        stories.append(f"🔄 With a {report.consistency_score}% consistency score, this website has evolved **steadily** over time.")
    elif report.consistency_score >= 40:
        stories.append(f"🌊 This website has seen **significant transformation** (consistency: {report.consistency_score}%).")
    else:
        stories.append(f"🎭 A website of **many faces** (consistency: {report.consistency_score}%).")
    
    if report.niche_changed:
        stories.append(f"⚠️ **Plot twist!** The website **changed its niche**. Early: *{report.top_keywords_first}*. Later: *{report.top_keywords_last}*.")
    else:
        stories.append(f"🎯 Throughout its history, the website stayed **true to its mission**. Themes: *{report.top_keywords_first}*.")
    
    if report.change_timeline:
        max_change = max(report.change_timeline, key=lambda x: x['change_percent'])
        if max_change['change_percent'] > 50:
            stories.append(f"📅 Biggest change: **{max_change['from']}** → **{max_change['to']}** ({max_change['change_percent']}% changed).")
    
    stories.append(f"🏁 Last captured in **{report.last_snapshot}**.")
    
    return "\n\n".join(stories)


def get_wayback_url(domain: str) -> str:
    domain_clean = domain.replace('https://', '').replace('http://', '')
    return f"https://web.archive.org/web/*/{domain_clean}"


def get_latest_snapshot_url(domain: str) -> str:
    domain_clean = domain.replace('https://', '').replace('http://', '')
    return f"https://web.archive.org/web/{domain_clean}"


def analyze_single_domain(session_id: str, domain: str, interval: int):
    """Analyze a single domain with progress tracking"""
    domain_clean = domain.strip()
    domain_id = domain_clean.replace('.', '_').replace('/', '_')
    
    # Initialize progress for this domain
    sessions[session_id]['domain_progress'][domain_id] = {
        'domain': domain_clean,
        'snapshots_checked': 0,
        'total_snapshots': 0,
        'step': 'starting',
        'percent': 0
    }
    sessions[session_id]['active_domains'].append(domain_clean)
    sessions[session_id]['logs'].append(f"🔍 Starting: {domain_clean}")
    
    checker = ProgressTrackingChecker(
        session_id=session_id,
        domain_id=domain_id,
        interval_months=interval,
        max_workers=1,
        quiet=True,
        retries=3
    )
    
    result = checker.analyze_domain(domain_clean)
    
    # Remove from active domains
    if domain_clean in sessions[session_id]['active_domains']:
        sessions[session_id]['active_domains'].remove(domain_clean)
    
    return result


def analyze_domains_parallel(session_id: str, domains: list, interval: int, max_workers: int = 3):
    """Run analysis in parallel"""
    session = sessions[session_id]
    session['logs'].append(f"🚀 Processing {len(domains)} domains with {max_workers} parallel workers...")
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_domain = {
            executor.submit(analyze_single_domain, session_id, domain, interval): domain
            for domain in domains
        }
        
        # Process results as they complete
        for future in as_completed(future_to_domain):
            domain = future_to_domain[future]
            domain_clean = domain.strip()
            
            try:
                result = future.result()
                
                if isinstance(result, FailedDomain):
                    session['results'].append({
                        'domain': result.domain,
                        'status': 'failed',
                        'error': result.error,
                        'wayback_url': get_wayback_url(result.domain)
                    })
                    session['logs'].append(f"❌ Failed: {domain_clean} - {result.error[:50]}")
                    session['failed'] += 1
                elif isinstance(result, DomainReport):
                    report_dict = {
                        'domain': result.domain,
                        'status': 'success',
                        'total_snapshots': result.total_snapshots,
                        'analyzed_snapshots': result.analyzed_snapshots,
                        'first_snapshot': result.first_snapshot,
                        'last_snapshot': result.last_snapshot,
                        'consistency_score': result.consistency_score,
                        'avg_change_percent': result.avg_change_percent,
                        'max_change_percent': result.max_change_percent,
                        'stability_rating': result.stability_rating,
                        'niche_changed': result.niche_changed,
                        'top_keywords_first': result.top_keywords_first,
                        'top_keywords_last': result.top_keywords_last,
                        'change_timeline': result.change_timeline,
                        'story': generate_story(result),
                        'wayback_url': get_wayback_url(result.domain),
                        'latest_snapshot_url': get_latest_snapshot_url(result.domain)
                    }
                    session['results'].append(report_dict)
                    session['logs'].append(f"✅ Done: {domain_clean} → {result.consistency_score}% ({result.analyzed_snapshots} snapshots)")
                    session['successful'] += 1
                    
            except Exception as e:
                session['results'].append({
                    'domain': domain_clean,
                    'status': 'failed',
                    'error': str(e),
                    'wayback_url': get_wayback_url(domain_clean)
                })
                session['logs'].append(f"❌ Error: {domain_clean} - {str(e)[:50]}")
                session['failed'] += 1
            
            # Update overall progress
            completed = session['successful'] + session['failed']
            session['progress'] = int((completed / len(domains)) * 100)
    
    session['progress'] = 100
    session['status'] = 'complete'
    session['logs'].append(f"🎉 Complete! {session['successful']} successful, {session['failed']} failed")


def gmb_analyze_domains_thread(session_id: str, domains: list, max_workers: int = 5):
    """Run GMB analysis in background thread - streams results as they complete"""
    session = sessions[session_id]
    session['logs'].append(f"🏢 Starting GMB check for {len(domains)} domains with {max_workers} workers...")
    
    async def check_single_domain(browser, domain: str, semaphore):
        """Check a single domain and return result immediately"""
        from gmb_checker_fast import check_gmb_listing, GMBResult, FailedGMB
        
        async with semaphore:
            session['current_domain'] = domain
            
            context = await browser.new_context(
                viewport={'width': 1280, 'height': 800},
                user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
            )
            page = await context.new_page()
            
            try:
                result = await check_gmb_listing(page, domain)
                return result
            except Exception as e:
                return FailedGMB(domain=domain, error=str(e))
            finally:
                await context.close()
    
    def process_result(result, captcha_count_ref):
        """Process a single result and update session immediately"""
        from gmb_checker_fast import GMBResult, FailedGMB
        
        if isinstance(result, Exception):
            session['results'].append({
                'domain': 'unknown',
                'has_gmb': False,
                'error': str(result)
            })
            session['failed'] += 1
            session['logs'].append(f"❌ Error: {str(result)[:40]}")
        elif isinstance(result, FailedGMB):
            if result.captcha_detected:
                captcha_count_ref[0] += 1
            session['results'].append({
                'domain': result.domain,
                'has_gmb': False,
                'error': result.error,
                'captcha_detected': result.captcha_detected
            })
            session['failed'] += 1
            if result.captcha_detected:
                session['logs'].append(f"⚠️ {result.domain}: CAPTCHA detected")
            else:
                session['logs'].append(f"❌ {result.domain}: {result.error[:40]}")
        else:
            # GMBResult
            if result.captcha_detected:
                captcha_count_ref[0] += 1
            
            result_dict = {
                'domain': result.domain,
                'has_gmb': result.has_gmb,
                'business_name': result.business_name,
                'address': result.address,
                'phone': result.phone,
                'rating': result.rating,
                'reviews_count': result.reviews_count,
                'website_url': result.website_url,
                'gmb_url': result.gmb_url,
                'category': result.category,
                'confidence': result.confidence,
                'captcha_detected': result.captcha_detected
            }
            session['results'].append(result_dict)
            
            if result.has_gmb:
                session['successful'] += 1
                biz_name = result.business_name[:30] if result.business_name else 'Unknown'
                session['logs'].append(f"✅ {result.domain}: Has GMB ({biz_name})")
            else:
                session['failed'] += 1
                if result.captcha_detected:
                    session['logs'].append(f"⚠️ {result.domain}: CAPTCHA detected")
                else:
                    session['logs'].append(f"❌ {result.domain}: No GMB found")
        
        # Update progress after each result
        completed = session['successful'] + session['failed']
        session['progress'] = int((completed / len(domains)) * 100)
        session['captcha_count'] = captcha_count_ref[0]
    
    async def run_gmb_check():
        from playwright.async_api import async_playwright
        
        captcha_count = [0]  # Use list for mutable reference
        semaphore = asyncio.Semaphore(max_workers)
        
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            
            try:
                # Create all tasks
                tasks = [
                    asyncio.create_task(check_single_domain(browser, domain, semaphore))
                    for domain in domains
                ]
                
                # Process results as they complete (not in order, but immediately)
                for coro in asyncio.as_completed(tasks):
                    try:
                        result = await coro
                        process_result(result, captcha_count)
                    except Exception as e:
                        session['failed'] += 1
                        session['logs'].append(f"❌ Task error: {str(e)[:40]}")
                        completed = session['successful'] + session['failed']
                        session['progress'] = int((completed / len(domains)) * 100)
                        
            finally:
                await browser.close()
        
        session['progress'] = 100
        session['status'] = 'complete'
        gmb_count = session['successful']
        session['logs'].append(f"🎉 Complete! {gmb_count} with GMB, {session['failed']} without")
        if captcha_count[0] > 0:
            session['logs'].append(f"⚠️ CAPTCHA detected {captcha_count[0]} times")
    
    # Run async function in thread
    asyncio.run(run_gmb_check())


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/playwright-status')
def playwright_status():
    """Check if Playwright is available"""
    return jsonify({
        'available': GMB_AVAILABLE,
        'message': 'GMB checker ready' if GMB_AVAILABLE else 'Playwright not installed'
    })


@app.route('/api/analyze', methods=['POST'])
def start_analysis():
    data = request.json
    domains = data.get('domains', [])
    interval = data.get('interval', 3)
    workers = min(data.get('workers', 3), 5)  # Max 5 parallel workers
    mode = data.get('mode', 'wayback')
    gmb_parallel = min(data.get('gmb_parallel', 5), GMB_MAX_WORKERS)
    
    if not domains:
        return jsonify({'error': 'No domains provided'}), 400
    
    session_id = str(uuid.uuid4())[:8]
    
    sessions[session_id] = {
        'status': 'running',
        'progress': 0,
        'total': len(domains),
        'successful': 0,
        'failed': 0,
        'results': [],
        'logs': [f"🚀 Starting {mode.upper()} analysis of {len(domains)} domains..."],
        'domain_progress': {},
        'active_domains': [],
        'mode': mode,
        'captcha_count': 0,
        'current_domain': ''
    }
    
    if mode == 'wayback':
        sessions[session_id]['logs'].append(f"📊 Using {workers} parallel workers")
        thread = threading.Thread(
            target=analyze_domains_parallel,
            args=(session_id, domains, interval, workers)
        )
    elif mode == 'gmb':
        if not GMB_AVAILABLE:
            return jsonify({'error': 'GMB checker not available. Install playwright.'}), 400
        sessions[session_id]['logs'].append(f"🏢 Using {gmb_parallel} parallel workers for GMB")
        thread = threading.Thread(
            target=gmb_analyze_domains_thread,
            args=(session_id, domains, gmb_parallel)
        )
    else:
        return jsonify({'error': 'Invalid mode'}), 400
    
    thread.daemon = True
    thread.start()
    
    return jsonify({
        'session_id': session_id,
        'domains_count': len(domains),
        'mode': mode,
        'workers': workers if mode == 'wayback' else gmb_parallel
    })


@app.route('/api/status/<session_id>')
def get_status(session_id):
    if session_id not in sessions:
        return jsonify({'error': 'Session not found'}), 404
    
    session = sessions[session_id]
    return jsonify({
        'status': session['status'],
        'progress': session['progress'],
        'total': session['total'],
        'successful': session['successful'],
        'failed': session['failed'],
        'results': session['results'],
        'logs': session['logs'][-50:],
        'domain_progress': session.get('domain_progress', {}),
        'active_domains': session.get('active_domains', []),
        'mode': session.get('mode', 'wayback'),
        'captcha_count': session.get('captcha_count', 0),
        'current_domain': session.get('current_domain', '')
    })


if __name__ == '__main__':
    print("Starting Domain Tools UI...")
    print(f"GMB Checker: {'✅ Available' if GMB_AVAILABLE else '❌ Not available (install playwright)'}")
    print("Open http://localhost:5001 in your browser")
    app.run(debug=False, port=5001, threaded=True)
