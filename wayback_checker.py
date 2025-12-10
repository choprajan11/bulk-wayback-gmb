#!/usr/bin/env python3
"""
Wayback Machine Website Consistency Checker
Analyzes website changes over time using archive.org snapshots
"""

import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
from collections import defaultdict
import difflib
import re
import json
import csv
import time
import argparse
import threading
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, as_completed


@dataclass
class Snapshot:
    """Represents a Wayback Machine snapshot"""
    timestamp: str
    url: str
    date: datetime
    text_content: str = ""
    title: str = ""
    
    @property
    def formatted_date(self) -> str:
        return self.date.strftime("%B %Y")


@dataclass
class DomainReport:
    """Report for a single domain's analysis"""
    domain: str
    total_snapshots: int
    analyzed_snapshots: int
    first_snapshot: Optional[str]
    last_snapshot: Optional[str]
    consistency_score: float  # 0-100, higher = more consistent
    niche_changed: bool
    change_timeline: List[Dict]
    summary: str
    avg_change_percent: float = 0.0
    max_change_percent: float = 0.0
    stability_rating: str = ""
    top_keywords_first: str = ""
    top_keywords_last: str = ""
    status: str = "success"
    error: str = ""


@dataclass
class FailedDomain:
    """Represents a failed domain analysis"""
    domain: str
    error: str
    status: str = "failed"


class WaybackChecker:
    """Main class for checking website consistency via Wayback Machine"""
    
    CDX_API_URL = "http://web.archive.org/cdx/search/cdx"
    WAYBACK_URL = "http://web.archive.org/web"
    
    def __init__(self, interval_months: int = 3, max_workers: int = 3, quiet: bool = False, retries: int = 3):
        self.interval_months = interval_months
        self.max_workers = max_workers
        self.quiet = quiet
        self.retries = retries
        self.print_lock = threading.Lock()
        # Create thread-local storage for sessions
        self._local = threading.local()
    
    def _get_session(self) -> requests.Session:
        """Get a thread-local session"""
        if not hasattr(self._local, 'session'):
            self._local.session = requests.Session()
            self._local.session.headers.update({
                'User-Agent': 'WaybackConsistencyChecker/1.0 (Research Tool)'
            })
        return self._local.session
    
    def _print(self, message: str, force: bool = False):
        """Thread-safe print"""
        if not self.quiet or force:
            with self.print_lock:
                print(message)
    
    def _request_with_retry(self, url: str, params: Optional[Dict] = None, timeout: int = 60) -> Optional[requests.Response]:
        """Make HTTP request with retry logic and exponential backoff"""
        session = self._get_session()
        last_exception = None
        
        for attempt in range(self.retries):
            try:
                response = session.get(url, params=params, timeout=timeout)
                response.raise_for_status()
                return response
            except requests.exceptions.Timeout as e:
                last_exception = e
                wait_time = (2 ** attempt) + 1  # Exponential backoff: 2, 3, 5 seconds
                self._print(f"    ⏳ Timeout (attempt {attempt + 1}/{self.retries}), retrying in {wait_time}s...")
                time.sleep(wait_time)
            except requests.exceptions.ConnectionError as e:
                last_exception = e
                wait_time = (2 ** attempt) + 1
                self._print(f"    🔌 Connection error (attempt {attempt + 1}/{self.retries}), retrying in {wait_time}s...")
                time.sleep(wait_time)
            except requests.exceptions.HTTPError as e:
                # Don't retry on 4xx errors (client errors), only on 5xx (server errors)
                if e.response is not None and 400 <= e.response.status_code < 500:
                    raise  # Don't retry client errors
                last_exception = e
                wait_time = (2 ** attempt) + 1
                self._print(f"    ⚠ HTTP error {e.response.status_code if e.response else 'unknown'} (attempt {attempt + 1}/{self.retries}), retrying in {wait_time}s...")
                time.sleep(wait_time)
            except Exception as e:
                last_exception = e
                wait_time = (2 ** attempt) + 1
                self._print(f"    ❌ Error (attempt {attempt + 1}/{self.retries}): {e}, retrying in {wait_time}s...")
                time.sleep(wait_time)
        
        # All retries exhausted
        if last_exception:
            raise last_exception
        return None
    
    def get_available_snapshots(self, domain: str) -> List[Dict]:
        """Fetch all available snapshots for a domain from CDX API"""
        params = {
            'url': domain,
            'output': 'json',
            'fl': 'timestamp,original,statuscode',
            'filter': 'statuscode:200',
            'collapse': 'timestamp:6'  # One per month
        }
        
        try:
            response = self._request_with_retry(self.CDX_API_URL, params=params, timeout=60)
            if response is None:
                return []
            
            data = response.json()
            
            if len(data) <= 1:  # First row is headers
                return []
            
            headers = data[0]
            snapshots = []
            for row in data[1:]:
                snapshot = dict(zip(headers, row))
                snapshots.append(snapshot)
            
            return snapshots
        except Exception as e:
            self._print(f"  ⚠ Error fetching snapshots for {domain} after {self.retries} retries: {e}", force=True)
            return []
    
    def select_snapshots_by_interval(self, snapshots: List[Dict]) -> List[Dict]:
        """Select snapshots approximately every N months"""
        if not snapshots:
            return []
        
        selected = []
        last_selected_date = None
        interval_days = self.interval_months * 30
        
        for snapshot in snapshots:
            timestamp = snapshot['timestamp']
            date = datetime.strptime(timestamp[:8], '%Y%m%d')
            
            if last_selected_date is None:
                selected.append(snapshot)
                last_selected_date = date
            elif (date - last_selected_date).days >= interval_days:
                selected.append(snapshot)
                last_selected_date = date
        
        # Always include the last snapshot if not already included
        if snapshots and (not selected or selected[-1] != snapshots[-1]):
            last_snapshot = snapshots[-1]
            last_date = datetime.strptime(last_snapshot['timestamp'][:8], '%Y%m%d')
            if last_selected_date and (last_date - last_selected_date).days >= 30:
                selected.append(last_snapshot)
        
        return selected
    
    def fetch_snapshot_content(self, domain: str, timestamp: str) -> Optional[Snapshot]:
        """Fetch content from a specific Wayback Machine snapshot"""
        url = f"{self.WAYBACK_URL}/{timestamp}/{domain}"
        
        try:
            response = self._request_with_retry(url, timeout=60)
            if response is None:
                return None
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Remove script, style, and Wayback Machine toolbar elements
            for element in soup(['script', 'style', 'noscript', 'iframe']):
                element.decompose()
            
            # Remove Wayback Machine specific elements
            for wb_element in soup.find_all(id=re.compile(r'wm-.*')):
                wb_element.decompose()
            
            # Extract text content
            text = soup.get_text(separator=' ', strip=True)
            # Clean up whitespace
            text = re.sub(r'\s+', ' ', text).strip()
            
            # Extract title
            title = ""
            title_tag = soup.find('title')
            if title_tag:
                title = title_tag.get_text(strip=True)
            
            date = datetime.strptime(timestamp[:8], '%Y%m%d')
            
            return Snapshot(
                timestamp=timestamp,
                url=url,
                date=date,
                text_content=text[:50000],  # Limit content size
                title=title
            )
        except Exception as e:
            self._print(f"    ⚠ Error fetching {timestamp} after {self.retries} retries: {e}")
            return None
    
    def calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity ratio between two texts (0-1)"""
        if not text1 or not text2:
            return 0.0
        
        # Use SequenceMatcher for similarity
        matcher = difflib.SequenceMatcher(None, text1[:10000], text2[:10000])
        return matcher.ratio()
    
    def extract_keywords(self, text: str, top_n: int = 20) -> List[str]:
        """Extract top keywords from text for niche detection"""
        # Common stop words to ignore
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'up', 'about', 'into', 'through', 'during',
            'before', 'after', 'above', 'below', 'between', 'under', 'again',
            'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why',
            'how', 'all', 'each', 'few', 'more', 'most', 'other', 'some', 'such',
            'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very',
            'can', 'will', 'just', 'should', 'now', 'this', 'that', 'these', 'those',
            'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has',
            'had', 'having', 'do', 'does', 'did', 'doing', 'would', 'could', 'might',
            'must', 'shall', 'may', 'need', 'dare', 'ought', 'used', 'get', 'got',
            'your', 'our', 'their', 'its', 'my', 'his', 'her', 'we', 'you', 'they',
            'it', 'i', 'me', 'him', 'us', 'them', 'what', 'which', 'who', 'whom',
            'web', 'archive', 'wayback', 'machine', 'page', 'site', 'click', 'home',
            'contact', 'copyright', 'rights', 'reserved', 'privacy', 'policy', 'terms'
        }
        
        # Tokenize and filter
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        word_freq = defaultdict(int)
        
        for word in words:
            if word not in stop_words:
                word_freq[word] += 1
        
        # Sort by frequency and return top N
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        return [word for word, _ in sorted_words[:top_n]]
    
    def detect_niche_change(self, snapshots: List[Snapshot]) -> Tuple[bool, List[Dict]]:
        """Detect if the website's niche/business changed over time"""
        if len(snapshots) < 2:
            return False, []
        
        changes = []
        first_keywords = set(self.extract_keywords(snapshots[0].text_content))
        prev_keywords = first_keywords
        
        for i in range(1, len(snapshots)):
            current_keywords = set(self.extract_keywords(snapshots[i].text_content))
            
            # Calculate keyword overlap
            if prev_keywords and current_keywords:
                overlap = len(prev_keywords & current_keywords) / max(len(prev_keywords), len(current_keywords))
                new_keywords = current_keywords - prev_keywords
                lost_keywords = prev_keywords - current_keywords
                
                changes.append({
                    'date': snapshots[i].formatted_date,
                    'overlap_with_previous': round(overlap * 100, 1),
                    'new_keywords': list(new_keywords)[:5],
                    'lost_keywords': list(lost_keywords)[:5]
                })
            
            prev_keywords = current_keywords
        
        # Check overall keyword overlap between first and last
        last_keywords = set(self.extract_keywords(snapshots[-1].text_content))
        overall_overlap = 0
        if first_keywords and last_keywords:
            overall_overlap = len(first_keywords & last_keywords) / max(len(first_keywords), len(last_keywords))
        
        # If less than 30% keyword overlap, consider it a niche change
        niche_changed = overall_overlap < 0.3
        
        return niche_changed, changes
    
    def analyze_domain(self, domain: str) -> DomainReport | FailedDomain:
        """Perform complete analysis for a single domain"""
        original_domain = domain
        self._print(f"\n{'='*60}")
        self._print(f"🔍 Analyzing: {domain}")
        self._print(f"{'='*60}")
        
        # Normalize domain
        if not domain.startswith(('http://', 'https://')):
            domain = f"https://{domain}"
        
        # Get available snapshots
        self._print("  📥 Fetching available snapshots...")
        try:
            all_snapshots = self.get_available_snapshots(domain)
        except Exception as e:
            error_msg = f"Failed to fetch snapshots: {str(e)}"
            self._print(f"  ❌ {error_msg}", force=True)
            return FailedDomain(domain=original_domain, error=error_msg)
        
        if not all_snapshots:
            error_msg = "No archived snapshots found in Wayback Machine"
            self._print(f"  ❌ {error_msg} for {original_domain}", force=True)
            return FailedDomain(domain=original_domain, error=error_msg)
        
        self._print(f"  ✓ Found {len(all_snapshots)} monthly snapshots")
        
        # Select snapshots by interval
        selected = self.select_snapshots_by_interval(all_snapshots)
        self._print(f"  ✓ Selected {len(selected)} snapshots (~{self.interval_months} month intervals)")
        
        # Fetch content for selected snapshots
        self._print("  📄 Fetching snapshot contents...")
        snapshots_data: List[Snapshot] = []
        
        for i, snap in enumerate(selected):
            self._print(f"    [{i+1}/{len(selected)}] Fetching {snap['timestamp'][:6]}...")
            snapshot = self.fetch_snapshot_content(domain, snap['timestamp'])
            if snapshot:
                snapshots_data.append(snapshot)
            time.sleep(0.5)  # Be nice to Wayback Machine
        
        if not snapshots_data:
            error_msg = "Could not fetch any snapshot content (all requests failed)"
            self._print(f"  ❌ {error_msg} for {original_domain}", force=True)
            return FailedDomain(domain=original_domain, error=error_msg)
        
        self._print(f"  ✓ Successfully fetched {len(snapshots_data)} snapshots")
        
        # Calculate changes between consecutive snapshots
        self._print("  📊 Analyzing content changes...")
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
        niche_changed, keyword_changes = self.detect_niche_change(snapshots_data)
        
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
        
        # Generate summary
        first_date = snapshots_data[0].formatted_date
        last_date = snapshots_data[-1].formatted_date
        
        if niche_changed:
            niche_status = "⚠️ NICHE/BUSINESS LIKELY CHANGED"
        else:
            niche_status = "✓ Niche/business remained consistent"
        
        if consistency_score >= 80:
            stability_rating = "Very Stable"
        elif consistency_score >= 60:
            stability_rating = "Moderately Stable"
        elif consistency_score >= 40:
            stability_rating = "Significant Changes"
        else:
            stability_rating = "Major Overhauls"
        
        summary = f"""
Domain: {domain}
Period: {first_date} → {last_date}
Consistency Score: {consistency_score}% ({stability_rating})
Niche Status: {niche_status}
Snapshots Analyzed: {len(snapshots_data)} out of {len(all_snapshots)} available
"""
        
        # Get top keywords for first and last snapshot
        first_keywords = self.extract_keywords(snapshots_data[0].text_content, top_n=10)
        last_keywords = self.extract_keywords(snapshots_data[-1].text_content, top_n=10)
        
        self._print(f"  ✓ Analysis complete for {original_domain}", force=True)
        
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
    
    def print_report(self, report: DomainReport):
        """Print a detailed report for a domain"""
        with self.print_lock:
            print(f"\n{'='*60}")
            print(f"📋 REPORT: {report.domain}")
            print(f"{'='*60}")
            print(report.summary)
            
            if report.change_timeline and not self.quiet:
                print("\n📅 Change Timeline:")
                print("-" * 50)
                for change in report.change_timeline:
                    change_indicator = "🟢" if change['similarity'] >= 80 else "🟡" if change['similarity'] >= 50 else "🔴"
                    print(f"  {change_indicator} {change['from']} → {change['to']}")
                    print(f"     Similarity: {change['similarity']}% (Changed: {change['change_percent']}%)")
                    if change['title_from'] != change['title_to']:
                        print(f"     Title: \"{change['title_from']}\" → \"{change['title_to']}\"")
                
            print(f"\n🏁 Final Snapshot: {report.last_snapshot}")
            print(f"{'='*60}\n")
    
    def analyze_multiple_domains(self, domains: List[str], parallel: bool = False) -> Tuple[List[DomainReport], List[FailedDomain]]:
        """Analyze multiple domains and return reports
        
        Args:
            domains: List of domain names to analyze
            parallel: If True, analyze domains in parallel using thread pool
            
        Returns:
            Tuple of (successful_reports, failed_domains)
        """
        reports: List[DomainReport] = []
        failed: List[FailedDomain] = []
        
        print(f"\n🚀 Starting analysis of {len(domains)} domain(s)")
        print(f"   Interval: ~{self.interval_months} months between snapshots")
        print(f"   Retries: {self.retries} (with exponential backoff)")
        if parallel:
            print(f"   Mode: PARALLEL (max {self.max_workers} concurrent)")
        else:
            print(f"   Mode: Sequential")
        print()
        
        if parallel and len(domains) > 1:
            # Parallel processing
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # Submit all tasks
                future_to_domain = {
                    executor.submit(self.analyze_domain, domain.strip()): domain 
                    for domain in domains
                }
                
                # Collect results as they complete
                for future in as_completed(future_to_domain):
                    domain = future_to_domain[future]
                    try:
                        result = future.result()
                        if isinstance(result, FailedDomain):
                            failed.append(result)
                        elif isinstance(result, DomainReport):
                            reports.append(result)
                            if not self.quiet:
                                self.print_report(result)
                    except Exception as e:
                        error_msg = f"Unexpected error: {str(e)}"
                        print(f"  ❌ {error_msg} for {domain}")
                        failed.append(FailedDomain(domain=domain, error=error_msg))
        else:
            # Sequential processing
            for domain in domains:
                result = self.analyze_domain(domain.strip())
                if isinstance(result, FailedDomain):
                    failed.append(result)
                elif isinstance(result, DomainReport):
                    reports.append(result)
                    if not self.quiet:
                        self.print_report(result)
        
        # Sort reports by domain for consistent output
        reports.sort(key=lambda r: r.domain)
        failed.sort(key=lambda r: r.domain)
        
        # Print summary
        print("\n" + "="*70)
        print("📊 SUMMARY OF ALL DOMAINS")
        print("="*70)
        
        if reports:
            print(f"\n✅ SUCCESSFUL ({len(reports)}):")
            print(f"{'Domain':<35} {'Score':<10} {'Niche':<10} {'Stability':<18} {'Last Seen':<15}")
            print("-"*88)
            for r in reports:
                niche_str = "Changed" if r.niche_changed else "Same"
                domain_short = r.domain.replace('https://', '').replace('http://', '')[:33]
                print(f"{domain_short:<35} {r.consistency_score:<10} {niche_str:<10} {r.stability_rating:<18} {r.last_snapshot:<15}")
        
        if failed:
            print(f"\n❌ FAILED ({len(failed)}):")
            print(f"{'Domain':<35} {'Error':<50}")
            print("-"*85)
            for f in failed:
                domain_short = f.domain.replace('https://', '').replace('http://', '')[:33]
                error_short = f.error[:48] if len(f.error) > 48 else f.error
                print(f"{domain_short:<35} {error_short:<50}")
        
        print(f"\n📈 Total: {len(reports)} successful, {len(failed)} failed out of {len(domains)} domains")
        
        return reports, failed
    
    def save_csv(self, reports: List[DomainReport], failed: List[FailedDomain], filepath: str):
        """Save a concise CSV summary of all domain reports including failures"""
        if not reports and not failed:
            return
        
        fieldnames = [
            'domain',
            'status',
            'first_snapshot',
            'last_snapshot',
            'total_snapshots',
            'analyzed_snapshots',
            'consistency_score',
            'avg_change_percent',
            'max_change_percent',
            'stability_rating',
            'niche_changed',
            'top_keywords_first',
            'top_keywords_last',
            'error'
        ]
        
        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            # Write successful reports
            for r in reports:
                writer.writerow({
                    'domain': r.domain.replace('https://', '').replace('http://', ''),
                    'status': 'success',
                    'first_snapshot': r.first_snapshot,
                    'last_snapshot': r.last_snapshot,
                    'total_snapshots': r.total_snapshots,
                    'analyzed_snapshots': r.analyzed_snapshots,
                    'consistency_score': r.consistency_score,
                    'avg_change_percent': r.avg_change_percent,
                    'max_change_percent': r.max_change_percent,
                    'stability_rating': r.stability_rating,
                    'niche_changed': 'Yes' if r.niche_changed else 'No',
                    'top_keywords_first': r.top_keywords_first,
                    'top_keywords_last': r.top_keywords_last,
                    'error': ''
                })
            
            # Write failed domains
            for f_domain in failed:
                writer.writerow({
                    'domain': f_domain.domain.replace('https://', '').replace('http://', ''),
                    'status': 'failed',
                    'first_snapshot': '',
                    'last_snapshot': '',
                    'total_snapshots': '',
                    'analyzed_snapshots': '',
                    'consistency_score': '',
                    'avg_change_percent': '',
                    'max_change_percent': '',
                    'stability_rating': '',
                    'niche_changed': '',
                    'top_keywords_first': '',
                    'top_keywords_last': '',
                    'error': f_domain.error
                })
        
        print(f"✓ CSV summary saved to {filepath} ({len(reports)} success, {len(failed)} failed)")


def main():
    parser = argparse.ArgumentParser(
        description='Check website consistency using Wayback Machine archives',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python wayback_checker.py example.com
  python wayback_checker.py example.com another-site.com --interval 6
  python wayback_checker.py -f domains.txt --interval 3
  python wayback_checker.py -f domains.txt --parallel --csv results.csv
  python wayback_checker.py -f domains.txt -p -w 5 --csv results.csv --quiet
  
The script will analyze snapshots from the Wayback Machine and report:
  - How consistent the website content has been over time
  - Whether the niche/business appears to have changed
  - A timeline of major changes
  - The last available snapshot date

Output formats:
  --output/-o : Full JSON with change timeline details
  --csv/-c    : Concise CSV summary (no timeline, faster to read)
        """
    )
    
    parser.add_argument(
        'domains',
        nargs='*',
        help='Domain(s) to analyze (e.g., example.com)'
    )
    
    parser.add_argument(
        '-f', '--file',
        help='File containing list of domains (one per line)'
    )
    
    parser.add_argument(
        '-i', '--interval',
        type=int,
        default=3,
        help='Interval in months between snapshots to analyze (default: 3)'
    )
    
    parser.add_argument(
        '-o', '--output',
        help='Output JSON file for full results (includes change timeline)'
    )
    
    parser.add_argument(
        '-c', '--csv',
        help='Output CSV file for concise summary (no timeline, easier to read)'
    )
    
    parser.add_argument(
        '-p', '--parallel',
        action='store_true',
        help='Analyze multiple domains in parallel (faster for many domains)'
    )
    
    parser.add_argument(
        '-w', '--workers',
        type=int,
        default=3,
        help='Number of parallel workers when using --parallel (default: 3, max recommended: 5)'
    )
    
    parser.add_argument(
        '-q', '--quiet',
        action='store_true',
        help='Quiet mode - minimal output, suppress detailed timeline in console'
    )
    
    parser.add_argument(
        '-r', '--retries',
        type=int,
        default=3,
        help='Number of retries for failed requests (default: 3)'
    )
    
    args = parser.parse_args()
    
    # Collect domains
    domains = list(args.domains) if args.domains else []
    
    if args.file:
        try:
            with open(args.file, 'r') as f:
                file_domains = [line.strip() for line in f if line.strip() and not line.startswith('#')]
                domains.extend(file_domains)
        except FileNotFoundError:
            print(f"❌ Error: File '{args.file}' not found")
            return
    
    if not domains:
        parser.print_help()
        print("\n❌ Error: Please provide at least one domain to analyze")
        return
    
    # Cap workers at 5 to be nice to Wayback Machine
    max_workers = min(args.workers, 5)
    
    # Run analysis
    checker = WaybackChecker(
        interval_months=args.interval,
        max_workers=max_workers,
        quiet=args.quiet,
        retries=args.retries
    )
    reports, failed = checker.analyze_multiple_domains(domains, parallel=args.parallel)
    
    # Save to CSV if requested (concise summary including failures)
    if args.csv:
        checker.save_csv(reports, failed, args.csv)
    
    # Save to JSON if requested (full details including failures)
    if args.output:
        output_data = {
            'summary': {
                'total_domains': len(domains),
                'successful': len(reports),
                'failed': len(failed)
            },
            'successful': [],
            'failed': []
        }
        
        for r in reports:
            output_data['successful'].append({
                'domain': r.domain,
                'status': 'success',
                'total_snapshots': r.total_snapshots,
                'analyzed_snapshots': r.analyzed_snapshots,
                'first_snapshot': r.first_snapshot,
                'last_snapshot': r.last_snapshot,
                'consistency_score': r.consistency_score,
                'avg_change_percent': r.avg_change_percent,
                'max_change_percent': r.max_change_percent,
                'stability_rating': r.stability_rating,
                'niche_changed': r.niche_changed,
                'top_keywords_first': r.top_keywords_first,
                'top_keywords_last': r.top_keywords_last,
                'change_timeline': r.change_timeline
            })
        
        for f in failed:
            output_data['failed'].append({
                'domain': f.domain,
                'status': 'failed',
                'error': f.error
            })
        
        with open(args.output, 'w') as f_out:
            json.dump(output_data, f_out, indent=2)
        print(f"✓ Full JSON results saved to {args.output} ({len(reports)} success, {len(failed)} failed)")


if __name__ == '__main__':
    main()

