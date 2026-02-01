import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta, timezone
import nltk
from nltk.tokenize import sent_tokenize
import time
import sys
import re
from selenium import webdriver
from selenium.webdriver.edge.options import Options
from selenium.webdriver.edge.service import Service
from webdriver_manager.microsoft import EdgeChromiumDriverManager
import json

# Optional: transformer summarizer
try:
    from transformers import pipeline
    summarizer = pipeline("summarization", model="google/pegasus-cnn_dailymail")
except Exception as e:
    summarizer = None

# Download required NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
except:
    pass

# ------------------------
# Config
# ------------------------

sites = [
    ("Coinpedia", "https://coinpedia.org/news/"),
    ("CoinDesk", "https://www.coindesk.com/latest-crypto-news"),
    ("Crypto.News", "https://crypto.news/news/"),
    ("The Block", "https://www.theblock.co/latest-crypto-news"),
    ("Decrypt", "https://decrypt.co/news"),
    ("Cointelegraph", "https://cointelegraph.com/news")
]

# Get articles from the last 48 hours
days_back = 2
cutoff_date = datetime.now(timezone.utc) - timedelta(days=days_back)

# ------------------------
# Selenium Setup (Edge)
# ------------------------

def create_driver():
    """Create a Selenium Edge driver with proper options"""
    edge_options = Options()
    edge_options.add_argument('--headless')
    edge_options.add_argument('--no-sandbox')
    edge_options.add_argument('--disable-dev-shm-usage')
    edge_options.add_argument('--disable-gpu')
    edge_options.add_argument('--window-size=1920,1080')
    edge_options.add_argument('--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 Edg/120.0.0.0')
    edge_options.add_argument('--disable-blink-features=AutomationControlled')
    edge_options.add_experimental_option("excludeSwitches", ["enable-automation"])
    edge_options.add_experimental_option('useAutomationExtension', False)
    
    try:
        # Try without webdriver-manager first (use system Edge driver)
        driver = webdriver.Edge(options=edge_options)
        return driver
    except:
        # Fallback: try with webdriver-manager
        try:
            service = Service(EdgeChromiumDriverManager().install())
            driver = webdriver.Edge(service=service, options=edge_options)
            return driver
        except Exception as e:
            print(f"Error creating Edge driver: {e}")
            return None

# ------------------------
# Helpers
# ------------------------

def fetch_html_selenium(url, driver, wait_time=5):
    """Fetch HTML using Selenium for JavaScript-rendered pages"""
    try:
        driver.get(url)
        time.sleep(wait_time)
        return driver.page_source
    except Exception as e:
        print(f"        Error fetching with Selenium: {e}")
        return None

def fetch_html_requests(url, session=None, referer=None):
    """Fetch HTML using requests for simple pages"""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 Edg/120.0.0.0",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
        "Sec-Fetch-Dest": "document",
        "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-Site": "none" if not referer else "same-origin",
        "Cache-Control": "max-age=0",
    }
    if referer:
        headers["Referer"] = referer
    
    try:
        if session:
            r = session.get(url, headers=headers, timeout=15, allow_redirects=True)
        else:
            r = requests.get(url, headers=headers, timeout=15, allow_redirects=True)
        r.raise_for_status()
        return r.text
    except Exception as e:
        print(f"        Error fetching with requests: {e}")
        return None

def clean_text(text):
    """Remove illegal characters for Excel"""
    if not text:
        return ""
    # Remove control characters and other illegal XML characters
    import re
    # Remove control characters except tab, newline, carriage return
    text = re.sub(r'[\x00-\x08\x0B-\x0C\x0E-\x1F\x7F-\x9F]', '', text)
    # Remove any remaining problematic characters
    text = ''.join(char for char in text if ord(char) >= 32 or char in '\t\n\r')
    return text

def parse_date(date_str):
    if not date_str:
        return None
    
    if isinstance(date_str, datetime):
        if date_str.tzinfo is None:
            return date_str.replace(tzinfo=timezone.utc)
        return date_str
    
    date_str = str(date_str).strip()
    
    # Try ISO format first
    try:
        if 'T' in date_str:
            if date_str.endswith('Z'):
                date_str = date_str[:-1] + '+00:00'
            dt = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
    except:
        pass
    
    # Try common formats
    formats = [
        "%Y-%m-%d",
        "%b %d, %Y",
        "%d %b %Y",
        "%Y/%m/%d",
        "%m/%d/%Y",
        "%d-%m-%Y",
        "%Y-%m-%d %H:%M:%S",
        "%B %d, %Y",
        "%d %B %Y",
        "%b %d, %Y %H:%M %Z",
        "%b %d, %Y %H:%M UTC",
    ]
    
    for fmt in formats:
        try:
            dt = datetime.strptime(date_str, fmt)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        except:
            continue
    
    return None

def is_recent(date):
    if isinstance(date, datetime):
        if date.tzinfo is None:
            date = date.replace(tzinfo=timezone.utc)
        return date >= cutoff_date
    return True

def remove_boilerplate(text):
    """Remove boilerplate - PARAGRAPH-BASED APPROACH"""
    if not text:
        return text
    
    # Split by double newlines or periods followed by newlines
    import re
    paragraphs = re.split(r'\n\n+|\. \n', text)
    
    cleaned_paragraphs = []
    
    # Skip patterns
    skip_patterns = [
        'log in', 'sign up', 'journalist', 'author', 'founder',
        'data collected', 'personalization', 'advertising',
        'cookies', 'privacy', 'disclaimer', 'not financial advice',
    ]
    
    for paragraph in paragraphs:
        para_lower = paragraph.lower().strip()
        
        # Skip short paragraphs
        if len(paragraph.strip()) < 50:
            continue
        
        # Skip if contains boilerplate
        if any(pattern in para_lower for pattern in skip_patterns):
            continue
        
        # Keep this paragraph
        cleaned_paragraphs.append(paragraph.strip())
    
    # Join paragraphs and take first 1000 chars
    result = ' '.join(cleaned_paragraphs)
    
    # Split into sentences and take first 10
    sentences = result.split('. ')
    sentences = [s.strip() for s in sentences if len(s.strip()) > 30][:10]
    
    return '. '.join(sentences) + '.' if sentences else text

def summarize_text(text):
    text = remove_boilerplate(text)
    text = clean_text(text.strip()) if text else ""
    if not text:
        return "No content available"
    
    if summarizer:
        try:
            if len(text) > 100:
                res = summarizer(text[:4000], max_length=500, min_length=200, do_sample=False)
                return clean_text(res[0]['summary_text'])
        except:
            pass
    
    try:
        sentences = sent_tokenize(text)
        summary = " ".join(sentences[:10])
        return clean_text(summary)
    except:
        sentences = text.split('. ')
        summary = ". ".join(sentences[:10])
        if summary:
            return clean_text(summary + ".")
        return clean_text(text[:500] + "..." if len(text) > 500 else text)

# ------------------------
# Scraping functions
# ------------------------

def scrape_coinpedia(driver):
    url = sites[0][1]
    print(f"    Fetching page with Selenium...")
    
    html = fetch_html_selenium(url, driver, wait_time=5)
    if not html:
        print(f"    ✗ Failed to fetch page")
        return []
    
    print(f"    ✓ Fetched page: {len(html)} characters")
    soup = BeautifulSoup(html, "html.parser")
    results = []
    
    # Look for article containers
    articles = soup.find_all('article')
    if not articles:
        articles = soup.find_all('li', class_=lambda x: x and 'post' in str(x))
    if not articles:
        articles = soup.find_all('div', class_=lambda x: x and 'post' in str(x))
    
    print(f"    Found {len(articles)} potential article containers")
    
    if len(articles) == 0:
        # Fallback: look for h2 with links
        h2_tags = soup.find_all('h2')
        print(f"    Fallback: Found {len(h2_tags)} h2 tags")
        
        processed = 0
        for i, h2 in enumerate(h2_tags):
            if processed >= 10:
                break
                
            link_tag = h2.find('a', href=True)
            if not link_tag:
                continue
            
            link = link_tag.get('href', '')
            if not link or 'news' not in link:
                continue
            
            if not link.startswith('http'):
                link = 'https://coinpedia.org' + link
            
            title = link_tag.get_text(strip=True)
            if not title or len(title) < 10:
                continue
            
            print(f"      Article {processed+1}: {title[:50]}...")
            
            try:
                article_html = fetch_html_selenium(link, driver, wait_time=3)
                if not article_html:
                    print(f"        ✗ Failed to fetch article content")
                    continue
                
                article_soup = BeautifulSoup(article_html, "html.parser")
                
                paragraphs = article_soup.find_all("p")
                text = " ".join(p.get_text(" ", strip=True) for p in paragraphs if len(p.get_text(strip=True)) > 20)
                
                if len(text) < 100:
                    print(f"        ✗ Insufficient content ({len(text)} chars)")
                    continue
                
                date = None
                date_meta = article_soup.find("meta", {"property": "article:published_time"})
                if date_meta:
                    date = parse_date(date_meta.get("content"))
                
                if not date:
                    date_meta = article_soup.find("meta", {"property": "og:published_time"})
                    if date_meta:
                        date = parse_date(date_meta.get("content"))
                
                print(f"        Date: {date if date else 'Unknown'}")
                
                # For Coinpedia, accept top 10 articles even without date
                if date is None or is_recent(date):
                    results.append((title, "Coinpedia", text))
                    print(f"        ✓ ADDED")
                    processed += 1
                else:
                    print(f"        ✗ Too old")
                    
            except Exception as e:
                print(f"        Error: {e}")
                continue
        
        return results
    
    # Process found articles
    print(f"    Processing {min(len(articles), 10)} articles...")
    processed = 0
    
    for i, article in enumerate(articles):
        if processed >= 10:
            break
            
        h2 = article.find('h2')
        if not h2:
            h2 = article.find('h3')
        if not h2:
            continue
        
        link_tag = h2.find('a', href=True)
        if not link_tag:
            link_tag = article.find('a', href=True)
        
        if not link_tag:
            continue
        
        link = link_tag.get('href', '')
        if not link:
            continue
        
        if not link.startswith('http'):
            link = 'https://coinpedia.org' + link
        
        title = link_tag.get_text(strip=True)
        if not title or len(title) < 10:
            continue
        
        print(f"      Article {processed+1}: {title[:50]}...")
        
        try:
            article_html = fetch_html_selenium(link, driver, wait_time=3)
            if not article_html:
                print(f"        ✗ Failed to fetch article content")
                continue
            
            article_soup = BeautifulSoup(article_html, "html.parser")
            
            h1 = article_soup.find("h1")
            if h1:
                title = h1.get_text(strip=True)
            
            paragraphs = article_soup.find_all("p")
            text = " ".join(p.get_text(" ", strip=True) for p in paragraphs if len(p.get_text(strip=True)) > 20)
            
            if len(text) < 100:
                print(f"        ✗ Insufficient content ({len(text)} chars)")
                continue
            
            date = None
            date_meta = article_soup.find("meta", {"property": "article:published_time"})
            if date_meta:
                date = parse_date(date_meta.get("content"))
            
            if not date:
                date_meta = article_soup.find("meta", {"property": "og:published_time"})
                if date_meta:
                    date = parse_date(date_meta.get("content"))
            
            print(f"        Date: {date if date else 'Unknown'}")
            
            # For Coinpedia, accept top 10 articles even without date
            if date is None or is_recent(date):
                results.append((title, "Coinpedia", text))
                print(f"        ✓ ADDED")
                processed += 1
            else:
                print(f"        ✗ Too old")
                
        except Exception as e:
            print(f"        Error: {e}")
            continue
    
    return results

def scrape_coindesk(driver):
    url = sites[1][1]
    print(f"    Fetching page with requests...")
    
    html = fetch_html_requests(url)
    if not html:
        print(f"    ✗ Failed to fetch page")
        return []
    
    print(f"    ✓ Fetched page: {len(html)} characters")
    soup = BeautifulSoup(html, "html.parser")
    results = []
    
    all_links = soup.find_all("a", href=True)
    article_links = []
    
    for link in all_links:
        href = link.get("href", "")
        if re.search(r'/\d{4}/\d{2}/\d{2}/', href):
            article_links.append(link)
    
    print(f"    Found {len(article_links)} article links with date pattern")
    
    if len(article_links) == 0:
        print(f"    ✗ No article links found!")
        return []
    
    seen_links = set()
    print(f"    Processing {min(len(article_links), 15)} articles...")
    
    for i, link_tag in enumerate(article_links[:15]):
        href = link_tag.get("href", "")
        
        if href in seen_links:
            continue
        seen_links.add(href)
        
        if not href.startswith("http"):
            href = "https://www.coindesk.com" + href
        
        title = link_tag.get_text(strip=True)
        if not title or len(title) < 10:
            continue
        
        print(f"      Article {i+1}: {title[:50]}...")
        
        date = None
        date_match = re.search(r'/(\d{4})/(\d{2})/(\d{2})/', href)
        if date_match:
            year, month, day = date_match.groups()
            date_str = f"{year}-{month}-{day}"
            date = parse_date(date_str)
            print(f"        Date from URL: {date}")
        
        try:
            article_html = fetch_html_requests(href)
            if not article_html:
                print(f"        ✗ Failed to fetch article")
                continue
            
            article_soup = BeautifulSoup(article_html, "html.parser")
            
            h1 = article_soup.find("h1")
            if h1:
                title = h1.get_text(strip=True)
            
            paragraphs = article_soup.find_all("p")
            text = " ".join(p.get_text(" ", strip=True) for p in paragraphs if len(p.get_text(strip=True)) > 20)
            
            if is_recent(date) and len(text) > 100:
                results.append((title, "CoinDesk", text))
                print(f"        ✓ ADDED")
            elif not is_recent(date):
                print(f"        ✗ Too old")
            else:
                print(f"        ✗ Insufficient content")
                
        except Exception as e:
            print(f"        Error: {e}")
            continue
    
    return results

def scrape_crypto_news(driver):
    url = sites[2][1]
    print(f"    Fetching page with Selenium...")
    
    html = fetch_html_selenium(url, driver, wait_time=5)
    if not html:
        print(f"    ✗ Failed to fetch page")
        return []
    
    print(f"    ✓ Fetched page: {len(html)} characters")
    soup = BeautifulSoup(html, "html.parser")
    results = []
    
    time_tags = soup.find_all('time', datetime=True)
    print(f"    Found {len(time_tags)} time tags with datetime")
    
    if len(time_tags) == 0:
        print(f"    ✗ No time tags found!")
        return []
    
    print(f"    Processing up to 10 articles...")
    
    processed = 0
    for time_tag in time_tags[:30]:  # Check more to get 10 good ones
        if processed >= 10:
            break
        
        parent = time_tag
        link_tag = None
        
        for _ in range(10):
            if parent is None:
                break
            link_tag = parent.find('a', href=lambda x: x and x.startswith('http'))
            if link_tag:
                break
            parent = parent.find_parent()
        
        if not link_tag:
            continue
        
        link = link_tag.get('href', '')
        title = link_tag.get_text(strip=True)
        
        if not title or len(title) < 10:
            continue
        
        print(f"      Article {processed+1}: {title[:50]}...")
        
        date = parse_date(time_tag.get('datetime'))
        print(f"        Date: {date if date else 'Unknown'}")
        
        if not is_recent(date):
            print(f"        ✗ Too old")
            continue
        
        try:
            # Use Selenium for Crypto.News articles
            article_html = fetch_html_selenium(link, driver, wait_time=3)
            if not article_html:
                print(f"        ✗ Failed to fetch article")
                continue
            
            article_soup = BeautifulSoup(article_html, "html.parser")
            
            h1 = article_soup.find("h1")
            if h1:
                title = h1.get_text(strip=True)
            
            paragraphs = article_soup.find_all("p")
            text = " ".join(p.get_text(" ", strip=True) for p in paragraphs if len(p.get_text(strip=True)) > 20)
            
            if len(text) > 100:
                results.append((title, "Crypto.News", text))
                print(f"        ✓ ADDED")
                processed += 1
            else:
                print(f"        ✗ Insufficient content ({len(text)} chars)")
                
        except Exception as e:
            print(f"        Error: {e}")
            continue
    
    return results

def scrape_theblock(driver):
    url = sites[3][1]
    print(f"    Fetching page with Selenium...")
    
    html = fetch_html_selenium(url, driver, wait_time=5)
    if not html:
        print(f"    ✗ Failed to fetch page")
        return []
    
    print(f"    ✓ Fetched page: {len(html)} characters")
    soup = BeautifulSoup(html, "html.parser")
    results = []
    
    articles = soup.find_all('article')
    print(f"    Found {len(articles)} article tags")
    
    if len(articles) == 0:
        headings = soup.find_all(['h2', 'h3'])
        print(f"    Fallback: Found {len(headings)} h2/h3 tags")
        
        processed = 0
        for i, heading in enumerate(headings):
            if processed >= 10:
                break
                
            link_tag = heading.find('a', href=True)
            if not link_tag:
                continue
            
            link = link_tag.get('href', '')
            if not link:
                continue
            
            if not link.startswith('http'):
                link = 'https://www.theblock.co' + link
            
            title = link_tag.get_text(strip=True)
            if not title or len(title) < 10:
                continue
            
            print(f"      Article {processed+1}: {title[:50]}...")
            
            try:
                time.sleep(2)  # Be polite
                # Use Selenium for The Block articles
                article_html = fetch_html_selenium(link, driver, wait_time=3)
                if not article_html:
                    print(f"        ✗ Failed to fetch article")
                    continue
                
                article_soup = BeautifulSoup(article_html, "html.parser")
                
                paragraphs = article_soup.find_all("p")
                text = " ".join(p.get_text(" ", strip=True) for p in paragraphs if len(p.get_text(strip=True)) > 20)
                
                if len(text) < 100:
                    print(f"        ✗ Insufficient content ({len(text)} chars)")
                    continue
                
                date = None
                time_tag = article_soup.find("time", datetime=True)
                if time_tag:
                    date = parse_date(time_tag.get("datetime"))
                
                if not date:
                    date_meta = article_soup.find("meta", {"property": "article:published_time"})
                    if date_meta:
                        date = parse_date(date_meta.get("content"))
                
                print(f"        Date: {date if date else 'Unknown'}")
                
                if date is None or is_recent(date):
                    results.append((title, "The Block", text))
                    print(f"        ✓ ADDED")
                    processed += 1
                else:
                    print(f"        ✗ Too old")
                    
            except Exception as e:
                print(f"        Error: {e}")
                continue
        
        return results
    
    print(f"    Processing {min(len(articles), 10)} articles...")
    processed = 0
    
    for i, article in enumerate(articles):
        if processed >= 10:
            break
            
        link_tag = article.find('a', href=True)
        if not link_tag:
            continue
        
        link = link_tag.get('href', '')
        if not link.startswith('http'):
            link = 'https://www.theblock.co' + link
        
        h2 = article.find('h2')
        if h2:
            title = h2.get_text(strip=True)
        else:
            title = article.get_text(strip=True)[:100]
        
        if not title or len(title) < 10:
            continue
        
        print(f"      Article {processed+1}: {title[:50]}...")
        
        try:
            time.sleep(2)  # Be polite
            # Use Selenium for The Block articles
            article_html = fetch_html_selenium(link, driver, wait_time=3)
            if not article_html:
                print(f"        ✗ Failed to fetch article")
                continue
            
            article_soup = BeautifulSoup(article_html, "html.parser")
            
            h1 = article_soup.find("h1")
            if h1:
                title = h1.get_text(strip=True)
            
            paragraphs = article_soup.find_all("p")
            text = " ".join(p.get_text(" ", strip=True) for p in paragraphs if len(p.get_text(strip=True)) > 20)
            
            if len(text) < 100:
                print(f"        ✗ Insufficient content ({len(text)} chars)")
                continue
            
            date = None
            time_tag = article_soup.find("time", datetime=True)
            if time_tag:
                date = parse_date(time_tag.get("datetime"))
            
            if not date:
                date_meta = article_soup.find("meta", {"property": "article:published_time"})
                if date_meta:
                    date = parse_date(date_meta.get("content"))
            
            print(f"        Date: {date if date else 'Unknown'}")
            
            if date is None or is_recent(date):
                results.append((title, "The Block", text))
                print(f"        ✓ ADDED")
                processed += 1
            else:
                print(f"        ✗ Too old")
                
        except Exception as e:
            print(f"        Error: {e}")
            continue
    
    return results

def scrape_decrypt(driver):
    url = sites[4][1]  # Decrypt is now index 4
    print(f"    Fetching page with Selenium...")
    
    html = fetch_html_selenium(url, driver, wait_time=5)
    if not html:
        print(f"    ✗ Failed to fetch page")
        return []
    
    print(f"    ✓ Fetched page: {len(html)} characters")
    soup = BeautifulSoup(html, "html.parser")
    results = []
    
    # Find article containers - they use <article class="linkbox">
    articles = soup.find_all('article', class_='linkbox')
    print(f"    Found {len(articles)} article containers")
    
    if len(articles) == 0:
        print(f"    ✗ No articles found!")
        return []
    
    print(f"    Processing {min(len(articles), 10)} articles...")
    processed = 0
    
    for i, article in enumerate(articles):
        if processed >= 10:
            break
        
        # Find the link with class="linkbox__overlay"
        link_tag = article.find('a', class_='linkbox__overlay')
        if not link_tag:
            continue
        
        link = link_tag.get('href', '')
        if not link:
            continue
        
        # Make full URL
        if not link.startswith('http'):
            link = 'https://decrypt.co' + link
        
        # Get title from the span inside h3
        h3 = article.find('h3')
        if h3:
            title_span = h3.find('span', class_='font-medium')
            if title_span:
                title = title_span.get_text(strip=True)
            else:
                title = h3.get_text(strip=True)
        else:
            continue
        
        if not title or len(title) < 10:
            continue
        
        print(f"      Article {processed+1}: {title[:50]}...")
        
        # Get date from the parent article structure
        # Date is in <h4> tag with format "Jan 28, 2026"
        parent_article = article.find_parent('article')
        date = None
        if parent_article:
            h4_date = parent_article.find('h4')
            if h4_date:
                date_text = h4_date.get_text(strip=True)
                date = parse_date(date_text)
                print(f"        Date: {date if date else 'Unknown'}")
        
        try:
            time.sleep(2)  # Be polite
            # Use Selenium for Decrypt articles
            article_html = fetch_html_selenium(link, driver, wait_time=3)
            if not article_html:
                print(f"        ✗ Failed to fetch article")
                continue
            
            article_soup = BeautifulSoup(article_html, "html.parser")
            
            # Get better title from article page
            h1 = article_soup.find("h1")
            if h1:
                title = h1.get_text(strip=True)
            
            # Get article text from paragraphs
            paragraphs = article_soup.find_all("p")
            text = " ".join(p.get_text(" ", strip=True) for p in paragraphs if len(p.get_text(strip=True)) > 20)
            
            if len(text) < 100:
                print(f"        ✗ Insufficient content ({len(text)} chars)")
                continue
            
            # Accept if date is None or recent
            if date is None or is_recent(date):
                results.append((title, "Decrypt", text))
                print(f"        ✓ ADDED")
                processed += 1
            else:
                print(f"        ✗ Too old")
                
        except Exception as e:
            print(f"        Error: {e}")
            continue
    
    return results

def scrape_cointelegraph(driver):
    url = sites[5][1]  # Cointelegraph is now index 5
    print(f"    Fetching page with Selenium...")
    
    html = fetch_html_selenium(url, driver, wait_time=8)
    if not html:
        print(f"    ✗ Failed to fetch page")
        return []
    
    print(f"    ✓ Fetched page: {len(html)} characters")
    soup = BeautifulSoup(html, "html.parser")
    print(f"    DEBUG: Found {len(soup.find_all('li'))} total <li> tags")
    print(f"    DEBUG: Found {len(soup.find_all('article'))} total <article> tags")
    print(f"    DEBUG: Found {len(soup.find_all('a', class_='post-card-inline__title-link'))} title links")
    results = []
    
    # Find article containers - they use <li data-testid="posts-listing__item">
    # Try multiple selectors
    # Find article containers - try multiple approaches
    articles = []

    # Try approach 1: data-testid attribute
    articles = soup.find_all('li', {'data-testid': 'posts-listing__item'})

    # Try approach 2: article with class
    if len(articles) == 0:
        print(f"    Trying alternative selector: article.post-card-inline")
        articles = soup.find_all('article', class_='post-card-inline')

    # Try approach 3: just find all article tags
    if len(articles) == 0:
        print(f"    Trying alternative selector: all article tags")
        all_articles = soup.find_all('article')
        # Filter to only those with post-card in class
        articles = [a for a in all_articles if a.get('class') and any('post-card' in c for c in a.get('class', []))]

    # Try approach 4: find by title link
    if len(articles) == 0:
        print(f"    Trying alternative selector: finding by title links")
        title_links = soup.find_all('a', class_='post-card-inline__title-link')
        # Get parent article for each
        articles = [link.find_parent('article') for link in title_links if link.find_parent('article')]
        # Remove duplicates
        articles = list(dict.fromkeys(articles))
    if len(articles) == 0:
        articles = soup.find_all('article', class_='post-card-inline')
    if len(articles) == 0:
        # Fallback: find all articles
        articles = soup.find_all('article')
    print(f"    Found {len(articles)} article containers")
    
    if len(articles) == 0:
        print(f"    ✗ No articles found!")
        return []
    
    print(f"    Processing {min(len(articles), 10)} articles...")
    processed = 0
    
    for i, article in enumerate(articles):
        if processed >= 10:
            break
        
        # Find the title link
        title_link = article.find('a', class_='post-card-inline__title-link')
        if not title_link:
            continue
        
        link = title_link.get('href', '')
        if not link:
            continue
        
        # Make full URL
        if not link.startswith('http'):
            link = 'https://cointelegraph.com' + link
        
        # Get title from span inside the link
        title_span = title_link.find('span', class_='post-card-inline__title')
        if title_span:
            title = title_span.get_text(strip=True)
        else:
            title = title_link.get_text(strip=True)
        
        if not title or len(title) < 10:
            continue
        
        print(f"      Article {processed+1}: {title[:50]}...")
        
        # Get date from time tag with datetime attribute
        time_tag = article.find('time', class_='post-card-inline__date')
        date = None
        if time_tag:
            datetime_attr = time_tag.get('datetime')
            if datetime_attr:
                date = parse_date(datetime_attr)
                print(f"        Date: {date if date else 'Unknown'}")
        
        try:
            time.sleep(2)  # Be polite
            # Use Selenium for Cointelegraph articles
            article_html = fetch_html_selenium(link, driver, wait_time=3)
            if not article_html:
                print(f"        ✗ Failed to fetch article")
                continue
            
            article_soup = BeautifulSoup(article_html, "html.parser")
            
            # Get better title from article page
            h1 = article_soup.find("h1")
            if h1:
                title = h1.get_text(strip=True)
            
            # Get article text from paragraphs
            paragraphs = article_soup.find_all("p")
            text = " ".join(p.get_text(" ", strip=True) for p in paragraphs if len(p.get_text(strip=True)) > 20)
            
            if len(text) < 100:
                print(f"        ✗ Insufficient content ({len(text)} chars)")
                continue
            
            # Accept if date is None or recent
            if date is None or is_recent(date):
                results.append((title, "Cointelegraph", text))
                print(f"        ✓ ADDED")
                processed += 1
            else:
                print(f"        ✗ Too old")
                
        except Exception as e:
            print(f"        Error: {e}")
            continue
    
    return results

# ------------------------
# Main Execution
# ------------------------

def main():
    all_articles = []
    
    print(f"\n{'='*60}")
    print("CRYPTO NEWS SCRAPER (with Microsoft Edge)")
    print(f"{'='*60}")
    print(f"Looking for articles published after: {cutoff_date.strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print(f"Current time: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print(f"Time window: Last {days_back} day(s) (48 hours)")
    print(f"{'='*60}\n")
    
    print("Initializing Microsoft Edge browser (this may take a moment)...")
    driver = create_driver()
    
    if not driver:
        print("✗ Failed to create Edge browser driver!")
        print("Falling back to requests-only mode (CoinDesk only)...\n")
        
        try:
            print(f"Scraping CoinDesk...")
            articles = scrape_coindesk(None)
            all_articles.extend(articles)
            print(f"  Found {len(articles)} recent articles from CoinDesk\n")
        except Exception as e:
            print(f"Error scraping CoinDesk: {e}\n")
    else:
        print("✓ Browser ready!\n")
        
        scrapers = [
            ("Coinpedia", lambda d: scrape_coinpedia(d)),
            ("CoinDesk", lambda d: scrape_coindesk(d)),
            ("Crypto.News", lambda d: scrape_crypto_news(d)),
            ("The Block", lambda d: scrape_theblock(d)),
            ("Decrypt", lambda d: scrape_decrypt(d)),
            ("Cointelegraph", lambda d: scrape_cointelegraph(d)) 
        ]
        
        for name, scraper_func in scrapers:
            try:
                print(f"Scraping {name}...")
                articles = scraper_func(driver)
                all_articles.extend(articles)
                print(f"  Found {len(articles)} recent articles from {name}\n")
            except Exception as e:
                print(f"Error scraping {name}: {e}\n")
                import traceback
                traceback.print_exc()
                continue
        
        try:
            driver.quit()
            print("Browser closed.\n")
        except:
            pass
    
    print(f"{'='*60}")
    print(f"Total articles collected: {len(all_articles)}")
    print(f"{'='*60}\n")
    
    if len(all_articles) == 0:
        print("⚠ WARNING: No articles were collected!")
        print("Creating empty JSON file...\n")

    # Build JSON structure
    articles_data = []

    for title, site, full_text in all_articles:
        try:
            # Clean all text
            clean_title = clean_text(title)
            clean_site = clean_text(site)
            summary = summarize_text(full_text)
            
            # Create article object
            article_obj = {
                "title": clean_title,
                "source": clean_site,
                "summary": summary,
                "full_text": clean_text(full_text[:5000])  # Limit full text to 5000 chars
            }
            
            articles_data.append(article_obj)
        except Exception as e:
            print(f"Warning: Failed to add article to JSON: {title[:50]}... Error: {e}")
            continue

    # Save to JSON file
    filename = "crypto_latest_news.json"
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump({
                "collected_at": datetime.now(timezone.utc).isoformat(),
                "total_articles": len(articles_data),
                "time_window_days": days_back,
                "articles": articles_data
            }, f, indent=2, ensure_ascii=False)
    except PermissionError:
        # File is open, create new filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"crypto_latest_news_{timestamp}.json"
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump({
                "collected_at": datetime.now(timezone.utc).isoformat(),
                "total_articles": len(articles_data),
                "time_window_days": days_back,
                "articles": articles_data
            }, f, indent=2, ensure_ascii=False)
        print(f"\n⚠ Original file was open, saved as: {filename}")
    
    print("="*60)
    print(f"✓ SUCCESS: Saved as {filename}")
    print(f"✓ Total articles: {len(all_articles)}")
    print("="*60)

if __name__ == "__main__":
    main()