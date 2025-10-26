"""
Web Search-Based Fact Checker
Uses Brave Search API for real-time fact verification
"""

import requests
import os
from urllib.parse import quote

class WebSearchFactChecker:
    """Fact checker using Brave Search API"""
    
    def __init__(self, api_key=None):
        """
        Initialize with Brave Search API key
        Get free API key: https://brave.com/search/api/
        Free tier: 2000 queries/month
        """
        self.api_key = api_key or os.environ.get("BRAVE_API_KEY")
        
        if not self.api_key:
            print("⚠️ Warning: No Brave API key found. Set BRAVE_API_KEY environment variable.")
            self.api_key = None
    
    def search_claim(self, claim, num_results=5):
        """Search web for claim verification"""
        
        if not self.api_key:
            return {
                'found': False,
                'error': 'No API key configured',
                'results': []
            }
        
        # Construct search query optimized for fact-checking
        query = f'"{claim}" fact check OR verify OR true OR false'
        encoded_query = quote(query)
        
        url = f"https://api.search.brave.com/res/v1/web/search?q={encoded_query}&count={num_results}"
        
        headers = {
            "Accept": "application/json",
            "X-Subscription-Token": self.api_key
        }
        
        try:
            response = requests.get(url, headers=headers, timeout=10)
            
            if response.status_code != 200:
                return {
                    'found': False,
                    'error': f'API error: {response.status_code}',
                    'results': []
                }
            
            data = response.json()
            
            results = []
            for result in data.get('web', {}).get('results', [])[:num_results]:
                results.append({
                    'title': result.get('title', ''),
                    'description': result.get('description', ''),
                    'url': result.get('url', ''),
                    'age': result.get('age', '')
                })
            
            return {
                'found': True,
                'results': results,
                'query': query,
                'total_results': len(results)
            }
        
        except requests.exceptions.Timeout:
            return {
                'found': False,
                'error': 'Search timeout',
                'results': []
            }
        except Exception as e:
            return {
                'found': False,
                'error': str(e),
                'results': []
            }
    
    def verify_claim(self, claim):
        """
        Verify claim using web search
        Returns structured verification result
        """
        
        search_results = self.search_claim(claim)
        
        if not search_results['found']:
            return {
                'claim': claim,
                'status': 'UNVERIFIABLE',
                'confidence': 0,
                'explanation': f"Could not search web: {search_results.get('error', 'Unknown error')}",
                'sources': []
            }
        
        if len(search_results['results']) == 0:
            return {
                'claim': claim,
                'status': 'UNVERIFIABLE',
                'confidence': 0,
                'explanation': 'No relevant sources found',
                'sources': []
            }
        
        # Analyze search results
        analysis = self._analyze_search_results(claim, search_results['results'])
        
        return {
            'claim': claim,
            'status': analysis['status'],
            'confidence': analysis['confidence'],
            'explanation': analysis['explanation'],
            'sources': analysis['sources']
        }
    
    def _analyze_search_results(self, claim, results):
        """Analyze search results to determine claim accuracy"""
        
        # Trusted fact-checking domains
        trusted_domains = {
            'snopes.com': 95,
            'factcheck.org': 95,
            'politifact.com': 90,
            'fullfact.org': 90,
            'reuters.com': 85,
            'apnews.com': 85,
            'bbc.com': 80,
            'nytimes.com': 75,
            'wikipedia.org': 70,
            'nasa.gov': 95,
            'noaa.gov': 90,
            'cdc.gov': 90,
            'nih.gov': 90
        }
        
        # Keywords indicating verification
        true_keywords = ['true', 'correct', 'accurate', 'verified', 'confirmed', 'yes']
        false_keywords = ['false', 'incorrect', 'wrong', 'fake', 'debunked', 'myth', 'no']
        
        trusted_sources = []
        total_trust_score = 0
        true_signals = 0
        false_signals = 0
        
        for result in results:
            url = result['url'].lower()
            title = result['title'].lower()
            desc = result['description'].lower()
            
            # Check if from trusted domain
            domain_trust = 0
            for domain, trust_level in trusted_domains.items():
                if domain in url:
                    domain_trust = trust_level
                    trusted_sources.append({
                        'title': result['title'],
                        'url': result['url'],
                        'trust_level': trust_level,
                        'snippet': result['description'][:150] + '...'
                    })
                    break
            
            if domain_trust > 0:
                # Analyze content
                content = title + ' ' + desc
                
                # Count true/false indicators
                true_count = sum(1 for keyword in true_keywords if keyword in content)
                false_count = sum(1 for keyword in false_keywords if keyword in content)
                
                if true_count > false_count:
                    true_signals += domain_trust
                elif false_count > true_count:
                    false_signals += domain_trust
                
                total_trust_score += domain_trust
        
        # Determine verdict
        if total_trust_score == 0:
            return {
                'status': 'UNVERIFIABLE',
                'confidence': 30,
                'explanation': 'No trusted sources found. Results from general web only.',
                'sources': []
            }
        
        # Calculate confidence based on trusted sources
        if true_signals > false_signals * 1.5:
            status = 'VERIFIED'
            confidence = min(95, int((true_signals / total_trust_score) * 100))
            explanation = f'Multiple trusted sources confirm this claim. Found {len(trusted_sources)} reliable source(s).'
        
        elif false_signals > true_signals * 1.5:
            status = 'FALSE'
            confidence = min(95, int((false_signals / total_trust_score) * 100))
            explanation = f'Trusted sources indicate this is false. Found {len(trusted_sources)} reliable source(s) debunking this.'
        
        else:
            status = 'UNCERTAIN'
            confidence = 50
            explanation = f'Mixed signals from {len(trusted_sources)} trusted source(s). Requires further verification.'
        
        return {
            'status': status,
            'confidence': confidence,
            'explanation': explanation,
            'sources': trusted_sources[:3]  # Top 3 sources
        }
    
    def quick_check(self, claim):
        """Quick verification - simpler output"""
        result = self.verify_claim(claim)
        
        print(f"\n{'='*60}")
        print(f"Claim: {claim}")
        print(f"Status: {result['status']}")
        print(f"Confidence: {result['confidence']}%")
        print(f"Explanation: {result['explanation']}")
        
        if result['sources']:
            print(f"\nTrusted Sources:")
            for i, source in enumerate(result['sources'], 1):
                print(f"{i}. {source['title']}")
                print(f"   {source['url']}")
        print(f"{'='*60}\n")
        
        return result


# Example usage and testing
if __name__ == "__main__":
    print("Web Search-Based Fact Checker")
    print("=" * 60)
    
    # Check for API key
    api_key = os.environ.get("BRAVE_API_KEY")
    
    if not api_key:
        print("\n⚠️  Setup Required:")
        print("1. Get free API key: https://brave.com/search/api/")
        print("2. Set environment variable:")
        print("   Windows: set BRAVE_API_KEY=your-key-here")
        print("   Linux/Mac: export BRAVE_API_KEY=your-key-here")
        print("\nOr pass API key directly:")
        print("   checker = WebSearchFactChecker(api_key='your-key')")
    else:
        print("\n✅ API key found! Running tests...\n")
        
        checker = WebSearchFactChecker()
        
        # Test claims
        test_claims = [
            "The sun sets in the east",
            "Donald Trump is the president of America",
            "Water boils at 100 degrees Celsius",
            "The Earth is flat"
        ]
        
        for claim in test_claims:
            result = checker.quick_check(claim)


# Free alternatives if Brave doesn't work
ALTERNATIVE_APIS = """
Free Web Search APIs:

1. Brave Search (Recommended)
   - FREE: 2,000 queries/month
   - https://brave.com/search/api/

2. SerpAPI
   - FREE: 100 searches/month
   - https://serpapi.com/

3. DuckDuckGo (No API key needed!)
   - Unlimited but rate-limited
   - pip install duckduckgo-search

4. Custom Google Search
   - FREE: 100 searches/day
   - https://developers.google.com/custom-search
"""

print(ALTERNATIVE_APIS)