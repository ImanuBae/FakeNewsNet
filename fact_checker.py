"""
Enhanced Fact-Checking System
Combines ML model with web search verification
"""

from web_search_checker import WebSearchFactChecker
import requests
from datetime import datetime
import re

class EnhancedFactChecker:
    def __init__(self, model, vectorizer):
        self.model = model
        self.vectorizer = vectorizer
        self.web_checker = WebSearchFactChecker()
    
    def analyze_text(self, text):
        """Comprehensive analysis combining ML + fact checking"""
        # Step 1: ML-based pattern detection
        ml_result = self.ml_prediction(text)
        
        # Step 2: Entity extraction
        entities = self.extract_entities(text)
        
        # Step 3: Fact verification
        fact_check = self.verify_facts(text, entities)
        
        # Step 4: Combine all signals
        final_result = self.combine_signals(ml_result, fact_check)
        
        return final_result
    
    def ml_prediction(self, text):
        """Original ML model prediction"""
        input_vec = self.vectorizer.transform([text])
        prediction = self.model.predict(input_vec)[0]
        probability = self.model.predict_proba(input_vec)[0]
        
        return {
            'prediction': 'fake' if prediction == 0 else 'real',
            'confidence': max(probability) * 100,
            'probability': probability,
            'method': 'ML Pattern Recognition'
        }
    
    def extract_entities(self, text):
        """Extract named entities (people, places, organizations)"""
        entities = {
            'people': [],
            'organizations': [],
            'locations': [],
            'dates': []
        }
        
        # Detect common political figures
        political_figures = ['Trump', 'Biden', 'Obama', 'Harris', 'Putin', 'Xi Jinping']
        for figure in political_figures:
            if figure.lower() in text.lower():
                entities['people'].append(figure)
        
        # Detect year mentions
        years = re.findall(r'\b(19|20)\d{2}\b', text)
        entities['dates'] = years
        
        return entities
    
    def verify_facts(self, text, entities):
        """Verify facts against knowledge base"""
        verifications = []
        text_lower = text.lower()
        
        # Current facts database (2025)
        facts_db = {
            'us_president': 'Donald Trump',
            'us_president_since': '2025-01-20',
            'previous_presidents': {
                'Biden': '2021-2025',
                'Trump': '2017-2021, 2025-present',
                'Obama': '2009-2017'
            }
        }
        
        # IMPROVED: More flexible president detection
        president_keywords = ['president', 'potus', 'commander in chief', 'leader']
        location_keywords = ['america', 'usa', 'united states', 'us', 'u.s.', 'american']
        
        # Check if text is about US president
        has_president_mention = any(keyword in text_lower for keyword in president_keywords)
        has_location_mention = any(keyword in text_lower for keyword in location_keywords)
        
        if has_president_mention and has_location_mention:
            current_president = facts_db['us_president']
            
            # Check if mentions current president correctly
            if current_president.lower() in text_lower or 'trump' in text_lower:
                verifications.append({
                    'claim': f'{current_president} is currently the US President',
                    'status': 'VERIFIED',
                    'source': 'Official US Government Records (2025)',
                    'confidence': 100,
                    'details': f"✓ {current_president} was inaugurated on {facts_db['us_president_since']} and is currently serving."
                })
            
            # Check for false claims about other people
            false_president_names = ['biden', 'obama', 'bush', 'clinton', 'harris', 'pence']
            for name in false_president_names:
                if name in text_lower and name != current_president.lower():
                    proper_name = name.title()
                    if proper_name in facts_db['previous_presidents']:
                        term = facts_db['previous_presidents'][proper_name]
                    else:
                        term = 'never served as president' if proper_name != 'Harris' else 'Vice President 2021-2025'
                    
                    verifications.append({
                        'claim': f'{proper_name} is the US President',
                        'status': 'FALSE',
                        'source': 'Official US Government Records',
                        'confidence': 100,
                        'details': f'✗ {proper_name} {term}. Current president: {current_president} (since Jan 20, 2025)'
                    })
        
        # NEW: Try web search for any remaining claims
        if len(verifications) == 0:
            # === SỬA LỖI: Các dòng sau đây cần được thụt vào trong ===
            # No rule-based fact found, try web search
            web_result = self.web_checker.verify_claim(text)
            
            if web_result['status'] != 'UNVERIFIABLE':
                verifications.append({
                    'claim': web_result['claim'],
                    'status': web_result['status'],
                    'source': 'Web Search Analysis',
                    'confidence': web_result['confidence'],
                    'details': web_result['explanation'],
                    'sources': web_result.get('sources', [])
                })
        
        # Dòng 'return' này phải nằm ở cấp độ này, bên trong hàm 'verify_facts'
        return verifications
    
    def combine_signals(self, ml_result, fact_check):
        """Combine ML and fact-checking for final decision"""
        # If facts are verified, they take priority
        verified_count = sum(1 for v in fact_check if v['status'] == 'VERIFIED')
        false_count = sum(1 for v in fact_check if v['status'] == 'FALSE')
        
        if verified_count > 0:
            return {
                'final_prediction': 'REAL',
                'confidence': 95,
                'ml_prediction': ml_result['prediction'],
                'ml_confidence': ml_result['confidence'],
                'fact_verification': fact_check,
                'override': True,
                'override_reason': f'{verified_count} fact(s) verified from trusted sources'
            }
        
        if false_count > 0:
            return {
                'final_prediction': 'FAKE',
                'confidence': 95,
                'ml_prediction': ml_result['prediction'],
                'ml_confidence': ml_result['confidence'],
                'fact_verification': fact_check,
                'override': True,
                'override_reason': f'{false_count} false claim(s) detected'
            }
        
        # No facts to check, trust ML model
        return {
            'final_prediction': ml_result['prediction'].upper(),
            'confidence': ml_result['confidence'],
            'ml_prediction': ml_result['prediction'],
            'ml_confidence': ml_result['confidence'],
            'fact_verification': fact_check,
            'override': False,
            'override_reason': None
        }


class WebFactChecker:
    """Web-based fact checking using external APIs"""
    
    def search_wikipedia(self, query):
        """Search Wikipedia for entity information"""
        try:
            # Clean query
            query = query.replace(' is ', ' ').replace(' the ', ' ').strip()
            words = query.split()
            
            # Find entity (capitalized words)
            entities = [w for w in words if len(w) > 1 and w[0].isupper()]
            
            if entities:
                entity = ' '.join(entities[:2])
                url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{entity.replace(' ', '_')}"
                response = requests.get(url, timeout=5)
                
                if response.status_code == 200:
                    data = response.json()
                    return {
                        'found': True,
                        'title': data.get('title'),
                        'extract': data.get('extract', '')[:400] + '...',
                        'url': data.get('content_urls', {}).get('desktop', {}).get('page', ''),
                        'thumbnail': data.get('thumbnail', {}).get('source', '')
                    }
        except Exception as e:
            print(f"Wikipedia search error: {str(e)}")
        
        return {'found': False}


if __name__ == "__main__":
    print("Enhanced Fact-Checking System")
    print("Module loaded successfully!")
# === SỬA LỖI: Đã xóa dấu '}' thừa ở đây ===