"""
Enhanced Fact-Checking System with Free Checker
Uses Wikipedia and public APIs - NO API KEY NEEDED
"""

import requests
from datetime import datetime
import re
from free_fact_checker import FreeFactChecker

class EnhancedFactChecker:
    def __init__(self, model, vectorizer):
        self.model = model
        self.vectorizer = vectorizer
        self.free_checker = FreeFactChecker()
    
    def analyze_text(self, text):
        """Comprehensive analysis combining ML + fact checking"""
        # Step 1: ML-based pattern detection
        ml_result = self.ml_prediction(text)
        
        # Step 2: Entity extraction
        entities = self.extract_entities(text)
        
        # Step 3: Fact verification using FREE checker
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
        """Verify facts using free checker - NO API KEY NEEDED"""
        # Use free fact checker (Wikipedia + Wikidata + basic facts)
        verifications = self.free_checker.verify_claim(text)
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
    print("Using FREE fact-checking (no API keys needed)")
    print("Module loaded successfully!")