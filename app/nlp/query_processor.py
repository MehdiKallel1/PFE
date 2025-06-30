import spacy
from .entity_extractor import FinancialEntityExtractor
from .intent_classifier import IntentClassifier
import logging

logger = logging.getLogger(__name__)

class QueryProcessor:
    def __init__(self):
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            logger.warning("spaCy model not found, using basic processing")
            self.nlp = None
        
        self.entity_extractor = FinancialEntityExtractor()
        self.intent_classifier = IntentClassifier()
    
    def process(self, query):
        """Main processing pipeline"""
        try:
            # Basic processing
            entities = self.entity_extractor.extract_entities(query)
            intent = self.intent_classifier.classify_intent(query)
            
            # Advanced processing if spaCy available
            pos_tags = []
            if self.nlp:
                doc = self.nlp(query)
                pos_tags = [(token.text, token.pos_) for token in doc]
            
            return {
                'original_query': query,
                'entities': entities,
                'intent': intent,
                'pos_tags': pos_tags,
                'processed': True
            }
        
        except Exception as e:
            logger.error(f"NLP processing failed: {e}")
            return {
                'original_query': query,
                'entities': {'metrics': [], 'indicators': [], 'time_periods': [], 'comparison_words': []},
                'intent': {'primary': 'DATA_QUERY', 'confidence': 0.5},
                'pos_tags': [],
                'processed': False,
                'error': str(e)
            }
    
    def enhance_context(self, dashboard_context, nlp_results):
        """Enhance dashboard context with NLP insights"""
        enhanced = dashboard_context.copy()
        
        # Add extracted entities as context
        enhanced['nlp_entities'] = nlp_results['entities']
        enhanced['detected_intent'] = nlp_results['intent']['primary']
        enhanced['intent_confidence'] = nlp_results['intent']['confidence']
        
        # Map entities to dashboard data
        if nlp_results['entities']['metrics']:
            enhanced['mentioned_metrics'] = nlp_results['entities']['metrics']
        
        if nlp_results['entities']['indicators']:
            enhanced['mentioned_indicators'] = nlp_results['entities']['indicators']
        
        return enhanced