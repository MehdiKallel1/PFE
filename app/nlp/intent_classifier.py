class IntentClassifier:
    def __init__(self):
        self.intent_patterns = {
            'TREND_ANALYSIS': [
                'trend', 'trends', 'trending', 'change', 'evolution', 'over time',
                'growth', 'decline', 'increase', 'decrease', 'pattern'
            ],
            'COMPARISON': [
                'compare', 'comparison', 'vs', 'versus', 'against', 'difference',
                'better', 'worse', 'higher', 'lower'
            ],
            'CORRELATION': [
                'correlation', 'correlate', 'relationship', 'impact', 'affect',
                'influence', 'connection', 'related', 'depends'
            ],
            'PREDICTION': [
                'predict', 'forecast', 'future', 'will', 'expect', 'projection',
                '2025', '2026', 'next', 'upcoming'
            ],
            'EXPLANATION': [
                'why', 'how', 'explain', 'reason', 'because', 'what causes',
                'factors', 'drivers'
            ],
            'DATA_QUERY': [
                'show', 'display', 'get', 'find', 'what is', 'current',
                'value', 'data', 'information'
            ]
        }
    
    def classify_intent(self, text):
        text_lower = text.lower()
        intent_scores = {}
        
        for intent, keywords in self.intent_patterns.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            if score > 0:
                intent_scores[intent] = score
        
        if intent_scores:
            primary_intent = max(intent_scores, key=intent_scores.get)
            confidence = intent_scores[primary_intent] / len(text_lower.split())
            return {
                'primary': primary_intent,
                'confidence': min(confidence, 1.0),
                'all_scores': intent_scores
            }
        
        return {
            'primary': 'DATA_QUERY',
            'confidence': 0.5,
            'all_scores': {'DATA_QUERY': 1}
        }