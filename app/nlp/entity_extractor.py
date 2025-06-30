import re
from datetime import datetime
from dateutil import parser

class FinancialEntityExtractor:
    def __init__(self):
        self.financial_metrics = {
            'revenue', 'sales', 'profit', 'income', 'earnings', 'costs', 'expenses',
            'margin', 'ebitda', 'cash_flow', 'debt', 'equity', 'assets', 'roi',
            'growth', 'risk_score', 'market_share'
        }
        
        self.economic_indicators = {
            'gdp', 'inflation', 'interest_rate', 'unemployment', 'cpi',
            'credit_interieur', 'impots_revenus', 'taux_interet',
            'rnb_par_habitant', 'masse_monetaire', 'pib_us_courants',
            'rnb_us_courants', 'paiements_interet'
        }
        
        self.time_patterns = [
            r'\b(q[1-4]|quarter [1-4])\s*\d{4}\b',
            r'\b\d{4}\b',
            r'\b(last|previous|next)\s+(year|quarter|month)\b',
            r'\b(20\d{2})-(\d{2})-(\d{2})\b'
        ]
    
    def extract_entities(self, text):
        text_lower = text.lower()
        
        return {
            'metrics': self._extract_metrics(text_lower),
            'indicators': self._extract_indicators(text_lower),
            'time_periods': self._extract_time_periods(text),
            'comparison_words': self._extract_comparison_words(text_lower)
        }
    
    def _extract_metrics(self, text):
        return [metric for metric in self.financial_metrics if metric in text]
    
    def _extract_indicators(self, text):
        indicators = []
        for indicator in self.economic_indicators:
            if indicator.replace('_', ' ') in text or indicator in text:
                indicators.append(indicator)
        return indicators
    
    def _extract_time_periods(self, text):
        periods = []
        for pattern in self.time_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            periods.extend([match if isinstance(match, str) else ' '.join(match) for match in matches])
        return periods
    
    def _extract_comparison_words(self, text):
        comparison_words = ['vs', 'versus', 'compared to', 'against', 'with', 'correlation', 'relationship']
        return [word for word in comparison_words if word in text]