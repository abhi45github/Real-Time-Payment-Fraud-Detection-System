"""
Dynamic Risk Scoring System
Implements adaptive thresholds with cost-sensitive optimization
Responds to fraud patterns, time variations, and business rules
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass
from collections import deque
from datetime import datetime, timedelta
import logging
from enum import Enum

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Decision(Enum):
    """Fraud decision types"""
    APPROVE = "APPROVE"
    REVIEW = "REVIEW"
    DECLINE = "DECLINE"
    CHALLENGE = "CHALLENGE"  # Additional verification required


@dataclass
class RiskConfig:
    """Configuration for risk scoring"""
    
    # Base thresholds
    base_approve_threshold: float = 0.3
    base_review_threshold: float = 0.6
    base_decline_threshold: float = 0.85
    
    # Target metrics
    target_fraud_rate: float = 0.02  # 2%
    target_false_positive_rate: float = 0.05  # 5%
    max_review_rate: float = 0.1  # 10% max manual review
    
    # Cost parameters (in USD)
    fraud_loss_cost: float = 100.0  # Average loss per fraud
    review_cost: float = 5.0  # Cost per manual review
    false_decline_cost: float = 20.0  # Lost revenue per false decline
    
    # Adaptation parameters
    adaptation_rate: float = 0.1  # How quickly thresholds adapt
    history_window: int = 1000  # Number of transactions for metrics
    
    # Time-based adjustments
    night_sensitivity_boost: float = 0.1
    weekend_sensitivity_boost: float = 0.05
    holiday_sensitivity_boost: float = 0.15
    
    # Business rules
    auto_decline_amount: float = 10000  # Auto-decline above this
    auto_approve_amount: float = 10  # Auto-approve below this
    high_risk_countries: List[str] = None
    trusted_merchants: List[str] = None
    
    def __post_init__(self):
        if self.high_risk_countries is None:
            self.high_risk_countries = ['NG', 'PK', 'RO', 'UA']
        if self.trusted_merchants is None:
            self.trusted_merchants = []


class DynamicRiskScorer:
    """
    Dynamic risk scoring with adaptive thresholds
    Optimizes for business objectives while maintaining fraud prevention
    """
    
    def __init__(self, config: RiskConfig = None):
        """Initialize risk scorer with configuration"""
        
        self.config = config or RiskConfig()
        
        # Current thresholds (will adapt over time)
        self.current_thresholds = {
            'approve': self.config.base_approve_threshold,
            'review': self.config.base_review_threshold,
            'decline': self.config.base_decline_threshold
        }
        
        # Performance tracking
        self.fraud_history = deque(maxlen=self.config.history_window)
        self.decision_history = deque(maxlen=self.config.history_window)
        self.feedback_buffer = deque(maxlen=1000)
        
        # Metrics
        self.metrics = {
            'total_transactions': 0,
            'fraud_detected': 0,
            'false_positives': 0,
            'true_positives': 0,
            'reviews_sent': 0
        }
        
        # Custom rules engine
        self.custom_rules = []
        
        logger.info("Dynamic risk scorer initialized")
    
    def calculate_risk_score(self,
                            transaction: Dict,
                            ml_score: float,
                            rule_scores: Dict[str, float],
                            features: Dict) -> Tuple[float, Dict]:
        """
        Calculate comprehensive risk score
        
        Args:
            transaction: Transaction data
            ml_score: ML model probability (0-1)
            rule_scores: Individual rule scores
            features: Extracted features
            
        Returns:
            Tuple of (final_risk_score, score_components)
        """
        
        # Start with ML score as base
        base_score = ml_score
        
        # Apply rule-based adjustments
        rule_adjustment = self._apply_rule_scores(rule_scores)
        
        # Cost-sensitive adjustment based on amount
        amount_weight = self._calculate_amount_weight(
            transaction.get('amount', 0)
        )
        
        # Velocity and behavioral adjustments
        velocity_adjustment = self._calculate_velocity_adjustment(features)
        behavioral_adjustment = self._calculate_behavioral_adjustment(features)
        
        # Network risk adjustment
        network_adjustment = self._calculate_network_adjustment(features)
        
        # Combine scores with weighted average
        weights = {
            'ml': 0.5,
            'rules': 0.2,
            'velocity': 0.1,
            'behavioral': 0.1,
            'network': 0.1
        }
        
        weighted_score = (
            weights['ml'] * base_score +
            weights['rules'] * rule_adjustment +
            weights['velocity'] * velocity_adjustment +
            weights['behavioral'] * behavioral_adjustment +
            weights['network'] * network_adjustment
        )
        
        # Apply amount-based scaling
        scaled_score = weighted_score * (1 + amount_weight * 0.2)
        
        # Apply custom business rules
        final_score = self._apply_business_rules(
            scaled_score, transaction, features
        )
        
        # Ensure score is in [0, 1]
        final_score = np.clip(final_score, 0, 1)
        
        # Track components for explainability
        components = {
            'ml_score': ml_score,
            'rule_adjustment': rule_adjustment,
            'velocity_adjustment': velocity_adjustment,
            'behavioral_adjustment': behavioral_adjustment,
            'network_adjustment': network_adjustment,
            'amount_weight': amount_weight,
            'pre_rules_score': scaled_score,
            'final_score': final_score
        }
        
        return final_score, components
    
    def get_decision(self,
                    risk_score: float,
                    transaction: Dict,
                    override_thresholds: Optional[Dict] = None) -> Tuple[Decision, Dict]:
        """
        Make fraud decision based on risk score and dynamic thresholds
        
        Args:
            risk_score: Calculated risk score (0-1)
            transaction: Transaction data
            override_thresholds: Optional threshold overrides
            
        Returns:
            Tuple of (decision, decision_details)
        """
        
        # Get effective thresholds
        thresholds = self._get_effective_thresholds(
            transaction,
            override_thresholds
        )
        
        # Make decision
        if risk_score < thresholds['approve']:
            decision = Decision.APPROVE
        elif risk_score < thresholds['review']:
            # Check if we're within review budget
            if self._can_send_to_review():
                decision = Decision.REVIEW
            else:
                # If review queue is full, make binary decision
                decision = (
                    Decision.APPROVE 
                    if risk_score < thresholds['review'] * 0.7
                    else Decision.DECLINE
                )
        elif risk_score < thresholds['decline']:
            decision = Decision.CHALLENGE
        else:
            decision = Decision.DECLINE
        
        # Apply hard rules that override ML decision
        decision = self._apply_hard_rules(decision, transaction, risk_score)
        
        # Track decision
        self._track_decision(decision, risk_score, transaction)
        
        # Prepare decision details
        details = {
            'risk_score': risk_score,
            'decision': decision.value,
            'thresholds_used': thresholds,
            'confidence': self._calculate_confidence(risk_score, thresholds),
            'reason_codes': self._generate_reason_codes(
                risk_score, transaction, thresholds
            )
        }
        
        return decision, details
    
    def _get_effective_thresholds(self,
                                 transaction: Dict,
                                 override_thresholds: Optional[Dict]) -> Dict:
        """Calculate effective thresholds with all adjustments"""
        
        if override_thresholds:
            return override_thresholds
        
        # Start with current adaptive thresholds
        thresholds = self.current_thresholds.copy()
        
        # Time-based adjustments
        timestamp = transaction.get('timestamp', datetime.now().timestamp())
        dt = datetime.fromtimestamp(timestamp)
        
        # Night time adjustment (0-6, 22-24)
        if 0 <= dt.hour <= 6 or dt.hour >= 22:
            for key in thresholds:
                thresholds[key] -= self.config.night_sensitivity_boost
        
        # Weekend adjustment
        if dt.weekday() >= 5:
            for key in thresholds:
                thresholds[key] -= self.config.weekend_sensitivity_boost
        
        # Holiday adjustment
        if self._is_holiday(dt):
            for key in thresholds:
                thresholds[key] -= self.config.holiday_sensitivity_boost
        
        # Merchant category adjustment
        merchant_category = transaction.get('merchant_category')
        if merchant_category in ['gambling', 'crypto', 'wire_transfer']:
            thresholds['approve'] -= 0.1
            thresholds['review'] -= 0.15
        
        # Country risk adjustment
        if transaction.get('country') in self.config.high_risk_countries:
            thresholds['approve'] -= 0.15
            thresholds['review'] -= 0.1
        
        # Amount-based adjustment
        amount = transaction.get('amount', 0)
        if amount > 5000:
            thresholds['approve'] -= 0.1
            thresholds['review'] -= 0.05
        
        # Ensure valid threshold ordering
        thresholds['approve'] = max(0.1, min(thresholds['approve'], 0.5))
        thresholds['review'] = max(thresholds['approve'] + 0.1, 
                                   min(thresholds['review'], 0.8))
        thresholds['decline'] = max(thresholds['review'] + 0.1,
                                   min(thresholds['decline'], 0.95))
        
        return thresholds
    
    def adapt_thresholds(self, feedback_batch: List[Dict]):
        """
        Adapt thresholds based on feedback
        
        Args:
            feedback_batch: List of feedback records with actual fraud labels
        """
        
        if not feedback_batch:
            return
        
        # Calculate current performance metrics
        metrics = self._calculate_performance_metrics(feedback_batch)
        
        # Calculate threshold adjustments
        adjustments = self._calculate_threshold_adjustments(metrics)
        
        # Apply adjustments with learning rate
        for key, adjustment in adjustments.items():
            self.current_thresholds[key] += (
                adjustment * self.config.adaptation_rate
            )
        
        # Ensure thresholds remain valid
        self._validate_thresholds()
        
        logger.info(f"Thresholds adapted: {self.current_thresholds}")
        logger.info(f"Current metrics: {metrics}")
    
    def _calculate_performance_metrics(self, feedback_batch: List[Dict]) -> Dict:
        """Calculate performance metrics from feedback"""
        
        total = len(feedback_batch)
        if total == 0:
            return {}
        
        frauds = sum(1 for f in feedback_batch if f['is_fraud'])
        detected = sum(1 for f in feedback_batch 
                      if f['is_fraud'] and f['decision'] != 'APPROVE')
        false_positives = sum(1 for f in feedback_batch
                             if not f['is_fraud'] and f['decision'] == 'DECLINE')
        reviews = sum(1 for f in feedback_batch if f['decision'] == 'REVIEW')
        
        return {
            'fraud_rate': frauds / total,
            'detection_rate': detected / frauds if frauds > 0 else 0,
            'false_positive_rate': false_positives / (total - frauds) if total > frauds else 0,
            'review_rate': reviews / total,
            'precision': detected / (detected + false_positives) if (detected + false_positives) > 0 else 0
        }
    
    def _calculate_threshold_adjustments(self, metrics: Dict) -> Dict:
        """Calculate how thresholds should be adjusted"""
        
        adjustments = {
            'approve': 0,
            'review': 0,
            'decline': 0
        }
        
        if not metrics:
            return adjustments
        
        # Adjust based on fraud rate
        fraud_rate_error = metrics.get('fraud_rate', 0) - self.config.target_fraud_rate
        if fraud_rate_error > 0.005:  # Too much fraud
            adjustments['approve'] -= 0.02
            adjustments['review'] -= 0.01
        elif fraud_rate_error < -0.005:  # Too conservative
            adjustments['approve'] += 0.01
            adjustments['review'] += 0.02
        
        # Adjust based on false positive rate
        fp_rate = metrics.get('false_positive_rate', 0)
        if fp_rate > self.config.target_false_positive_rate:
            adjustments['decline'] += 0.02
            adjustments['review'] += 0.01
        
        # Adjust based on review rate
        review_rate = metrics.get('review_rate', 0)
        if review_rate > self.config.max_review_rate:
            adjustments['review'] += 0.02  # Increase review threshold
        
        return adjustments
    
    def _apply_rule_scores(self, rule_scores: Dict[str, float]) -> float:
        """Combine multiple rule scores into single adjustment"""
        
        if not rule_scores:
            return 0.5
        
        # Weighted combination of rule scores
        weights = {
            'velocity': 0.3,
            'frequency': 0.2,
            'blacklist': 0.3,
            'whitelist': -0.2,  # Negative weight for whitelist
            'pattern': 0.2
        }
        
        total_score = 0
        total_weight = 0
        
        for rule, score in rule_scores.items():
            weight = weights.get(rule, 0.1)
            total_score += score * abs(weight)
            total_weight += abs(weight)
        
        return total_score / total_weight if total_weight > 0 else 0.5
    
    def _calculate_amount_weight(self, amount: float) -> float:
        """Calculate weight based on transaction amount"""
        
        # Logarithmic scaling for amount
        if amount <= 0:
            return 0
        
        # Normalize amount (assuming typical range 1-10000)
        normalized = np.log1p(amount) / np.log1p(10000)
        
        # Apply cost-sensitive weighting
        if amount > 5000:
            return min(normalized * 2, 1.0)
        elif amount > 1000:
            return normalized * 1.5
        else:
            return normalized
    
    def _calculate_velocity_adjustment(self, features: Dict) -> float:
        """Calculate adjustment based on velocity features"""
        
        velocity_score = features.get('velocity_score', 0)
        tx_count_1h = features.get('tx_count_1h', 0)
        
        # High velocity is suspicious
        if velocity_score > 0.7 or tx_count_1h > 10:
            return 0.8
        elif velocity_score > 0.5 or tx_count_1h > 5:
            return 0.6
        else:
            return 0.3
    
    def _calculate_behavioral_adjustment(self, features: Dict) -> float:
        """Calculate adjustment based on behavioral features"""
        
        deviation = features.get('spending_pattern_deviation', 0)
        consistency = features.get('consistency_score', 0.5)
        
        # High deviation or low consistency is suspicious
        if deviation > 0.7 or consistency < 0.3:
            return 0.7
        elif deviation > 0.5 or consistency < 0.5:
            return 0.5
        else:
            return 0.3
    
    def _calculate_network_adjustment(self, features: Dict) -> float:
        """Calculate adjustment based on network features"""
        
        ip_risk = features.get('ip_risk_score', 0)
        fraud_ring_prob = features.get('fraud_ring_probability', 0)
        
        # High network risk indicators
        if ip_risk > 0.7 or fraud_ring_prob > 0.6:
            return 0.8
        elif ip_risk > 0.5 or fraud_ring_prob > 0.4:
            return 0.6
        else:
            return 0.3
    
    def _apply_business_rules(self,
                             score: float,
                             transaction: Dict,
                             features: Dict) -> float:
        """Apply business rules that can override scores"""
        
        amount = transaction.get('amount', 0)
        
        # Auto-decline very high amounts
        if amount > self.config.auto_decline_amount:
            return 0.95
        
        # Auto-approve very low amounts
        if amount < self.config.auto_approve_amount:
            return 0.1
        
        # Trusted merchant override
        if transaction.get('merchant_id') in self.config.trusted_merchants:
            return score * 0.5  # Reduce risk score by half
        
        # Apply custom rules
        for rule in self.custom_rules:
            score = rule(score, transaction, features)
        
        return score
    
    def _apply_hard_rules(self,
                         decision: Decision,
                         transaction: Dict,
                         risk_score: float) -> Decision:
        """Apply hard rules that override ML decisions"""
        
        # Blacklist check
        if transaction.get('ip_blacklisted') or transaction.get('card_blacklisted'):
            return Decision.DECLINE
        
        # Whitelist check
        if transaction.get('user_whitelisted'):
            return Decision.APPROVE
        
        # Amount-based hard rules
        amount = transaction.get('amount', 0)
        if amount > 50000:  # Very high amount always needs review
            return Decision.REVIEW if decision == Decision.APPROVE else decision
        
        # First transaction from new device
        if transaction.get('is_new_device') and risk_score > 0.4:
            return Decision.CHALLENGE
        
        return decision
    
    def _can_send_to_review(self) -> bool:
        """Check if we can send transaction to manual review"""
        
        recent_reviews = sum(
            1 for d in list(self.decision_history)[-100:]
            if d == Decision.REVIEW
        )
        
        review_rate = recent_reviews / 100 if len(self.decision_history) >= 100 else 0
        
        return review_rate < self.config.max_review_rate
    
    def _calculate_confidence(self, risk_score: float, thresholds: Dict) -> float:
        """Calculate confidence in the decision"""
        
        # Distance from nearest threshold
        distances = [
            abs(risk_score - thresholds['approve']),
            abs(risk_score - thresholds['review']),
            abs(risk_score - thresholds['decline'])
        ]
        
        min_distance = min(distances)
        
        # Convert distance to confidence (closer to threshold = lower confidence)
        confidence = 1 - np.exp(-5 * min_distance)
        
        return round(confidence, 3)
    
    def _generate_reason_codes(self,
                              risk_score: float,
                              transaction: Dict,
                              thresholds: Dict) -> List[str]:
        """Generate human-readable reason codes for decision"""
        
        reasons = []
        
        if risk_score > thresholds['decline']:
            reasons.append("VERY_HIGH_RISK")
        elif risk_score > thresholds['review']:
            reasons.append("HIGH_RISK")
        
        amount = transaction.get('amount', 0)
        if amount > 5000:
            reasons.append("HIGH_AMOUNT")
        
        if transaction.get('country') in self.config.high_risk_countries:
            reasons.append("HIGH_RISK_COUNTRY")
        
        if transaction.get('is_new_device'):
            reasons.append("NEW_DEVICE")
        
        if transaction.get('velocity_score', 0) > 0.7:
            reasons.append("HIGH_VELOCITY")
        
        return reasons
    
    def _track_decision(self, decision: Decision, risk_score: float, transaction: Dict):
        """Track decision for metrics and adaptation"""
        
        self.decision_history.append(decision)
        self.metrics['total_transactions'] += 1
        
        if decision == Decision.REVIEW:
            self.metrics['reviews_sent'] += 1
    
    def _validate_thresholds(self):
        """Ensure thresholds are valid and properly ordered"""
        
        self.current_thresholds['approve'] = np.clip(
            self.current_thresholds['approve'], 0.1, 0.5
        )
        self.current_thresholds['review'] = np.clip(
            self.current_thresholds['review'],
            self.current_thresholds['approve'] + 0.1,
            0.8
        )
        self.current_thresholds['decline'] = np.clip(
            self.current_thresholds['decline'],
            self.current_thresholds['review'] + 0.1,
            0.95
        )
    
    def _is_holiday(self, dt: datetime) -> bool:
        """Check if date is a holiday"""
        # Simplified - would use proper holiday calendar
        holidays = [(1, 1), (7, 4), (12, 25)]
        return (dt.month, dt.day) in holidays
    
    def add_custom_rule(self, rule: Callable):
        """Add custom business rule"""
        self.custom_rules.append(rule)
    
    def get_metrics(self) -> Dict:
        """Get current performance metrics"""
        return self.metrics.copy()
    
    def get_current_thresholds(self) -> Dict:
        """Get current adaptive thresholds"""
        return self.current_thresholds.copy()


if __name__ == "__main__":
    # Example usage
    scorer = DynamicRiskScorer()
    
    # Sample transaction
    transaction = {
        'transaction_id': 'tx_123',
        'amount': 500,
        'merchant_category': 'retail',
        'country': 'US',
        'timestamp': datetime.now().timestamp()
    }
    
    # Sample scores
    ml_score = 0.65
    rule_scores = {
        'velocity': 0.3,
        'frequency': 0.4,
        'pattern': 0.5
    }
    
    features = {
        'velocity_score': 0.4,
        'tx_count_1h': 3,
        'spending_pattern_deviation': 0.2
    }
    
    # Calculate risk score
    risk_score, components = scorer.calculate_risk_score(
        transaction, ml_score, rule_scores, features
    )
    
    # Get decision
    decision, details = scorer.get_decision(risk_score, transaction)
    
    logger.info(f"Risk Score: {risk_score:.3f}")
    logger.info(f"Decision: {decision.value}")
    logger.info(f"Details: {details}")

# Update: 2025-10-01T15:32:00.601173 - 4279

# Update: 2025-10-01T15:32:01.247734 - 9597

# Update: 2025-10-01T15:32:02.032991 - 6846

# Update: 2025-10-01T15:32:02.614298 - 8001

# Update: 2025-10-01T15:32:03.574812 - 1329

# Update: 2025-10-01T15:32:04.411866 - 9617

# Update: 2025-10-01T15:32:04.949988 - 9975

# Update: 2025-10-01T15:32:06.104560 - 3728

# Update: 2025-10-01T15:32:06.243224 - 3096

# Update: 2025-10-01T15:32:06.775943 - 2874

# Update: 2025-10-01T15:32:06.965290 - 7417

# Update: 2025-10-01T15:32:09.164794 - 9930

# Update: 2025-10-01T15:32:09.371092 - 6961

# Update: 2025-10-01T15:32:12.148241 - 4318

# Update: 2025-10-01T15:32:12.657841 - 8789

# Update: 2025-10-01T15:32:12.866230 - 7599

# Update: 2025-10-01T15:32:13.257403 - 8318

# Update: 2025-10-01T15:32:13.336502 - 2110

# Update: 2025-10-01T15:32:13.524114 - 9583

# Update: 2025-10-01T15:32:14.093490 - 1852

# Update: 2025-10-01T15:32:15.043355 - 1747

# Update: 2025-10-01T15:32:15.755608 - 2497

# Update: 2025-10-01T15:32:15.958869 - 9975

# Update: 2025-10-01T15:32:16.148167 - 4397

# Update: 2025-10-01T15:32:16.352960 - 6804

# Update: 2025-10-01T15:32:16.731245 - 7714

# Update: 2025-10-01T15:32:17.398482 - 6606

# Update: 2025-10-01T15:32:17.586675 - 4175

# Update: 2025-10-01T15:32:17.856877 - 8601

# Update: 2025-10-01T15:32:18.775236 - 6902

# Update: 2025-10-01T15:32:19.281833 - 9321

# Update: 2025-10-01T15:32:19.946951 - 5933

# Update: 2025-10-01T15:32:20.519727 - 1945

# Update: 2025-10-01T15:32:20.582360 - 2106

# Update: 2025-10-01T15:32:21.771776 - 6711

# Update: 2025-10-01T15:32:23.901952 - 3920

# Update: 2025-10-01T15:32:24.044514 - 8410

# Update: 2025-10-01T15:32:24.611871 - 7666

# Update: 2025-10-01T15:32:26.202135 - 9545

# Update: 2025-10-01T15:32:26.866540 - 1424

# Update: 2025-10-01T15:32:27.546508 - 9612

# Update: 2025-10-01T15:32:27.863005 - 6275

# Update: 2025-10-01T15:32:29.637655 - 6299

# Update: 2025-10-01T15:32:30.565147 - 5544

# Update: 2025-10-01T15:32:31.410255 - 7023

# Update: 2025-10-01T15:32:32.499525 - 8009

# Update: 2025-10-01T15:32:32.958406 - 6580

# Update: 2025-10-01T15:32:33.020898 - 1404

# Update: 2025-10-01T15:32:34.005489 - 5961

# Update: 2025-10-01T15:32:34.939714 - 9099

# Update: 2025-10-01T15:32:35.399504 - 7995

# Update: 2025-10-01T15:32:36.002295 - 6340

# Update: 2025-10-01T15:32:36.457762 - 9600

# Update: 2025-10-01T15:32:37.043464 - 1149

# Update: 2025-10-01T15:32:37.437803 - 8979

# Update: 2025-10-01T15:32:38.100288 - 1626

# Update: 2025-10-01T15:32:38.496012 - 5348

# Update: 2025-10-01T15:32:39.126412 - 5613

# Update: 2025-10-01T15:32:39.315142 - 1761

# Update: 2025-10-01T15:32:39.570257 - 7736

# Update: 2025-10-01T15:32:39.841364 - 9306

# Update: 2025-10-01T15:32:40.296827 - 2519

# Update: 2025-10-01T15:32:40.359542 - 4259

# Update: 2025-10-01T15:32:41.700978 - 5852

# Update: 2025-10-01T15:32:41.763099 - 3263

# Update: 2025-10-01T15:32:43.729533 - 3391

# Update: 2025-10-01T15:32:43.996467 - 5040

# Update: 2025-10-01T15:32:44.455240 - 4755

# Update: 2025-10-01T15:32:46.021819 - 3059

# Updated: 2025-01-03 16:25:30 - Enhancement #2028

# Updated: 2025-01-29 22:14:31 - Enhancement #1326

# Updated: 2025-02-20 19:31:26 - Enhancement #7713

# Updated: 2025-03-16 13:24:34 - Enhancement #9939

# Updated: 2025-03-17 21:54:31 - Enhancement #1840

# Updated: 2025-04-09 19:30:48 - Enhancement #1928

# Updated: 2025-04-10 13:28:28 - Enhancement #8134

# Updated: 2025-04-13 17:05:16 - Enhancement #5915

# Updated: 2025-05-02 13:02:04 - Enhancement #2107

# Updated: 2025-05-06 12:18:11 - Enhancement #7832

# Updated: 2025-05-16 16:01:40 - Enhancement #2995

# Updated: 2025-01-16 12:21:49 - Enhancement #2372

# Updated: 2025-01-16 14:05:22 - Enhancement #6906

# Updated: 2025-01-29 18:59:05 - Enhancement #8974

# Updated: 2025-01-30 14:10:23 - Enhancement #3745

# Updated: 2025-02-07 12:39:36 - Enhancement #9323

# Updated: 2025-02-10 10:20:37 - Enhancement #6673

# Updated: 2025-02-12 20:31:54 - Enhancement #7458

# Updated: 2025-02-15 21:14:09 - Enhancement #7826

# Updated: 2025-02-17 20:16:02 - Enhancement #7823

# Updated: 2025-02-17 08:55:25 - Enhancement #6766

# Updated: 2025-02-19 17:29:06 - Enhancement #5730

# Updated: 2025-02-23 18:28:19 - Enhancement #3220

# Updated: 2025-02-26 09:51:41 - Enhancement #5128

# Updated: 2025-02-27 10:41:54 - Enhancement #5362

# Updated: 2025-02-28 11:53:50 - Enhancement #4731

# Updated: 2025-03-03 17:23:48 - Enhancement #8808
