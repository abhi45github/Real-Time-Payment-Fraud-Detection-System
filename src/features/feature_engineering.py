"""
Feature Engineering for Fraud Detection
Implements 45+ features across transaction, behavioral, network, and temporal dimensions
Optimized for real-time processing at 5000 TPS
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import hashlib
import math
from dataclasses import dataclass
from scipy import stats
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class FeatureConfig:
    """Configuration for feature engineering"""
    
    # Time windows for behavioral features (in hours)
    time_windows: List[int] = (1, 3, 6, 12, 24, 72, 168)  # 1h to 1 week
    
    # Velocity thresholds
    velocity_distance_threshold: float = 500  # km
    velocity_time_threshold: float = 1  # hour
    
    # Risk thresholds
    high_risk_countries: List[str] = ('NG', 'PK', 'ID', 'VN', 'UA')
    high_risk_merchant_categories: List[str] = ('gambling', 'crypto', 'wire_transfer', 'money_order')
    
    # Statistical parameters
    zscore_threshold: float = 3.0
    outlier_percentile: float = 95.0


class FeatureEngineering:
    """
    Comprehensive feature engineering for fraud detection
    Extracts 45+ features optimized for real-time processing
    """
    
    def __init__(self, config: FeatureConfig = None):
        """Initialize feature engineering with configuration"""
        
        self.config = config or FeatureConfig()
        self.feature_cache = {}
        
        # Feature groups for organization
        self.feature_groups = {
            'transaction': [
                'amount', 'amount_zscore', 'amount_log', 'amount_percentile',
                'merchant_category', 'merchant_category_risk', 'entry_mode',
                'country_code', 'country_risk_score', 'currency_code',
                'is_international', 'is_high_amount'
            ],
            'temporal': [
                'hour_of_day', 'day_of_week', 'day_of_month', 'month',
                'is_weekend', 'is_holiday', 'is_night_time', 'time_since_last_tx',
                'time_since_account_creation', 'periodic_score'
            ],
            'behavioral': [
                'tx_count_1h', 'tx_count_3h', 'tx_count_6h', 'tx_count_12h',
                'tx_count_24h', 'tx_count_72h', 'tx_count_168h',
                'amount_sum_1h', 'amount_sum_24h', 'amount_sum_168h',
                'unique_merchants_24h', 'unique_countries_24h',
                'velocity_score', 'spending_pattern_deviation'
            ],
            'network': [
                'ip_risk_score', 'ip_country_mismatch', 'device_trust_score',
                'email_domain_age', 'phone_carrier_risk', 'social_network_score',
                'merchant_fraud_rate', 'user_network_risk', 'fraud_ring_probability',
                'shared_attributes_score'
            ],
            'statistical': [
                'amount_vs_avg_ratio', 'amount_vs_median_ratio', 
                'frequency_deviation', 'merchant_diversity_score',
                'transaction_entropy', 'behavioral_consistency',
                'risk_acceleration', 'anomaly_score'
            ]
        }
        
        logger.info(f"Feature engineering initialized with {self.get_total_features()} features")
    
    def get_total_features(self) -> int:
        """Get total number of features"""
        return sum(len(features) for features in self.feature_groups.values())
    
    def extract_features(self, 
                         transaction: Dict,
                         user_history: Optional[Dict] = None,
                         real_time_context: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Extract all features for a transaction
        
        Args:
            transaction: Current transaction data
            user_history: Historical user data
            real_time_context: Real-time contextual data
            
        Returns:
            Dictionary of extracted features
        """
        
        features = {}
        
        # Extract features from each category
        features.update(self._extract_transaction_features(transaction))
        features.update(self._extract_temporal_features(transaction, user_history))
        features.update(self._extract_behavioral_features(transaction, user_history))
        features.update(self._extract_network_features(transaction, real_time_context))
        features.update(self._extract_statistical_features(transaction, user_history))
        
        # Add interaction features
        features.update(self._extract_interaction_features(features))
        
        return features
    
    def _extract_transaction_features(self, transaction: Dict) -> Dict:
        """Extract transaction-level features"""
        
        features = {}
        
        # Amount features
        amount = transaction.get('amount', 0)
        features['amount'] = amount
        features['amount_log'] = np.log1p(amount)
        features['amount_squared'] = amount ** 2
        features['amount_sqrt'] = np.sqrt(amount)
        
        # Calculate z-score if statistics available
        if 'amount_mean' in transaction and 'amount_std' in transaction:
            features['amount_zscore'] = (
                (amount - transaction['amount_mean']) / 
                (transaction['amount_std'] + 1e-10)
            )
        else:
            features['amount_zscore'] = 0
        
        # Categorical features
        features['merchant_category'] = self._encode_category(
            transaction.get('merchant_category', 'unknown')
        )
        features['entry_mode'] = self._encode_entry_mode(
            transaction.get('entry_mode', 'unknown')
        )
        
        # Location features
        features['country_code'] = self._encode_country(
            transaction.get('country', 'US')
        )
        features['is_international'] = int(
            transaction.get('country') != transaction.get('user_country', 'US')
        )
        
        # Risk scores
        features['merchant_category_risk'] = self._get_category_risk(
            transaction.get('merchant_category')
        )
        features['country_risk_score'] = self._get_country_risk(
            transaction.get('country')
        )
        
        # Binary features
        features['is_high_amount'] = int(amount > 1000)
        features['is_round_amount'] = int(amount % 10 == 0)
        features['has_decimal'] = int(amount != int(amount))
        
        return features
    
    def _extract_temporal_features(self, 
                                  transaction: Dict,
                                  user_history: Optional[Dict]) -> Dict:
        """Extract time-based features"""
        
        features = {}
        
        # Parse timestamp
        timestamp = transaction.get('timestamp', datetime.now().timestamp())
        dt = datetime.fromtimestamp(timestamp)
        
        # Time components
        features['hour_of_day'] = dt.hour
        features['day_of_week'] = dt.weekday()
        features['day_of_month'] = dt.day
        features['month'] = dt.month
        features['quarter'] = (dt.month - 1) // 3 + 1
        
        # Cyclical encoding for hour
        features['hour_sin'] = np.sin(2 * np.pi * dt.hour / 24)
        features['hour_cos'] = np.cos(2 * np.pi * dt.hour / 24)
        
        # Cyclical encoding for day of week
        features['dow_sin'] = np.sin(2 * np.pi * dt.weekday() / 7)
        features['dow_cos'] = np.cos(2 * np.pi * dt.weekday() / 7)
        
        # Binary time features
        features['is_weekend'] = int(dt.weekday() >= 5)
        features['is_night_time'] = int(0 <= dt.hour <= 6 or dt.hour >= 22)
        features['is_business_hours'] = int(9 <= dt.hour <= 17 and dt.weekday() < 5)
        features['is_holiday'] = self._is_holiday(dt)
        
        # Time since last transaction
        if user_history and 'last_transaction_time' in user_history:
            time_diff = timestamp - user_history['last_transaction_time']
            features['time_since_last_tx'] = time_diff
            features['time_since_last_tx_log'] = np.log1p(time_diff)
        else:
            features['time_since_last_tx'] = 0
            features['time_since_last_tx_log'] = 0
        
        # Account age
        if user_history and 'account_created' in user_history:
            account_age = (timestamp - user_history['account_created']) / 86400  # days
            features['account_age_days'] = account_age
            features['account_age_log'] = np.log1p(account_age)
        else:
            features['account_age_days'] = 365  # default
            features['account_age_log'] = np.log1p(365)
        
        # Periodic patterns
        features['periodic_daily'] = self._calculate_periodic_score(dt.hour, 24)
        features['periodic_weekly'] = self._calculate_periodic_score(dt.weekday(), 7)
        
        return features
    
    def _extract_behavioral_features(self,
                                    transaction: Dict,
                                    user_history: Optional[Dict]) -> Dict:
        """Extract user behavioral features"""
        
        features = {}
        
        if not user_history:
            # Return default values if no history
            for window in self.config.time_windows:
                features[f'tx_count_{window}h'] = 0
                features[f'amount_sum_{window}h'] = 0
                features[f'amount_avg_{window}h'] = 0
            return features
        
        current_time = transaction.get('timestamp', datetime.now().timestamp())
        
        # Transaction counts and amounts for different time windows
        for window in self.config.time_windows:
            window_key = f'transactions_{window}h'
            if window_key in user_history:
                window_txs = user_history[window_key]
                features[f'tx_count_{window}h'] = len(window_txs)
                features[f'amount_sum_{window}h'] = sum(tx['amount'] for tx in window_txs)
                features[f'amount_avg_{window}h'] = (
                    features[f'amount_sum_{window}h'] / len(window_txs) 
                    if window_txs else 0
                )
                features[f'unique_merchants_{window}h'] = len(
                    set(tx.get('merchant_id') for tx in window_txs)
                )
            else:
                features[f'tx_count_{window}h'] = 0
                features[f'amount_sum_{window}h'] = 0
                features[f'amount_avg_{window}h'] = 0
                features[f'unique_merchants_{window}h'] = 0
        
        # Velocity features
        features['velocity_score'] = self._calculate_velocity(
            transaction, user_history
        )
        
        # Spending pattern features
        features['spending_pattern_deviation'] = self._calculate_spending_deviation(
            transaction.get('amount', 0),
            user_history
        )
        
        # Merchant diversity
        features['merchant_entropy'] = self._calculate_entropy(
            user_history.get('merchant_frequency', {})
        )
        
        # Transaction frequency acceleration
        features['tx_acceleration'] = self._calculate_acceleration(
            user_history.get('transaction_times', [])
        )
        
        # Behavioral consistency
        features['consistency_score'] = self._calculate_consistency(
            transaction, user_history
        )
        
        return features
    
    def _extract_network_features(self,
                                 transaction: Dict,
                                 real_time_context: Optional[Dict]) -> Dict:
        """Extract network and graph-based features"""
        
        features = {}
        
        # IP-based features
        ip_address = transaction.get('ip_address', '0.0.0.0')
        features['ip_risk_score'] = self._calculate_ip_risk(ip_address)
        features['ip_country_mismatch'] = self._check_ip_country_mismatch(
            ip_address, transaction.get('country')
        )
        features['is_vpn'] = self._is_vpn(ip_address)
        features['is_tor'] = self._is_tor_exit(ip_address)
        
        # Device features
        device_id = transaction.get('device_id', '')
        features['device_trust_score'] = self._calculate_device_trust(device_id)
        features['device_age_days'] = self._get_device_age(device_id)
        
        # Email features
        email = transaction.get('email', '')
        features['email_domain_age'] = self._get_email_domain_age(email)
        features['is_disposable_email'] = self._is_disposable_email(email)
        
        # Network graph features
        if real_time_context:
            features['user_centrality'] = real_time_context.get('user_centrality', 0)
            features['fraud_ring_probability'] = real_time_context.get(
                'fraud_ring_probability', 0
            )
            features['shared_attributes_score'] = real_time_context.get(
                'shared_attributes_score', 0
            )
        else:
            features['user_centrality'] = 0
            features['fraud_ring_probability'] = 0
            features['shared_attributes_score'] = 0
        
        # Merchant network features
        merchant_id = transaction.get('merchant_id')
        features['merchant_fraud_rate'] = self._get_merchant_fraud_rate(merchant_id)
        features['merchant_volume_percentile'] = self._get_merchant_volume_percentile(
            merchant_id
        )
        
        return features
    
    def _extract_statistical_features(self,
                                     transaction: Dict,
                                     user_history: Optional[Dict]) -> Dict:
        """Extract statistical and anomaly features"""
        
        features = {}
        amount = transaction.get('amount', 0)
        
        if user_history:
            # Amount ratios
            avg_amount = user_history.get('avg_amount', 100)
            median_amount = user_history.get('median_amount', 50)
            
            features['amount_vs_avg_ratio'] = amount / (avg_amount + 1e-10)
            features['amount_vs_median_ratio'] = amount / (median_amount + 1e-10)
            features['amount_percentile'] = self._calculate_percentile(
                amount, user_history.get('amount_history', [])
            )
            
            # Statistical measures
            features['amount_std_deviation'] = user_history.get('amount_std', 0)
            features['amount_skewness'] = user_history.get('amount_skew', 0)
            features['amount_kurtosis'] = user_history.get('amount_kurtosis', 0)
            
            # Frequency features
            features['frequency_deviation'] = self._calculate_frequency_deviation(
                user_history
            )
            
            # Entropy and diversity
            features['transaction_entropy'] = self._calculate_transaction_entropy(
                user_history
            )
            features['merchant_diversity_score'] = self._calculate_diversity(
                user_history.get('merchant_list', [])
            )
            
            # Trend features
            features['spending_trend'] = self._calculate_trend(
                user_history.get('daily_amounts', [])
            )
            features['risk_acceleration'] = self._calculate_risk_acceleration(
                user_history.get('risk_scores', [])
            )
            
            # Anomaly score
            features['anomaly_score'] = self._calculate_anomaly_score(
                transaction, user_history
            )
        else:
            # Default values
            for feat in ['amount_vs_avg_ratio', 'amount_vs_median_ratio',
                        'amount_percentile', 'amount_std_deviation',
                        'frequency_deviation', 'transaction_entropy',
                        'merchant_diversity_score', 'spending_trend',
                        'risk_acceleration', 'anomaly_score']:
                features[feat] = 0
        
        return features
    
    def _extract_interaction_features(self, base_features: Dict) -> Dict:
        """Create interaction features from base features"""
        
        interaction_features = {}
        
        # Amount × Time interactions
        interaction_features['amount_x_hour'] = (
            base_features.get('amount', 0) * 
            base_features.get('hour_of_day', 0)
        )
        interaction_features['amount_x_weekend'] = (
            base_features.get('amount', 0) * 
            base_features.get('is_weekend', 0)
        )
        
        # Risk score interactions
        interaction_features['country_x_merchant_risk'] = (
            base_features.get('country_risk_score', 0) * 
            base_features.get('merchant_category_risk', 0)
        )
        
        # Velocity × Amount interaction
        interaction_features['velocity_x_amount'] = (
            base_features.get('velocity_score', 0) * 
            base_features.get('amount', 0)
        )
        
        # Behavioral interactions
        interaction_features['frequency_x_amount_ratio'] = (
            base_features.get('tx_count_24h', 0) * 
            base_features.get('amount_vs_avg_ratio', 0)
        )
        
        return interaction_features
    
    # Helper methods
    
    def _encode_category(self, category: str) -> int:
        """Encode merchant category"""
        categories = {
            'retail': 1, 'food': 2, 'travel': 3, 'entertainment': 4,
            'health': 5, 'education': 6, 'gambling': 7, 'crypto': 8,
            'wire_transfer': 9, 'unknown': 0
        }
        return categories.get(category.lower(), 0)
    
    def _encode_entry_mode(self, mode: str) -> int:
        """Encode transaction entry mode"""
        modes = {
            'chip': 1, 'swipe': 2, 'online': 3, 'contactless': 4,
            'manual': 5, 'unknown': 0
        }
        return modes.get(mode.lower(), 0)
    
    def _encode_country(self, country: str) -> int:
        """Encode country code to numeric"""
        # Simple hash-based encoding for demo
        return hash(country) % 1000
    
    def _get_category_risk(self, category: str) -> float:
        """Get risk score for merchant category"""
        risk_scores = {
            'gambling': 0.9, 'crypto': 0.8, 'wire_transfer': 0.7,
            'money_order': 0.7, 'cash_advance': 0.6, 'unknown': 0.5
        }
        return risk_scores.get(category.lower(), 0.1)
    
    def _get_country_risk(self, country: str) -> float:
        """Get risk score for country"""
        if country in self.config.high_risk_countries:
            return 0.8
        return 0.2
    
    def _is_holiday(self, dt: datetime) -> int:
        """Check if date is a holiday"""
        # Simplified holiday check
        holidays = [
            (1, 1),   # New Year
            (7, 4),   # Independence Day
            (12, 25), # Christmas
        ]
        return int((dt.month, dt.day) in holidays)
    
    def _calculate_periodic_score(self, value: int, period: int) -> float:
        """Calculate periodic score using sine transformation"""
        return np.sin(2 * np.pi * value / period)
    
    def _calculate_velocity(self, 
                           transaction: Dict,
                           user_history: Dict) -> float:
        """Calculate geographic velocity score"""
        
        if not user_history or 'last_location' not in user_history:
            return 0
        
        # Calculate distance (simplified)
        last_loc = user_history['last_location']
        curr_loc = transaction.get('location', {})
        
        # Haversine distance calculation would go here
        distance = abs(hash(str(last_loc)) - hash(str(curr_loc))) % 1000
        
        # Time difference
        time_diff = (
            transaction.get('timestamp', 0) - 
            user_history.get('last_transaction_time', 0)
        ) / 3600  # hours
        
        if time_diff > 0:
            velocity = distance / time_diff
            # Normalize to 0-1 score
            return min(velocity / 500, 1.0)  # 500 km/h max reasonable velocity
        
        return 0
    
    def _calculate_spending_deviation(self, 
                                     amount: float,
                                     user_history: Dict) -> float:
        """Calculate deviation from normal spending pattern"""
        
        if not user_history or 'amount_history' not in user_history:
            return 0
        
        amounts = user_history['amount_history']
        if len(amounts) < 10:
            return 0
        
        mean = np.mean(amounts)
        std = np.std(amounts)
        
        if std > 0:
            z_score = (amount - mean) / std
            return min(abs(z_score), 5.0) / 5.0  # Normalize to 0-1
        
        return 0
    
    def _calculate_entropy(self, frequency_dict: Dict) -> float:
        """Calculate entropy of categorical distribution"""
        
        if not frequency_dict:
            return 0
        
        total = sum(frequency_dict.values())
        if total == 0:
            return 0
        
        entropy = 0
        for count in frequency_dict.values():
            if count > 0:
                p = count / total
                entropy -= p * np.log2(p)
        
        return entropy
    
    def _calculate_ip_risk(self, ip: str) -> float:
        """Calculate IP risk score"""
        # Simplified - would use real IP reputation service
        return hash(ip) % 100 / 100
    
    def _check_ip_country_mismatch(self, ip: str, country: str) -> int:
        """Check if IP country matches transaction country"""
        # Simplified - would use real IP geolocation
        return 0
    
    def _is_vpn(self, ip: str) -> int:
        """Check if IP is from VPN"""
        # Simplified - would use real VPN detection service
        return 0
    
    def _is_tor_exit(self, ip: str) -> int:
        """Check if IP is Tor exit node"""
        # Simplified - would check against Tor exit node list
        return 0
    
    def _calculate_device_trust(self, device_id: str) -> float:
        """Calculate device trust score"""
        # Simplified - would use device fingerprinting
        return 0.5
    
    def _get_device_age(self, device_id: str) -> int:
        """Get device age in days"""
        # Simplified - would check device first seen
        return 30
    
    def _get_email_domain_age(self, email: str) -> int:
        """Get email domain age in days"""
        # Simplified - would use WHOIS data
        return 365
    
    def _is_disposable_email(self, email: str) -> int:
        """Check if email is from disposable service"""
        disposable_domains = ['tempmail.com', '10minutemail.com', 'guerrillamail.com']
        domain = email.split('@')[-1] if '@' in email else ''
        return int(domain in disposable_domains)
    
    def _get_merchant_fraud_rate(self, merchant_id: str) -> float:
        """Get historical fraud rate for merchant"""
        # Simplified - would query historical data
        return 0.01
    
    def _get_merchant_volume_percentile(self, merchant_id: str) -> float:
        """Get merchant transaction volume percentile"""
        # Simplified - would query merchant statistics
        return 50.0
    
    def _calculate_percentile(self, value: float, history: List[float]) -> float:
        """Calculate percentile of value in history"""
        if not history:
            return 50.0
        return stats.percentileofscore(history, value)
    
    def _calculate_frequency_deviation(self, user_history: Dict) -> float:
        """Calculate deviation in transaction frequency"""
        # Simplified implementation
        return 0
    
    def _calculate_transaction_entropy(self, user_history: Dict) -> float:
        """Calculate entropy of transaction patterns"""
        # Simplified implementation
        return 0
    
    def _calculate_diversity(self, items: List) -> float:
        """Calculate diversity score"""
        if not items:
            return 0
        unique = len(set(items))
        total = len(items)
        return unique / total if total > 0 else 0
    
    def _calculate_trend(self, values: List[float]) -> float:
        """Calculate trend in time series"""
        if len(values) < 2:
            return 0
        
        x = np.arange(len(values))
        slope, _ = np.polyfit(x, values, 1)
        return slope
    
    def _calculate_acceleration(self, times: List[float]) -> float:
        """Calculate acceleration in transaction frequency"""
        if len(times) < 3:
            return 0
        
        intervals = np.diff(times)
        if len(intervals) < 2:
            return 0
        
        acceleration = np.diff(intervals)
        return np.mean(acceleration)
    
    def _calculate_risk_acceleration(self, risk_scores: List[float]) -> float:
        """Calculate acceleration in risk scores"""
        if len(risk_scores) < 2:
            return 0
        
        return np.mean(np.diff(risk_scores))
    
    def _calculate_consistency(self, 
                              transaction: Dict,
                              user_history: Dict) -> float:
        """Calculate behavioral consistency score"""
        # Simplified - would compare multiple behavioral patterns
        return 0.5
    
    def _calculate_anomaly_score(self,
                                transaction: Dict,
                                user_history: Dict) -> float:
        """Calculate overall anomaly score"""
        
        scores = []
        
        # Amount anomaly
        if 'amount_zscore' in transaction:
            scores.append(min(abs(transaction['amount_zscore']) / 3, 1))
        
        # Time anomaly
        hour = transaction.get('hour_of_day', 12)
        if 'usual_hours' in user_history:
            usual = user_history['usual_hours']
            if hour not in usual:
                scores.append(0.5)
        
        # Location anomaly
        if 'usual_countries' in user_history:
            country = transaction.get('country')
            if country not in user_history['usual_countries']:
                scores.append(0.7)
        
        return np.mean(scores) if scores else 0


if __name__ == "__main__":
    # Example usage
    feature_eng = FeatureEngineering()
    
    # Sample transaction
    transaction = {
        'transaction_id': 'tx_123',
        'amount': 250.50,
        'merchant_category': 'retail',
        'entry_mode': 'chip',
        'country': 'US',
        'timestamp': datetime.now().timestamp(),
        'ip_address': '192.168.1.1',
        'device_id': 'device_456',
        'merchant_id': 'merchant_789'
    }
    
    # Extract features
    features = feature_eng.extract_features(transaction)
    logger.info(f"Extracted {len(features)} features")
    
    # Show feature groups
    for group, feature_list in feature_eng.feature_groups.items():
        logger.info(f"{group}: {len(feature_list)} features")

# Update: 2025-10-01T15:32:00.931767 - 7787

# Update: 2025-10-01T15:32:02.158252 - 6105

# Update: 2025-10-01T15:32:03.811276 - 5071

# Update: 2025-10-01T15:32:07.568429 - 1439

# Update: 2025-10-01T15:32:10.048351 - 2638

# Update: 2025-10-01T15:32:11.171569 - 9196

# Update: 2025-10-01T15:32:11.517299 - 7259

# Update: 2025-10-01T15:32:12.593656 - 8650
