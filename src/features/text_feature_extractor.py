"""
Text Feature Extractor for semantic embeddings and statistical features

This module implements comprehensive text feature extraction including pre-trained
language model embeddings, statistical text features, and categorical feature extraction.
"""

import re
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass
import string
from collections import Counter
import math

try:
    from transformers import AutoTokenizer, AutoModel
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logging.warning("Transformers library not available. Text embeddings will use fallback methods.")

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import TruncatedSVD
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logging.warning("Scikit-learn not available. Some features will be limited.")


@dataclass
class TextFeatures:
    """Container for extracted text features"""
    embeddings: Optional[np.ndarray] = None
    statistical_features: Optional[Dict[str, float]] = None
    categorical_features: Optional[Dict[str, Any]] = None
    readability_features: Optional[Dict[str, float]] = None


class TextFeatureExtractor:
    """
    Comprehensive text feature extractor for product catalog content.
    
    Integrates pre-trained language models (BERT/RoBERTa) for semantic embeddings,
    creates statistical text features, and implements categorical feature extraction.
    """
    
    def __init__(self, 
                 model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 max_length: int = 512,
                 device: str = "auto",
                 logger: Optional[logging.Logger] = None):
        """
        Initialize text feature extractor.
        
        Args:
            model_name: Name of the pre-trained model to use
            max_length: Maximum sequence length for tokenization
            device: Device to use for model inference ("auto", "cpu", "cuda")
            logger: Optional logger instance
        """
        self.logger = logger or logging.getLogger(__name__)
        self.model_name = model_name
        self.max_length = max_length
        
        # Initialize device
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() and TRANSFORMERS_AVAILABLE else "cpu"
        else:
            self.device = device
        
        # Initialize model and tokenizer
        self.tokenizer = None
        self.model = None
        self._initialize_model()
        
        # Initialize TF-IDF vectorizer for fallback embeddings
        self.tfidf_vectorizer = None
        self.tfidf_svd = None
        if SKLEARN_AVAILABLE:
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=1000,
                stop_words='english',
                ngram_range=(1, 2),
                min_df=2
            )
            self.tfidf_svd = TruncatedSVD(n_components=384)  # Match embedding dimension
        
        # Brand and category patterns
        self.brand_patterns = self._initialize_brand_patterns()
        self.category_patterns = self._initialize_category_patterns()
        
        # Statistical feature extractors
        self.readability_calculator = ReadabilityCalculator()
    
    def _initialize_model(self):
        """Initialize the pre-trained language model."""
        if not TRANSFORMERS_AVAILABLE:
            self.logger.warning("Transformers not available, using fallback methods")
            return
        
        try:
            self.logger.info(f"Loading model: {self.model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()
            self.logger.info(f"Model loaded successfully on {self.device}")
        except Exception as e:
            self.logger.error(f"Failed to load model {self.model_name}: {e}")
            self.tokenizer = None
            self.model = None
    
    def _initialize_brand_patterns(self) -> List[Dict[str, Any]]:
        """Initialize brand detection patterns."""
        return [
            {'pattern': r'\b(apple|samsung|sony|lg|microsoft|google|amazon|nike|adidas)\b', 'category': 'tech_fashion'},
            {'pattern': r'\b(coca-cola|pepsi|nestle|unilever|procter|johnson|kraft)\b', 'category': 'consumer_goods'},
            {'pattern': r'\b(toyota|ford|bmw|mercedes|honda|volkswagen)\b', 'category': 'automotive'},
            {'pattern': r'\b(walmart|target|costco|ikea|home depot)\b', 'category': 'retail'},
        ]
    
    def _initialize_category_patterns(self) -> List[Dict[str, Any]]:
        """Initialize product category detection patterns."""
        return [
            {'pattern': r'\b(phone|smartphone|tablet|laptop|computer|electronics?)\b', 'category': 'electronics'},
            {'pattern': r'\b(shirt|pants|dress|shoes|clothing|apparel|fashion)\b', 'category': 'clothing'},
            {'pattern': r'\b(food|snack|beverage|drink|grocery|organic)\b', 'category': 'food_beverage'},
            {'pattern': r'\b(book|magazine|novel|textbook|literature)\b', 'category': 'books'},
            {'pattern': r'\b(toy|game|puzzle|doll|action figure)\b', 'category': 'toys'},
            {'pattern': r'\b(furniture|chair|table|sofa|bed|home)\b', 'category': 'furniture'},
            {'pattern': r'\b(beauty|cosmetic|skincare|makeup|shampoo)\b', 'category': 'beauty'},
            {'pattern': r'\b(tool|hardware|drill|hammer|screwdriver)\b', 'category': 'tools'},
            {'pattern': r'\b(car|auto|vehicle|tire|engine|automotive)\b', 'category': 'automotive'},
            {'pattern': r'\b(health|vitamin|supplement|medicine|medical)\b', 'category': 'health'},
        ]
    
    def generate_text_embeddings(self, text: str) -> np.ndarray:
        """
        Generate text embeddings using pre-trained language models.
        
        Args:
            text: Input text to embed
            
        Returns:
            Numpy array of embeddings
        """
        if not text or pd.isna(text):
            return np.zeros(384)  # Default embedding dimension
        
        text = str(text).strip()
        
        # Use transformer model if available
        if self.model is not None and self.tokenizer is not None:
            return self._generate_transformer_embeddings(text)
        
        # Fallback to TF-IDF embeddings
        elif SKLEARN_AVAILABLE and self.tfidf_vectorizer is not None:
            return self._generate_tfidf_embeddings(text)
        
        # Final fallback to simple statistical features
        else:
            return self._generate_statistical_embeddings(text)
    
    def _generate_transformer_embeddings(self, text: str) -> np.ndarray:
        """Generate embeddings using transformer model."""
        try:
            # Tokenize text
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                max_length=self.max_length,
                truncation=True,
                padding=True
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate embeddings
            with torch.no_grad():
                outputs = self.model(**inputs)
                # Use mean pooling of last hidden states
                embeddings = outputs.last_hidden_state.mean(dim=1)
                embeddings = embeddings.cpu().numpy().flatten()
            
            return embeddings
            
        except Exception as e:
            self.logger.error(f"Error generating transformer embeddings: {e}")
            return np.zeros(384)
    
    def _generate_tfidf_embeddings(self, text: str) -> np.ndarray:
        """Generate embeddings using TF-IDF and SVD."""
        try:
            # Transform text to TF-IDF
            tfidf_vector = self.tfidf_vectorizer.transform([text])
            
            # Reduce dimensions using SVD
            if self.tfidf_svd is not None:
                embeddings = self.tfidf_svd.transform(tfidf_vector)
                return embeddings.flatten()
            else:
                return tfidf_vector.toarray().flatten()[:384]  # Truncate to 384 dimensions
                
        except Exception as e:
            self.logger.error(f"Error generating TF-IDF embeddings: {e}")
            return np.zeros(384)
    
    def _generate_statistical_embeddings(self, text: str) -> np.ndarray:
        """Generate simple statistical embeddings as final fallback."""
        features = self.extract_statistical_features(text)
        
        # Convert statistical features to fixed-size vector
        embedding = np.zeros(384)
        feature_values = list(features.values())
        
        # Fill embedding with statistical features (repeated to reach 384 dimensions)
        for i in range(384):
            if i < len(feature_values):
                embedding[i] = feature_values[i]
            else:
                embedding[i] = feature_values[i % len(feature_values)]
        
        return embedding
    
    def extract_statistical_features(self, text: str) -> Dict[str, float]:
        """
        Extract statistical text features.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary of statistical features
        """
        if not text or pd.isna(text):
            return self._empty_statistical_features()
        
        text = str(text).strip()
        words = text.split()
        sentences = re.split(r'[.!?]+', text)
        
        features = {
            # Basic length features
            'char_count': len(text),
            'word_count': len(words),
            'sentence_count': len([s for s in sentences if s.strip()]),
            
            # Average features
            'avg_word_length': np.mean([len(word) for word in words]) if words else 0,
            'avg_sentence_length': len(words) / max(len(sentences), 1),
            
            # Character type ratios
            'uppercase_ratio': sum(1 for c in text if c.isupper()) / max(len(text), 1),
            'lowercase_ratio': sum(1 for c in text if c.islower()) / max(len(text), 1),
            'digit_ratio': sum(1 for c in text if c.isdigit()) / max(len(text), 1),
            'punctuation_ratio': sum(1 for c in text if c in string.punctuation) / max(len(text), 1),
            'whitespace_ratio': sum(1 for c in text if c.isspace()) / max(len(text), 1),
            
            # Word complexity features
            'unique_word_ratio': len(set(words)) / max(len(words), 1),
            'long_word_ratio': sum(1 for word in words if len(word) > 6) / max(len(words), 1),
            'short_word_ratio': sum(1 for word in words if len(word) <= 3) / max(len(words), 1),
            
            # Special character features
            'has_numbers': 1.0 if any(c.isdigit() for c in text) else 0.0,
            'has_special_chars': 1.0 if any(c in '!@#$%^&*()' for c in text) else 0.0,
            'has_urls': 1.0 if re.search(r'http[s]?://|www\.', text) else 0.0,
            'has_email': 1.0 if re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text) else 0.0,
        }
        
        # Add readability features
        readability_features = self.readability_calculator.calculate_readability(text)
        features.update(readability_features)
        
        return features
    
    def _empty_statistical_features(self) -> Dict[str, float]:
        """Return empty statistical features for missing text."""
        return {
            'char_count': 0.0, 'word_count': 0.0, 'sentence_count': 0.0,
            'avg_word_length': 0.0, 'avg_sentence_length': 0.0,
            'uppercase_ratio': 0.0, 'lowercase_ratio': 0.0, 'digit_ratio': 0.0,
            'punctuation_ratio': 0.0, 'whitespace_ratio': 0.0,
            'unique_word_ratio': 0.0, 'long_word_ratio': 0.0, 'short_word_ratio': 0.0,
            'has_numbers': 0.0, 'has_special_chars': 0.0, 'has_urls': 0.0, 'has_email': 0.0,
            'flesch_reading_ease': 0.0, 'flesch_kincaid_grade': 0.0,
            'automated_readability_index': 0.0, 'coleman_liau_index': 0.0,
        }
    
    def extract_categorical_features(self, text: str) -> Dict[str, Any]:
        """
        Extract categorical features like brands and product categories.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary of categorical features
        """
        if not text or pd.isna(text):
            return {'detected_brands': [], 'detected_categories': [], 'brand_count': 0, 'category_count': 0}
        
        text_lower = str(text).lower()
        
        # Detect brands
        detected_brands = []
        for brand_info in self.brand_patterns:
            matches = re.findall(brand_info['pattern'], text_lower, re.IGNORECASE)
            for match in matches:
                detected_brands.append({
                    'brand': match,
                    'category': brand_info['category']
                })
        
        # Detect product categories
        detected_categories = []
        for category_info in self.category_patterns:
            if re.search(category_info['pattern'], text_lower, re.IGNORECASE):
                detected_categories.append(category_info['category'])
        
        # Remove duplicates
        detected_categories = list(set(detected_categories))
        
        return {
            'detected_brands': detected_brands,
            'detected_categories': detected_categories,
            'brand_count': len(detected_brands),
            'category_count': len(detected_categories),
            'has_brand': len(detected_brands) > 0,
            'has_category': len(detected_categories) > 0,
        }
    
    def create_text_features(self, df: pd.DataFrame, 
                           content_column: str = 'catalog_content') -> np.ndarray:
        """
        Create comprehensive text feature matrix for DataFrame.
        
        Args:
            df: DataFrame containing text content
            content_column: Name of the column containing text content
            
        Returns:
            Feature matrix as numpy array
        """
        if content_column not in df.columns:
            raise ValueError(f"Column '{content_column}' not found in DataFrame")
        
        self.logger.info(f"Creating text features for {len(df)} samples")
        
        # Fit TF-IDF vectorizer if using sklearn fallback
        if SKLEARN_AVAILABLE and self.tfidf_vectorizer is not None and self.model is None:
            texts = df[content_column].fillna("").astype(str).tolist()
            self.logger.info("Fitting TF-IDF vectorizer...")
            tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)
            self.tfidf_svd.fit(tfidf_matrix)
        
        # Extract features for each sample
        all_embeddings = []
        all_statistical = []
        all_categorical = []
        
        for idx, row in df.iterrows():
            try:
                text = row[content_column]
                
                # Generate embeddings
                embeddings = self.generate_text_embeddings(text)
                all_embeddings.append(embeddings)
                
                # Extract statistical features
                statistical = self.extract_statistical_features(text)
                all_statistical.append(list(statistical.values()))
                
                # Extract categorical features (convert to numerical)
                categorical = self.extract_categorical_features(text)
                categorical_numerical = [
                    categorical['brand_count'],
                    categorical['category_count'],
                    float(categorical['has_brand']),
                    float(categorical['has_category']),
                ]
                all_categorical.append(categorical_numerical)
                
            except Exception as e:
                self.logger.error(f"Error processing row {idx}: {e}")
                # Add empty features for failed rows
                all_embeddings.append(np.zeros(384))
                all_statistical.append([0.0] * len(self._empty_statistical_features()))
                all_categorical.append([0.0, 0.0, 0.0, 0.0])
        
        # Combine all features
        embeddings_matrix = np.array(all_embeddings)
        statistical_matrix = np.array(all_statistical)
        categorical_matrix = np.array(all_categorical)
        
        # Concatenate all feature types
        feature_matrix = np.concatenate([
            embeddings_matrix,
            statistical_matrix,
            categorical_matrix
        ], axis=1)
        
        self.logger.info(f"Text feature extraction completed. "
                        f"Feature matrix shape: {feature_matrix.shape}")
        
        return feature_matrix
    
    def get_feature_names(self) -> List[str]:
        """Get names of all extracted features."""
        # Embedding feature names
        embedding_names = [f'embedding_{i}' for i in range(384)]
        
        # Statistical feature names
        statistical_names = list(self._empty_statistical_features().keys())
        
        # Categorical feature names
        categorical_names = ['brand_count', 'category_count', 'has_brand', 'has_category']
        
        return embedding_names + statistical_names + categorical_names


class ReadabilityCalculator:
    """Calculate readability scores for text."""
    
    def calculate_readability(self, text: str) -> Dict[str, float]:
        """Calculate various readability metrics."""
        if not text or pd.isna(text):
            return {
                'flesch_reading_ease': 0.0,
                'flesch_kincaid_grade': 0.0,
                'automated_readability_index': 0.0,
                'coleman_liau_index': 0.0,
            }
        
        text = str(text).strip()
        
        # Count sentences, words, and syllables
        sentences = len(re.split(r'[.!?]+', text))
        words = len(text.split())
        syllables = self._count_syllables(text)
        characters = len(re.sub(r'\s', '', text))
        
        if sentences == 0 or words == 0:
            return {
                'flesch_reading_ease': 0.0,
                'flesch_kincaid_grade': 0.0,
                'automated_readability_index': 0.0,
                'coleman_liau_index': 0.0,
            }
        
        # Calculate readability scores
        flesch_reading_ease = 206.835 - (1.015 * words / sentences) - (84.6 * syllables / words)
        flesch_kincaid_grade = (0.39 * words / sentences) + (11.8 * syllables / words) - 15.59
        
        automated_readability_index = (4.71 * characters / words) + (0.5 * words / sentences) - 21.43
        coleman_liau_index = (0.0588 * (characters / words * 100)) - (0.296 * (sentences / words * 100)) - 15.8
        
        return {
            'flesch_reading_ease': max(0, min(100, flesch_reading_ease)),
            'flesch_kincaid_grade': max(0, flesch_kincaid_grade),
            'automated_readability_index': max(0, automated_readability_index),
            'coleman_liau_index': max(0, coleman_liau_index),
        }
    
    def _count_syllables(self, text: str) -> int:
        """Estimate syllable count in text."""
        words = re.findall(r'\b\w+\b', text.lower())
        syllable_count = 0
        
        for word in words:
            syllables = self._count_word_syllables(word)
            syllable_count += syllables
        
        return max(1, syllable_count)
    
    def _count_word_syllables(self, word: str) -> int:
        """Estimate syllables in a single word."""
        word = word.lower()
        vowels = 'aeiouy'
        syllable_count = 0
        prev_was_vowel = False
        
        for char in word:
            is_vowel = char in vowels
            if is_vowel and not prev_was_vowel:
                syllable_count += 1
            prev_was_vowel = is_vowel
        
        # Handle silent 'e'
        if word.endswith('e') and syllable_count > 1:
            syllable_count -= 1
        
        return max(1, syllable_count)