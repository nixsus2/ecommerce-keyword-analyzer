import requests
import json
import logging
from collections import Counter
import re
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime
import os
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'keyword_analysis_{datetime.now().strftime("%Y%m%d")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# NLTK kurulumu ve veri indirme
def setup_nltk():
    """NLTK verilerini yükler ve gerekli dizinleri oluşturur"""
    import nltk
    
    try:
        # Temel NLTK paketlerini indir
        nltk.download('punkt')
        nltk.download('stopwords')
        
        # Özellikle Türkçe desteği için gerekli paketleri kontrol et
        try:
            from nltk.tokenize import word_tokenize
            word_tokenize("Türkçe test cümlesi")
        except LookupError:
            # Eğer Türkçe tokenizer yoksa, tekrar indirmeyi dene
            nltk.download('punkt')
        
        # Türkçe stopwords'ü kontrol et
        try:
            from nltk.corpus import stopwords
            stopwords.words('turkish')
        except (LookupError, OSError):
            nltk.download('stopwords')
        
        return True
        
    except Exception as e:
        logger.error(f"NLTK kurulum hatası: {e}")
        logger.info("Manuel NLTK kurulumu için: python -m nltk.downloader punkt stopwords")
        return False

@dataclass
class SearchResult:
    """Data class to store search result information"""
    title: str
    snippet: str
    url: str

class KeywordAnalyzer:
    """Enhanced keyword analysis tool for Turkish language"""
    
    def __init__(self, serp_api_key: str, num_results: int = 10, min_word_length: int = 3):
        if not serp_api_key or len(serp_api_key.strip()) == 0:
            raise ValueError("SERP API anahtarı boş olamaz")
            
        if not self.validate_api_key():
            raise ValueError("Geçersiz SERP API anahtarı")
        """
        Initialize the KeywordAnalyzer.
        
        Args:
            serp_api_key (str): SERP API key for Google search
            num_results (int): Number of search results to analyze
            min_word_length (int): Minimum length for words to be considered
        """
        self.serp_api_key = serp_api_key
        self.num_results = num_results
        self.min_word_length = min_word_length
        
        # Initialize NLTK
        if not setup_nltk():
            raise RuntimeError("NLTK kurulumu başarısız oldu!")
            
        # NLTK modüllerini import et
        from nltk.corpus import stopwords
        from nltk.tokenize import word_tokenize
        
        self.word_tokenize = word_tokenize
        try:
            self.stop_words = set(stopwords.words('turkish'))
        except Exception as e:
            logger.error(f"Türkçe stop words yüklenemedi: {e}")
            self.stop_words = set()

    def validate_api_key(self) -> bool:
        """API anahtarının geçerliliğini kontrol eder"""
        test_url = "https://serpapi.com/account"
        try:
            # API anahtarını temizle
            clean_api_key = self.serp_api_key.strip()
            response = requests.get(test_url, params={"api_key": clean_api_key})
            if response.status_code == 200:
                logger.info("API anahtarı başarıyla doğrulandı")
                return True
            else:
                logger.error(f"API doğrulama hatası: {response.status_code}")
                return False
        except Exception as e:
            logger.error(f"API doğrulama sırasında hata: {str(e)}")
            return False

    def search_google(self, keyword: str) -> Optional[List[SearchResult]]:
        """
        Perform Google search using SERP API.
        
        Args:
            keyword (str): Search keyword
            
        Returns:
            Optional[List[SearchResult]]: List of search results or None if error occurs
        """
        search_url = "https://serpapi.com/search"
        
        # API anahtarından fazladan karakterleri temizle
        clean_api_key = self.serp_api_key.strip()
        
        # URL encode edilmiş parametreler
        params = {
            "q": keyword.strip(),
            "api_key": clean_api_key,
            "num": self.num_results,
            "gl": "tr",
            "hl": "tr",
            "engine": "google"
        }
        
        try:
            response = requests.get(search_url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            results = []
            for result in data.get("organic_results", []):
                results.append(SearchResult(
                    title=result.get("title", ""),
                    snippet=result.get("snippet", ""),
                    url=result.get("link", "")
                ))
            return results
            
        except requests.RequestException as e:
            logger.error(f"SERP API error for keyword '{keyword}': {e}")
            return None
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing error: {e}")
            return None

    def clean_text(self, text: str) -> str:
        """
        Clean and normalize text.
        
        Args:
            text (str): Input text
            
        Returns:
            str: Cleaned text
        """
        # Convert to lowercase and remove special characters
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        # Remove extra whitespace
        text = ' '.join(text.split())
        return text

    def extract_keywords(self, text: str) -> List[str]:
        """
        Extract keywords from text.
        
        Args:
            text (str): Input text
            
        Returns:
            List[str]: List of extracted keywords
        """
        # Clean the text first
        clean_text = self.clean_text(text)
        
        # Türkçe karakterleri destekleyen tokenization
        try:
            words = self.word_tokenize(clean_text)
        except Exception as e:
            logger.warning(f"Tokenization hatası: {e}")
            # Basit kelime bölme ile devam et
            words = clean_text.split()
            
        # Türkçe karakterleri koruyarak filtreleme
        words = [
            word for word in words 
            if (len(word) >= self.min_word_length and 
                any(c.isalnum() for c in word) and 
                word not in self.stop_words)
        ]
        
        return words

    def analyze_keywords(self, keyword: str) -> List[Tuple[str, int]]:
        """
        Perform keyword analysis on search results.
        
        Args:
            keyword (str): Search keyword
            
        Returns:
            List[Tuple[str, int]]: List of keyword frequencies
        """
        search_results = self.search_google(keyword)
        if not search_results:
            logger.warning(f"No search results found for keyword: {keyword}")
            return []

        # Combine all text from search results
        all_text = " ".join(
            result.title + " " + result.snippet 
            for result in search_results
        )

        # Extract and count keywords
        keywords = self.extract_keywords(all_text)
        keyword_counts = Counter(keywords)

        # Return top 5 keywords
        return keyword_counts.most_common(5)

    def generate_report(self, keyword: str, analysis_results: List[Tuple[str, int]]) -> str:
        """
        Generate a formatted report of analysis results.
        
        Args:
            keyword (str): Analyzed keyword
            analysis_results: List of keyword frequencies
            
        Returns:
            str: Formatted report
        """
        report = [
            f"Anahtar Kelime Analiz Raporu: {keyword}",
            "=" * 50,
            ""
        ]

        if analysis_results:
            report.append("En Sık Geçen Anahtar Kelimeler:")
            for word, count in analysis_results:
                report.append(f"- {word}: {count} kez")
        else:
            report.append("Bu anahtar kelime için sonuç bulunamadı.")

        return "\n".join(report)

def main():
    """Main function to run the keyword analysis"""
    SERP_API_KEY = "94527f54420d26c21b9048060a90049f00f9461d630294134dcf99ec25fab903"  # Gerçek API anahtarınız
    keywords = [
        "araba kiralama",
        "evcil hayvan ürünleri",
        "online eğitim"
    ]

    try:
        analyzer = KeywordAnalyzer(SERP_API_KEY, num_results=5)
        
        for keyword in keywords:
            logger.info(f"Analyzing keyword: {keyword}")
            
            analysis_results = analyzer.analyze_keywords(keyword)
            report = analyzer.generate_report(keyword, analysis_results)
            
            print(report)
            print("\n")
            
            logger.info(f"Analysis completed for keyword: {keyword}")
            
    except Exception as e:
        logger.error(f"An error occurred during analysis: {e}")
        raise

if __name__ == "__main__":
    main()