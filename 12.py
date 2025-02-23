import requests
import json
import logging
from collections import Counter
import re
from nltk.corpus import stopwords
from nltk import ngrams
from pytrends.request import TrendReq
from bs4 import BeautifulSoup

import nltk
nltk.download('stopwords')

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class KeywordAnalyzer:
    def __init__(self, serp_api_key, num_results=10):
        self.serp_api_key = serp_api_key
        self.num_results = num_results
        self.session = requests.Session()
        self.stop_words = set(stopwords.words('turkish'))  # Türkçe stop words
        self.pytrends = TrendReq(hl='tr-TR', tz=360) # Google Trends bağlantısı

        if not self.serp_api_key:
            raise ValueError("94527f54420d26c21b9048060a90049f00f9461d630294134dcf99ec25fab903")

    def search_google(self, keyword):
        """Google SERP API'si kullanarak arama yapar."""
        search_url = "https://serpapi.com/search"
        params = {
            "q": keyword,
            "api_key": self.serp_api_key,
            "num": self.num_results,
            "gl": "tr",
            "hl": "tr"
        }

        try:
            response = self.session.get(search_url, params=params, timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            logger.error(f"Google SERP API error for keyword '{keyword}': {e}")
            return None

    def extract_keywords_from_results(self, search_results):
        """Arama sonuçlarından anahtar kelimeleri çıkarır."""
        keywords = []
        if not search_results:
            return keywords

        organic_results = search_results.get("organic_results", [])
        for result in organic_results:
            if "snippet" in result:
                snippet_keywords = re.findall(r'\b\w+\b', result["snippet"].lower())
                keywords.extend(snippet_keywords)
            if "title" in result:
                title_keywords = re.findall(r'\b\w+\b', result["title"].lower())
                keywords.extend(title_keywords)

        # İlgili sorular ve aramalar bölümünden de anahtar kelimeler ekle
        related_questions = search_results.get("related_questions", [])
        for question in related_questions:
            question_keywords = re.findall(r'\b\w+\b', question["question"].lower())
            keywords.extend(question_keywords)

        related_searches = search_results.get("related_searches", [])
        if related_searches:
            for search in related_searches:
                search_keywords = re.findall(r'\b\w+\b', search["query"].lower())
                keywords.extend(search_keywords)

        return keywords

    def get_google_trends_data(self, keyword):
        """Google Trends'ten arama hacmi verilerini alır."""
        try:
            self.pytrends.build_payload([keyword], cat=0, timeframe='today 3-m', geo='TR', gprop='')
            data = self.pytrends.interest_over_time()
            if not data.empty:
                return int(data[keyword].mean())  # Ortalama arama hacmi
            else:
                return 0
        except Exception as e:
            logger.warning(f"Google Trends error for keyword '{keyword}': {e}")
            return 0

    def scrape_amazon_suggestions(self, keyword):
        """Amazon'dan arama önerilerini çek."""
        url = f"https://www.amazon.com/s?k={keyword}"
        try:
            response = self.session.get(url)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            # Amazon'un HTML yapısına göre bu kısmı güncelleyin
            suggestions = [a.text.strip() for a in soup.find_all('a', class_='s-suggestion-ellipsis-span')]
            return suggestions
        except Exception as e:
            logger.warning(f"Amazon öneri çekme hatası: {e}")
            return []

    def analyze_keywords(self, keyword):
        """Anahtar kelime analizi yapar."""
        search_results = self.search_google(keyword)
        if not search_results:
            return {}

        extracted_keywords = self.extract_keywords_from_results(search_results)
        amazon_suggestions = self.scrape_amazon_suggestions(keyword)
        all_keywords = extracted_keywords + amazon_suggestions

        # Stop words'leri filtrele
        filtered_keywords = [word for word in all_keywords if word not in self.stop_words]

        # Tek kelimelerin sıklığı
        keyword_counts = Counter(filtered_keywords)

        # İki kelimelik (bigram) kombinasyonları
        bigrams = ngrams(filtered_keywords, 2)
        bigram_counts = Counter(bigrams)

        # Üç kelimelik (trigram) kombinasyonları
        trigrams = ngrams(filtered_keywords, 3)
        trigram_counts = Counter(trigrams)

        # En sık geçen tek kelimeler, bigramlar ve trigramlar
        most_common_keywords = keyword_counts.most_common(5)
        most_common_bigrams = bigram_counts.most_common(3)
        most_common_trigrams = trigram_counts.most_common(3)

        # Potansiyel anahtar kelimeleri birleştir ve Google Trends verilerini ekle
        potential_keywords = []
        for word, count in most_common_keywords:
            trend_volume = self.get_google_trends_data(word)
            potential_keywords.append({"keyword": word, "count": count, "trend_volume": trend_volume})
        for bigram, count in most_common_bigrams:
            trend_volume = self.get_google_trends_data(" ".join(bigram))
            potential_keywords.append({"keyword": " ".join(bigram), "count": count, "trend_volume": trend_volume})
        for trigram, count in most_common_trigrams:
            trend_volume = self.get_google_trends_data(" ".join(trigram))
            potential_keywords.append({"keyword": " ".join(trigram), "count": count, "trend_volume": trend_volume})

        return potential_keywords

    def print_report(self, keyword, analysis_results):
        """Analiz sonuçlarını terminal ekranına yazdırır."""
        print(f"Keyword Analysis Report for: {keyword}")
        print("----------------------------------------")
        if analysis_results:
            print("Potential Keywords (with Google Trends Volume):")
            for item in analysis_results:
                print(f"- {item['keyword']}: Count={item['count']}, Trend Volume={item['trend_volume']}")
        else:
            print("No results found for this keyword.")
        print("\n")

if __name__ == "__main__":
    # Örnek Kullanım:
    SERP_API_KEY = "94527f54420d26c21b9048060a90049f00f9461d630294134dcf99ec25fab903"  # SERP API anahtarınızı buraya girin
    keywords = ["araba kiralama", "evcil hayvan ürünleri", "online eğitim"]  # Analiz edilecek anahtar kelimeler

    analyzer = KeywordAnalyzer(SERP_API_KEY, num_results=5)

    for keyword in keywords:
        analysis_results = analyzer.analyze_keywords(keyword)
        analyzer.print_report(keyword, analysis_results)  # Sonuçları ekrana yazdır
        print(f"Analysis completed for keyword: {keyword}")

    print("All keyword analysis completed.")