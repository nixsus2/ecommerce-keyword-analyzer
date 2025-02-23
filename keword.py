import requests
import json
import logging
from collections import Counter
import re
from nltk.corpus import stopwords
from nltk import ngrams

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

        if not self.serp_api_key:
            raise ValueError("Google SERP API anahtarı gereklidir.")

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

    def analyze_keywords(self, keyword):
        """Anahtar kelime analizi yapar."""
        search_results = self.search_google(keyword)
        if not search_results:
            return {}

        extracted_keywords = self.extract_keywords_from_results(search_results)

        # Stop words'leri filtrele
        filtered_keywords = [word for word in extracted_keywords if word not in self.stop_words]

        # Tek kelimelerin sıklığı
        keyword_counts = Counter(filtered_keywords)

        # İki kelimelik (bigram) kombinasyonları
        bigrams = ngrams(filtered_keywords, 2)
        bigram_counts = Counter(bigrams)

        # Üç kelimelik (trigram) kombinasyonları
        trigrams = ngrams(filtered_keywords, 3)
        trigram_counts = Counter(trigrams)

        # En sık geçen tek kelimeler, bigramlar ve trigramlar
        most_common_keywords = keyword_counts.most_common(10)
        most_common_bigrams = bigram_counts.most_common(5)
        most_common_trigrams = trigram_counts.most_common(5)

        # Potansiyel anahtar kelimeleri birleştir
        potential_keywords = []
        for word, count in most_common_keywords:
            potential_keywords.append({"keyword": word, "count": count})
        for bigram, count in most_common_bigrams:
            potential_keywords.append({"keyword": " ".join(bigram), "count": count})
        for trigram, count in most_common_trigrams:
            potential_keywords.append({"keyword": " ".join(trigram), "count": count})

        return potential_keywords

    def print_report(self, keyword, analysis_results):
        """Analiz sonuçlarını terminal ekranına yazdırır."""
        print(f"Keyword Analysis Report for: {keyword}")
        print("----------------------------------------")
        if analysis_results:
            print("Potential Keywords:")
            for item in analysis_results:
                print(f"- {item['keyword']}: Count={item['count']}")
        else:
            print("No results found for this keyword.")
        print("\n")

if __name__ == "__main__":
    # Örnek Kullanım:
    SERP_API_KEY = "94527f54420d26c21b9048060a90049f00f9461d630294134dcf99ec25fab903"  # SERP API anahtarınızı buraya girin
    keywords = ["mavibet", "mavibetgir", "mavibetslot"]  # Analiz edilecek anahtar kelimeler

    analyzer = KeywordAnalyzer(SERP_API_KEY)

    for keyword in keywords:
        analysis_results = analyzer.analyze_keywords(keyword)
        analyzer.print_report(keyword, analysis_results)  # Sonuçları ekrana yazdır
        print(f"Analysis completed for keyword: {keyword}")

    print("All keyword analysis completed.")