import requests
import json
import logging
from collections import Counter
import re
from nltk.corpus import stopwords
from nltk import ngrams
from pytrends.request import TrendReq
from duckduckgo_search import DDGS
from bs4 import BeautifulSoup
import csv
import os  # Dosya yolu oluşturma için

import nltk
nltk.download('stopwords')

# Logging Ayarı
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class KeywordAnalyzer:
    def __init__(self):
        self.stop_words = set(stopwords.words('turkish'))
        self.pytrends = TrendReq(hl='tr-TR', tz=360)

    def get_google_trends(self, keyword):
        """Google Trends verilerini getirir."""
        try:
            self.pytrends.build_payload([keyword], cat=0, timeframe='today 3-m', geo='TR')
            interest_over_time = self.pytrends.interest_over_time()
            if not interest_over_time.empty:
                logger.info(f"Trend Verisi: {interest_over_time.tail(1).to_dict()}")
                return interest_over_time.to_dict()
            return {}
        except Exception as e:
            logger.error(f"Google Trends hatası: {e}")
            return {}

    def get_duckduckgo_results(self, keyword):
        """DuckDuckGo üzerinden ilgili anahtar kelimeleri getirir."""
        related_keywords = []
        try:
            with DDGS() as ddgs:
                results = list(ddgs.text(keyword, region='tr-tr', max_results=10))
                for result in results:
                    title_keywords = re.findall(r'\b\w+\b', result['title'].lower())
                    related_keywords.extend(title_keywords)
        except Exception as e:
            logger.error(f"DuckDuckGo araması hatası: {e}")
        return related_keywords

    def get_keywordtool_io_results(self, keyword):
        """Keywordtool.io üzerinden ilgili anahtar kelimeleri getirir."""
        url = f"https://keywordtool.io/google-autocomplete/{keyword}"
        try:
            response = requests.get(url)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            # Buradaki seçim, Keywordtool.io'nun HTML yapısına göre değişebilir.
            keyword_elements = soup.find_all('div', class_='keyword-element')  # Örnek seçim
            keywords = [element.text.strip() for element in keyword_elements]
            return keywords
        except Exception as e:
            logger.error(f"Keywordtool.io scraper hatası: {e}")
            return []

    def get_people_also_ask(self, keyword):
        """Google'dan 'People Also Ask' sorularını getirir."""
        url = f"https://www.google.com/search?q={keyword}"
        try:
            response = requests.get(url)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            # Buradaki seçim, Google'ın HTML yapısına göre değişebilir.
            question_elements = soup.find_all('div', class_='related-question-pair')  # Örnek seçim
            questions = [element.text.strip() for element in question_elements]
            return questions
        except Exception as e:
            logger.error(f"People Also Ask scraper hatası: {e}")
            return []

    def analyze_keywords(self, keyword):
        trends = self.get_google_trends(keyword)
        duck_keywords = self.get_duckduckgo_results(keyword)
        keywordtool_keywords = self.get_keywordtool_io_results(keyword)
        paa_questions = self.get_people_also_ask(keyword)

        all_keywords = duck_keywords + keywordtool_keywords + paa_questions

        # Stop words'leri filtrele
        filtered_keywords = [word for word in all_keywords if word not in self.stop_words]

        # Anahtar Kelime Sıklığı
        keyword_counts = Counter(filtered_keywords)
        most_common_keywords = keyword_counts.most_common(10)

        return {
            "trends": trends,
            "most_common_keywords": most_common_keywords
        }

    def print_report(self, keyword, analysis_results):
        print(f"Keyword Analysis Report for: {keyword}")
        print("----------------------------------------")
        print(f"Google Trends: {analysis_results['trends']}")
        print("Most Common Keywords:")
        for item, count in analysis_results['most_common_keywords']:
            print(f"- {item}: {count} kez")
        print("\n")

    def save_to_json(self, keyword, analysis_results, filepath="output.json"):
        """Sonuçları JSON dosyasına kaydet."""
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump({"keyword": keyword, "analysis": analysis_results}, f, ensure_ascii=False, indent=4)
            logger.info(f"JSON dosyası oluşturuldu: {filepath}")
        except Exception as e:
            logger.error(f"JSON dosyası oluşturma hatası: {e}")

    def save_to_csv(self, keyword, analysis_results, filepath="output.csv"):
        """Sonuçları CSV dosyasına kaydet."""
        try:
            with open(filepath, 'w', newline='', encoding='utf-8') as csvfile:
                fieldnames = ['keyword', 'count', 'source']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                
                for item, count in analysis_results['most_common_keywords']:
                    writer.writerow({'keyword': item, 'count': count, 'source': 'most_common'})

                logger.info(f"CSV dosyası oluşturuldu: {filepath}")
        except Exception as e:
            logger.error(f"CSV dosyası oluşturma hatası: {e}")

    def send_to_llm_api(self, keyword, analysis_results, api_endpoint):
        """Sonuçları LLM API'sine gönder."""
        try:
            headers = {'Content-Type': 'application/json'}
            data = json.dumps({"keyword": keyword, "analysis": analysis_results}, ensure_ascii=False)
            response = requests.post(api_endpoint, headers=headers, data=data)
            response.raise_for_status()  # Hata durumunda exception yükselt
            logger.info(f"LLM API'sine veri gönderildi. Durum kodu: {response.status_code}")
            return response.json()  # API'den dönen cevabı işle
        except requests.exceptions.RequestException as e:
            logger.error(f"LLM API'sine veri gönderme hatası: {e}")
            return None

# Kullanım
if __name__ == "__main__":
    analyzer = KeywordAnalyzer()
    keyword = "araba kiralama"
    results = analyzer.analyze_keywords(keyword)
    analyzer.print_report(keyword, results)

    # Çıktıları kaydet
    analyzer.save_to_json(keyword, results, filepath="araba_kiralama.json")
    analyzer.save_to_csv(keyword, results, filepath="araba_kiralama.csv")

    # LLM API'sine gönder (API endpoint'ini kendi endpoint'inizle değiştirin)
    # llm_response = analyzer.send_to_llm_api(keyword, results, "http://your-llm-api.com/analyze")
    # if llm_response:
    #     print("LLM API Cevabı:", llm_response)