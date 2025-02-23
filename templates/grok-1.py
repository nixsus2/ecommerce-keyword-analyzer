from flask import Flask, render_template, request
import requests
import json
import logging
from collections import Counter
from nltk.corpus import stopwords
from nltk import ngrams
from trnlp import TrnlpWord  # Türkçe analiz için trnlp
from pytrends.request import TrendReq  # Google Trends için
import nltk

nltk.download('stopwords')

# Logging ayarları
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)

class KeywordAnalyzer:
    def __init__(self, serp_api_key, num_results=10):
        self.serp_api_key = serp_api_key
        self.num_results = num_results
        self.session = requests.Session()
        self.stop_words = set(stopwords.words('turkish'))
        self.synonym_dict = {
            "mavibet": ["bahis", "iddaa", "kumar"],
            "mavibetgir": ["mavibet giriş", "mavibet erişim", "mavibet bağlan"],
            "mavibetslot": ["mavibet slot oyunları", "mavibet slot makineleri", "mavibet slotları"]
        }
        self.trnlp = TrnlpWord()  # trnlp nesnesi
        self.pytrends = TrendReq(hl='tr-TR', tz=180)  # Türkiye için ayarlanmış pytrends

        if not self.serp_api_key:
            raise ValueError("Google SERP API anahtarı gereklidir.")

    def search_google(self, keyword):
        """Google SERP API ile arama yapar."""
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
            logger.error(f"Google SERP API hatası '{keyword}': {e}")
            return None

    def preprocess_text(self, text):
        """Metni ön işleme yapar."""
        words = text.split()
        words = [word.lower() for word in words if word.isalnum()]
        words = [word for word in words if word not in self.stop_words]
        return words

    def lemmatize_word(self, word):
        """Kelimenin kökünü bulur (trnlp ile)."""
        self.trnlp.setword(word)
        return self.trnlp.get_stem or word

    def extract_keywords_from_results(self, search_results):
        """Arama sonuçlarından anahtar kelimeleri çıkarır."""
        keywords = []
        if not search_results:
            return keywords

        organic_results = search_results.get("organic_results", [])
        for result in organic_results:
            if "snippet" in result:
                keywords.extend(self.preprocess_text(result["snippet"]))
            if "title" in result:
                keywords.extend(self.preprocess_text(result["title"]))

        related_questions = search_results.get("related_questions", [])
        for question in related_questions:
            keywords.extend(self.preprocess_text(question["question"]))

        related_searches = search_results.get("related_searches", [])
        for search in related_searches:
            if "query" in search:
                keywords.extend(self.preprocess_text(search["query"]))

        return [self.lemmatize_word(word) for word in keywords]

    def get_trend_data(self, keyword):
        """Google Trends verilerini çeker."""
        try:
            self.pytrends.build_payload([keyword], cat=0, timeframe='today 12-m', geo='TR')
            trend_data = self.pytrends.interest_over_time()
            if keyword in trend_data.columns and not trend_data.empty:
                avg_interest = trend_data[keyword].mean()
                return {
                    "status": "Mevcut",
                    "average_interest": round(avg_interest, 2)  # Ortalama ilgi düzeyi
                }
            return {"status": "Veri Yok"}
        except Exception as e:
            logger.error(f"Trends verisi alınamadı: {e}")
            return {"status": "Hata"}

    def calculate_competition(self, search_results):
        """Anahtar kelime rekabet seviyesini hesaplar."""
        if not search_results or "organic_results" not in search_results:
            return "Bilinmiyor"
        organic_results = search_results["organic_results"]
        domain_authority = len(set(r.get("domain", "") for r in organic_results))
        return "Yüksek" if domain_authority > 5 else "Düşük"

    def suggest_keywords(self, keywords):
        """Basit bir öneri motoru: en sık kelimeleri öner."""
        return [k for k, _ in Counter(keywords).most_common(3)]

    def analyze_keywords(self, keyword):
        """Anahtar kelime analizi yapar."""
        search_results = self.search_google(keyword)
        if not search_results:
            return {"error": f"'{keyword}' için arama sonucu bulunamadı"}

        extracted_keywords = self.extract_keywords_from_results(search_results)

        # Kelime sıklıkları
        keyword_counts = Counter(extracted_keywords)
        bigrams = Counter(ngrams(extracted_keywords, 2))
        trigrams = Counter(ngrams(extracted_keywords, 3))

        # En sık geçenler
        most_common_keywords = keyword_counts.most_common(10)
        most_common_bigrams = bigrams.most_common(5)
        most_common_trigrams = trigrams.most_common(5)

        # Sonuçları birleştir
        potential_keywords = [
            {"keyword": k, "count": c, "type": "single"} for k, c in most_common_keywords
        ] + [
            {"keyword": " ".join(k), "count": c, "type": "bigram"} for k, c in most_common_bigrams
        ] + [
            {"keyword": " ".join(k), "count": c, "type": "trigram"} for k, c in most_common_trigrams
        ]

        # Eş anlamlılar
        if keyword in self.synonym_dict:
            for synonym in self.synonym_dict[keyword]:
                potential_keywords.append({"keyword": synonym, "count": "eş anlamlı", "type": "synonym"})

        # Rekabet analizi
        competition = self.calculate_competition(search_results)

        # Trend analizi
        trend_data = self.get_trend_data(keyword)

        # Öneri
        suggestions = self.suggest_keywords(extracted_keywords)

        return {
            "keywords": potential_keywords,
            "competition": competition,
            "trend": trend_data,
            "suggestions": suggestions
        }

SERP_API_KEY = "94527f54420d26c21b9048060a90049f00f9461d630294134dcf99ec25fab903"
analyzer = KeywordAnalyzer(SERP_API_KEY)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        keyword = request.form["keyword"]
        filter_type = request.form.get("filter", "all")
        analysis_results = analyzer.analyze_keywords(keyword)

        if "error" not in analysis_results:
            filtered_results = analysis_results["keywords"]
            if filter_type != "all":
                filtered_results = [r for r in filtered_results if r["type"] == filter_type or (filter_type == "single" and r["type"] == "single")]
            analysis_results["keywords"] = filtered_results

        return render_template("index.html", results=analysis_results, keyword=keyword)
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)