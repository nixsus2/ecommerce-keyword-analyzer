from flask import Flask, render_template, request
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

app = Flask(__name__)

class KeywordAnalyzer:
    def __init__(self, serp_api_key, num_results=10):
        self.serp_api_key = serp_api_key
        self.num_results = num_results
        self.session = requests.Session()
        self.stop_words = set(stopwords.words('turkish'))  # Türkçe stop words
        self.synonym_dict = {  # Basit eş anlamlı kelime sözlüğü
            "mavibet": ["bahis", "iddaa", "kumar"],
            "mavibetgir": ["mavibet giriş", "mavibet erişim", "mavibet bağlan"],
            "mavibetslot": ["mavibet slot oyunları", "mavibet slot makineleri", "mavibet slotları"]
        }

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
                if "query" in search:  # "query" anahtarının varlığını kontrol et
                    search_keywords = re.findall(r'\b\w+\b', search["query"].lower())
                    keywords.extend(search_keywords)

        return keywords

    def analyze_keywords(self, keyword):
        """Anahtar kelime analizi yapar."""
        search_results = self.search_google(keyword)
        if not search_results:
            return {"error": f"No search results found for keyword '{keyword}'"}

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

        # Eş anlamlı kelimeleri ekle
        if keyword in self.synonym_dict:
            synonyms = self.synonym_dict[keyword]
            for synonym in synonyms:
                potential_keywords.append({"keyword": synonym, "count": "eş anlamlı"})

        return potential_keywords

    def print_report(self, keyword, analysis_results):
        """Analiz sonuçlarını terminal ekranına yazdırır."""
        print(f"Keyword Analysis Report for: {keyword}")
        print("----------------------------------------")
        if "error" in analysis_results:
            print(analysis_results["error"])
        else:
            print("{:<30} {:<10}".format('Keyword', 'Count'))
            print("-" * 40)
            for item in analysis_results:
                print("{:<30} {:<10}".format(item['keyword'], str(item['count']))) # count değerini stringe çeviriyoruz.
        print("\n")

SERP_API_KEY = "94527f54420d26c21b9048060a90049f00f9461d630294134dcf99ec25fab903"  # SERP API anahtarınızı buraya girin
analyzer = KeywordAnalyzer(SERP_API_KEY)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        keyword = request.form["keyword"]
        filter_type = request.form.get("filter", "all")  # "filter" anahtarını kontrol et ve varsayılan olarak "all" değerini kullan
        analysis_results = analyzer.analyze_keywords(keyword)

        if "error" not in analysis_results:
            if filter_type != "all":
                filtered_results = []
                for result in analysis_results:
                    if filter_type == "single" and len(result["keyword"].split()) == 1:
                        filtered_results.append(result)
                    elif filter_type == "bigram" and len(result["keyword"].split()) == 2:
                        filtered_results.append(result)
                    elif filter_type == "trigram" and len(result["keyword"].split()) == 3:
                        filtered_results.append(result)
                analysis_results = filtered_results

        return render_template("index.html", results=analysis_results, keyword=keyword)
    return render_template("index.html")
if __name__ == "__main__":
    app.run(debug=True)