import requests
from datetime import datetime, timedelta
from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.cluster import KMeans
from pymongo import MongoClient
import logging
import spacy

# MongoDB setup
client = MongoClient('mongodb://localhost:27017/')
db = client['football_news']
news_collection = db['articles']

# Transformers setup
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

# Logger setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class ScraperModule:
    def __init__(self, api_key, news_collection):
        self.api_key = api_key
        self.news_collection = news_collection
        # Define football-related keywords
        self.football_keywords = ['football', 'soccer', 'goal', 'match', 'championship', 'league', 'Arsenal', 
                                  'Manchester United', 'Liverpool', 'Premier League', 'World Cup']

    def is_football_related(self, article_content):
        """Check if the article content contains football-related keywords."""
        content_lower = article_content.lower()
        return any(keyword.lower() in content_lower for keyword in self.football_keywords)

    def fetch_news(self, team_name):
        """Fetch football news articles for a specific team."""
        yesterday = datetime.now() - timedelta(days=1)
        formatted_yesterday = yesterday.strftime('%Y-%m-%d')

        url = 'https://newsapi.org/v2/everything'
        query_params = {
            'q': f'"{team_name}"',
            'from': formatted_yesterday,
            'to': formatted_yesterday,
            'language': 'en',
            'sortBy': 'publishedAt',
            'pageSize': 10,
            'apiKey': self.api_key
        }

        response = requests.get(url, params=query_params)
        
        if response.status_code == 200:
            articles = response.json().get('articles', [])
            # Filter the articles based on football-related content
            football_related_articles = [article for article in articles if self.is_football_related(article.get('content', '') or article.get('description', ''))]
            return football_related_articles
        else:
            logging.error(f"Failed to retrieve news for {team_name}. Status code: {response.status_code}")
            return []
        

    def store_articles(self, articles):
        today = datetime.now().strftime('%Y-%m-%d')
        for article in articles:
            article['id'] = article['url']
            article_to_insert = {
                'id': article['id'],
                'source': article['source']['name'],
                'content': article['content'] or article['description'] or "",
                'datePublished': article['publishedAt'],
                'dateScraped': today
            }
            self.news_collection.insert_one(article_to_insert)


class PreprocessingModule:
    def __init__(self):
        # Load spaCy's small English model
        try:
            self.nlp = spacy.load('en_core_web_sm')
        except OSError:
            print("Downloading spaCy model 'en_core_web_sm'...")
            spacy.cli.download("en_core_web_sm")
            self.nlp = spacy.load('en_core_web_sm')
        
        # Load a tokenizer and model from Huggingface transformers
        self.tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
        self.model = AutoModel.from_pretrained('distilbert-base-uncased')
        
    def preprocess_text(self, text):
        """Clean and preprocess the text data using spaCy."""
        # Process the text using spaCy
        doc = self.nlp(text)
        
        # Keep only alphanumeric tokens, lowercase them, and remove stopwords and punctuation
        cleaned_tokens = [token.text.lower() for token in doc 
                          if not token.is_stop and token.is_alpha]
        
        # Join tokens back into a cleaned string
        cleaned_text = ' '.join(cleaned_tokens)
        return cleaned_text

    def vectorize_texts(self, texts):
        """Convert the list of texts to embeddings using a transformer model."""
        inputs = self.tokenizer(texts, return_tensors='pt', padding=True, truncation=True)
        
        with torch.no_grad():
            embeddings = self.model(**inputs).last_hidden_state.mean(dim=1)
        
        return embeddings



class ClusteringModule:
    def apply_clustering(self, embeddings, articles, n_clusters=2):
        # Ensure the number of clusters does not exceed the number of samples
        if len(embeddings) < n_clusters:
            n_clusters = len(embeddings)
            logging.info(f"Reduced number of clusters to {n_clusters} due to insufficient samples.")

        if n_clusters == 0:
            raise ValueError("No embeddings available for clustering.")

        # Apply K-Means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(embeddings)

        # Print information about clusters and the articles within them
        print("\nCluster Information:")
        for cluster_id in range(n_clusters):
            print(f"\nCluster {cluster_id}:")
            # Find articles belonging to this cluster
            cluster_articles = [articles[idx] for idx, label in enumerate(cluster_labels) if label == cluster_id]
            for article in cluster_articles:
                print(f"- Article URL: {article['url']}")
                print(f"  Source: {article['source']['name']}")
                print(f"  Content snippet: {article['content'][:150]}...")  # Print the first 150 characters
            print(f"Number of articles in this cluster: {len(cluster_articles)}")

        return cluster_labels


class DeduplicationModule:
    def select_representatives(self, articles, cluster_labels):
        """Select one representative article per cluster."""
        cluster_dict = {}
        for idx, label in enumerate(cluster_labels):
            if label not in cluster_dict:
                cluster_dict[label] = articles[idx]
        return list(cluster_dict.values())


class ProcessingModule:
    def analyze_articles(self, articles):
        """Perform further analysis on the articles (e.g., tagging, summarization)."""
        for article in articles:
            logging.info(f"Analyzing article: {article['id']} - {article['source']}")
            # Example: You can add a tag or perform summarization here
            article['tags'] = ['football', 'news', 'team']
            # Add additional processing logic as needed
        return articles


class FootballNewsScraper:
    def __init__(self, api_key, news_collection):
        self.api_key = api_key
        self.news_collection = news_collection
        self.scraper = ScraperModule(api_key, news_collection)
        self.preprocessing = PreprocessingModule()
        self.clustering = ClusteringModule()
        self.deduplication = DeduplicationModule()
        self.processing = ProcessingModule()

    def run(self, football_teams):
        """Main function to orchestrate the scraping, preprocessing, clustering, and processing of news articles."""
        for team in football_teams:
            logging.info(f"Fetching news for {team}...")

            # Step 1: Scrape the news articles
            articles = self.scraper.fetch_news(team)
            if not articles:
                logging.info(f"No articles fetched for {team}.")
                continue

            # Step 2: Preprocess and vectorize the articles
            preprocessed_articles = []
            article_texts = []
            for article in articles:
                preprocessed_text = self.preprocessing.preprocess_text(article['content'])
                if preprocessed_text:
                    article_texts.append(preprocessed_text)
                    preprocessed_articles.append(article)

            if not article_texts:
                logging.info(f"No valid articles after preprocessing for {team}.")
                continue

            # Step 3: Vectorize the preprocessed articles
            embeddings = self.preprocessing.vectorize_texts(article_texts)

            # Ensure there are embeddings before applying clustering
            if len(embeddings) == 0:
                logging.info(f"No embeddings available for clustering for {team}.")
                continue

            # Step 4: Apply clustering to group similar articles
            try:
                # Pass both embeddings and preprocessed_articles to the clustering module
                cluster_labels = self.clustering.apply_clustering(embeddings, preprocessed_articles, n_clusters=5)
            except ValueError as e:
                logging.warning(f"Clustering failed: {e}")
                continue

            # Step 5: Select representative articles from each cluster
            representative_articles = self.deduplication.select_representatives(preprocessed_articles, cluster_labels)

            # Step 6: Store representative articles in MongoDB
            self.scraper.store_articles(representative_articles)

            # Step 7: Perform further processing/analysis on the articles
            processed_articles = self.processing.analyze_articles(representative_articles)

            logging.info(f"Processed {len(processed_articles)} articles for {team}.")
