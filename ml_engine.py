import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity


class MLEngine:
    """
    Machine Learning Engine for Feedback Intelligence
    
    Handles text vectorization, clustering, similarity analysis,
    and intelligent feedback categorization.
    """
    
    def __init__(self, n_clusters=4):
        """
        Initialize the ML Engine
        
        Args:
            n_clusters (int): Number of clusters for feedback grouping
        """
        self.vectorizer = TfidfVectorizer(
            stop_words='english',
            max_features=1000,
            token_pattern=r'[a-zA-Z]{3,}',  # Only match words with 3+ letters, no numbers
            ngram_range=(1, 2)  # Include bigrams for better context
        )
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        # Dynamic clustering based on dataset size
        self.raw_n_clusters = n_clusters

    def vectorize(self, texts):
        """
        Convert text feedback into TF-IDF vectors
        
        Args:
            texts (list): List of feedback text strings
            
        Returns:
            sparse matrix: TF-IDF feature vectors
        """
        return self.vectorizer.fit_transform(texts)

    def cluster(self, X):
        """
        Cluster feedback into groups using K-Means
        
        Args:
            X (sparse matrix): TF-IDF vectors
            
        Returns:
            array: Cluster labels for each feedback
        """
        n_samples = X.shape[0]
        # Adjust clusters if we have very little data
        if n_samples < self.raw_n_clusters:
            n_c = max(1, n_samples)
        else:
            n_c = self.raw_n_clusters
            
        kmeans = KMeans(n_clusters=n_c, random_state=42)
        return kmeans.fit_predict(X)

    def similarity(self, X):
        """
        Calculate cosine similarity between all feedback pairs
        
        Args:
            X (sparse matrix): TF-IDF vectors
            
        Returns:
            array: Similarity matrix
        """
        return cosine_similarity(X)

    # -------- Feature 1: Auto Cluster Naming --------
    def name_clusters(self, X, texts, clusters):
        """
        Automatically generate descriptive names for feedback clusters
        
        Args:
            X (sparse matrix): TF-IDF vectors
            texts (list): Original feedback texts
            clusters (array): Cluster assignments
            
        Returns:
            dict: Mapping of cluster ID to descriptive name
        """
        feature_names = self.vectorizer.get_feature_names_out()
        names = {}

        for c in np.unique(clusters):
            idxs = np.where(clusters == c)[0]
            
            # Safety check for empty clusters
            if len(idxs) == 0:
                continue
                
            sub_X = X[idxs].mean(axis=0)
            # Get Top 10 keywords for better matching
            top_indices = np.argsort(sub_X.A1)[-10:]
            keywords = [feature_names[i] for i in top_indices]

            # Enhanced naming logic with more categories
            if any(k in keywords for k in ["delivery", "late", "delay", "shipping", "ship", "arrived"]):
                names[c] = "Delivery Issues"
            elif any(k in keywords for k in ["price", "cost", "expensive", "cheap", "money", "value", "worth"]):
                names[c] = "Pricing Issues"
            elif any(k in keywords for k in ["bug", "error", "crash", "issue", "problem", "broken", "fix", "glitch"]):
                names[c] = "App Problems"
            elif any(k in keywords for k in ["support", "help", "service", "customer", "contact", "reply", "disappointed"]):
                names[c] = "Customer Support"
            elif any(k in keywords for k in ["quality", "bad", "poor", "terrible", "worst", "broken", "meal", "food"]):
                names[c] = "Quality Concerns"
            elif any(k in keywords for k in ["feature", "need", "want", "add", "missing", "request"]):
                names[c] = "Feature Requests"
            elif any(k in keywords for k in ["love", "great", "good", "excellent", "amazing", "best", "awesome", "perfect", "happy", "thanks", "thank", "helpful", "satisfied", "impressive", "smooth", "easy", "fast"]):
                names[c] = "Positive Feedback"
            else:
                # Fallback: Check sentiment of the cluster items
                cluster_texts = [texts[i] for i in idxs]
                pos_count = sum(1 for t in cluster_texts if self.get_sentiment_indication(t) == "Positive")
                if pos_count > len(cluster_texts) / 2:
                    names[c] = "Positive Feedback"
                else:
                    # Use the top keyword as the name instead of "General"
                    top_kw = keywords[-1] if keywords else "General"
                    names[c] = f"Feedback about '{top_kw}'"
            
        return names

    # -------- Feature 2: Priority Level --------
    def priority_level(self, score):
        """
        Determine priority level based on similarity score
        
        Args:
            score (float): Priority score (sum of similarities)
            
        Returns:
            str: Priority level (High/Medium/Low)
        """
        # Adjusted thresholds for small dataset
        if score > 1.15:
            return "High"
        elif score > 1.05:
            return "Medium"
        else:
            return "Low"

    # -------- Feature 3: Auto Tags --------
    def generate_tags(self, text):
        """
        Extract important keywords as tags from feedback text
        
        Args:
            text (str): Feedback text
            
        Returns:
            list: List of relevant tags (keywords)
        """
        if not isinstance(text, str):
            return []
            
        words = text.lower().split()
        # Filter for meaningful words (length > 4) and remove duplicates
        keywords = list(set([w for w in words if len(w) > 4]))
        return keywords[:5]  # Return top 5 tags

    # -------- Feature 4: AI Summary --------
    def generate_summary(self, df):
        """
        Generate an AI summary of feedback distribution
        
        Args:
            df (DataFrame): Feedback dataframe with cluster names
            
        Returns:
            str: Human-readable summary of main issues
        """
        cluster_counts = df['cluster_name'].value_counts().to_dict()
        summary = "🧠 **AI-Generated Insights:**\n\n"
        summary += "Main issues detected:\n"
        for k, v in sorted(cluster_counts.items(), key=lambda x: x[1], reverse=True):
            percentage = (v / len(df)) * 100
            summary += f"- **{k}**: {v} feedbacks ({percentage:.1f}%)\n"
        
        # Add recommendations
        top_issue = max(cluster_counts, key=cluster_counts.get)
        summary += f"\n💡 **Recommendation:** Focus on addressing '{top_issue}' as it represents the largest category."
        
        return summary

    # -------- Feature 5: Similar Feedback --------
    def find_similar(self, sim_matrix, texts, idx, n=3):
        """
        Find most similar feedback items to a given feedback
        
        Args:
            sim_matrix (array): Similarity matrix
            texts (list): List of all feedback texts
            idx (int): Index of the reference feedback
            n (int): Number of similar items to return
            
        Returns:
            list: Top n most similar feedback texts
        """
        sims = list(enumerate(sim_matrix[idx]))
        sims = sorted(sims, key=lambda x: x[1], reverse=True)[1:n+1]
        return [texts[i[0]] for i in sims]

    # -------- Existing Priority Score --------
    def priority_score(self, similarity_matrix, idx):
        """
        Calculate priority score based on similarity to other feedback
        
        Higher score means this feedback is similar to many others,
        indicating a common/important issue.
        
        Args:
            similarity_matrix (array): Similarity matrix
            idx (int): Index of the feedback
            
        Returns:
            float: Priority score
        """
        return float(np.sum(similarity_matrix[idx]))
    
    # -------- Additional Feature: Sentiment Indication --------
    def get_sentiment_indication(self, text):
        """
        Get a basic sentiment indication from text
        (Simple keyword-based approach)
        
        Args:
            text (str): Feedback text
            
        Returns:
            str: Sentiment indication (Positive/Negative/Neutral)
        """
        positive_words = ['good', 'great', 'excellent', 'amazing', 'love', 'perfect', 'wonderful']
        negative_words = ['bad', 'terrible', 'awful', 'poor', 'hate', 'worst', 'horrible']
        
        text_lower = text.lower()
        pos_count = sum(1 for word in positive_words if word in text_lower)
        neg_count = sum(1 for word in negative_words if word in text_lower)
        
        if pos_count > neg_count:
            return "Positive"
        elif neg_count > pos_count:
            return "Negative"
        else:
            return "Neutral"