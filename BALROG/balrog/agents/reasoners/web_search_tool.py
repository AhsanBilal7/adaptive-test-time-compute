"""Web-search tool stub; no external search backend is connected."""

class WebSearchTool:
    """Placeholder for web search/retrieval functionality."""
    
    def __init__(self, client=None):
        self.client = client
        self.search_history = []
    
    def search(self, query):
        """
        Placeholder for web search functionality.
        
        Args:
            query: Search query (observation, question, etc.)
            
        Returns:
            str: Search results (placeholder)
        """
        # Placeholder: no external search backend is connected.
        result = f"[Web search placeholder for query: {str(query)[:100]}]"
        
        self.search_history.append({
            "query": str(query),
            "result": result
        })
        
        return result
    
    def reset(self):
        """Reset search history."""
        self.search_history = []