"""Web-search tool stub; no external search backend is connected."""

class WebSearchTool:
    """Web-search stub; returns a placeholder result."""
    
    def __init__(self, client=None):
        self.client = client
        self.search_history = []
    
    def search(self, query):
        result = f"[Web search placeholder for query: {str(query)[:100]}]"
        
        self.search_history.append({
            "query": str(query),
            "result": result
        })
        
        return result
    
    def reset(self):
        self.search_history = []
