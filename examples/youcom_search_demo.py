#!/usr/bin/env python3
"""
You.com Search Integration Example for Second Brain

This example demonstrates how to use the You.com search tool
in different modes and scenarios.
"""

def demonstrate_youcom_search():
    """Example usage patterns for the youcom_search tool."""
    
    # Example 1: Automatic mode selection
    print("=== Automatic Mode Selection ===")
    print("Query: 'What are the latest developments in quantum computing?'")
    print("→ Tool automatically selects SMART mode (question-like query)")
    print("→ Returns AI-powered answer with citations")
    print()
    
    print("Query: 'python pandas tutorial'") 
    print("→ Tool automatically selects WEB mode (keyword search)")
    print("→ Returns traditional search results with links")
    print()
    
    # Example 2: Explicit mode usage
    print("=== Explicit Mode Usage ===")
    print("youcom_search(query='machine learning algorithms', mode='web', count=8)")
    print("→ Forces web search mode with 8 results")
    print()
    
    print("youcom_search(query='explain neural networks', mode='smart')")
    print("→ Forces smart search for AI-powered explanation")
    print()
    
    # Example 3: Content extraction
    print("=== Content Extraction ===")
    print("youcom_search(query='https://arxiv.org/abs/2301.00001')")
    print("→ Extracts and returns readable content from the URL")
    print()
    
    # Example 4: Advanced parameters
    print("=== Advanced Parameters ===")
    print("""youcom_search(
        query='renewable energy policies',
        mode='web', 
        count=10,
        country='DE',
        safesearch='strict'
    )""")
    print("→ German-localized search with strict filtering")
    print()

def setup_instructions():
    """Configuration setup for You.com integration."""
    print("=== Setup Instructions ===")
    print()
    print("1. Obtain You.com API Key:")
    print("   - Visit https://you.com") 
    print("   - Sign up or log in")
    print("   - Generate a YDC API key")
    print()
    print("2. Configure Second Brain:")
    print("   - Open service configuration")
    print("   - Find 'youcom_search_provider'")  
    print("   - Set 'You.com API Key' to your YDC key")
    print("   - Save configuration")
    print()
    print("3. Verify Installation:")
    print("   - Tool 'youcom_search' should be available")
    print("   - Test with: youcom_search(query='test search')")
    print()

if __name__ == "__main__":
    print("You.com Search Integration for Second Brain")
    print("==========================================")
    print()
    
    setup_instructions()
    print()
    demonstrate_youcom_search()
    
    print("For full documentation, see docs/youcom-integration.md")