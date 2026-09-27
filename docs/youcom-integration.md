# You.com Search Integration

This integration adds You.com's premium web search capabilities to Second Brain, providing real-time web search, smart AI-powered answers, and URL content extraction.

## Features

- **Intelligent Mode Selection**: Automatically chooses between web search and smart search based on query type
- **Web Search**: Traditional search results with titles, URLs, and descriptions
- **Smart Search**: AI-powered answers with citations for question-like queries  
- **Content Extraction**: Extract and read content from any URL
- **Real-time Results**: Access to current web information

## Setup

1. **Get a You.com API Key**: Sign up at [you.com](https://you.com) and obtain a YDC API key

2. **Configure the Service**: Set your API key in Second Brain's service configuration:
   - Service: `youcom_search_provider`
   - Setting: "You.com API Key" (`secret_ydc_api_key`)
   - Value: Your YDC API key

3. **Install the Tool**: The `youcom_search` tool will be available once the service is configured

## Usage

### Automatic Mode (Recommended)
```
Search for: "latest developments in AI research"
```
The tool automatically selects smart search for questions and web search for keywords.

### Explicit Web Search
```
youcom_search(query="python tutorial", mode="web", count=5)
```

### Explicit Smart Search  
```
youcom_search(query="How does machine learning work?", mode="smart")
```

### Content Extraction
```  
youcom_search(query="https://example.com/article")
```

### Advanced Options
```
youcom_search(
    query="climate change research", 
    mode="web",
    count=10,
    country="US", 
    safesearch="moderate"
)
```

## Parameters

- **query** (required): Search terms or URL to extract content from
- **mode**: "auto" (default), "web", or "smart"  
- **count**: Max results for web search (1-20, default 6)
- **country**: 2-letter country code for localized results (e.g., "US", "GB")
- **safesearch**: "strict", "moderate" (default), or "off"
- **narration**: Optional description of what you're looking for (shown to user)

## Mode Selection Logic

When `mode="auto"` (default), the tool intelligently selects:

- **Smart Search** for:
  - Questions ending with "?"
  - Queries starting with question words (what, why, how, etc.)  
  - Long conversational queries (8+ words)
  - Comparison queries ("X vs Y", "compare X and Y")

- **Web Search** for:
  - Short keyword searches
  - Specific product/name lookups
  - Technical queries where multiple sources are valuable

## Error Handling

The tool provides clear error messages for common issues:
- Missing API key configuration
- Network connectivity problems  
- API rate limits or quota exceeded
- Invalid URLs for content extraction

If smart search fails in auto mode, the tool automatically falls back to web search.

## Security

- API keys are stored securely using Second Brain's secret system
- The service never logs or exposes API credentials  
- All web content is treated as untrusted external data
- URLs are normalized and validated before content extraction