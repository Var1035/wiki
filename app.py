import streamlit as st
import requests

# Function to get search results from Wikipedia
def get_wikipedia_search_results(search_term):
    endpoint = "https://en.wikipedia.org/w/api.php"
    params = {
        'action': 'query',
        'list': 'search',
        'srsearch': search_term,
        'format': 'json'
    }
    response = requests.get(endpoint, params=params)
    return response.json()['query']['search'][:3]

# Streamlit app layout
st.title('Wikipedia Search App')

# Search input
search_term = st.text_input('Enter a search term:', '')

# Search button
if st.button('Search'):
    if search_term:
        # Get the top 3 search results from Wikipedia
        results = get_wikipedia_search_results(search_term)

        # Display the results
        for result in results:
            title = result['title']
            snippet = result['snippet']
            url = f"https://en.wikipedia.org/?curid={result['pageid']}"

            # Display each result as a clickable link with a snippet
            st.markdown(f"### {title}")
            st.markdown(snippet)
    else:
        st.error('Please enter a search term.')

# Run the Streamlit app by saving this script and running streamlit run your_script.py
