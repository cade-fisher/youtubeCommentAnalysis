import requests

# Function to get youtube comments
def get_youtube_comments(video_id, api_key, max_results=50):
    url = "https://www.googleapis.com/youtube/v3/commentThreads"

# Parameters for API
    params = {
        'part': 'snippet',
        'videoId': video_id,
        'maxResults': max_results,
        'key': api_key
    }

# Sends a GET request
    response = requests.get(url, params=params)

# List to store the comments
    comment_list = []

    if response.status_code == 200:
        data = response.json()
        comments = data.get('items', [])
        for comment in comments:
            snippet = comment['snippet']['topLevelComment']['snippet']
            text = snippet['textDisplay']
            comment_list.append(text)

        next_page_token = data.get('nextPageToken')
        if next_page_token:
            print(f"Next Page Token: {next_page_token}")

    else:
        print(f"Error: {response.status_code} - {response.text}")

# Returns list of comments
    return comment_list
