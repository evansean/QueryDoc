from langchain_community.document_loaders import YoutubeLoader

def load_youtube_transcript(video_url: str):
    """Load YouTube video transcript as documents."""
    loader = YoutubeLoader.from_youtube_url(video_url, add_video_info=False)
    docs = loader.load()
    return docs