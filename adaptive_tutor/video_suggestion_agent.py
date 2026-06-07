"""
video_suggestion_agent.py
──────────────────────────
Responsible for:
  1. Extracting key concepts from the lecture
  2. Searching YouTube for relevant educational videos
  3. Returning a ranked list of video suggestions

Uses YouTube Data API to search for videos related to lecture concepts.

Perceives:  concept_graph, lecture_title  (from blackboard)
Acts:       videos  (written to blackboard)
"""

import requests
import os
from base_agent import BaseAgent


class VideoSuggestionAgent(BaseAgent):

    def __init__(self, blackboard, youtube_api_key: str = ""):
        super().__init__("VideoSuggestionAgent", blackboard)
        self.youtube_api_key = youtube_api_key or os.getenv("YOUTUBE_API_KEY", "")
        self.youtube_api_url = "https://www.googleapis.com/youtube/v3/search"

    # ── PERCEIVE ─────────────────────────────────────────────
    def perceive(self) -> dict:
        concept_graph = self.blackboard.read("concept_graph") or []
        lecture_title = self.blackboard.read("lecture_title") or "lecture"
        
        print(f"  [VideoSuggestionAgent] PERCEIVE → lecture: {lecture_title}")
        self.blackboard.log_thinking(
            "VideoSuggestionAgent",
            f"I see {len(concept_graph)} concepts. I will search YouTube for relevant videos."
        )
        
        return {
            "concept_graph": concept_graph,
            "lecture_title": lecture_title,
        }

    # ── REASON ───────────────────────────────────────────────
    def reason(self, perception: dict) -> dict:
        """
        Extract key concepts and search YouTube for relevant videos.
        Returns a list of video suggestions.
        """
        concept_graph = perception["concept_graph"]
        lecture_title = perception["lecture_title"]
        
        # Extract top concepts to search for
        print("\n  [VideoSuggestionAgent] REASON: Extracting top concepts...")
        top_concepts = self._extract_top_concepts(concept_graph)
        print(f"  [VideoSuggestionAgent] Top concepts: {top_concepts}")
        
        # Search for videos for each concept
        print("\n  [VideoSuggestionAgent] REASON: Searching YouTube for videos...")
        all_videos = []
        
        if self.youtube_api_key:
            # Search by top concepts first
            for concept in top_concepts[:3]:  # Limit to top 3 concepts
                videos = self._search_youtube_videos(concept)
                all_videos.extend(videos)
            
            # Also search by lecture title
            if lecture_title:
                videos = self._search_youtube_videos(lecture_title)
                all_videos.extend(videos)
        else:
            print("  [VideoSuggestionAgent] No YouTube API key found. Using fallback suggestions.")
            all_videos = self._generate_fallback_videos(top_concepts, lecture_title)
        
        # Deduplicate and rank videos
        videos = self._deduplicate_and_rank(all_videos)
        print(f"  [VideoSuggestionAgent] Found {len(videos)} unique videos")
        
        return {"videos": videos}

    # ── ACT ──────────────────────────────────────────────────
    def act(self, decision: dict):
        videos = decision.get("videos", [])
        self.blackboard.write("videos", videos)
        self.blackboard.log_thinking(
            "VideoSuggestionAgent",
            f"Compiled {len(videos)} educational videos for the lecture concepts."
        )

    # ── HELPER METHODS ───────────────────────────────────────
    def _extract_top_concepts(self, concept_graph: list) -> list:
        """Extract the top 5 most important concepts from the graph."""
        # Sort by dependency order (concepts with no prerequisites are more fundamental)
        top_concepts = []
        for concept_data in concept_graph[:5]:
            if isinstance(concept_data, dict):
                concept_name = concept_data.get("concept", "")
            else:
                concept_name = str(concept_data)
            
            if concept_name:
                top_concepts.append(concept_name)
        
        return top_concepts

    def _search_youtube_videos(self, query: str) -> list:
        """Search YouTube for videos related to the query."""
        if not self.youtube_api_key:
            return []
        
        try:
            params = {
                "q": query,
                "type": "video",
                "part": "snippet",
                "maxResults": 5,
                "relevanceLanguage": "en",
                "key": self.youtube_api_key,
                "videoCategoryId": "27",  # Education category
            }
            
            response = requests.get(self.youtube_api_url, params=params, timeout=5)
            response.raise_for_status()
            data = response.json()
            
            videos = []
            for item in data.get("items", []):
                snippet = item.get("snippet", {})
                video_id = item.get("id", {}).get("videoId", "")
                
                if video_id:
                    videos.append({
                        "id": video_id,
                        "title": snippet.get("title", ""),
                        "description": snippet.get("description", "")[:200],
                        "thumbnail": snippet.get("thumbnails", {}).get("default", {}).get("url", ""),
                        "channel": snippet.get("channelTitle", ""),
                        "url": f"https://www.youtube.com/watch?v={video_id}",
                    })
            
            return videos
        except Exception as e:
            print(f"  [VideoSuggestionAgent] Error searching YouTube: {e}")
            return []

    def _generate_fallback_videos(self, concepts: list, lecture_title: str) -> list:
        """Generate fallback video suggestions when API is unavailable."""
        # Return generic educational video recommendations
        fallback_videos = [
            {
                "id": "fallback_1",
                "title": f"Understanding {concepts[0] if concepts else 'the Topic'}",
                "description": "A comprehensive educational video to help you understand the key concepts.",
                "thumbnail": "",
                "channel": "Educational Content",
                "url": "#",
                "is_fallback": True,
            },
            {
                "id": "fallback_2",
                "title": f"{concepts[1] if len(concepts) > 1 else 'Deep Dive'}: Advanced Topics",
                "description": "Explore advanced concepts and real-world applications.",
                "thumbnail": "",
                "channel": "Educational Content",
                "url": "#",
                "is_fallback": True,
            },
            {
                "id": "fallback_3",
                "title": f"Practice and Examples: {concepts[2] if len(concepts) > 2 else 'Common Topics'}",
                "description": "Learn through practical examples and problem-solving.",
                "thumbnail": "",
                "channel": "Educational Content",
                "url": "#",
                "is_fallback": True,
            },
        ]
        return fallback_videos

    def _deduplicate_and_rank(self, videos: list) -> list:
        """Remove duplicate videos and rank by relevance."""
        seen_ids = set()
        unique_videos = []
        
        for video in videos:
            video_id = video.get("id", "")
            if video_id and video_id not in seen_ids:
                seen_ids.add(video_id)
                unique_videos.append(video)
            
            # Limit to top 10 videos
            if len(unique_videos) >= 10:
                break
        
        return unique_videos
