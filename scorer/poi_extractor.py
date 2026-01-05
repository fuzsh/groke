from typing import Dict, List
import json
import re
from .utils import extract_poi_keywords, build_poi_description

class POIExtractor:
    @staticmethod
    def prepare_poi_context(area_pois: Dict, max_samples: int = 20) -> str:
        poi_samples = []
        count = 0

        for poi_id, poi_data in area_pois.items():
            if count >= max_samples:
                break

            try:
                tags = json.loads(poi_data['tags']) if isinstance(poi_data['tags'], str) else poi_data['tags']
                description = build_poi_description({'tags': tags})
                poi_samples.append(f"- {description} (tags: {json.dumps(tags)})")
                count += 1
            except:
                continue

        return "\n".join(poi_samples)

    @staticmethod
    def parse_poi_extraction(extraction_result: str) -> Dict:
        try:
            # Try to extract JSON from the response
            json_match = re.search(r'\{.*\}', extraction_result, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            return json.loads(extraction_result)
        except:
            # Return empty structure if parsing fails
            return {
                "mentioned_landmarks": [],
                "general_keywords": [],
                "navigation_actions": []
            }

    @staticmethod
    def score_poi_match(poi: Dict, extraction_data: Dict) -> float:
        score = 0.0
        tags = poi.get('tags', {})
        poi_keywords = extract_poi_keywords(tags)

        # Check against general keywords
        general_keywords = extraction_data.get('general_keywords', [])
        keyword_matches = sum(1 for kw in general_keywords if kw.lower() in ' '.join(poi_keywords).lower())
        if general_keywords:
            score += 0.3 * (keyword_matches / len(general_keywords))

        # Check against mentioned landmarks
        mentioned = extraction_data.get('mentioned_landmarks', [])
        for landmark in mentioned:
            landmark_keywords = landmark.get('keywords', [])
            landmark_matches = sum(1 for kw in landmark_keywords if kw.lower() in ' '.join(poi_keywords).lower())

            if landmark_keywords and landmark_matches > 0:
                match_ratio = landmark_matches / len(landmark_keywords)

                # Boost score based on role
                role = landmark.get('role', 'reference')
                if role == 'waypoint':
                    score += 0.5 * match_ratio
                elif role in ['start', 'end']:
                    score += 0.7 * match_ratio
                else:
                    score += 0.3 * match_ratio

        # Boost for named POIs
        if 'name' in tags and tags['name']:
            score *= 1.2

        # Normalize to [0, 1]
        return min(score, 1.0)

    @staticmethod
    def rank_pois(area_pois: Dict, extraction_data: Dict, top_k: int = 50) -> List[Dict]:
        scored_pois = []

        for poi_id, poi_data in area_pois.items():
            try:
                tags = json.loads(poi_data['tags']) if isinstance(poi_data['tags'], str) else poi_data['tags']
                poi_dict = {
                    'poi_id': poi_id,
                    'lat': poi_data['lat'],
                    'lng': poi_data['lng'],
                    'tags': tags
                }

                score = POIExtractor.score_poi_match(poi_dict, extraction_data)

                if score > 0.1:  # Only include POIs with some relevance
                    scored_pois.append({
                        **poi_dict,
                        'relevance_score': score,
                        'description': build_poi_description(poi_dict)
                    })
            except:
                continue

        # Sort by score and return top K
        scored_pois.sort(key=lambda x: x['relevance_score'], reverse=True)
        return scored_pois[:top_k]