"""
kb/seed_data.py
Sample KB documents matching the 3-type schema from the assignment document.
Used to bootstrap the vector store and BM25 index for testing.
"""

# ── Type 1: Video Suggestion Logs ─────────────────────────────────────────────
COACHING_LOGS = [
    {
        "log_id": "SWM-2024-0312",
        "doc_type": "coaching_log",
        "athlete_profile": {"age": 19, "gender": "M", "event": "200m Butterfly", "weakness": ["dolphin kick"]},
        "videos_recommended": [
            {"video_id": "V-0091", "watch_pct": 92},
            {"video_id": "V-0044", "watch_pct": 78},
        ],
        "performance_delta": {"before_s": 121.4, "after_s": 119.8, "improvement_pct": 1.3},
        "weeks_to_competition": 6,
        "notes": "Athlete showed marked improvement in breakout distance after V-0091 series.",
    },
    {
        "log_id": "SWM-2024-0289",
        "doc_type": "coaching_log",
        "athlete_profile": {"age": 21, "gender": "M", "event": "200m Butterfly", "weakness": ["dolphin kick", "hip rotation"]},
        "videos_recommended": [
            {"video_id": "V-0091", "watch_pct": 88},
            {"video_id": "V-0103", "watch_pct": 95},
        ],
        "performance_delta": {"before_s": 118.2, "after_s": 117.1, "improvement_pct": 0.93},
        "weeks_to_competition": 8,
        "notes": "V-0103 hip-drive drill combined with V-0091 produced compound improvement.",
    },
    {
        "log_id": "SWM-2024-0401",
        "doc_type": "coaching_log",
        "athlete_profile": {"age": 18, "gender": "M", "event": "100m Freestyle", "weakness": ["flip turn", "underwater streamline"]},
        "videos_recommended": [
            {"video_id": "V-0017", "watch_pct": 97},
            {"video_id": "V-0022", "watch_pct": 84},
        ],
        "performance_delta": {"before_s": 52.3, "after_s": 51.8, "improvement_pct": 0.96},
        "weeks_to_competition": 4,
        "notes": "Flip turn timing corrected; underwater dolphin kick extended to 7m.",
    },
    {
        "log_id": "SWM-2024-0455",
        "doc_type": "coaching_log",
        "athlete_profile": {"age": 22, "gender": "F", "event": "200m Backstroke", "weakness": ["catch mechanics", "shoulder rotation"]},
        "videos_recommended": [
            {"video_id": "V-0067", "watch_pct": 91},
            {"video_id": "V-0071", "watch_pct": 73},
        ],
        "performance_delta": {"before_s": 133.5, "after_s": 132.1, "improvement_pct": 1.05},
        "weeks_to_competition": 10,
        "notes": "Overhead camera angle in V-0067 proved critical for catch awareness.",
    },
    {
        "log_id": "SWM-2024-0512",
        "doc_type": "coaching_log",
        "athlete_profile": {"age": 20, "gender": "M", "event": "200m Butterfly", "weakness": ["race strategy", "pacing"]},
        "videos_recommended": [
            {"video_id": "V-0091", "watch_pct": 80},
            {"video_id": "V-0115", "watch_pct": 90},
        ],
        "performance_delta": {"before_s": 120.0, "after_s": 118.5, "improvement_pct": 1.25},
        "weeks_to_competition": 5,
        "notes": "Race-strategy video V-0115 helped athlete manage split pacing more effectively.",
    },
]

# ── Type 2: Video Master Catalog ──────────────────────────────────────────────
VIDEO_CATALOG = [
    {
        "video_id": "V-0091",
        "doc_type": "video_catalog",
        "title": "Elite Butterfly Breakout & Dolphin Kick (Underwater Side Angle)",
        "topic_tags": ["butterfly", "dolphin kick", "breakout", "underwater", "advanced"],
        "difficulty": "advanced",
        "duration_s": 420,
        "camera_angle": "underwater side",
        "embedding_text": (
            "Elite butterfly breakout technique filmed from underwater side angle. "
            "Demonstrates hip-driven dolphin kick with high amplitude and minimal drag. "
            "Athlete achieves 7.5 m underwater breakout distance post-turn. "
            "Focus on tight streamline position, undulation initiation from core, "
            "and timing of first stroke pull relative to surfacing. "
            "Advanced drill suitable for competitive butterfly swimmers targeting sub-2:00 200m."
        ),
    },
    {
        "video_id": "V-0103",
        "doc_type": "video_catalog",
        "title": "Hip Drive & Body Undulation for Butterfly",
        "topic_tags": ["butterfly", "hip rotation", "body undulation", "intermediate"],
        "difficulty": "intermediate",
        "duration_s": 360,
        "camera_angle": "poolside side",
        "embedding_text": (
            "Poolside side-angle footage demonstrating hip drive mechanics for butterfly stroke. "
            "Slow-motion segments highlight the two-beat kick rhythm synchronized with arm entry. "
            "Body undulation sourced from hips and core, not shoulders. "
            "Prerequisite: basic butterfly arm cycle mastery. Intermediate level."
        ),
    },
    {
        "video_id": "V-0017",
        "doc_type": "video_catalog",
        "title": "Freestyle Flip Turn — Timing & Underwater Streamline",
        "topic_tags": ["freestyle", "flip turn", "tumble turn", "underwater streamline", "beginner"],
        "difficulty": "beginner",
        "duration_s": 300,
        "camera_angle": "underwater end wall",
        "embedding_text": (
            "Underwater end-wall camera captures freestyle flip turn from approach through push-off. "
            "Highlights: touch-to-flip timing (0.8 s), tight tuck radius, wall contact at 0.45 m depth, "
            "streamline lock before dolphin kicks. Common error corrections included. "
            "Suitable for swimmers new to competitive flip turn technique."
        ),
    },
    {
        "video_id": "V-0022",
        "doc_type": "video_catalog",
        "title": "Underwater Dolphin Kick Extension Post-Turn (Freestyle)",
        "topic_tags": ["freestyle", "dolphin kick", "underwater", "post-turn", "intermediate"],
        "difficulty": "intermediate",
        "duration_s": 390,
        "camera_angle": "underwater side",
        "embedding_text": (
            "Side-angle underwater footage of elite swimmer executing 7-8 m dolphin kick sequence "
            "after freestyle flip turn. Body depth maintained at 0.5 m for reduced drag. "
            "Kick amplitude narrows progressively toward surface. Timing cue for breakout pull shown. "
            "Intermediate drill; assumes solid flip turn mechanics."
        ),
    },
    {
        "video_id": "V-0044",
        "doc_type": "video_catalog",
        "title": "Butterfly Underwater Pull Phase — Catch to Finish",
        "topic_tags": ["butterfly", "pull phase", "catch", "underwater", "advanced"],
        "difficulty": "advanced",
        "duration_s": 450,
        "camera_angle": "underwater front",
        "embedding_text": (
            "Frontal underwater angle isolating butterfly pull phase from high-elbow catch "
            "through in-sweep and finish. Pressure points marked with coaching overlays. "
            "Common faults: dropped elbow, early breathing disrupting pull line. "
            "Advanced video; best viewed after V-0103 hip-drive foundation."
        ),
    },
    {
        "video_id": "V-0067",
        "doc_type": "video_catalog",
        "title": "Backstroke Catch Mechanics — High-Elbow Entry",
        "topic_tags": ["backstroke", "catch mechanics", "high elbow", "shoulder rotation", "intermediate"],
        "difficulty": "intermediate",
        "duration_s": 410,
        "camera_angle": "overhead",
        "embedding_text": (
            "Overhead aerial camera shows backstroke high-elbow catch from hand entry to catch lock. "
            "Shoulder rotation 45-55 degrees per stroke cycle measured. "
            "Drill progressions: single-arm, fingertip drag, full stroke with tempo trainer. "
            "Targets swimmers with dropped-wrist catch fault. Intermediate difficulty."
        ),
    },
    {
        "video_id": "V-0071",
        "doc_type": "video_catalog",
        "title": "Backstroke Shoulder Roll & Body Rotation",
        "topic_tags": ["backstroke", "shoulder rotation", "body roll", "beginner"],
        "difficulty": "beginner",
        "duration_s": 280,
        "camera_angle": "poolside side",
        "embedding_text": (
            "Side-poolside footage demonstrating optimal shoulder and hip rotation for backstroke. "
            "60-degree body roll shown in slow motion. Connection between rotation and pull power explained. "
            "Beginner-friendly with drill progressions using kickboard for isolation."
        ),
    },
    {
        "video_id": "V-0115",
        "doc_type": "video_catalog",
        "title": "Race Strategy — 200m Butterfly Split Pacing",
        "topic_tags": ["butterfly", "race strategy", "pacing", "split times", "mental conditioning"],
        "difficulty": "intermediate",
        "duration_s": 520,
        "camera_angle": "above water overhead",
        "embedding_text": (
            "Race analysis video breaking down 200m butterfly split pacing strategy. "
            "Optimal split ratios: 25-26 / 29-30 / 31-32 / 30-31 seconds per 50m. "
            "Common mistake: going out too fast on first 50m causing catastrophic fade on final 50m. "
            "Includes elite race footage comparison and stroke rate data. "
            "Intermediate mental conditioning and race awareness content."
        ),
    },
]

# ── Type 3: Athlete Profiles ──────────────────────────────────────────────────
ATHLETE_PROFILES = [
    {
        "athlete_id": "ATH-001",
        "doc_type": "athlete_profile",
        "age": 19,
        "gender": "M",
        "primary_event": "200m Butterfly",
        "pb_seconds": 121.4,
        "weaknesses": ["dolphin kick"],
        "videos_watched": ["V-0103"],
        "embedding_text": (
            "Male butterfly swimmer aged 19. Primary event 200m butterfly, PB 2:01.4. "
            "Identified weakness: dolphin kick depth and amplitude during breakout. "
            "Has watched hip-drive drill video. Needs advanced breakout and kick extension content. "
            "6 weeks to State Championships."
        ),
    },
]
