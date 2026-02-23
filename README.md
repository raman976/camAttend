# camAttend - Face Recognition Attendance System

> **Note:** This is Version 1.0 - A Minimum Viable Product (MVP) focused on core functionality and proof of concept.

## What is this?

camAttend is a face recognition-based attendance tracking system that I built to automate the tedious process of marking attendance. Instead of calling out names or passing around sign-up sheets, you just take a photo of the class or group, and the system automatically identifies who's present.

The idea is simple: students enroll once with their photo, and from then on, the system can recognize them in group photos. No more manual attendance sheets, no more proxies.

## Architecture

I've designed this system with a modular approach, where each component handles a specific part of the recognition pipeline. Here's how it all works together:

### Core Components

**1. Face Detection (MTCNN)**

- Uses Multi-task Cascaded Convolutional Networks for detecting faces in images
- Handles multiple faces in a single frame
- Pretty robust against different lighting conditions and angles
- Returns bounding boxes and facial landmarks

**2. Face Recognition (InsightFace)**

- Leverages ArcFace models for generating face embeddings
- Converts detected faces into 512-dimensional vectors
- These embeddings capture unique facial features
- State-of-the-art accuracy for face verification

**3. Vector Search Engine (FAISS)**

- Facebook's library for efficient similarity search
- Stores and indexes all enrolled student embeddings
- Performs lightning-fast nearest neighbor searches
- Scales well even with thousands of enrolled students

**4. Database Layer (Supabase)**

- PostgreSQL-based backend for storing student records
- Handles user authentication securely with bcrypt
- Stores attendance logs with timestamps
- Cloud-hosted, so no need to manage servers

### How It Works

```
Input Image → Face Detection → Face Alignment → Embedding Generation →
FAISS Search → Match Found → Mark Attendance → Update Database
```

1. **Enrollment Phase**: When a student enrolls, their photo is processed to extract facial embeddings, which are stored in both FAISS index and Supabase database.

2. **Recognition Phase**: During attendance, a group photo is captured. The system:
   - Detects all faces in the image
   - Generates embeddings for each detected face
   - Searches the FAISS index for nearest matches
   - Applies a confidence threshold to prevent false positives
   - Marks attendance for matched students

3. **Data Management**: All student information, embeddings, and attendance logs are stored in Supabase for easy retrieval and analytics.

## Tech Stack

- **Language**: Python 3.12
- **Face Detection**: PyTorch + MTCNN (facenet-pytorch)
- **Face Recognition**: InsightFace (ArcFace)
- **Vector Database**: FAISS (CPU version)
- **Backend Database**: Supabase (PostgreSQL)
- **Image Processing**: OpenCV, PIL
- **Password Hashing**: bcrypt
- **Environment Management**: python-dotenv

## Setup Instructions

### Prerequisites

- Python 3.12 or higher
- A Supabase account (free tier works fine)
- Basic understanding of virtual environments

### Installation

1. **Clone the repository**

   ```bash
   git clone <your-repo-url>
   cd camAttend
   ```

2. **Create and activate virtual environment**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**

   Create a `.env` file in the project root:

   ```env
   SUPABASE_URL=your_supabase_project_url
   SUPABASE_KEY=your_supabase_anon_key
   ```

   Get these credentials from your Supabase project dashboard.

5. **Run the application**
   ```bash
   streamlit run app.py
   ```

## Project Structure

```
camAttend/
├── app.py                 # Main application interface
├── core/
│   ├── detector.py       # MTCNN face detection
│   ├── embedder.py       # InsightFace embedding generation
│   └── matcher.py        # FAISS-based face matching
├── database/
│   └── supabase_db.py    # Database operations
├── requirements.txt       # Python dependencies
└── README.md
```

## Current Limitations (v1.0 MVP)

This is an MVP, so there are some things I'm planning to improve in future versions:

- **Performance**: Face detection and embedding generation can be slow on CPU. GPU support would help.
- **Accuracy**: The confidence threshold is fixed. Adaptive thresholding could reduce false matches.
- **Scalability**: FAISS index is rebuilt on every restart. Persistent indexing needed.
- **UI/UX**: The interface is functional but could be more polished.
- **Edge Cases**: Poor lighting, multiple similar faces, or masks can affect accuracy.
- **Analytics**: Basic attendance tracking works, but detailed reports and visualizations are missing.

## Future Plans

- [ ] GPU acceleration for faster processing
- [ ] Persistent FAISS index storage
- [ ] Advanced analytics dashboard
- [ ] Attendance reports (CSV export, monthly summaries)
- [ ] Multi-camera support
- [ ] Real-time video stream recognition
- [ ] Mobile app integration
- [ ] Better error handling and logging

## Contributing

This is a personal project and an MVP, but if you find bugs or have suggestions, feel free to open an issue or reach out.

## License

MIT License - feel free to use this for your own projects.

---

**Version**: 1.0.0 (MVP)  
**Status**: Active Development  
**Last Updated**: February 2026
