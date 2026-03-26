"""
TrustLens - Flask Backend API
Endpoints:
  POST /api/analyze        → Upload & analyze image/video
  GET  /api/history        → Recent analysis results
  GET  /api/stats          → System statistics
  GET  /api/health         → Health check
  GET  /                   → Serve frontend
"""

import os
import time
import hashlib
import logging
import tempfile
from pathlib import Path
from dotenv import load_dotenv
from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename

load_dotenv()

# ─────────────────────────────────────────────
#  App Setup
# ─────────────────────────────────────────────

app = Flask(__name__, template_folder='templates', static_folder='static')
CORS(app)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)

# Config
UPLOAD_FOLDER = Path('uploads')
UPLOAD_FOLDER.mkdir(exist_ok=True)
MAX_FILE_SIZE = 100 * 1024 * 1024  # 100 MB
ALLOWED_IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
ALLOWED_VIDEO_EXTS = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}
ALLOWED_EXTS = ALLOWED_IMAGE_EXTS | ALLOWED_VIDEO_EXTS

app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

# ─────────────────────────────────────────────
#  Lazy-load ML model & DB (avoids slow startup)
# ─────────────────────────────────────────────

_pipeline = None
_db = None


def get_pipeline():
    global _pipeline
    if _pipeline is None:
        from model.deepfake_detector import DeepfakeDetectionPipeline
        model_path = os.getenv('MODEL_PATH', 'checkpoints/best_model.pth')
        _pipeline = DeepfakeDetectionPipeline(model_path=model_path if os.path.exists(model_path) else None)
        logger.info("ML pipeline loaded")
    return _pipeline


def get_db():
    global _db
    if _db is None:
        from database.db import get_db as _get_db
        _db = _get_db()
        logger.info("Database connected")
    return _db


# ─────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────

def allowed_file(filename: str) -> bool:
    return Path(filename).suffix.lower() in ALLOWED_EXTS


def get_media_type(filename: str) -> str:
    return 'video' if Path(filename).suffix.lower() in ALLOWED_VIDEO_EXTS else 'image'


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def format_file_size(size_bytes: int) -> str:
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} TB"


# ─────────────────────────────────────────────
#  Routes
# ─────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/health')
def health():
    groq_configured = bool(os.getenv('GROQ_API_KEY'))
    return jsonify({
        'status': 'ok',
        'model': os.getenv('GROQ_MODEL', 'meta-llama/llama-4-scout-17b-16e-instruct'),
        'version': '2.0.0',
        'engine': 'Groq Vision AI',
        'groq_configured': groq_configured
    })


@app.route('/api/analyze', methods=['POST'])
def analyze():
    """Main analysis endpoint."""
    start_time = time.time()

    if 'file' not in request.files:
        return jsonify({'success': False, 'error': 'No file provided'}), 400

    file = request.files['file']
    if not file.filename:
        return jsonify({'success': False, 'error': 'Empty filename'}), 400

    if not allowed_file(file.filename):
        return jsonify({
            'success': False,
            'error': f'Unsupported file type. Allowed: {", ".join(ALLOWED_EXTS)}'
        }), 400

    filename = secure_filename(file.filename)
    media_type = get_media_type(filename)
    file_data = file.read()
    file_size = len(file_data)
    file_hash = sha256_bytes(file_data)

    try:
        db = get_db()

        # Check for duplicate analysis
        existing = db.check_duplicate(file_hash)
        if existing and not request.form.get('force_reanalyze'):
            processing_time = (time.time() - start_time) * 1000
            return jsonify({
                **existing,
                'cached': True,
                'message': 'Previously analyzed file — cached result returned',
                'processing_time_ms': round(processing_time, 1)
            })

        pipeline = get_pipeline()

        # Run detection
        if media_type == 'image':
            result = pipeline.predict_image(file_data)
        else:
            # Save video temporarily for processing
            with tempfile.NamedTemporaryFile(
                suffix=Path(filename).suffix, delete=False
            ) as tmp:
                tmp.write(file_data)
                tmp_path = tmp.name

            try:
                result = pipeline.predict_video(tmp_path)
            finally:
                os.unlink(tmp_path)

        if not result.get('success'):
            return jsonify({'success': False, 'error': result.get('error', 'Analysis failed')}), 500

        processing_time = (time.time() - start_time) * 1000

        # Save to database
        try:
            saved = db.save_analysis(
                result=result,
                filename=filename,
                file_size=file_size,
                media_type=media_type,
                processing_time_ms=round(processing_time, 1),
                file_hash=file_hash
            )
        except Exception as db_error:
            logger.warning(f"DB save failed (non-critical): {db_error}")
            saved = {}

        return jsonify({
            **result,
            'cached': False,
            'file_size': format_file_size(file_size),
            'processing_time_ms': round(processing_time, 1),
            'analysis_id': saved.get('id'),
        })

    except Exception as e:
        logger.exception(f"Analysis error: {e}")
        return jsonify({'success': False, 'error': 'Internal server error'}), 500


@app.route('/api/history')
def history():
    """Returns recent analysis history."""
    try:
        limit = min(int(request.args.get('limit', 50)), 200)
        db = get_db()
        results = db.get_recent_analyses(limit=limit)
        return jsonify({'success': True, 'results': results, 'count': len(results)})
    except Exception as e:
        logger.error(f"History error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/stats')
def stats():
    """Returns aggregate statistics."""
    try:
        db = get_db()
        return jsonify({'success': True, **db.get_stats()})
    except Exception as e:
        logger.error(f"Stats error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.errorhandler(413)
def too_large(e):
    return jsonify({'success': False, 'error': 'File too large. Maximum size is 100MB'}), 413


@app.errorhandler(404)
def not_found(e):
    return jsonify({'success': False, 'error': 'Endpoint not found'}), 404


# ─────────────────────────────────────────────
#  Entry Point
# ─────────────────────────────────────────────

if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    debug = os.getenv('FLASK_DEBUG', 'false').lower() == 'true'
    logger.info(f"🚀 TrustLens running at http://localhost:{port}")
    app.run(host='0.0.0.0', port=port, debug=debug)
