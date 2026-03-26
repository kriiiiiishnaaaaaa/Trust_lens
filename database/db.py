"""
TrustLens - Database Layer
SQLAlchemy models + connection manager.
Supports: PostgreSQL (production) and SQLite (development/fallback).

Configure via .env:
  DATABASE_URL=postgresql://user:pass@localhost:5432/trustlens
  (or leave blank to use SQLite)
"""

import os
import uuid
import logging
from datetime import datetime
from contextlib import contextmanager

from sqlalchemy import (
    create_engine, Column, String, Float, Integer,
    Boolean, DateTime, Text, Enum, ForeignKey, Index
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
import enum

logger = logging.getLogger(__name__)
Base = declarative_base()


# ─────────────────────────────────────────────
#  Enums
# ─────────────────────────────────────────────

class MediaType(str, enum.Enum):
    IMAGE = 'image'
    VIDEO = 'video'


class DetectionLabel(str, enum.Enum):
    REAL = 'REAL'
    FAKE = 'FAKE'
    INCONCLUSIVE = 'INCONCLUSIVE'


# ─────────────────────────────────────────────
#  Models
# ─────────────────────────────────────────────

class AnalysisResult(Base):
    """
    Stores each deepfake analysis result.
    """
    __tablename__ = 'analysis_results'

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    # File info
    filename = Column(String(255), nullable=False)
    file_size_bytes = Column(Integer)
    media_type = Column(String(10), nullable=False)  # 'image' or 'video'
    file_hash = Column(String(64), index=True)  # SHA256 for deduplication

    # Detection results
    label = Column(String(20), nullable=False)         # REAL / FAKE / INCONCLUSIVE
    confidence = Column(Float, nullable=False)          # 0.0 – 100.0
    fake_probability = Column(Float, nullable=False)    # 0.0 – 100.0
    real_probability = Column(Float, nullable=False)    # 0.0 – 100.0

    # Video-specific
    frames_analyzed = Column(Integer, nullable=True)
    total_frames = Column(Integer, nullable=True)
    duration_seconds = Column(Float, nullable=True)

    # Metadata
    analysis_method = Column(String(100), default='EfficientNet-B4')
    model_version = Column(String(50), default='1.0.0')
    processing_time_ms = Column(Float)                  # Time taken for analysis
    face_detected = Column(Boolean, default=True)
    notes = Column(Text, nullable=True)

    # User association (optional — for future auth)
    user_id = Column(String(36), ForeignKey('users.id', ondelete='SET NULL'), nullable=True)
    user = relationship('User', back_populates='analyses')

    __table_args__ = (
        Index('idx_created_at', 'created_at'),
        Index('idx_label', 'label'),
        Index('idx_user_id', 'user_id'),
    )

    def to_dict(self):
        return {
            'id': self.id,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'filename': self.filename,
            'file_size_bytes': self.file_size_bytes,
            'media_type': self.media_type,
            'label': self.label,
            'confidence': self.confidence,
            'fake_probability': self.fake_probability,
            'real_probability': self.real_probability,
            'frames_analyzed': self.frames_analyzed,
            'duration_seconds': self.duration_seconds,
            'analysis_method': self.analysis_method,
            'processing_time_ms': self.processing_time_ms,
            'face_detected': self.face_detected,
        }

    def __repr__(self):
        return f"<AnalysisResult id={self.id} label={self.label} conf={self.confidence:.1f}%>"


class User(Base):
    """
    Optional user accounts for tracking analysis history per user.
    """
    __tablename__ = 'users'

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    created_at = Column(DateTime, default=datetime.utcnow)
    email = Column(String(255), unique=True, nullable=True)
    username = Column(String(100), unique=True, nullable=True)
    api_key = Column(String(64), unique=True, nullable=True, index=True)
    is_active = Column(Boolean, default=True)
    total_analyses = Column(Integer, default=0)

    analyses = relationship('AnalysisResult', back_populates='user', cascade='all, delete-orphan')

    def to_dict(self):
        return {
            'id': self.id,
            'username': self.username,
            'email': self.email,
            'total_analyses': self.total_analyses,
            'created_at': self.created_at.isoformat()
        }


class SystemStats(Base):
    """
    Aggregate statistics snapshot (updated periodically).
    """
    __tablename__ = 'system_stats'

    id = Column(Integer, primary_key=True, autoincrement=True)
    recorded_at = Column(DateTime, default=datetime.utcnow)
    total_analyses = Column(Integer, default=0)
    total_fake_detected = Column(Integer, default=0)
    total_real_detected = Column(Integer, default=0)
    avg_confidence = Column(Float, default=0.0)
    total_images = Column(Integer, default=0)
    total_videos = Column(Integer, default=0)


# ─────────────────────────────────────────────
#  Connection Manager
# ─────────────────────────────────────────────

class DatabaseManager:
    """
    Manages database connection and sessions.
    Auto-detects PostgreSQL vs SQLite from DATABASE_URL env var.
    """

    def __init__(self, database_url: str = None):
        self.database_url = database_url or os.getenv(
            'DATABASE_URL', 'sqlite:///trustlens.db'
        )

        # Fix Heroku-style postgres:// URLs
        if self.database_url.startswith('postgres://'):
            self.database_url = self.database_url.replace('postgres://', 'postgresql://', 1)

        is_sqlite = self.database_url.startswith('sqlite')
        connect_args = {'check_same_thread': False} if is_sqlite else {}

        self.engine = create_engine(
            self.database_url,
            connect_args=connect_args,
            pool_pre_ping=True,
            pool_size=5 if not is_sqlite else 1,
            max_overflow=10 if not is_sqlite else 0,
            echo=os.getenv('DB_ECHO', 'false').lower() == 'true'
        )

        self.SessionLocal = sessionmaker(
            autocommit=False, autoflush=False, bind=self.engine
        )

        db_type = 'SQLite' if is_sqlite else 'PostgreSQL'
        logger.info(f"Database: {db_type} | URL: {self._safe_url()}")

    def _safe_url(self) -> str:
        """Returns DB URL with password masked."""
        if '@' in self.database_url:
            parts = self.database_url.split('@')
            creds = parts[0].split('://')
            return f"{creds[0]}://***:***@{parts[1]}"
        return self.database_url

    def create_tables(self):
        """Create all tables if they don't exist."""
        Base.metadata.create_all(bind=self.engine)
        logger.info("Database tables created/verified")

    def drop_tables(self):
        """Drop all tables (use with caution!)."""
        Base.metadata.drop_all(bind=self.engine)
        logger.warning("All database tables dropped")

    @contextmanager
    def get_session(self):
        """Context manager for database sessions with auto-rollback on error."""
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Database session error: {e}")
            raise
        finally:
            session.close()

    # ── CRUD Operations ──

    def save_analysis(self, result: dict, filename: str, file_size: int,
                      media_type: str, processing_time_ms: float,
                      user_id: str = None, file_hash: str = None) -> AnalysisResult:
        """Save a detection result to the database."""
        with self.get_session() as session:
            record = AnalysisResult(
                filename=filename,
                file_size_bytes=file_size,
                media_type=media_type,
                file_hash=file_hash,
                label=result.get('label', 'INCONCLUSIVE'),
                confidence=result.get('confidence', 0.0),
                fake_probability=result.get('fake_probability', 0.0),
                real_probability=result.get('real_probability', 0.0),
                frames_analyzed=result.get('frames_analyzed'),
                total_frames=result.get('total_frames'),
                duration_seconds=result.get('duration_seconds'),
                analysis_method=result.get('analysis_method', 'EfficientNet-B4'),
                processing_time_ms=processing_time_ms,
                face_detected=result.get('face_detected', True),
                user_id=user_id,
            )
            session.add(record)
            session.flush()
            return record.to_dict()

    def get_recent_analyses(self, limit: int = 50, user_id: str = None) -> list:
        """Fetch recent analysis results."""
        with self.get_session() as session:
            query = session.query(AnalysisResult).order_by(AnalysisResult.created_at.desc())
            if user_id:
                query = query.filter(AnalysisResult.user_id == user_id)
            results = query.limit(limit).all()
            return [r.to_dict() for r in results]

    def get_stats(self) -> dict:
        """Returns aggregate statistics."""
        with self.get_session() as session:
            total = session.query(AnalysisResult).count()
            fake_count = session.query(AnalysisResult).filter(
                AnalysisResult.label == 'FAKE'
            ).count()
            real_count = session.query(AnalysisResult).filter(
                AnalysisResult.label == 'REAL'
            ).count()

            from sqlalchemy import func
            avg_conf = session.query(
                func.avg(AnalysisResult.confidence)
            ).scalar() or 0.0

            return {
                'total_analyses': total,
                'fake_detected': fake_count,
                'real_detected': real_count,
                'avg_confidence': round(float(avg_conf), 2),
                'fake_rate': round(fake_count / total * 100, 1) if total > 0 else 0
            }

    def check_duplicate(self, file_hash: str) -> dict | None:
        """Check if this file was analyzed before (by hash)."""
        if not file_hash:
            return None
        with self.get_session() as session:
            result = session.query(AnalysisResult).filter(
                AnalysisResult.file_hash == file_hash
            ).order_by(AnalysisResult.created_at.desc()).first()
            return result.to_dict() if result else None


# ─────────────────────────────────────────────
#  Singleton instance
# ─────────────────────────────────────────────
_db_manager = None


def get_db() -> DatabaseManager:
    global _db_manager
    if _db_manager is None:
        _db_manager = DatabaseManager()
        _db_manager.create_tables()
    return _db_manager
