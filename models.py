from datetime import datetime
from app import db
from sqlalchemy import Column, Integer, String, DateTime, Float, Text, Boolean, ForeignKey
from sqlalchemy.orm import relationship


class ImageUpload(db.Model):
    """Model for tracking uploaded images"""
    __tablename__ = 'image_uploads'
    
    id = Column(Integer, primary_key=True)
    filename = Column(String(255), unique=True, nullable=False)
    original_filename = Column(String(255), nullable=False)
    file_size = Column(Integer, nullable=False)
    width = Column(Integer, nullable=False)
    height = Column(Integer, nullable=False)
    channels = Column(Integer, nullable=False)
    format = Column(String(20), nullable=False)
    upload_date = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    analyses = relationship('ImageAnalysis', back_populates='image', cascade='all, delete-orphan')
    comparisons = relationship('ImageComparison', back_populates='image1', foreign_keys='ImageComparison.image1_id')
    
    def __repr__(self):
        return f'<ImageUpload {self.filename}>'


class ImageAnalysis(db.Model):
    """Model for storing image analysis results"""
    __tablename__ = 'image_analyses'
    
    id = Column(Integer, primary_key=True)
    image_id = Column(Integer, ForeignKey('image_uploads.id'), nullable=False)
    analysis_type = Column(String(50), nullable=False)  # 'statistics', 'histogram', 'conversion'
    analysis_date = Column(DateTime, default=datetime.utcnow)
    
    # Statistics data
    mean_brightness = Column(Float)
    std_brightness = Column(Float)
    min_brightness = Column(Float)
    max_brightness = Column(Float)
    
    # Channel statistics (JSON stored as text)
    channel_stats = Column(Text)
    
    # Histogram data (JSON stored as text)
    histogram_data = Column(Text)
    
    # Texture features
    contrast = Column(Float)
    homogeneity = Column(Float)
    energy = Column(Float)
    correlation = Column(Float)
    
    # Conversion info
    conversion_type = Column(String(20))  # 'grayscale', 'binary', 'hsv'
    converted_filename = Column(String(255))
    
    # Relationship
    image = relationship('ImageUpload', back_populates='analyses')
    
    def __repr__(self):
        return f'<ImageAnalysis {self.analysis_type} for {self.image.filename}>'


class ImageComparison(db.Model):
    """Model for storing image comparison results"""
    __tablename__ = 'image_comparisons'
    
    id = Column(Integer, primary_key=True)
    image1_id = Column(Integer, ForeignKey('image_uploads.id'), nullable=False)
    image2_filename = Column(String(255), nullable=False)  # For external comparison images
    comparison_date = Column(DateTime, default=datetime.utcnow)
    
    # Similarity metrics
    ssim_score = Column(Float)
    ssim_interpretation = Column(String(50))
    psnr_score = Column(Float)
    psnr_interpretation = Column(String(50))
    mse_score = Column(Float)
    mse_interpretation = Column(String(50))
    
    # Overall similarity
    overall_similarity = Column(Float)
    overall_interpretation = Column(String(50))
    
    # Difference map filename
    difference_map = Column(String(255))
    
    # Histogram comparison data (JSON stored as text)
    histogram_comparison = Column(Text)
    
    # Relationship
    image1 = relationship('ImageUpload', back_populates='comparisons')
    
    def __repr__(self):
        return f'<ImageComparison {self.image1.filename} vs {self.image2_filename}>'


class PixelExploration(db.Model):
    """Model for storing pixel exploration data"""
    __tablename__ = 'pixel_explorations'
    
    id = Column(Integer, primary_key=True)
    image_id = Column(Integer, ForeignKey('image_uploads.id'), nullable=False)
    x_coordinate = Column(Integer, nullable=False)
    y_coordinate = Column(Integer, nullable=False)
    exploration_date = Column(DateTime, default=datetime.utcnow)
    
    # Pixel values
    rgb_r = Column(Integer)
    rgb_g = Column(Integer)
    rgb_b = Column(Integer)
    hsv_h = Column(Float)
    hsv_s = Column(Float)
    hsv_v = Column(Float)
    lab_l = Column(Float)
    lab_a = Column(Float)
    lab_b = Column(Float)
    brightness = Column(Float)
    hex_color = Column(String(7))
    
    # Relationship
    image = relationship('ImageUpload')
    
    def __repr__(self):
        return f'<PixelExploration ({self.x_coordinate}, {self.y_coordinate}) for {self.image.filename}>'


class ImageProcessingLog(db.Model):
    """Model for logging image processing operations"""
    __tablename__ = 'processing_logs'
    
    id = Column(Integer, primary_key=True)
    image_id = Column(Integer, ForeignKey('image_uploads.id'), nullable=False)
    operation = Column(String(50), nullable=False)  # 'crop', 'rotate', 'flip', 'convert'
    operation_date = Column(DateTime, default=datetime.utcnow)
    
    # Operation parameters (JSON stored as text)
    parameters = Column(Text)
    
    # Result
    success = Column(Boolean, default=True)
    result_filename = Column(String(255))
    error_message = Column(Text)
    
    # Relationship
    image = relationship('ImageUpload')
    
    def __repr__(self):
        return f'<ProcessingLog {self.operation} for {self.image.filename}>'