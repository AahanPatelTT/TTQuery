#!/usr/bin/env python3
"""
Web interface for real-time document uploads and processing.

This Flask app provides endpoints for:
- Uploading documents to specific folders
- Checking processing status
- Triggering background embedding processing
- Managing the knowledge base through a web interface

Usage:
    python web_upload.py
    # Then open http://localhost:5000 in your browser
"""

import os
import sys
import logging
import time
import threading
from pathlib import Path
from typing import Dict, List, Optional
from flask import Flask, request, jsonify, render_template_string, send_from_directory
from werkzeug.utils import secure_filename

# Add pipeline to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'pipeline'))

from pipeline.database import SynapseDB
from pipeline.fast_embed import FastEmbeddingService


# Flask app configuration
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size
app.config['UPLOAD_FOLDER'] = 'Data'

# Global services
db = None
embedding_service = None
background_thread = None

# Allowed file extensions
ALLOWED_EXTENSIONS = {
    'pdf', 'pptx', 'docx', 'txt', 'md', 'csv',
    'png', 'jpg', 'jpeg', 'tiff', 'bmp'
}


def allowed_file(filename):
    """Check if file extension is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def initialize_services():
    """Initialize database and embedding services."""
    global db, embedding_service, background_thread
    
    try:
        # Initialize database
        db = SynapseDB()
        logging.info("Database initialized")
        
        # Initialize embedding service
        embedding_service = FastEmbeddingService(
            provider="local",
            model_name="BAAI/bge-large-en-v1.5",
            batch_size=32,
            max_workers=1  # Single worker for web uploads
        )
        logging.info("Embedding service initialized")
        
        # Start background processing
        background_thread = embedding_service.start_background_processing()
        logging.info("Background processing started")
        
        return True
        
    except Exception as e:
        logging.error(f"Failed to initialize services: {e}")
        return False


# HTML template for the upload interface
UPLOAD_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Synapse Document Upload</title>
    <style>
        body { 
            font-family: Arial, sans-serif; 
            max-width: 800px; 
            margin: 0 auto; 
            padding: 20px;
            background-color: #f5f5f5;
        }
        .header { 
            text-align: center; 
            margin-bottom: 30px;
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }
        .upload-form { 
            background: white; 
            padding: 30px; 
            border-radius: 10px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }
        .form-group { 
            margin-bottom: 20px; 
        }
        label { 
            display: block; 
            margin-bottom: 5px; 
            font-weight: bold;
        }
        input[type="file"], select { 
            width: 100%; 
            padding: 10px; 
            border: 1px solid #ddd; 
            border-radius: 5px;
        }
        button { 
            background-color: #007bff; 
            color: white; 
            padding: 12px 24px; 
            border: none; 
            border-radius: 5px; 
            cursor: pointer;
            font-size: 16px;
        }
        button:hover { 
            background-color: #0056b3; 
        }
        .status-section {
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }
        .status-item {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 10px;
            border-bottom: 1px solid #eee;
        }
        .status-item:last-child {
            border-bottom: none;
        }
        .progress {
            background-color: #e9ecef;
            border-radius: 10px;
            height: 20px;
            margin-top: 5px;
        }
        .progress-bar {
            background-color: #28a745;
            height: 100%;
            border-radius: 10px;
            transition: width 0.3s ease;
        }
        .message {
            padding: 10px;
            margin: 10px 0;
            border-radius: 5px;
        }
        .success { 
            background-color: #d4edda; 
            border: 1px solid #c3e6cb; 
            color: #155724; 
        }
        .error { 
            background-color: #f8d7da; 
            border: 1px solid #f5c6cb; 
            color: #721c24; 
        }
        .info {
            background-color: #d1ecf1;
            border: 1px solid #bee5eb;
            color: #0c5460;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>🚀 Synapse Document Upload</h1>
        <p>Upload documents for real-time processing and embedding generation</p>
    </div>

    <div class="upload-form">
        <h2>📁 Upload Documents</h2>
        <form id="uploadForm" enctype="multipart/form-data">
            <div class="form-group">
                <label for="files">Select Files:</label>
                <input type="file" id="files" name="files" multiple accept=".pdf,.pptx,.docx,.txt,.md,.csv,.png,.jpg,.jpeg,.tiff,.bmp" required>
                <small>Supported formats: PDF, PPTX, DOCX, TXT, MD, CSV, PNG, JPG, JPEG, TIFF, BMP</small>
            </div>
            
            <div class="form-group">
                <label for="folder">Target Folder:</label>
                <select id="folder" name="folder">
                    <option value="uploads">uploads</option>
                    <option value="documents">documents</option>
                    <option value="research">research</option>
                    <option value="technical">technical</option>
                </select>
                <small>Or type a custom folder name</small>
            </div>
            
            <button type="submit">📤 Upload & Process</button>
        </form>
        
        <div id="uploadMessages"></div>
    </div>

    <div class="status-section">
        <h2>📊 Processing Status</h2>
        <button onclick="refreshStatus()">🔄 Refresh Status</button>
        <div id="statusContainer">
            <p>Loading status...</p>
        </div>
    </div>

    <script>
        // Upload form handling
        document.getElementById('uploadForm').addEventListener('submit', async function(e) {
            e.preventDefault();
            
            const formData = new FormData();
            const files = document.getElementById('files').files;
            const folder = document.getElementById('folder').value;
            
            if (files.length === 0) {
                showMessage('Please select at least one file', 'error');
                return;
            }
            
            // Add files to form data
            for (let file of files) {
                formData.append('files', file);
            }
            formData.append('folder', folder);
            
            try {
                showMessage('Uploading files...', 'info');
                
                const response = await fetch('/api/upload', {
                    method: 'POST',
                    body: formData
                });
                
                const result = await response.json();
                
                if (response.ok) {
                    showMessage(result.message, 'success');
                    refreshStatus();
                    document.getElementById('uploadForm').reset();
                } else {
                    showMessage(result.error || 'Upload failed', 'error');
                }
                
            } catch (error) {
                showMessage('Upload failed: ' + error.message, 'error');
            }
        });
        
        // Status refresh
        async function refreshStatus() {
            try {
                const response = await fetch('/api/status');
                const status = await response.json();
                
                let html = '';
                if (status.folders && status.folders.length > 0) {
                    for (let folder of status.folders) {
                        const completionRate = folder.completion_rate || 0;
                        html += `
                            <div class="status-item">
                                <div>
                                    <strong>📁 ${folder.name}</strong><br>
                                    <small>${folder.total_chunks} chunks, ${folder.embedded_chunks} embedded</small>
                                </div>
                                <div style="width: 200px;">
                                    <div class="progress">
                                        <div class="progress-bar" style="width: ${completionRate}%"></div>
                                    </div>
                                    <small>${completionRate.toFixed(1)}% complete</small>
                                </div>
                            </div>
                        `;
                    }
                } else {
                    html = '<p>No folders found. Upload some documents to get started!</p>';
                }
                
                document.getElementById('statusContainer').innerHTML = html;
                
            } catch (error) {
                document.getElementById('statusContainer').innerHTML = 
                    '<p class="error">Failed to load status: ' + error.message + '</p>';
            }
        }
        
        // Message display
        function showMessage(message, type) {
            const container = document.getElementById('uploadMessages');
            const div = document.createElement('div');
            div.className = 'message ' + type;
            div.textContent = message;
            container.appendChild(div);
            
            // Auto-remove after 5 seconds
            setTimeout(() => {
                div.remove();
            }, 5000);
        }
        
        // Load status on page load
        refreshStatus();
        
        // Auto-refresh status every 30 seconds
        setInterval(refreshStatus, 30000);
    </script>
</body>
</html>
"""


@app.route('/')
def index():
    """Main upload interface."""
    return render_template_string(UPLOAD_TEMPLATE)


@app.route('/api/upload', methods=['POST'])
def upload_files():
    """Upload files to a specific folder."""
    try:
        if 'files' not in request.files:
            return jsonify({'error': 'No files provided'}), 400
        
        files = request.files.getlist('files')
        folder = request.form.get('folder', 'uploads')
        
        if not files or all(file.filename == '' for file in files):
            return jsonify({'error': 'No files selected'}), 400
        
        # Sanitize folder name
        folder = secure_filename(folder) or 'uploads'
        folder_path = os.path.join(app.config['UPLOAD_FOLDER'], folder)
        os.makedirs(folder_path, exist_ok=True)
        
        uploaded_files = []
        failed_files = []
        
        for file in files:
            if file and file.filename and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file_path = os.path.join(folder_path, filename)
                
                try:
                    file.save(file_path)
                    
                    # Mark for processing in database
                    if db:
                        db.mark_document_processing(file_path, folder)
                    
                    uploaded_files.append({
                        'filename': filename,
                        'path': file_path,
                        'folder': folder
                    })
                    
                    logging.info(f"Uploaded file: {file_path}")
                    
                except Exception as e:
                    failed_files.append({
                        'filename': file.filename,
                        'error': str(e)
                    })
                    logging.error(f"Failed to upload {file.filename}: {e}")
            else:
                failed_files.append({
                    'filename': file.filename or 'Unknown',
                    'error': 'Invalid file type or empty file'
                })
        
        # Trigger background processing
        if uploaded_files and embedding_service:
            try:
                # Process the folder in background (will handle parsing and embedding)
                threading.Thread(
                    target=lambda: process_folder_async(folder),
                    daemon=True
                ).start()
            except Exception as e:
                logging.warning(f"Failed to trigger background processing: {e}")
        
        message = f"Successfully uploaded {len(uploaded_files)} files"
        if failed_files:
            message += f", {len(failed_files)} files failed"
        
        return jsonify({
            'success': True,
            'message': message,
            'uploaded_files': uploaded_files,
            'failed_files': failed_files,
            'folder': folder
        })
    
    except Exception as e:
        logging.error(f"Upload error: {e}")
        return jsonify({'error': str(e)}), 500


def process_folder_async(folder_key: str):
    """Process a folder asynchronously in the background."""
    try:
        # Give a moment for files to be written
        time.sleep(2)
        
        # Process pending embeddings for this folder
        if embedding_service:
            processed = embedding_service.process_folder_immediately(folder_key)
            logging.info(f"Background processing completed for {folder_key}: {processed} chunks processed")
            
    except Exception as e:
        logging.error(f"Background processing failed for {folder_key}: {e}")


@app.route('/api/status')
def get_status():
    """Get processing status for all folders."""
    try:
        if not db:
            return jsonify({'error': 'Database not available'}), 500
        
        folders = db.get_all_folders()
        folder_status = []
        
        for folder_key in folders:
            stats = db.get_folder_stats(folder_key)
            
            if embedding_service:
                embed_status = embedding_service.get_embedding_status(folder_key)
                completion_rate = embed_status['completion_rate']
            else:
                completion_rate = 0
            
            folder_status.append({
                'name': folder_key,
                'display_name': folder_key.replace('_', ' ').replace('hash_', '#'),
                'total_docs': stats['completed_docs'],
                'pending_docs': stats['pending_docs'],
                'failed_docs': stats['failed_docs'],
                'total_chunks': stats['total_chunks'],
                'embedded_chunks': stats['embedded_chunks'],
                'pending_embeddings': stats['pending_embeddings'],
                'completion_rate': completion_rate
            })
        
        return jsonify({
            'folders': folder_status,
            'total_folders': len(folders)
        })
    
    except Exception as e:
        logging.error(f"Status error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/status/<folder_key>')
def get_folder_status(folder_key):
    """Get processing status for a specific folder."""
    try:
        if not db:
            return jsonify({'error': 'Database not available'}), 500
        
        stats = db.get_folder_stats(folder_key)
        
        if embedding_service:
            embed_status = embedding_service.get_embedding_status(folder_key)
            completion_rate = embed_status['completion_rate']
        else:
            completion_rate = 0
        
        return jsonify({
            'folder': folder_key,
            'display_name': folder_key.replace('_', ' ').replace('hash_', '#'),
            'total_docs': stats['completed_docs'],
            'pending_docs': stats['pending_docs'],
            'failed_docs': stats['failed_docs'],
            'total_chunks': stats['total_chunks'],
            'embedded_chunks': stats['embedded_chunks'],
            'pending_embeddings': stats['pending_embeddings'],
            'completion_rate': completion_rate,
            'status': 'processing' if stats['pending_docs'] > 0 or stats['pending_embeddings'] > 0 else 'complete'
        })
    
    except Exception as e:
        logging.error(f"Folder status error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/process/<folder_key>', methods=['POST'])
def trigger_processing(folder_key):
    """Manually trigger processing for a specific folder."""
    try:
        if not embedding_service:
            return jsonify({'error': 'Embedding service not available'}), 500
        
        # Process the folder immediately
        processed = embedding_service.process_folder_immediately(folder_key)
        
        return jsonify({
            'success': True,
            'message': f'Processed {processed} chunks for folder {folder_key}',
            'processed_count': processed
        })
    
    except Exception as e:
        logging.error(f"Processing trigger error: {e}")
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('web_upload.log')
        ]
    )
    
    # Initialize services
    if not initialize_services():
        print("❌ Failed to initialize services. Check the logs for details.")
        sys.exit(1)
    
    print("\n" + "="*60)
    print("🚀 SYNAPSE WEB UPLOAD SERVER")
    print("="*60)
    print("📍 Server: http://localhost:5000")
    print("📤 Upload documents and get real-time processing")
    print("📊 Monitor processing status")
    print("⏹️  Press Ctrl+C to stop")
    print("="*60 + "\n")
    
    try:
        # Run Flask app
        app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
    except KeyboardInterrupt:
        print("\n🛑 Stopping server...")
        if embedding_service:
            embedding_service.stop_background_processing()
        print("✅ Server stopped")
    except Exception as e:
        logging.error(f"Server error: {e}")
        if embedding_service:
            embedding_service.stop_background_processing()
        sys.exit(1)

