#!/usr/bin/env python3
"""
Image extraction module for Synapse

This module extracts images/diagrams from PDFs and other documents, saving them as 
independent files with rich metadata for better retrieval and contextual understanding.

Key features:
- Extract images from PDFs with page context
- Generate descriptive filenames based on document and page context
- Store metadata about image source, context, and technical content
- Support for various image formats and OCR enhancement
- Integration with existing Synapse parsing pipeline
"""

import hashlib
import json
import logging
import os
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import base64


@dataclass
class ExtractedImage:
    """Metadata for an extracted image."""
    image_id: str                    # Unique identifier
    source_document: str             # Original document path
    source_type: str                 # pdf, pptx, etc.
    page_number: Optional[int]       # Page/slide where image was found
    image_path: str                  # Path to extracted image file
    image_format: str                # png, jpg, etc.
    width: int                       # Image width in pixels
    height: int                      # Image height in pixels
    file_size: int                   # Image file size in bytes
    extraction_method: str           # Method used to extract (pymupdf, unstructured, etc.)
    document_context: str            # Surrounding text from the same page
    ocr_text: str                    # OCR text from the image
    caption_text: str                # AI-generated caption
    technical_keywords: List[str]    # Extracted technical terms
    image_type: str                  # diagram, chart, photo, etc.
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


class PDFImageExtractor:
    """Extract images from PDF documents."""
    
    def __init__(self, output_dir: str = "artifacts/extracted_images", enable_captioning: bool = True):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.enable_captioning = enable_captioning
        
    def extract_images_pymupdf(self, pdf_path: str, page_context: Optional[Dict[int, str]] = None) -> List[ExtractedImage]:
        """Extract images using PyMuPDF (fast, good for most PDFs)."""
        try:
            import fitz  # PyMuPDF
        except ImportError:
            logging.warning("PyMuPDF not available for image extraction")
            return []
        
        extracted_images = []
        
        try:
            doc = fitz.open(pdf_path)
            doc_name = Path(pdf_path).stem
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                image_list = page.get_images()
                
                # Get page text for context
                page_text = page_context.get(page_num + 1, "") if page_context else page.get_text()
                
                for img_index, img in enumerate(image_list):
                    try:
                        # Get image data
                        xref = img[0]
                        pix = fitz.Pixmap(doc, xref)
                        
                        # Skip if image is too small (likely not a meaningful diagram)
                        if pix.width < 100 or pix.height < 100:
                            pix = None
                            continue
                        
                        # Generate unique filename
                        image_id = self._generate_image_id(pdf_path, page_num + 1, img_index)
                        image_filename = f"{doc_name}_p{page_num + 1:03d}_img{img_index:02d}_{image_id[:8]}.png"
                        image_path = self.output_dir / image_filename
                        
                        # Convert to PNG and save
                        if pix.n - pix.alpha < 4:  # GRAY or RGB
                            pix.save(str(image_path))
                        else:  # CMYK: convert to RGB first
                            pix1 = fitz.Pixmap(fitz.csRGB, pix)
                            pix1.save(str(image_path))
                            pix1 = None
                        
                        # Extract text using OCR if available
                        ocr_text = self._extract_ocr_text(str(image_path))
                        
                        # Generate caption
                        caption_text = self._generate_caption(str(image_path))
                        
                        # Detect image type and extract keywords
                        image_type, keywords = self._analyze_image_content(ocr_text, caption_text, page_text)
                        
                        # Create metadata
                        extracted_image = ExtractedImage(
                            image_id=image_id,
                            source_document=os.path.abspath(pdf_path),
                            source_type="pdf",
                            page_number=page_num + 1,
                            image_path=str(image_path.absolute()),
                            image_format="png",
                            width=pix.width,
                            height=pix.height,
                            file_size=os.path.getsize(image_path),
                            extraction_method="pymupdf",
                            document_context=page_text[:2000],  # Limit context size
                            ocr_text=ocr_text,
                            caption_text=caption_text,
                            technical_keywords=keywords,
                            image_type=image_type
                        )
                        
                        extracted_images.append(extracted_image)
                        logging.debug(f"Extracted image: {image_filename}")
                        
                        pix = None
                        
                    except Exception as e:
                        logging.warning(f"Failed to extract image {img_index} from page {page_num + 1}: {e}")
                        continue
                        
            doc.close()
            
        except Exception as e:
            logging.error(f"Failed to extract images from {pdf_path}: {e}")
            
        return extracted_images
    
    def extract_images_unstructured(self, pdf_path: str) -> List[ExtractedImage]:
        """Extract images using unstructured.io (more comprehensive but slower)."""
        try:
            from unstructured.partition.pdf import partition_pdf
        except ImportError:
            logging.warning("unstructured library not available for image extraction")
            return []
        
        extracted_images = []
        
        try:
            # Use unstructured to get elements with image detection
            elements = partition_pdf(
                filename=pdf_path,
                strategy="hi_res",
                extract_images_in_pdf=True,
                infer_table_structure=True
            )
            
            doc_name = Path(pdf_path).stem
            
            for element in elements:
                if hasattr(element, 'metadata') and hasattr(element.metadata, 'image_path'):
                    try:
                        source_image_path = element.metadata.image_path
                        
                        if not os.path.exists(source_image_path):
                            continue
                        
                        # Get image info
                        from PIL import Image
                        with Image.open(source_image_path) as img:
                            width, height = img.size
                            img_format = img.format or "PNG"
                        
                        # Generate unique filename and copy to our output directory
                        page_num = getattr(element.metadata, 'page_number', 0)
                        image_id = self._generate_image_id(pdf_path, page_num, 0)
                        image_filename = f"{doc_name}_p{page_num:03d}_unstr_{image_id[:8]}.{img_format.lower()}"
                        image_path = self.output_dir / image_filename
                        
                        # Copy image to our directory
                        import shutil
                        shutil.copy2(source_image_path, image_path)
                        
                        # Extract context from nearby elements
                        page_context = self._get_page_context_unstructured(elements, page_num)
                        
                        # Extract text using OCR
                        ocr_text = self._extract_ocr_text(str(image_path))
                        
                        # Generate caption
                        caption_text = self._generate_caption(str(image_path))
                        
                        # Analyze content
                        image_type, keywords = self._analyze_image_content(ocr_text, caption_text, page_context)
                        
                        extracted_image = ExtractedImage(
                            image_id=image_id,
                            source_document=os.path.abspath(pdf_path),
                            source_type="pdf",
                            page_number=page_num,
                            image_path=str(image_path.absolute()),
                            image_format=img_format.lower(),
                            width=width,
                            height=height,
                            file_size=os.path.getsize(image_path),
                            extraction_method="unstructured",
                            document_context=page_context,
                            ocr_text=ocr_text,
                            caption_text=caption_text,
                            technical_keywords=keywords,
                            image_type=image_type
                        )
                        
                        extracted_images.append(extracted_image)
                        logging.debug(f"Extracted image via unstructured: {image_filename}")
                        
                    except Exception as e:
                        logging.warning(f"Failed to process unstructured image: {e}")
                        continue
                        
        except Exception as e:
            logging.error(f"Failed to extract images via unstructured from {pdf_path}: {e}")
            
        return extracted_images
    
    def _generate_image_id(self, source_path: str, page_num: int, img_index: int) -> str:
        """Generate a unique ID for an extracted image."""
        source_str = f"{source_path}:page_{page_num}:img_{img_index}"
        return hashlib.sha256(source_str.encode()).hexdigest()
    
    def _extract_ocr_text(self, image_path: str) -> str:
        """Extract text from image using OCR."""
        try:
            import pytesseract
            from PIL import Image
            
            with Image.open(image_path) as img:
                # Use enhanced OCR configuration for technical diagrams
                custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz ×x-/()[]{}:.,;'
                ocr_text = pytesseract.image_to_string(img, config=custom_config)
                
                # Apply common technical OCR corrections
                ocr_text = self._correct_technical_ocr_errors(ocr_text)
                
                return ocr_text.strip()
                
        except Exception as e:
            logging.debug(f"OCR extraction failed for {image_path}: {e}")
            return ""
    
    def _generate_caption(self, image_path: str) -> str:
        """Generate an AI caption for the image."""
        # Skip captioning if disabled
        if not self.enable_captioning:
            return ""
            
        try:
            # Cache the captioner pipeline to avoid reloading for each image
            if not hasattr(self, '_captioner'):
                try:
                    from transformers import pipeline
                    from PIL import Image
                    
                    logging.info("Loading BLIP image captioning model (this may take a moment)...")
                    self._captioner = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")
                    logging.info("BLIP model loaded successfully")
                except Exception as model_error:
                    logging.warning(f"Failed to load BLIP model (likely due to rate limits): {model_error}")
                    # Fallback: disable captioning for this session
                    self._captioner = None
                    return ""
            
            if self._captioner is None:
                return ""
                
            from PIL import Image
            with Image.open(image_path) as img:
                result = self._captioner(img, max_new_tokens=64)
                if isinstance(result, list) and result:
                    return result[0].get("generated_text", "").strip()
                    
        except Exception as e:
            logging.debug(f"Image captioning failed for {image_path}: {e}")
            
        return ""
    
    def _correct_technical_ocr_errors(self, ocr_text: str) -> str:
        """Apply corrections for common OCR errors in technical diagrams."""
        if not ocr_text:
            return ocr_text
        
        corrections = {
            # Common OCR errors in technical diagrams
            "12C": "I2C",
            "13C": "I2C", 
            "SPI ": "SPI ",
            "UARTx": "UART x",
            "GPIOx": "GPIO x",
            "CANFD": "CAN-FD",
            "CANFDx": "CAN-FD x",
            "SPIx": "SPI x",
            "I2Cx": "I2C x",
            " x ": " x",  # normalize spacing around x
            "×": "x",      # replace multiplication symbol with x
            "Timer": "Timer",
            "WDT": "WDT",
            "RAS": "RAS",
            "Cluster": "Cluster",
            "CPU": "CPU",
            "Core": "Core",
            "Cache": "Cache",
            "L1": "L1",
            "L2": "L2",
            "L3": "L3"
        }
        
        result = ocr_text
        for error, correction in corrections.items():
            result = result.replace(error, correction)
        
        return result
    
    def _analyze_image_content(self, ocr_text: str, caption_text: str, page_context: str) -> Tuple[str, List[str]]:
        """Analyze image content to determine type and extract technical keywords."""
        
        # Combine all text sources for analysis
        all_text = f"{ocr_text} {caption_text} {page_context}".lower()
        
        # Determine image type based on content
        image_type = "diagram"  # default
        
        if any(word in all_text for word in ["chart", "graph", "plot", "performance", "benchmark"]):
            image_type = "chart"
        elif any(word in all_text for word in ["block", "architecture", "system", "connection", "interface"]):
            image_type = "block_diagram"
        elif any(word in all_text for word in ["flow", "process", "pipeline", "state"]):
            image_type = "flowchart"
        elif any(word in all_text for word in ["table", "matrix", "grid"]):
            image_type = "table"
        elif any(word in all_text for word in ["photo", "image", "picture"]):
            image_type = "photo"
        
        # Extract technical keywords
        technical_patterns = [
            r'\b(?:CPU|GPU|TPU|DSP|RISC-V|ARM|x86)\b',
            r'\b(?:I2C|SPI|UART|USB|PCIe|CAN|Ethernet)\b',
            r'\b(?:L1|L2|L3)\s*(?:cache|Cache)\b',
            r'\b(?:Timer|WDT|RAS|Cluster|Core|Cache)\b',
            r'\b(?:MHz|GHz|MB|GB|TB|Mbps|Gbps)\b',
            r'\b(?:interrupt|vector|peripheral|interface)\b',
            r'\b(?:Ascalon|Tensix|Alexandria|Blackhole)\b',  # TT-specific terms
            r'\b\d+(?:x\d+|\s*x\s*\d+)\b',  # Dimensions like "2x4", "8 x 8"
        ]
        
        keywords = []
        for pattern in technical_patterns:
            matches = re.findall(pattern, all_text, re.IGNORECASE)
            keywords.extend([match.strip() for match in matches])
        
        # Remove duplicates and normalize
        keywords = list(set([kw.strip() for kw in keywords if len(kw.strip()) > 1]))
        
        return image_type, keywords
    
    def _get_page_context_unstructured(self, elements: List, target_page: int) -> str:
        """Extract context text from the same page using unstructured elements."""
        page_texts = []
        
        for element in elements:
            if hasattr(element, 'metadata') and hasattr(element.metadata, 'page_number'):
                if element.metadata.page_number == target_page:
                    if hasattr(element, 'text') and element.text:
                        page_texts.append(element.text.strip())
        
        return " ".join(page_texts)[:2000]  # Limit context size


class PPTXImageExtractor:
    """Extract images from PowerPoint presentations."""
    
    def __init__(self, output_dir: str = "artifacts/extracted_images", enable_captioning: bool = True):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.enable_captioning = enable_captioning
    
    def extract_images(self, pptx_path: str) -> List[ExtractedImage]:
        """Extract images from PPTX file."""
        try:
            from pptx import Presentation
            from PIL import Image
        except ImportError:
            logging.warning("python-pptx or PIL not available for PPTX image extraction")
            return []
        
        extracted_images = []
        
        try:
            pres = Presentation(pptx_path)
            doc_name = Path(pptx_path).stem
            
            for slide_idx, slide in enumerate(pres.slides):
                # Get slide text for context
                slide_texts = []
                for shape in slide.shapes:
                    if hasattr(shape, "text") and shape.text:
                        slide_texts.append(shape.text.strip())
                slide_context = " ".join(slide_texts)
                
                # Extract images from slide
                img_idx = 0
                for shape in slide.shapes:
                    try:
                        if hasattr(shape, "image"):
                            # Extract embedded image
                            image_stream = shape.image.blob
                            
                            # Generate filename
                            image_id = self._generate_image_id(pptx_path, slide_idx + 1, img_idx)
                            image_filename = f"{doc_name}_s{slide_idx + 1:03d}_img{img_idx:02d}_{image_id[:8]}.png"
                            image_path = self.output_dir / image_filename
                            
                            # Save image
                            with open(image_path, 'wb') as f:
                                f.write(image_stream)
                            
                            # Get image dimensions
                            with Image.open(image_path) as img:
                                width, height = img.size
                            
                            # Process image content
                            ocr_text = self._extract_ocr_text(str(image_path))
                            caption_text = self._generate_caption(str(image_path))
                            image_type, keywords = self._analyze_image_content(ocr_text, caption_text, slide_context)
                            
                            extracted_image = ExtractedImage(
                                image_id=image_id,
                                source_document=os.path.abspath(pptx_path),
                                source_type="pptx",
                                page_number=slide_idx + 1,
                                image_path=str(image_path.absolute()),
                                image_format="png",
                                width=width,
                                height=height,
                                file_size=os.path.getsize(image_path),
                                extraction_method="python-pptx",
                                document_context=slide_context[:2000],
                                ocr_text=ocr_text,
                                caption_text=caption_text,
                                technical_keywords=keywords,
                                image_type=image_type
                            )
                            
                            extracted_images.append(extracted_image)
                            logging.debug(f"Extracted PPTX image: {image_filename}")
                            
                            img_idx += 1
                            
                    except Exception as e:
                        logging.warning(f"Failed to extract image from slide {slide_idx + 1}: {e}")
                        continue
                        
        except Exception as e:
            logging.error(f"Failed to extract images from {pptx_path}: {e}")
            
        return extracted_images
    
    def _generate_image_id(self, source_path: str, slide_num: int, img_index: int) -> str:
        """Generate a unique ID for an extracted image."""
        source_str = f"{source_path}:slide_{slide_num}:img_{img_index}"
        return hashlib.sha256(source_str.encode()).hexdigest()
    
    def _extract_ocr_text(self, image_path: str) -> str:
        """Extract text from image using OCR."""
        try:
            import pytesseract
            from PIL import Image
            
            with Image.open(image_path) as img:
                custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz ×x-/()[]{}:.,;'
                ocr_text = pytesseract.image_to_string(img, config=custom_config)
                return self._correct_technical_ocr_errors(ocr_text.strip())
                
        except Exception as e:
            logging.debug(f"OCR extraction failed for {image_path}: {e}")
            return ""
    
    def _generate_caption(self, image_path: str) -> str:
        """Generate an AI caption for the image."""
        try:
            from transformers import pipeline
            from PIL import Image
            
            captioner = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")
            
            with Image.open(image_path) as img:
                result = captioner(img, max_new_tokens=64)
                if isinstance(result, list) and result:
                    return result[0].get("generated_text", "").strip()
                    
        except Exception as e:
            logging.debug(f"Image captioning failed for {image_path}: {e}")
            
        return ""
    
    def _correct_technical_ocr_errors(self, ocr_text: str) -> str:
        """Apply corrections for common OCR errors in technical diagrams."""
        if not ocr_text:
            return ocr_text
        
        corrections = {
            "12C": "I2C", "13C": "I2C", "RAS": "RAS", "Cluster": "Cluster",
            "CPU": "CPU", "Core": "Core", "Cache": "Cache", "Timer": "Timer"
        }
        
        result = ocr_text
        for error, correction in corrections.items():
            result = result.replace(error, correction)
        
        return result
    
    def _analyze_image_content(self, ocr_text: str, caption_text: str, slide_context: str) -> Tuple[str, List[str]]:
        """Analyze image content to determine type and extract technical keywords."""
        all_text = f"{ocr_text} {caption_text} {slide_context}".lower()
        
        # Determine image type
        image_type = "diagram"
        if any(word in all_text for word in ["chart", "graph", "performance"]):
            image_type = "chart"
        elif any(word in all_text for word in ["block", "architecture", "system"]):
            image_type = "block_diagram"
        elif any(word in all_text for word in ["flow", "process", "pipeline"]):
            image_type = "flowchart"
        
        # Extract technical keywords using similar patterns as PDF extractor
        technical_patterns = [
            r'\b(?:CPU|GPU|TPU|DSP|RISC-V|ARM|x86)\b',
            r'\b(?:I2C|SPI|UART|USB|PCIe|CAN|Ethernet)\b',
            r'\b(?:L1|L2|L3)\s*(?:cache|Cache)\b',
            r'\b(?:Timer|WDT|RAS|Cluster|Core|Cache)\b',
            r'\b(?:Ascalon|Tensix|Alexandria|Blackhole)\b',
        ]
        
        keywords = []
        for pattern in technical_patterns:
            matches = re.findall(pattern, all_text, re.IGNORECASE)
            keywords.extend([match.strip() for match in matches])
        
        return image_type, list(set([kw.strip() for kw in keywords if len(kw.strip()) > 1]))


def save_image_metadata(extracted_images: List[ExtractedImage], metadata_path: str):
    """Save image metadata to JSON file."""
    metadata = {
        "extraction_timestamp": json.dumps(None),  # Will be set by caller
        "total_images": len(extracted_images),
        "images": [img.to_dict() for img in extracted_images]
    }
    
    os.makedirs(os.path.dirname(metadata_path), exist_ok=True)
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logging.info(f"Saved metadata for {len(extracted_images)} images to {metadata_path}")


def load_image_metadata(metadata_path: str) -> List[ExtractedImage]:
    """Load image metadata from JSON file."""
    if not os.path.exists(metadata_path):
        return []
    
    try:
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        images = []
        for img_data in metadata.get("images", []):
            images.append(ExtractedImage(**img_data))
        
        return images
        
    except Exception as e:
        logging.warning(f"Failed to load image metadata from {metadata_path}: {e}")
        return []


if __name__ == "__main__":
    # Test the image extractor
    import sys
    logging.basicConfig(level=logging.DEBUG)
    
    if len(sys.argv) > 1:
        test_file = sys.argv[1]
        
        if test_file.endswith('.pdf'):
            extractor = PDFImageExtractor()
            images = extractor.extract_images_pymupdf(test_file)
        elif test_file.endswith('.pptx'):
            extractor = PPTXImageExtractor()
            images = extractor.extract_images(test_file)
        else:
            print("Unsupported file type. Use .pdf or .pptx")
            sys.exit(1)
        
        print(f"Extracted {len(images)} images from {test_file}")
        for img in images:
            print(f"  - {img.image_path} ({img.image_type}, {len(img.technical_keywords)} keywords)")
    else:
        print("Usage: python image_extractor.py <pdf_or_pptx_file>")
