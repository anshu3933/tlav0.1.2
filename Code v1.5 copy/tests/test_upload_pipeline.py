import unittest
from pathlib import Path
import logging
from langchain.schema import Document
from dspy_pipeline import process_iep_to_lesson_plans, ProcessingConfig, IEPPipeline
from utils.logging_config import logger
from loaders import load_documents
import os

class TestUploadPipelineErrors(unittest.TestCase):
    def test_trace_upload_error(self):
        """Trace the exact upload and processing flow"""
        logger.debug("=== Starting Upload Flow Test ===")
        
        # 1. Create test document similar to Streamlit upload
        test_doc = Document(
            page_content="Assessment Report Content",
            metadata={"source": "Assessment report for Renuga (1).pdf"}
        )
        logger.debug(f"1. Created test document: {test_doc.metadata}")
        
        # 2. Initialize pipeline
        iep_pipeline = IEPPipeline()
        logger.debug("2. Initialized IEP Pipeline")
        
        # 3. Process document
        logger.debug("3. Starting document processing")
        result = process_iep_to_lesson_plans([test_doc])
        
        # Verify we got results even if IEP processing failed
        self.assertIsNotNone(result)
        self.assertGreater(len(result), 0)
        logger.debug(f"3. Pipeline result length: {len(result)}")

    def test_empty_iep_processing(self):
        """Test handling of empty IEP processing results"""
        logger.debug("=== Starting Empty IEP Processing Test ===")
        
        test_doc = Document(
            page_content="Assessment Report Content",
            metadata={"source": "test.pdf"}
        )
        
        class MockIEPPipeline:
            def process_documents(self, docs):
                logger.debug("Mock IEP pipeline returning empty list")
                return []
        
        original_pipeline = IEPPipeline
        try:
            import dspy_pipeline
            dspy_pipeline.IEPPipeline = MockIEPPipeline
            
            result = process_iep_to_lesson_plans([test_doc])
            
            # Should return at least one document since we fall back to original
            self.assertGreater(len(result), 0)
            logger.debug("Test completed successfully")
            
        finally:
            dspy_pipeline.IEPPipeline = original_pipeline

    def test_pdf_upload_flow(self):
        """Test with actual PDF upload"""
        logger.debug("=== Testing PDF Upload Flow ===")
        
        # Create a test PDF with IEP-like content
        pdf_content = """
        Student Assessment Report
        Name: John Doe
        Grade: 5
        
        Current Performance:
        - Reading level: 3rd grade
        - Math skills: Needs support with fractions
        
        Accommodations:
        1. Extended time on tests
        2. Visual aids for math
        3. Preferential seating
        
        Goals:
        - Improve reading comprehension
        - Master basic fraction operations
        """
        
        pdf_path = "tests/test_files/test_iep.pdf"
        try:
            # Create PDF file
            from reportlab.pdfgen import canvas
            c = canvas.Canvas(pdf_path)
            c.drawString(72, 720, pdf_content)
            c.save()
            
            # Test upload flow
            logger.debug(f"Loading PDF from {pdf_path}")
            documents = load_documents([pdf_path])
            self.assertGreater(len(documents), 0)
            logger.debug(f"Loaded {len(documents)} documents")
            
            # Process through pipeline
            logger.debug("Processing PDF through pipeline")
            config = ProcessingConfig(use_only_uploaded=True)
            result = process_iep_to_lesson_plans(documents, config=config)
            
            # Verify results
            self.assertIsNotNone(result)
            self.assertGreater(len(result), 0)
            logger.debug(f"Generated {len(result)} lesson plans")
            
            # Check content
            for doc in result:
                logger.debug(f"Lesson plan type: {doc.metadata.get('type')}")
                logger.debug(f"Timeframe: {doc.metadata.get('timeframe')}")
                logger.debug(f"Content length: {len(doc.page_content)}")
                
        finally:
            if os.path.exists(pdf_path):
                os.remove(pdf_path)

if __name__ == '__main__':
    unittest.main(verbosity=2) 