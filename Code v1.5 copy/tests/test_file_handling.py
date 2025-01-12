import unittest
from pathlib import Path
import tempfile
import os
from io import BytesIO
import dspy
import dsp
from loaders import FileUploadHandler, DocumentLoader
from utils.logging_config import logger
from dspy_pipeline import LessonPlanPipeline, process_uploaded_document
import json

# Set up OpenAI API key for testing
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY environment variable must be set to run tests")

class MockUploadedFile:
    """Mock class for testing file uploads."""
    def __init__(self, content: bytes, name: str = "test.txt"):
        self.content = content
        self.name = name
    
    def getvalue(self):
        return self.content

class TestFileUpload(unittest.TestCase):
    """Test file upload handling."""
    
    def setUp(self):
        """Set up test environment."""
        self.upload_handler = FileUploadHandler()
        self.doc_loader = DocumentLoader()
        self.test_content = b"Test document content"
        self.test_dir = Path("test_files")
        self.test_dir.mkdir(exist_ok=True)
    
    def tearDown(self):
        """Clean up test environment."""
        self.upload_handler.cleanup()
        if self.test_dir.exists():
            import shutil
            shutil.rmtree(self.test_dir)
    
    def test_upload_processing(self):
        """Test processing of uploaded file."""
        # Create mock uploaded file
        uploaded_file = MockUploadedFile(self.test_content)
        
        # Process upload
        temp_path = self.upload_handler.process_uploaded_file(uploaded_file)
        
        # Verify temp file created
        self.assertIsNotNone(temp_path)
        self.assertTrue(os.path.exists(temp_path))
        
        # Verify content
        with open(temp_path, 'rb') as f:
            content = f.read()
        self.assertEqual(content, self.test_content)
    
    def test_document_loading_from_upload(self):
        """Test loading document from uploaded file."""
        # Create mock uploaded file with text content
        text_content = "Test document content with some meaningful text."
        uploaded_file = MockUploadedFile(
            text_content.encode(),
            name="test_doc.txt"  # Ensure proper extension
        )
        
        # Process upload
        temp_path = self.upload_handler.process_uploaded_file(uploaded_file)
        
        # Load document
        documents = self.doc_loader.load_documents([temp_path])
        
        # Verify document loaded
        self.assertEqual(len(documents), 1)
        self.assertEqual(documents[0].page_content.strip(), text_content)
    
    def test_multiple_file_uploads(self):
        """Test handling multiple file uploads."""
        files = [
            ("doc1.txt", b"Content of first document"),
            ("doc2.txt", b"Content of second document")
        ]
        
        paths = []
        for name, content in files:
            uploaded_file = MockUploadedFile(content, name)
            path = self.upload_handler.process_uploaded_file(uploaded_file)
            self.assertIsNotNone(path)
            paths.append(path)
        
        # Verify all files exist
        for path in paths:
            self.assertTrue(os.path.exists(path))
    
    def test_cleanup(self):
        """Test temporary file cleanup."""
        uploaded_file = MockUploadedFile(b"Test content")
        path = self.upload_handler.process_uploaded_file(uploaded_file)
        
        # Verify file exists
        self.assertTrue(os.path.exists(path))
        
        # Clean up
        self.upload_handler.cleanup()
        
        # Verify file was deleted
        self.assertFalse(os.path.exists(path))

class TestDSPyPipeline(unittest.TestCase):
    """Test DSPy pipeline functionality."""
    
    def setUp(self):
        """Set up test data."""
        self.test_data = {
            "iep_content": "Student requires visual aids and extended time for math tasks. "
                          "Shows difficulty with multi-step problems.",
            "subject": "Mathematics",
            "grade_level": "5th Grade",
            "duration": "45 minutes",
            "specific_goals": ["Master basic algebra concepts"],
            "materials": ["Textbook", "Worksheets"],
            "additional_accommodations": ["Visual aids"],
            "timeframe": "daily",
            "days": ["Monday", "Wednesday", "Friday"]
        }
    
    def test_lesson_plan_generation(self):
        """Test lesson plan generation."""
        pipeline = LessonPlanPipeline()
        plan = pipeline.generate_lesson_plan(self.test_data)
        
        # Verify plan structure
        self.assertIsNotNone(plan)
        self.assertIn('schedule', plan)
        self.assertIn('learning_objectives', plan)
        self.assertIn('assessment_criteria', plan)
        self.assertIn('modifications', plan)
        
        # Verify plan content
        self.assertGreater(len(plan['learning_objectives']), 0)
        self.assertGreater(len(plan['assessment_criteria']), 0)
        self.assertGreater(len(plan['modifications']), 0)
    
    def test_document_processing(self):
        """Test document processing pipeline."""
        # Create test IEP document content
        iep_content = """
        Student Assessment Results:
        - Reading comprehension below grade level
        - Strong verbal communication skills
        - Requires extended time for assignments
        
        Recommendations:
        1. Visual learning aids
        2. Break tasks into smaller steps
        3. Provide frequent feedback
        """
        
        # Create mock uploaded file
        uploaded_file = MockUploadedFile(
            iep_content.encode(),
            "test_iep.txt"
        )
        
        # Process document
        result = process_uploaded_document(
            uploaded_file.getvalue(),
            uploaded_file.name
        )
        
        # Verify processing result
        self.assertIsNotNone(result)
        self.assertIn('processed_with', result.metadata)
        self.assertEqual(result.metadata['processed_with'], 'dspy')
        
        # Verify document content was processed
        self.assertIn('assessment', result.page_content.lower())
        self.assertIn('recommendations', result.page_content.lower())

if __name__ == '__main__':
    unittest.main() 