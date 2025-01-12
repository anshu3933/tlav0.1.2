import unittest
from pathlib import Path
from dspy_pipeline import LessonPlanPipeline, ProcessingConfig
from utils.logging_config import logger
import json

class TestData:
    """Test data management."""
    
    @staticmethod
    def get_test_data() -> dict:
        """Get standard test data for lesson plans."""
        return {
            "iep_content": "Student requires visual aids and extended time",
            "subject": "Mathematics",
            "grade_level": "5th Grade",
            "duration": "45 minutes",
            "specific_goals": ["Master basic algebra concepts"],
            "materials": ["Textbook", "Worksheets"],
            "additional_accommodations": ["Visual aids"],
            "timeframe": "daily",
            "days": ["Monday", "Wednesday", "Friday"]
        }

class TestEnvironment:
    """Test environment setup and teardown."""
    
    def __init__(self):
        self.test_dir = Path(__file__).parent / "test_files"
        self.test_file = self.test_dir / "test_doc.pdf"
        self.logger = logger
    
    def setup(self):
        """Set up test environment."""
        try:
            self.test_dir.mkdir(exist_ok=True)
            return True
        except Exception as e:
            self.logger.error(f"Error setting up test environment: {e}")
            return False
    
    def cleanup(self):
        """Clean up test environment."""
        try:
            if self.test_dir.exists():
                import shutil
                shutil.rmtree(self.test_dir)
            return True
        except Exception as e:
            self.logger.error(f"Error cleaning up test environment: {e}")
            return False

class LessonPlanValidator:
    """Validates lesson plan outputs."""
    
    @staticmethod
    def validate_plan_structure(plan: dict) -> bool:
        """Validate lesson plan has required components."""
        required_fields = [
            'schedule',
            'learning_objectives',
            'assessment_criteria',
            'modifications'
        ]
        return all(field in plan for field in required_fields)
    
    @staticmethod
    def validate_plan_content(plan: dict) -> bool:
        """Validate lesson plan content meets requirements."""
        try:
            # Check minimum content requirements
            has_sufficient_objectives = len(plan['learning_objectives']) >= 3
            has_sufficient_criteria = len(plan['assessment_criteria']) >= 3
            has_sufficient_modifications = len(plan['modifications']) >= 2
            
            return all([
                has_sufficient_objectives,
                has_sufficient_criteria,
                has_sufficient_modifications
            ])
        except Exception:
            return False

class TestLessonPlanGeneration(unittest.TestCase):
    """Test lesson plan generation functionality."""
    
    def setUp(self):
        """Set up test environment."""
        self.env = TestEnvironment()
        self.assertTrue(self.env.setup())
        self.test_data = TestData.get_test_data()
        self.validator = LessonPlanValidator()
    
    def tearDown(self):
        """Clean up after tests."""
        self.assertTrue(self.env.cleanup())
    
    def test_lesson_plan_initialization(self):
        """Test lesson plan pipeline initialization."""
        try:
            pipeline = LessonPlanPipeline()
            self.assertIsNotNone(pipeline)
            self.assertIsNotNone(pipeline.lm)
            logger.debug("LessonPlanPipeline initialized successfully")
        except Exception as e:
            self.fail(f"Pipeline initialization failed: {e}")
    
    def test_lesson_plan_generation(self):
        """Test generating a lesson plan."""
        pipeline = LessonPlanPipeline()
        plan = pipeline.generate_lesson_plan(
            data=self.test_data,
            timeframe="daily"
        )
        
        # Validate plan structure
        self.assertIsNotNone(plan)
        self.assertTrue(self.validator.validate_plan_structure(plan))
        
        # Validate plan content
        self.assertTrue(self.validator.validate_plan_content(plan))
    
    def test_lesson_plan_evaluation(self):
        """Test lesson plan evaluation."""
        pipeline = LessonPlanPipeline()
        plan = pipeline.generate_lesson_plan(self.test_data)
        
        score = pipeline.evaluate_lesson_plan(plan)
        self.assertIsInstance(score, float)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)
    
    def test_error_handling(self):
        """Test error handling with invalid input."""
        pipeline = LessonPlanPipeline()
        
        # Test with empty data
        empty_result = pipeline.generate_lesson_plan({})
        self.assertIsNone(empty_result)
        
        # Test with invalid timeframe
        invalid_data = self.test_data.copy()
        invalid_data['timeframe'] = 'invalid'
        invalid_result = pipeline.generate_lesson_plan(invalid_data)
        self.assertIsNone(invalid_result)

class TestLessonPlanIntegration(unittest.TestCase):
    """Integration tests for lesson plan generation."""
    
    def setUp(self):
        self.env = TestEnvironment()
        self.assertTrue(self.env.setup())
        self.test_data = TestData.get_test_data()
    
    def tearDown(self):
        self.assertTrue(self.env.cleanup())
    
    def test_end_to_end_generation(self):
        """Test end-to-end lesson plan generation process."""
        try:
            # Initialize pipeline
            pipeline = LessonPlanPipeline()
            
            # Generate plan
            plan = pipeline.generate_lesson_plan(self.test_data)
            self.assertIsNotNone(plan)
            
            # Evaluate plan
            score = pipeline.evaluate_lesson_plan(plan)
            self.assertGreater(score, 0.0)
            
            # Verify plan can be serialized
            plan_json = json.dumps(plan)
            self.assertIsInstance(plan_json, str)
            
        except Exception as e:
            self.fail(f"End-to-end test failed: {e}") 