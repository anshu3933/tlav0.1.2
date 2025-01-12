import unittest
from dspy_pipeline import LessonPlanGenerator, LessonPlanSignature
import dspy
from utils.logging_config import logger

class TestLessonPlanGeneration(unittest.TestCase):
    def setUp(self):
        self.lm = dspy.OpenAI(model="o1-mini", max_tokens=4000)
        dspy.configure(lm=self.lm)
        self.generator = LessonPlanGenerator(self.lm)
        
    def test_generator_initialization(self):
        """Test proper generator initialization"""
        self.assertIsNotNone(self.generator.generator)
        self.assertIsInstance(self.generator.generator, dspy.ChainOfThought)
        
    def test_data_preparation(self):
        """Test data preparation for generator"""
        test_data = {
            "iep_content": "Student needs visual aids",
            "subject": "Math",
            "grade_level": "5th",
            "duration": "45 min",
            "specific_goals": ["Goal 1", "Goal 2"],
            "materials": ["Book", "Paper"],
            "additional_accommodations": ["Extra time"],
            "timeframe": "daily",
            "days": ["Monday"]
        }
        
        prepared_data = self.generator._prepare_input_data(test_data)
        
        # Verify all required fields are present
        for field in LessonPlanSignature.__annotations__:
            self.assertIn(field, prepared_data)
            
    def test_generation_with_depth(self):
        """Test generation with proper depth configuration"""
        test_data = {
            "iep_content": "Student requires visual aids",
            "subject": "Mathematics",
            "grade_level": "5th Grade",
            "duration": "45 minutes",
            "specific_goals": ["Master basic algebra"],
            "materials": ["Textbook"],
            "additional_accommodations": ["Visual aids"],
            "timeframe": "daily",
            "days": ["Monday"]
        }
        
        try:
            # Configure DSPy with proper depth
            dspy.settings.configure(max_depth=3)
            
            result = self.generator.generate(test_data, "daily")
            self.assertIsNotNone(result)
            
        except Exception as e:
            logger.error(f"Generation error: {str(e)}", exc_info=True)
            raise 