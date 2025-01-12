import dspy
from typing import List, Dict, Any, Optional, Union
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
import os
from datetime import datetime
import re
import json
import logging
from utils.logging_config import logger
import dsp  # Ensure dsp is correctly imported

class ProcessingConfig:
    """Configuration for document processing."""
    def __init__(self, use_only_uploaded: bool = False):
        self.use_only_uploaded = use_only_uploaded

class IEPSignature(dspy.Signature):
    """Signature for analyzing assessments and generating IEP components."""
    content = dspy.InputField(desc="Document content (assessment or IEP)")
    
    # Analysis outputs - must be grounded in input content
    academic_analysis = dspy.OutputField(desc="""
        Analysis of academic performance based ONLY on provided document:
        - Current performance levels with specific examples
        - Identified skill gaps with evidence
        - Observed error patterns with examples
        - Learning barriers noted in document
        Include specific quotes or references from the document.
    """)
    
    iep_components = dspy.OutputField(desc="""
        IEP components based ONLY on document evidence:
        - SMART learning objectives tied to identified needs
        - Specific accommodations supported by assessment data
        - Modifications based on documented challenges
        - Progress monitoring aligned with identified gaps
        Reference specific findings from the document.
    """)
    
    evidence = dspy.OutputField(desc="""
        Document evidence supporting analysis:
        - Direct quotes supporting each finding
        - Page or section references
        - Specific test scores or observations cited
        - Context for each recommendation
    """)

class DocumentProcessor:
    """Handles document processing and validation."""
    
    def __init__(self):
        self.logger = logger
    
    def validate_document(self, doc: Document) -> bool:
        """Validate a document has required content."""
        if not doc:
            self.logger.warning("Empty document object encountered")
            return False
            
        if not doc.page_content or not doc.page_content.strip():
            self.logger.warning(f"Empty content in document: {doc.metadata.get('source', 'unknown')}")
            return False
            
        return True
    
    def verify_extraction_result(self, result: Any) -> bool:
        """Verify extraction result has required fields."""
        return (hasattr(result, 'academic_analysis') and 
                hasattr(result, 'iep_components') and
                hasattr(result, 'evidence'))
    
    def create_enhanced_document(self, doc: Document, result: Any) -> Document:
        """Create enhanced document with extraction results."""
        return Document(
            page_content=doc.page_content,
            metadata={
                **doc.metadata,
                "processed_with": "dspy",
                "has_verified_content": True,
                "extraction_result": result
            }
        )

class EvidenceValidator:
    """Handles evidence validation and scoring."""
    
    def verify_evidence(self, result: Any, original_content: str) -> bool:
        """Verify that generated content is supported by document evidence."""
        try:
            evidence_quotes = self._extract_quotes(result.evidence)
            return all(
                quote.lower() in original_content.lower()
                for quote in evidence_quotes
                if quote.strip()
            )
        except Exception as e:
            logger.error(f"Error verifying evidence: {e}")
            return False
    
    def _extract_quotes(self, evidence_text: str) -> List[str]:
        """Extract quoted passages from evidence text."""
        quotes = re.findall(r'"([^"]*)"', evidence_text)
        if not quotes:
            quotes = re.findall(r"'([^']*)'", evidence_text)
        return quotes
    
    def calculate_evidence_score(self, result: Any) -> float:
        """Calculate evidence score based on verification results."""
        try:
            evidence_parts = result.evidence.split('\n')
            valid_evidence = [part for part in evidence_parts if part.strip()]
            return len(valid_evidence) / max(len(result.academic_analysis.split('\n')), 1)
        except Exception as e:
            logger.error(f"Error calculating evidence score: {e}")
            return 0.0

class ContentFormatter:
    """Handles formatting of processed content."""
    
    def format_verified_content(self, result: Any) -> str:
        """Format content with evidence references."""
        return f"""
Academic Analysis:
{result.academic_analysis}

Supporting Evidence:
{result.evidence}

IEP Components:
{result.iep_components}
"""
    
    def format_content(self,
                      original_content: str,
                      analysis: str,
                      iep_components: str,
                      recommendations: str) -> str:
        """Format processed content into structured output."""
        return f"""
Original Content:
{original_content}

Academic Analysis:
{analysis}

IEP Components:
{iep_components}

Educational Recommendations:
{recommendations}
"""

class IEPPipeline:
    """Processes educational documents to generate IEP components."""
    
    def __init__(self, model_name: str = "gpt-4o-mini", config: Optional[ProcessingConfig] = None, test_mode: bool = False):
        """Initialize the IEP pipeline with configuration."""
        self._initialize_pipeline(model_name, config, test_mode)
        self.doc_processor = DocumentProcessor()
        self.evidence_validator = EvidenceValidator()
        self.content_formatter = ContentFormatter()
    
    def _initialize_pipeline(self, model_name: str, config: Optional[ProcessingConfig], test_mode: bool):
        """Initialize DSPy components."""
        try:
            self.config = config or ProcessingConfig()
            
            if test_mode:
                # Use mock LLM for testing
                from tests.test_file_handling import MockLLM
                self.lm = MockLLM()
            else:
                # Use real LLM
                logger.debug("Initializing OpenAI LM...")
                self.lm = dspy.OpenAI(model=model_name, max_tokens=4000)
            
            logger.debug("Configuring DSPy settings...")
            dspy.configure(lm=self.lm, max_depth=5)
            
            logger.debug("Creating IEP extractor...")
            self.extractor = dspy.Predict(IEPSignature)
            
            logger.debug("IEP pipeline initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize IEP pipeline: {e}")
            raise
    
    def process_documents(self, documents: List[Document]) -> List[Document]:
        """Process documents through DSPy pipeline."""
        try:
            processed_docs = []
            for doc in documents:
                # Process document content
                processed_content = self._process_content(doc.page_content)
                
                # Create new document with processed content and metadata
                processed_doc = Document(
                    page_content=processed_content,
                    metadata={
                        **doc.metadata,
                        'processed_with': 'dspy',
                        'processing_timestamp': datetime.now().isoformat()
                    }
                )
                processed_docs.append(processed_doc)
            
            return processed_docs
            
        except Exception as e:
            logger.error(f"Error processing document: {e}")
            return documents
    
    def _process_single_document(self, doc: Document) -> Document:
        """Process a single document."""
        try:
            if not self.doc_processor.validate_document(doc):
                return doc
                
            result = self.extractor(content=doc.page_content)
            
            if self.doc_processor.verify_extraction_result(result):
                logger.debug("Successfully extracted IEP components")
                return self.doc_processor.create_enhanced_document(doc, result)
            else:
                logger.warning(f"Incomplete extraction result for document: {doc.metadata.get('source', 'unknown')}")
                return doc
                
        except Exception as e:
            logger.error(f"Error processing document: {e}")
            return doc
    
    def _verify_evidence(self, result: Any, original_content: str) -> bool:
        """Verify that generated content is supported by document evidence."""
        try:
            # Check if evidence quotes exist in original content
            evidence_quotes = self._extract_quotes(result.evidence)
            return all(
                quote.lower() in original_content.lower()
                for quote in evidence_quotes
                if quote.strip()
            )
        except Exception as e:
            logger.error(f"Error verifying evidence: {e}")
            return False
    
    def _extract_quotes(self, evidence_text: str) -> List[str]:
        """Extract quoted passages from evidence text."""
        quotes = re.findall(r'"([^"]*)"', evidence_text)
        if not quotes:
            # Try single quotes if no double quotes found
            quotes = re.findall(r"'([^']*)'", evidence_text)
        return quotes
    
    def _calculate_evidence_score(self, result: Any) -> float:
        """Calculate evidence score based on verification results."""
        try:
            evidence_parts = result.evidence.split('\n')
            valid_evidence = [part for part in evidence_parts if part.strip()]
            return len(valid_evidence) / max(len(result.academic_analysis.split('\n')), 1)
        except Exception as e:
            logger.error(f"Error calculating evidence score: {e}")
            return 0.0
    
    def _format_verified_content(self, result: Any) -> str:
        """Format content with evidence references."""
        return f"""
Academic Analysis:
{result.academic_analysis}

Supporting Evidence:
{result.evidence}

IEP Components:
{result.iep_components}
"""
    
    def _format_content(self,
                       original_content: str,
                       analysis: str,
                       iep_components: str,
                       recommendations: str) -> str:
        """Format processed content into structured output."""
        return f"""
Original Content:
{original_content}

Academic Analysis:
{analysis}

IEP Components:
{iep_components}

Educational Recommendations:
{recommendations}
"""
    
    def _create_metadata(self, 
                        doc: Document, 
                        result: Any) -> Dict[str, Any]:
        """Create metadata for processed document."""
        return {
            **doc.metadata,
            "processed_with": "dspy",
            "processing_timestamp": datetime.now().isoformat(),
            "has_analysis": bool(result.academic_analysis),
            "has_iep_components": bool(result.iep_components),
            "has_recommendations": bool(result.recommendations)
        }
    
    def _process_content(self, content: str) -> str:
        """Process document content using DSPy pipeline."""
        try:
            # Extract key information using DSPy
            result = self.pipeline(content)
            
            # Format the processed content
            processed_content = f"""
            Assessment Results:
            {result.get('assessment_results', '')}
            
            Recommendations:
            {result.get('recommendations', '')}
            
            Goals:
            {result.get('goals', '')}
            """
            
            return processed_content.strip()
            
        except Exception as e:
            logger.error(f"Error processing content: {e}")
            return content

def build_faiss_index_with_dspy(documents: List[Document], 
                               persist_directory: str,
                               model_name: str = "o1-mini") -> Optional[FAISS]:
    """Build a FAISS index with DSPy-enhanced documents."""
    builder = FAISSIndexBuilder(persist_directory)
    return builder.build_index_with_dspy(documents, model_name)

class LessonPlanRM(dspy.Signature):
    """Signature for lesson plan reasoning module."""
    context = dspy.InputField()
    reasoning = dspy.OutputField()

class LessonPlanSignature(dspy.Signature):
    """Signature for generating lesson plans with reasoning."""
    
    # Input fields with detailed descriptions
    iep_content = dspy.InputField(desc="Full IEP content including student needs and accommodations")
    subject = dspy.InputField(desc="Subject area (e.g., Math, Science)")
    grade_level = dspy.InputField(desc="Student's grade level")
    duration = dspy.InputField(desc="Length of each lesson")
    specific_goals = dspy.InputField(desc="Specific learning objectives to be achieved")
    materials = dspy.InputField(desc="Required teaching materials and resources")
    additional_accommodations = dspy.InputField(desc="Additional accommodations beyond IEP requirements")
    timeframe = dspy.InputField(desc="Daily or weekly planning timeframe")
    days = dspy.InputField(desc="Days of the week for instruction")
    
    # Output fields with detailed structure requirements
    schedule = dspy.OutputField(desc="""
        Detailed daily schedule including:
        - Warm-up activities (5-10 minutes)
        - Main concept introduction with visual aids
        - Guided practice with accommodations
        - Independent work time
        - Assessment and closure
        Minimum length: 200 words
    """)
    
    lesson_plan = dspy.OutputField(desc="""
        Comprehensive lesson plan including:
        1. Detailed teaching strategies
        2. Step-by-step instructions
        3. Differentiation methods
        4. IEP accommodations integration
        5. Real-world connections
        6. Student engagement techniques
        7. Time management details
        Minimum length: 300 words
    """)
    
    learning_objectives = dspy.OutputField(desc="""
        Specific, measurable objectives including:
        - Knowledge acquisition goals
        - Skill development targets
        - Application objectives
        - Assessment criteria
        Minimum 5 detailed objectives
    """)
    
    assessment_criteria = dspy.OutputField(desc="""
        Detailed assessment criteria including:
        - Understanding checks
        - Skill demonstration requirements
        - Progress monitoring methods
        - Success indicators
        Minimum 5 specific criteria
    """)
    
    modifications = dspy.OutputField(desc="""
        Specific IEP-aligned modifications including:
        - Learning accommodations
        - Assessment modifications
        - Environmental adjustments
        - Support strategies
        Minimum 5 detailed modifications
    """)
    
    instructional_strategies = dspy.OutputField(desc="""
        Detailed teaching strategies including:
        - Visual learning techniques
        - Hands-on activities
        - Technology integration
        - Differentiation methods
        - Student engagement approaches
        Minimum 5 specific strategies
    """)

class LessonPlanGenerator:
    """Handles the generation of lesson plans."""
    
    def __init__(self, lm: Any):
        """Initialize with consistent DSPy configuration."""
        try:
            self.lm = lm
            
            # Set consistent DSPy configuration with explicit max_depth
            dspy.configure(
                lm=self.lm,
                max_tokens=4000,
                temperature=0.7,
                max_depth=5  # Explicitly set max_depth
            )
            
            self.generator = dspy.ChainOfThought(LessonPlanSignature)
            logger.debug("Successfully initialized LessonPlanGenerator")
            
        except Exception as e:
            logger.error(f"Error in LessonPlanGenerator initialization: {str(e)}")
            raise
    
    def generate(self, data: Dict[str, Any], timeframe: str) -> Optional[Dict[str, Any]]:
        """Generate a lesson plan based on input data."""
        try:
            prepared_data = self._prepare_input_data(data)
            
            # Use context manager for generation to ensure max_depth
            with dsp.settings.context(max_depth=5):
                logger.debug("Generating lesson plan with prepared data")
                result = self.generator(**prepared_data)
                
            return self._format_result(result, timeframe)
            
        except Exception as e:
            logger.error(f"Error generating lesson plan: {str(e)}")
            return None
    
    def _prepare_input_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare input data for generation."""
        return {
            "iep_content": data["iep_content"],
            "subject": data["subject"],
            "grade_level": data["grade_level"],
            "duration": data["duration"],
            "specific_goals": self._format_list_field(data["specific_goals"]),
            "materials": self._format_list_field(data["materials"]),
            "additional_accommodations": self._format_list_field(data["additional_accommodations"]),
            "timeframe": data.get("timeframe", "daily"),
            "days": self._format_days(data["days"])
        }
        
    def _format_list_field(self, field: List[str]) -> str:
        """Format list field into string."""
        if not field:
            return ""
        return "\n".join(f"- {item}" for item in field if item.strip())
        
    def _format_days(self, days: List[str]) -> str:
        """Format days list into string."""
        if not days or days == ["Daily"]:
            return "Daily"
        return ", ".join(days)
        
    def _format_result(self, result: Any, timeframe: str) -> Dict[str, Any]:
        """Format the generator result into a structured lesson plan."""
        try:
            return {
                "schedule": self._format_schedule(result, timeframe),
                "lesson_plan": result.lesson_plan,
                "learning_objectives": self._format_list_output(result.learning_objectives),
                "assessment_criteria": self._format_list_output(result.assessment_criteria),
                "modifications": self._format_list_output(result.modifications)
            }
        except Exception as e:
            logger.error(f"Error formatting result: {str(e)}")
            return None
            
    def _format_schedule(self, result: Any, timeframe: str) -> str:
        """Format schedule based on timeframe."""
        if timeframe.lower() == "daily":
            return self._format_daily_schedule(result.lesson_plan)
        return self._format_weekly_schedule(result.lesson_plan)
        
    def _format_daily_schedule(self, lesson_plan: str) -> str:
        """Extract and format daily schedule from lesson plan."""
        schedule_parts = []
        for line in lesson_plan.split('\n'):
            if any(time_indicator in line.lower() for time_indicator in ['min', 'minutes', ':']): 
                schedule_parts.append(line.strip())
        return '\n'.join(schedule_parts) if schedule_parts else lesson_plan
        
    def _format_weekly_schedule(self, lesson_plan: str) -> str:
        """Extract and format weekly schedule from lesson plan."""
        days = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday']
        schedule_parts = []
        current_day = None
        
        for line in lesson_plan.split('\n'):
            line_lower = line.lower()
            if any(day in line_lower for day in days):
                current_day = line.strip()
                schedule_parts.append(current_day)
            elif current_day and line.strip():
                schedule_parts.append(f"  {line.strip()}")
                
        return '\n'.join(schedule_parts) if schedule_parts else lesson_plan
        
    def _format_list_output(self, output: str) -> List[str]:
        """Format string output into list of items."""
        if not output:
            return []
            
        items = []
        for line in output.split('\n'):
            line = line.strip()
            if line and not line.startswith(('•', '-', '*')):
                items.append(line)
            elif line:
                items.append(line[1:].strip())
        return [item for item in items if item]

class LessonPlanEvaluator:
    """Handles evaluation of generated lesson plans."""
    
    def evaluate(self, plan: Dict[str, Any]) -> float:
        """Evaluate the quality of a generated lesson plan."""
        try:
            score = 0.0
            score += self._evaluate_segmentation(plan)
            score += self._evaluate_real_world_anchoring(plan)
            score += self._evaluate_iep_alignment(plan)
            return score
        except Exception as e:
            logger.error(f"Error in lesson plan evaluation: {str(e)}")
            return 0.0
    
    def _evaluate_segmentation(self, plan: Dict[str, Any]) -> float:
        """Evaluate plan segmentation."""
        if isinstance(plan.get('schedule'), str) and len(plan.get('schedule', '').split('\n')) > 2:
            return 0.3
        return 0.0
    
    def _evaluate_real_world_anchoring(self, plan: Dict[str, Any]) -> float:
        """Evaluate real-world connections."""
        lesson_plan_text = str(plan.get('lesson_plan', ''))
        if 'real-world' in lesson_plan_text.lower() or 'practical application' in lesson_plan_text.lower():
            return 0.3
        return 0.0
    
    def _evaluate_iep_alignment(self, plan: Dict[str, Any]) -> float:
        """Evaluate IEP alignment."""
        if plan.get('modifications') and len(plan.get('modifications', [])) > 0:
            return 0.4
        return 0.0

class LessonPlanPipeline:
    """Pipeline for generating adaptive lesson plans from IEPs."""
    
    def __init__(self, model_name: str = "gpt-4o-mini", test_mode: bool = False):
        """Initialize the lesson plan pipeline."""
        try:
            logger.debug("Initializing LessonPlanPipeline")
            
            if test_mode:
                # Use mock LLM for testing
                from tests.test_file_handling import MockLLM
                self.lm = MockLLM()
            else:
                # Use real LLM
                self.lm = dspy.OpenAI(model=model_name, max_tokens=4000)
            
            # Configure DSPy settings
            dspy.configure(
                lm=self.lm,
                max_depth=5,
                max_tokens=4000,
                temperature=0.7
            )
            
            # Initialize generator
            self.generator = LessonPlanGenerator(self.lm)
            logger.debug("Successfully initialized LessonPlanPipeline")
            
        except Exception as e:
            logger.error(f"Error initializing LessonPlanPipeline: {str(e)}", exc_info=True)
            raise
    
    def generate_lesson_plan(self, data: Dict[str, Any], timeframe: str = "daily") -> Optional[Dict[str, Any]]:
        """Generate a lesson plan."""
        try:
            logger.debug("=== Starting lesson plan generation pipeline ===")
            logger.debug(f"Input data: {json.dumps(data, indent=2)}")
            logger.debug(f"Timeframe: {timeframe}")
            
            # Generate with explicit context
            with dsp.settings.context(max_depth=10):
                logger.debug("Calling generator.generate()")
                result = self.generator.generate(data, timeframe)
                logger.debug(f"Generation complete. Result type: {type(result)}")
                logger.debug(f"Result: {json.dumps(result, indent=2) if result else None}")
                
            return result
            
        except Exception as e:
            logger.error(f"Error in lesson plan generation pipeline: {str(e)}", exc_info=True)
            return None
    
    def evaluate_lesson_plan(self, plan: Dict[str, Any]) -> float:
        """Evaluate the quality of a generated lesson plan."""
        return self.evaluator.evaluate(plan)

class IEPLessonPlanProcessor:
    """Handles processing of IEP documents into lesson plans."""
    
    def __init__(self, config: ProcessingConfig = None):
        """Initialize the processor."""
        self.config = config or ProcessingConfig()
        self.iep_pipeline = IEPPipeline(config=self.config)
        self.lesson_pipeline = LessonPlanPipeline()
        self.logger = logger
    
    def process_documents(self, documents: List[Document], timeframes: List[str] = ["daily"]) -> List[Document]:
        """Process IEP documents into lesson plans."""
        if not documents:
            self.logger.warning("No documents provided")
            return []
            
        self.logger.debug(f"Starting IEP to lesson plan processing for {len(documents)} documents")
        return [
            plan 
            for doc in documents
            for plan in self._process_single_document(doc, timeframes)
        ]
    
    def _process_single_document(self, doc: Document, timeframes: List[str]) -> List[Document]:
        """Process a single document for all timeframes."""
        try:
            iep_result = self._get_processed_iep(doc)
            return self._generate_plans_for_timeframes(iep_result, timeframes)
        except Exception as e:
            self.logger.error(f"Error processing document: {e}")
            return []
    
    def _get_processed_iep(self, doc: Document) -> Document:
        """Get processed IEP document."""
        processed_docs = self.iep_pipeline.process_documents([doc])
        if not processed_docs:
            raise ValueError("Failed to process IEP document")
        return processed_docs[0]
    
    def _generate_plans_for_timeframes(self, iep_doc: Document, timeframes: List[str]) -> List[Document]:
        """Generate lesson plans for all timeframes."""
        plans = []
        for timeframe in timeframes:
            try:
                if not iep_doc.page_content.strip():
                    self.logger.warning(f"Empty content in IEP result for {timeframe}")
                    continue
                
                data = self._prepare_lesson_data(iep_doc, timeframe)
                lesson_plan = self.lesson_pipeline.generate_lesson_plan(data, timeframe)
                
                if lesson_plan:
                    plans.append(self._create_lesson_plan_document(lesson_plan, iep_doc, timeframe))
                    
            except Exception as e:
                self.logger.error(f"Error generating {timeframe} lesson plan: {e}")
                continue
                
        return plans
    
    def _prepare_lesson_data(self, doc: Document, timeframe: str) -> Dict[str, Any]:
        """Prepare data for lesson plan generation."""
        return {
            "iep_content": doc.page_content,
            "subject": "Mathematics",
            "grade_level": "5th Grade",
            "duration": "45 minutes",
            "specific_goals": ["Master basic algebra concepts"],
            "materials": ["Textbook", "Worksheets"],
            "additional_accommodations": ["Visual aids"],
            "timeframe": timeframe,
            "days": ["Monday", "Wednesday", "Friday"],
            "assessment_data": {}
        }
    
    def _create_lesson_plan_document(self, 
                                   lesson_plan: Dict[str, Any], 
                                   source_doc: Document,
                                   timeframe: str) -> Document:
        """Create a Document object from a lesson plan."""
        return Document(
            page_content=str(lesson_plan),
            metadata={
                **source_doc.metadata,
                "type": "lesson_plan",
                "timeframe": timeframe,
                "source_iep": source_doc.metadata.get("source")
            }
        )

class AssessmentDataExtractor:
    """Handles extraction of assessment data from documents."""
    
    def __init__(self):
        self.logger = logger
    
    def extract_data(self, doc: Document) -> Dict[str, Any]:
        """Extract assessment data from processed document."""
        empty_result = {
            "academic_analysis": "",
            "skill_gaps": "",
            "learning_objectives": ""
        }
        
        if not self._validate_document(doc):
            return empty_result
            
        try:
            content = doc.page_content
            return {
                "academic_analysis": self._extract_section(content, "Academic Analysis"),
                "skill_gaps": self._extract_section(content, "Skill Gaps"),
                "learning_objectives": self._extract_section(content, "Learning Objectives")
            }
            
        except Exception as e:
            self.logger.error(f"Error extracting assessment data: {e}")
            return empty_result
    
    def _validate_document(self, doc: Document) -> bool:
        """Validate document for data extraction."""
        if not doc or not isinstance(doc, Document):
            self.logger.warning("Invalid document for assessment data extraction")
            return False
            
        if not doc.page_content:
            self.logger.warning("Empty content in document")
            return False
            
        return True
    
    def _extract_section(self, content: str, section_name: str) -> str:
        """Extract a specific section from the content."""
        pattern = f"{section_name}:\n(.*?)(?=\n\n|$)"
        match = re.search(pattern, content, re.DOTALL)
        return match.group(1).strip() if match else ""

class FAISSIndexBuilder:
    """Handles building and managing FAISS indexes."""
    
    def __init__(self, persist_directory: str):
        self.persist_directory = persist_directory
        self.logger = logger
    
    def build_index_with_dspy(self, 
                             documents: List[Document], 
                             model_name: str = "o1-mini") -> Optional[FAISS]:
        """Build a FAISS index with DSPy-enhanced documents."""
        try:
            enhanced_docs = self._enhance_documents(documents, model_name)
            all_docs = documents + enhanced_docs
            return self._build_index(all_docs)
            
        except Exception as e:
            self.logger.error(f"Error in DSPy-enhanced indexing: {e}")
            return self._build_index(documents)  # Fallback to regular indexing
    
    def _enhance_documents(self, 
                          documents: List[Document], 
                          model_name: str) -> List[Document]:
        """Enhance documents using DSPy pipeline."""
        pipeline = IEPPipeline(model_name=model_name)
        return pipeline.process_documents(documents)
    
    def _build_index(self, documents: List[Document]) -> Optional[FAISS]:
        """Build FAISS index from documents."""
        try:
            from embeddings import build_faiss_index
            return build_faiss_index(documents, self.persist_directory)
        except Exception as e:
            self.logger.error(f"Error building FAISS index: {e}")
            return None

def process_iep_to_lesson_plans(documents: List[Document], 
                               timeframes: List[str] = ["daily"], 
                               config: Optional[ProcessingConfig] = None) -> List[Document]:
    """Process IEP documents into lesson plans."""
    processor = IEPLessonPlanProcessor(config)
    return processor.process_documents(documents, timeframes)

def process_uploaded_document(file_data: bytes, filename: str, test_mode: bool = False) -> Optional[Document]:
    """Process uploaded document with diagnostics."""
    try:
        logger.debug(f"Processing uploaded file: {filename}")
        
        # Create document
        doc = Document(
            page_content=file_data.decode('utf-8'),
            metadata={
                "source": filename,
                "type": "uploaded",
                "timestamp": datetime.now().isoformat()
            }
        )
        logger.debug(f"Created document object with metadata: {doc.metadata}")
        
        # Process through IEP pipeline
        processor = IEPPipeline(test_mode=test_mode)
        processed_docs = processor.process_documents([doc])
        
        if not processed_docs:
            logger.error("No documents returned from IEP pipeline")
            return None
            
        logger.debug(f"Successfully processed document: {processed_docs[0].metadata}")
        return processed_docs[0]
        
    except Exception as e:
        logger.error(f"Error processing uploaded document: {str(e)}", exc_info=True)
        return None

class FileUploadHandler:
    """Handles file upload processing and temporary file management."""
    
    def __init__(self):
        self.logger = logger
        self.temp_files = []
    
    def process_uploaded_file(self, uploaded_file) -> Optional[str]:
        """Process an uploaded file and return its temporary path."""
        try:
            import tempfile
            
            # Create temporary file
            temp_file = tempfile.NamedTemporaryFile(delete=False)
            self.temp_files.append(temp_file.name)
            
            # Write uploaded file content to temporary file
            temp_file.write(uploaded_file.getvalue())
            temp_file.close()
            
            return temp_file.name
            
        except Exception as e:
            self.logger.error(f"Error processing uploaded file: {e}")
            return None
    
    def cleanup(self):
        """Clean up temporary files."""
        for temp_path in self.temp_files:
            try:
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
            except Exception as e:
                self.logger.error(f"Error cleaning up temp file {temp_path}: {e}")

