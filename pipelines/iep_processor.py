from typing import Optional, Dict, Any
from langchain.schema import Document
from utils.logging_config import logger
from dspy_pipeline import IEPPipeline, LessonPlanPipeline
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from io import BytesIO
from datetime import datetime
import json

class IEPProcessor:
    def __init__(self):
        self.iep_pipeline = IEPPipeline()
        
    def process_document(self, doc: Document) -> Optional[Document]:
        """Process single document through IEP pipeline."""
        try:
            processed_docs = self.iep_pipeline.process_documents([doc])
            if not processed_docs:
                logger.error("IEP pipeline returned no documents")
                return None
                
            logger.debug(f"Generated IEP content length: {len(processed_docs[0].page_content)}")
            return processed_docs[0]
            
        except Exception as e:
            logger.error(f"Error in IEP pipeline: {str(e)}", exc_info=True)
            return None

class LessonPlanProcessor:
    def __init__(self):
        self.lesson_pipeline = LessonPlanPipeline()
        
    def generate_lesson_plan(self, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Generate lesson plan from prepared data."""
        try:
            logger.debug("Starting lesson plan generation")
            logger.debug(f"Input data: {json.dumps(data, indent=2)}")
            
            # Ensure timeframe is properly formatted
            timeframe = data.get("timeframe", "daily").lower()
            logger.debug(f"Using timeframe: {timeframe}")
            
            # Prepare data according to expected format
            prepared_data = {
                "iep_content": data["iep_content"],
                "subject": data["subject"],
                "grade_level": data["grade_level"],
                "duration": data["duration"],
                "specific_goals": data["specific_goals"],
                "materials": data.get("materials", []),
                "additional_accommodations": data.get("additional_accommodations", []),
                "timeframe": timeframe,
                "days": data.get("days", ["Daily"])
            }
            logger.debug(f"Prepared data: {json.dumps(prepared_data, indent=2)}")
            
            # Generate plan
            logger.debug("Calling lesson pipeline generate_lesson_plan")
            result = self.lesson_pipeline.generate_lesson_plan(prepared_data, timeframe)
            
            if not result:
                logger.error("Lesson pipeline returned None")
                return None
                
            logger.debug(f"Generated lesson plan: {json.dumps(result, indent=2)}")
            return result
            
        except Exception as e:
            logger.error(f"Error generating lesson plan: {str(e)}", exc_info=True)
            return None

class IEPDocumentGenerator:
    def __init__(self):
        self.styles = getSampleStyleSheet()
        self._setup_custom_styles()
    
    def _setup_custom_styles(self):
        """Setup custom paragraph styles for IEP document."""
        self.styles.add(ParagraphStyle(
            'IEPHeading1',
            parent=self.styles['Heading1'],
            fontSize=14,
            spaceAfter=20
        ))
        self.styles.add(ParagraphStyle(
            'IEPNormal',
            parent=self.styles['Normal'],
            fontSize=10,
            spaceAfter=12
        ))
        self.styles.add(ParagraphStyle(
            'IEPSection',
            parent=self.styles['Heading2'],
            fontSize=12,
            spaceAfter=10
        ))
    
    def generate_pdf(self, iep_data: dict) -> BytesIO:
        """Generate PDF from IEP data."""
        buffer = BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=72, leftMargin=72, topMargin=72, bottomMargin=72)
        
        story = []
        
        # Header
        story.append(Paragraph("Individualized Education Program (IEP)", self.styles['IEPHeading1']))
        story.append(Spacer(1, 12))
        
        # Add each section
        for section, content in iep_data["sections"].items():
            if content.strip():
                story.append(Paragraph(section, self.styles['IEPSection']))
                story.append(Spacer(1, 6))
                
                # Split content into paragraphs
                paragraphs = content.split('\n\n')
                for para in paragraphs:
                    if para.strip():
                        story.append(Paragraph(para.strip(), self.styles['IEPNormal']))
                story.append(Spacer(1, 12))
        
        # Build PDF
        doc.build(story)
        buffer.seek(0)
        return buffer 