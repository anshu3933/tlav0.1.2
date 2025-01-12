import streamlit as st
from chains import build_rag_chain
from embeddings import build_faiss_index, load_faiss_index
from loaders import load_documents
from dspy_pipeline import IEPPipeline, LessonPlanPipeline, ProcessingConfig, process_iep_to_lesson_plans
from pipelines.iep_processor import IEPProcessor, LessonPlanProcessor, IEPDocumentGenerator
import os
import tempfile
import shutil
import time
import json
from datetime import datetime
from langchain.schema import Document
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
import io
import zipfile
from utils.logging_config import logger
from typing import List, Optional, Dict, Any
from main import initialize_system
from dotenv import load_dotenv
from search_augmentation import SearchAugmentedRetrieval
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
import uuid

# Load environment variables
load_dotenv()

# Get API key from environment
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    st.error("OpenAI API key not found in environment variables. Please add it to your .env file.")
    st.stop()

# Page Configuration
st.set_page_config(
    page_title="Educational Assistant",
    page_icon=":books:",
    layout="wide"
)

# Initialize directories
DATA_DIR = "data"
INDEX_DIR = "models/faiss_index"
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(INDEX_DIR, exist_ok=True)

# Initialize session state with default values if not exists
def init_session_state():
    defaults = {
        "chain": None,
        "documents_processed": False,
        "messages": [],
        "iep_results": [],
        "documents": [],
        "lesson_plans": [],
        "current_plan": None
    }
    
    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value

# Initialize session state
init_session_state()

# Initialize system only if not already done
if st.session_state["chain"] is None:
    st.session_state["chain"] = initialize_system(
        data_dir=DATA_DIR,
        use_dspy=False
    )

def process_uploaded_file(uploaded_file):
    """Process an uploaded file with unified state management."""
    try:
        logger.debug(f"=== Processing uploaded file: {uploaded_file.name} ===")
        
        # Create a temporary file to save the uploaded file
        suffix = os.path.splitext(uploaded_file.name)[1]  # e.g., '.pdf'
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_file_path = tmp_file.name
            logger.debug(f"Temporary file created at: {tmp_file_path}")
        
        # Load document using the temporary file path
        documents = load_documents([tmp_file_path])
        logger.debug(f"Loaded documents: {documents}")
        
        if not documents:
            st.error("No text could be extracted from the uploaded file.")
            os.unlink(tmp_file_path)  # Clean up
            return False
            
        # Process through IEP pipeline
        processed_docs = process_iep_to_lesson_plans(documents)
        logger.debug(f"Processed documents: {processed_docs}")
        
        if processed_docs:
            # Update session state
            st.session_state["documents"].extend(processed_docs)
            logger.debug("Processed documents added to session state.")
            
            # Generate IEP for each processed document
            for doc in processed_docs:
                save_iep_to_session(doc, {
                    "name": uploaded_file.name,
                    "source": "uploaded"
                })
            logger.debug("IEP results saved to session state.")
            
            # Clean up temporary file
            os.unlink(tmp_file_path)
            logger.debug(f"Temporary file {tmp_file_path} deleted.")
            
            st.success(f"Successfully processed {uploaded_file.name}")
            st.rerun()
            return True
        
        # Clean up if processing failed
        os.unlink(tmp_file_path)
        logger.debug(f"Temporary file {tmp_file_path} deleted due to processing failure.")
        return False
        
    except Exception as e:
        logger.error(f"Error processing upload: {str(e)}", exc_info=True)
        st.error("Failed to process file")
        return False

def create_lesson_plan_pdf(plan_data):
    """Create a formatted PDF from lesson plan data."""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []

    # Title
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        spaceAfter=30
    )
    story.append(Paragraph(f"Lesson Plan - {plan_data['timeframe'].title()}", title_style))
    story.append(Spacer(1, 12))

    # Sections
    for section, content in plan_data.items():
        if section not in ['timeframe', 'timestamp', 'source_iep', 'quality_score']:
            # Section header
            story.append(Paragraph(section.replace('_', ' ').title(), styles['Heading2']))
            story.append(Spacer(1, 6))
            
            # Section content
            if isinstance(content, list):
                for item in content:
                    story.append(Paragraph(f"• {item}", styles['Normal']))
            else:
                story.append(Paragraph(str(content), styles['Normal']))
            story.append(Spacer(1, 12))

    # Build PDF
    doc.build(story)
    return buffer

def serialize_iep_data(iep_data):
    """Convert IEP data to JSON-serializable format."""
    return {
        "source": str(iep_data.get("source", "Unknown")),
        "timestamp": iep_data.get("timestamp", datetime.now().isoformat()),
        "content": str(iep_data.get("content", "")),
        "metadata": {
            k: str(v) for k, v in iep_data.get("metadata", {}).items()
        }
    }

def format_iep_content(content: str, max_length: int = 1000) -> dict:
    """Format IEP content into standardized sections."""
    sections = {
        "Student Information": "",
        "Present Levels": "",
        "Annual Goals": "",
        "Special Education Services": "",
        "Accommodations & Modifications": "",
        "Assistive Technology": "",
        "Behavior Intervention": "",
        "Transition Plan": "",
        "Progress Monitoring": "",
        "Team Members": "",
        "Additional Notes": ""
    }
    
    # Split content into sections based on headers
    current_section = "Additional Notes"  # Default section
    current_paragraph = []
    
    # Split into sentences and clean up
    sentences = [s.strip() for s in content.replace('\n', ' ').split('.') if s.strip()]
    
    for sentence in sentences:
        lower_sentence = sentence.lower()
        
        # Section detection logic
        if any(keyword in lower_sentence for keyword in ["name", "birth", "grade", "school", "case manager"]):
            current_section = "Student Information"
            if not sections[current_section]:
                sections[current_section] += "\n\n### Student Details\n"
        elif any(keyword in lower_sentence for keyword in ["present level", "current performance", "assessment", "evaluation"]):
            current_section = "Present Levels"
            if not sections[current_section]:
                sections[current_section] += "\n\n### Present Levels of Performance\n"
        elif any(keyword in lower_sentence for keyword in ["goal", "objective", "benchmark", "target"]):
            current_section = "Annual Goals"
            if not sections[current_section]:
                sections[current_section] += "\n\n### Measurable Annual Goals\n"
        elif any(keyword in lower_sentence for keyword in ["special education", "related service", "instruction"]):
            current_section = "Special Education Services"
            if not sections[current_section]:
                sections[current_section] += "\n\n### Special Education and Related Services\n"
        elif any(keyword in lower_sentence for keyword in ["accommodat", "modif", "support"]):
            current_section = "Accommodations & Modifications"
            if not sections[current_section]:
                sections[current_section] += "\n\n### Accommodations and Modifications\n"
        elif any(keyword in lower_sentence for keyword in ["technology", "device", "software"]):
            current_section = "Assistive Technology"
            if not sections[current_section]:
                sections[current_section] += "\n\n### Assistive Technology Needs\n"
        elif any(keyword in lower_sentence for keyword in ["behavior", "intervention", "positive support"]):
            current_section = "Behavior Intervention"
            if not sections[current_section]:
                sections[current_section] += "\n\n### Behavior Intervention Plan\n"
        elif any(keyword in lower_sentence for keyword in ["transition", "postsecondary", "career"]):
            current_section = "Transition Plan"
            if not sections[current_section]:
                sections[current_section] += "\n\n### Transition Planning\n"
        elif any(keyword in lower_sentence for keyword in ["progress", "monitor", "report"]):
            current_section = "Progress Monitoring"
            if not sections[current_section]:
                sections[current_section] += "\n\n### Progress Monitoring Plan\n"
        elif any(keyword in lower_sentence for keyword in ["team", "signature", "participant"]):
            current_section = "Team Members"
            if not sections[current_section]:
                sections[current_section] += "\n\n### IEP Team Members\n"
        
        # Add sentence to current paragraph
        current_paragraph.append(sentence)
        
        # Create paragraph breaks
        if len(current_paragraph) >= 3 or any(keyword in sentence.lower() for keyword in 
            ["therefore", "however", "additionally", "furthermore", "moreover"]):
            sections[current_section] += " ".join(current_paragraph) + ".\n\n"
            current_paragraph = []
    
    # Add remaining content
    if current_paragraph:
        sections[current_section] += " ".join(current_paragraph) + ".\n\n"
    
    # Clean up and format sections
    for section in sections:
        sections[section] = sections[section].strip()
        if sections[section]:
            sections[section] = sections[section].replace('\n- ', '\n• ')
            sections[section] = sections[section].replace('\n###', '\n\n###')
    
    return {
        "sections": {k: v for k, v in sections.items() if v.strip()},
        "truncated": any(len(content) > max_length for content in sections.values())
    }

def handle_iep_generation(selected_doc: dict) -> None:
    """Handle IEP generation UI flow."""
    if not selected_doc:
        st.error("Please select a document first.")
        return

    with st.spinner("Generating IEP..."):
        try:
            doc = get_document_for_processing(selected_doc)
            if not doc:
                st.error("Could not retrieve document for processing.")
                return
                
            processor = IEPProcessor()
            processed_doc = processor.process_document(doc)
            
            if processed_doc:
                save_iep_to_session(processed_doc, selected_doc)
                display_iep_content(processed_doc.page_content, selected_doc["name"])
            else:
                st.error("Failed to generate IEP content.")
                
        except Exception as e:
            logger.error(f"Error during IEP generation: {str(e)}", exc_info=True)
            st.error("Error generating IEP. Please try again.")

def get_document_for_processing(selected_doc: dict) -> Optional[Document]:
    """Retrieve document based on selection source."""
    if selected_doc["source"] == "uploaded":
        return next(
            (doc for doc in st.session_state["documents"] 
             if doc.metadata["source"] == selected_doc["name"]),
            None
        )
    return load_documents([os.path.join(DATA_DIR, selected_doc["name"])])[0]

def process_document_with_pipeline(doc: Document) -> Optional[Document]:
    """Process document through IEP pipeline."""
    try:
        pipeline = IEPPipeline()
        processed_docs = pipeline.process_documents([doc])
        
        if not processed_docs:
            logger.error("IEP pipeline returned no documents")
            return None
            
        logger.debug(f"Generated IEP content length: {len(processed_docs[0].page_content)}")
        return processed_docs[0]
        
    except Exception as e:
        logger.error(f"Error in IEP pipeline: {str(e)}", exc_info=True)
        return None

def display_iep_content(content: str, doc_name: str) -> None:
    """Display formatted IEP content with download option."""
    formatted_content = format_iep_content(content)
    
    # Display sections
    for section, content in formatted_content["sections"].items():
        if content.strip():
            with st.expander(section, expanded=True):
                st.markdown(content)
    
    # Add download button
    if formatted_content["sections"]:
        generator = IEPDocumentGenerator()
        pdf_buffer = generator.generate_pdf(formatted_content)
        
        st.download_button(
            label="Download IEP as PDF",
            data=pdf_buffer,
            file_name=f"IEP_{doc_name}_{datetime.now().strftime('%Y%m%d')}.pdf",
            mime="application/pdf"
        )

def save_and_display_iep(processed_doc: Document, selected_doc: dict) -> None:
    """Save IEP result to session state and display success."""
    try:
        iep_result = {
            "source": selected_doc["name"],
            "content": processed_doc.page_content,
            "metadata": {
                **processed_doc.metadata,
                "timestamp": datetime.now().isoformat()
            }
        }
        
        if "iep_results" not in st.session_state:
            st.session_state["iep_results"] = []
            
        st.session_state["iep_results"].append(iep_result)
        st.success("IEP generated successfully!")
        
        # Display sections
        st.subheader("Generated IEP Content")
        st.markdown(processed_doc.page_content)
        
    except Exception as e:
        logger.error(f"Error saving/displaying IEP: {str(e)}")
        st.error("Error displaying IEP content")

def get_available_documents():
    """Get list of available documents from all sources."""
    available_documents = []
    
    # Get uploaded documents first
    if st.session_state.get("documents"):
        available_documents.extend([
            {
                "name": doc.metadata.get("source", "Untitled Document"),
                "source": "uploaded",
                "display": f"📤 {doc.metadata.get('source', 'Untitled Document')}"
            }
            for doc in st.session_state["documents"]
            if "source" in doc.metadata
        ])
    
    # Get documents from data directory
    if os.path.exists(DATA_DIR):
        data_files = [f for f in os.listdir(DATA_DIR) 
                     if f.endswith(('.pdf', '.docx', '.txt'))]
        available_documents.extend([
            {"name": f, "source": "data_dir", "display": f"📁 {f}"} 
            for f in data_files
        ])
    
    return available_documents

def handle_lesson_plan_generation(form_data: dict) -> Optional[Dict[str, Any]]:
    """Handle lesson plan generation with proper error handling."""
    try:
        processor = LessonPlanProcessor()
        lesson_plan = processor.generate_lesson_plan(form_data)
        
        if not lesson_plan:
            logger.error("Failed to generate lesson plan")
            return None
            
        return lesson_plan
        
    except Exception as e:
        logger.error(f"Error in lesson plan generation: {str(e)}", exc_info=True)
        return None

def save_iep_to_session(processed_doc: Document, selected_doc: dict) -> None:
    """Save IEP result to session state."""
    iep_result = {
        "source": selected_doc["name"],
        "content": processed_doc.page_content,
        "metadata": processed_doc.metadata,
        "timestamp": datetime.now().isoformat()
    }
    
    if "iep_results" not in st.session_state:
        st.session_state["iep_results"] = []
        
    # Convert to JSON-serializable format
    serialized_result = serialize_iep_data(iep_result)
    st.session_state["iep_results"].append(serialized_result)
    st.success("IEP generated successfully!")

def handle_file_upload(uploaded_file) -> None:
    """Handle file upload with proper session state management."""
    try:
        logger.debug(f"Handling upload for file: {uploaded_file.name}")
        
        # Create temporary file to handle PDF
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name
            
        # Load document using file path
        documents = load_documents([tmp_file_path])
        
        if not documents:
            st.error("Could not load document")
            return
            
        # Process through IEP pipeline
        processed_docs = process_iep_to_lesson_plans(documents)
        if not processed_docs:
            st.error("Could not process document")
            return
            
        # Update session state
        if "documents" not in st.session_state:
            st.session_state.documents = []
            
        # Add processed document
        for doc in processed_docs:
            doc_entry = {
                "name": uploaded_file.name,
                "content": doc.page_content,
                "metadata": doc.metadata,
                "timestamp": datetime.now().isoformat()
            }
            st.session_state.documents.append(doc_entry)
            
        st.success(f"Successfully processed {uploaded_file.name}")
        
        # Cleanup temp file
        os.unlink(tmp_file_path)
        st.rerun()
        
    except Exception as e:
        logger.error(f"Error in file upload handler: {str(e)}", exc_info=True)
        st.error("Failed to process file")

def handle_chat_input(user_input: str):
    """Handle chat input with search augmentation."""
    try:
        if not st.session_state.get("chain"):
            return {
                "role": "assistant",
                "content": "I can help answer questions about documents once they're uploaded. For now, I can assist with general educational questions.",
                "sources": []
            }
            
        # Initialize search augmented retrieval
        augmented_retrieval = SearchAugmentedRetrieval(
            vector_store=st.session_state.chain["vectorstore"],
            search_limit=3
        )
        
        # Get relevant documents from vector store using invoke
        docs = st.session_state.chain["chain"].retriever.invoke(user_input)
        
        # Augment with search results
        augmented_docs = augmented_retrieval.augment_context(
            query=user_input,
            doc_results=docs,
            search_weight=0.3
        )
        
        # Create new vector store for temporary use
        temp_vectorstore = FAISS.from_documents(
            augmented_docs,
            OpenAIEmbeddings()
        )
        
        # Create temporary retriever
        temp_retriever = temp_vectorstore.as_retriever()
        
        # Generate response using temporary retriever
        response = st.session_state.chain["chain"].invoke(
            {"query": user_input},
            {"retriever": temp_retriever}
        )
        
        message_data = {
            "role": "assistant",
            "content": response["result"],
            "sources": augmented_docs
        }
        
        return message_data
        
    except Exception as e:
        logger.error(f"Error in chat handling: {str(e)}", exc_info=True)
        return {
            "role": "assistant",
            "content": "I encountered an error while processing your question. Please try again.",
            "sources": []
        }

st.title("Educational Assistant")

# Sidebar for API Key and File Upload
with st.sidebar:
    st.title("Document Upload")
    # Set hardcoded API key

    #use_dspy = st.checkbox("Use DSPy Processing", value=False)

    uploaded_files = st.file_uploader(
        "Upload educational documents",
        type=["txt", "docx", "pdf", "md"],
        accept_multiple_files=True
    )

    if uploaded_files and not st.session_state["documents_processed"]:
        processing_success = True  # Track overall success
        with st.spinner("Processing documents..."):
            st.write("### Processing Files")
            for file in uploaded_files:
                status_container = st.empty()
                status_container.info(f"Processing {file.name}...")
                
                try:
                    # Save uploaded file to temp directory
                    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.name)[1]) as tmp_file:
                        tmp_file.write(file.getbuffer())
                        file_path = tmp_file.name
                    
                    # Load and process document
                    documents = load_documents([file_path])
                    if not documents:
                        status_container.error(f"Could not extract text from {file.name}")
                        processing_success = False
                        continue
                        
                    # Process with IEP pipeline
                    try:
                        processed_docs = process_iep_to_lesson_plans(documents)
                        if processed_docs and len(processed_docs) > 0:
                            st.session_state["documents"].extend(processed_docs)
                            status_container.success(f"Successfully processed {file.name}")
                        else:
                            status_container.warning(f"No lesson plans generated for {file.name}")
                            processing_success = False
                    except IndexError:
                        status_container.error(f"Error processing content from {file.name}. File may be empty or corrupted.")
                        processing_success = False
                    except Exception as e:
                        status_container.error(f"Error processing {file.name}: {str(e)}")
                        processing_success = False
                        
                    # Clean up temp file
                    os.unlink(file_path)
                    
                except Exception as e:
                    status_container.error(f"Error handling {file.name}: {str(e)}")
                    processing_success = False
            
            if processing_success:
                st.session_state["documents_processed"] = True
                st.success("All documents processed successfully!")
            else:
                st.error("Error processing some documents.")

    if st.session_state["documents_processed"]:
        if st.button("Clear Documents"):
            st.session_state["documents_processed"] = False
            st.session_state["chain"] = None
            st.session_state["documents"] = []  # Clear documents
            st.session_state["iep_results"] = []  # Clear IEP results
            if os.path.exists(INDEX_DIR):
                shutil.rmtree(INDEX_DIR)
            st.rerun()

    #st.title("System Status")
#    if st.button("Check System Health"):
 #       status = check_system_health()
        
  #      st.write("### System Components Status")
   #     for component, is_healthy in status.items():
    #        if is_healthy:
    #            st.success(f"✅ {component}: OK")
    #        else:
    #            st.error(f"❌ {component}: Failed")
                
        # Show detailed status
        #with st.expander("System Details"):
         #   st.write("- Directories:", "Created" if status["directories"] else "Failed")
          #  st.write("- Document Loading:", "Working" if status["document_loading"] else "Failed")
           # st.write("- Vector Store:", "Operational" if status["vectorstore"] else "Failed")
            #st.write("- RAG Chain:", "Functional" if status["chain"] else "Failed")
            st.write("- DSPy Integration:", "Available" if status["dspy"] else "Not Available")

# Change from four tabs to three
tab1, tab2, tab3 = st.tabs(["Chat", "IEP Generation", "Lesson Plans"])

# Chat Interface Tab
with tab1:
    st.header("Chat with your documents")
    
    # Initialize chat history and messages
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("sources"):
                with st.expander("View Sources"):
                    for i, source in enumerate(message["sources"], 1):
                        st.write(f"Source {i}:")
                        st.write(source.page_content)
                        if source.metadata.get('source'):
                            st.write(f"Source URL: {source.metadata['source']}")
                        st.write("---")

    if prompt := st.chat_input("Ask a question about the documents..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    if st.session_state.get("chain"):
                        # Use search augmented retrieval
                        augmented_retrieval = SearchAugmentedRetrieval(
                            vector_store=st.session_state.chain["vectorstore"],
                            search_limit=3
                        )
                        
                        # Get documents using invoke
                        docs = st.session_state.chain["chain"].retriever.invoke(prompt)
                        
                        # Augment with search results
                        augmented_docs = augmented_retrieval.augment_context(
                            query=prompt,
                            doc_results=docs,
                            search_weight=0.3
                        )
                        
                        # Create new temporary vector store
                        temp_vectorstore = FAISS.from_documents(
                            augmented_docs,
                            OpenAIEmbeddings()
                        )
                        
                        # Use temporary retriever for this query
                        temp_retriever = temp_vectorstore.as_retriever()
                        
                        # Generate response
                        response = st.session_state.chain["chain"].invoke(
                            {"query": prompt},
                            {"retriever": temp_retriever}
                        )
                        
                        # Format message with sources
                        message_data = {
                            "role": "assistant",
                            "content": response["result"],
                            "sources": augmented_docs
                        }
                    else:
                        message_data = {
                            "role": "assistant",
                            "content": "I can help answer questions about documents once they're uploaded. For now, I can assist with general educational questions.",
                            "sources": []
                        }
                    
                    # Add response to chat history
                    st.session_state.messages.append(message_data)
                    st.markdown(message_data["content"])
                    
                    # Show sources if available
                    if message_data["sources"]:
                        with st.expander("View Sources"):
                            for i, doc in enumerate(message_data["sources"], 1):
                                st.write(f"Source {i}:")
                                st.write(doc.page_content)
                                if doc.metadata.get('source'):
                                    st.write(f"Source URL: {doc.metadata['source']}")
                                st.write("---")
                                
                except Exception as e:
                    logger.error(f"Error generating response: {str(e)}", exc_info=True)
                    error_message = {
                        "role": "assistant",
                        "content": f"I encountered an error while processing your question. Please try again.",
                        "sources": []
                    }
                    st.session_state.messages.append(error_message)
                    st.markdown(error_message["content"])

    # Add clear chat button with confirmation
    if st.session_state.messages and st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

# IEP Generation Tab
with tab2:
    st.header("IEP Generation")
    
    # Document Selection Section
    st.markdown("### Document Selection")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Upload a new document",
        type=["pdf", "docx", "txt"],
        help="Upload assessment reports or existing IEPs"
    )
    
    if uploaded_file:
        if process_uploaded_file(uploaded_file):
            st.success(f"Successfully processed {uploaded_file.name}")
            st.session_state["documents_processed"] = True
    
    # Document selector
    available_documents = get_available_documents()
    if available_documents:
        selected_doc = st.selectbox(
            "Select document to process",
            options=available_documents,
            format_func=lambda x: x["display"],
            key="iep_doc_selector"  # Unique key for the selector
        )
        
        if st.button("Generate IEP"):
            handle_iep_generation(selected_doc)
    else:
        st.info("No documents available. Please upload a document or add documents to the data directory.")

# Lesson Plan Generation Tab
with tab3:
    st.header("Lesson Plan Generation")
    
    # Combined form for all lesson plan generation
    st.subheader("Lesson Plan Details")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        with st.form("lesson_plan_form"):
            # Required form fields
            st.markdown("### Basic Information")
            subject = st.text_input("Subject *", placeholder="e.g., Mathematics, Reading, Science")
            grade_level = st.text_input("Grade Level *", placeholder="e.g., 3rd Grade, High School")
            
            # Timeframe selection
            timeframe = st.radio(
                "Schedule Type *",
                ["Daily", "Weekly"],
                help="Choose between a daily lesson plan or a weekly schedule"
            )
            
            duration = st.text_input(
                "Daily Duration *", 
                placeholder="e.g., 45 minutes per session"
            )
            
            if timeframe == "Weekly":
                days_per_week = st.multiselect(
                    "Select Days *",
                    ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"],
                    default=["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
                )
            
            st.markdown("### Learning Details")
            specific_goals = st.text_area(
                "Specific Learning Goals *",
                placeholder="Enter specific goals for this lesson, one per line"
            )
            
            materials = st.text_area(
                "Materials Needed",
                placeholder="List required materials, one per line"
            )
            
            st.markdown("### Additional Support")
            additional_accommodations = st.text_area(
                "Additional Accommodations",
                placeholder="Enter any specific accommodations beyond those in the IEP"
            )
            
            # IEP Selection
            st.markdown("### IEP Integration")
            if st.session_state.get("iep_results"):
                selected_iep = st.selectbox(
                    "Select IEP to Integrate *",
                    options=[iep["source"] for iep in st.session_state["iep_results"]],
                    format_func=lambda x: f"IEP from {x}"
                )
            else:
                st.error("No IEPs available. Please generate an IEP first.")
                selected_iep = None
            
            st.markdown("*Required fields")
            
            generate_button = st.form_submit_button("Generate Enhanced Lesson Plan")

            if generate_button:
                if not all([subject, grade_level, duration, specific_goals, selected_iep]):
                    st.error("Please fill in all required fields.")
                else:
                    try:
                        # Get selected IEP data
                        iep_data = next(
                            iep for iep in st.session_state["iep_results"] 
                            if iep["source"] == selected_iep
                        )
                        
                        # Prepare data for lesson plan generation
                        combined_data = {
                            "iep_content": iep_data["content"],
                            "subject": subject,
                            "grade_level": grade_level,
                            "duration": duration,
                            "specific_goals": specific_goals.split('\n'),
                            "materials": materials.split('\n') if materials else [],
                            "additional_accommodations": additional_accommodations.split('\n') if additional_accommodations else [],
                            "timeframe": timeframe.lower(),
                            "days": days_per_week if timeframe == "Weekly" else ["Daily"]
                        }
                        
                        # Generate lesson plan
                        pipeline = LessonPlanPipeline()
                        lesson_plan = pipeline.generate_lesson_plan(combined_data)
                        
                        if lesson_plan:
                            # Create complete plan data structure
                            plan_data = {
                                # Input data
                                "subject": subject,
                                "grade_level": grade_level,
                                "duration": duration,
                                "timeframe": timeframe,
                                "days": days_per_week if timeframe == "Weekly" else ["Daily"],
                                "specific_goals": specific_goals.split('\n'),
                                "materials": materials.split('\n') if materials else [],
                                "additional_accommodations": additional_accommodations.split('\n') if additional_accommodations else [],
                                # Generated content
                                "schedule": lesson_plan.get('schedule', []),
                                "lesson_content": lesson_plan.get('lesson_plan', []),
                                "learning_objectives": lesson_plan.get('learning_objectives', []),
                                "assessment_criteria": lesson_plan.get('assessment_criteria', []),
                                "modifications": lesson_plan.get('modifications', []),
                                # Metadata
                                "source_iep": selected_iep,
                                "timestamp": datetime.now().isoformat()
                            }
                            
                            if "lesson_plans" not in st.session_state:
                                st.session_state["lesson_plans"] = []
                            
                            st.session_state["lesson_plans"].append(plan_data)
                            st.success("Lesson plan generated successfully!")
                            st.rerun()
                        else:
                            st.error("Failed to generate lesson plan.")
                            
                    except Exception as e:
                        st.error(f"Error generating lesson plan: {str(e)}")

# Move PDF download outside form
if st.session_state.get("current_plan"):
    pdf_buffer = create_lesson_plan_pdf(st.session_state["current_plan"])
    st.download_button(
        label="Download PDF",
        data=pdf_buffer.getvalue(),
        file_name=f"lesson_plan_{timeframe.lower()}_{subject.lower().replace(' ', '_')}.pdf",
        mime="application/pdf"
    )

# Display generated lesson plans
if st.session_state.get("lesson_plans"):
    st.markdown("### Generated Lesson Plans")
    
    for idx, plan in enumerate(st.session_state["lesson_plans"]):
        with st.expander(f"Lesson Plan {idx + 1} - {plan.get('subject', 'Untitled')}", expanded=False):
            # Basic info
            st.markdown(f"**Subject**: {plan.get('subject', 'Not specified')}")
            st.markdown(f"**Grade Level**: {plan.get('grade_level', 'Not specified')}")
            st.markdown(f"**Duration**: {plan.get('duration', 'Not specified')}")
            
            # PDF download with safe access
            pdf_buffer = create_lesson_plan_pdf(plan)
            st.download_button(
                label=f"Download Plan {idx + 1} (PDF)",
                data=pdf_buffer.getvalue(),
                file_name=f"lesson_plan_{idx + 1}_{plan.get('subject', 'untitled').lower().replace(' ', '_')}.pdf",
                mime="application/pdf",
                key=f"download_plan_{idx}"
            )

# Footer
st.markdown("---")
st.markdown("Educational Assistant")
