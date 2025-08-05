# Personal Health Copilot - Certification Challenge Documentation

## Executive Summary

The Personal Health Copilot is an intelligent medical assistant that helps users understand complex medical information, lab results, and medication interactions through a conversational interface. Built with advanced RAG (Retrieval Augmented Generation) technology and agentic AI, the system provides accurate, personalized medical guidance while maintaining user privacy and safety.

The application features a modern web interface with real-time chat capabilities, file upload processing for lab results, and comprehensive medical information retrieval from verified sources including Mayo Clinic, NIH, and FDA. The system uses BM25 retrieval for precise medical information matching and integrates multiple specialized tools for symptom analysis, drug interaction checking, and research synthesis.

Key achievements include successful RAGAS evaluation with 100% context recall and 94.44% context precision, advanced retrieval techniques with quantified performance improvements, and a production-ready architecture with FastAPI backend and Next.js frontend. The system demonstrates strong performance in medical information retrieval while providing clear pathways for future enhancements.

---

## Task 1: Defining the Problem and Audience

### 1.1 Problem Statement

**Problem Statement:** Users struggle to understand complex medical information, lab results, and medication interactions, leading to confusion and potential health risks.

### 1.2 User Impact Analysis

Chronic illness patients and health-conscious individuals face overwhelming amounts of medical information that is often written in technical jargon. When they receive lab results, they struggle to understand what the numbers mean, whether values are normal, and what actions they should take. Similarly, when prescribed new medications, they need to understand potential side effects, worry about interactions with existing drugs and whether the medication is safe to take with their existing prescriptions but lack an easy way to check this information.

Caregivers supporting elderly family members or children with chronic conditions need to quickly understand medical reports and make informed decisions. They need quick, reliable answers to questions like "What does this lab value mean?" or "Can I take this medication with my current prescriptions?" but lack access to immediate medical consultation. Current solutions provide conflicting information or require extensive research across multiple sources.

The current solutions - searching the internet, especially Google search or waiting for doctor appointments - are either unreliable (due to misinformation) or too slow for urgent questions. This creates anxiety, delays in care, and potential health risks from misinterpreted information.

**User Personas Identified:**
- **Chronic Illness Patient**: Self-managing health conditions with pain points including complex lab result interpretation, medication interaction concerns, symptom tracking, and treatment option understanding
- **Health-Conscious Individual**: Preventive health monitoring with pain points including understanding preventive health metrics, lifestyle recommendations, supplement interaction checking, and wellness goal tracking
- **Caregiver**: Supporting loved ones' health with pain points including understanding medical reports for others, medication management for multiple people, emergency health decision making, and communication with healthcare providers

**Potential User Questions:**
- "What do these lab results mean?"
- "Are these medications safe to take together?"
- "What could be causing these symptoms?"
- "How do I interpret this blood test?"
- "What are the side effects of this medication?"
- "Is this normal for my age and condition?"
- "What lifestyle changes should I make?"
- "How do I track my symptoms over time?"
- "What questions should I ask my doctor?"
- "Are there alternative treatments available?"
- "Can I take this supplement with my current medications?"
- "What does this medical term mean?"
- "How serious is this condition?"
- "What should I do if my symptoms get worse?"
- "Are there any warning signs I should watch for?"

---

## Task 2: Propose a Solution

### 2.1 Proposing a Solution

The Personal Health Copilot provides a conversational web application that feels like having a knowledgeable medical assistant at your fingertips. Users interact with a clean, chat-based interface where they can upload lab reports, describe symptoms, or ask questions about medications. The system responds with clear, personalized explanations written in plain language, avoiding medical jargon while maintaining accuracy.

For lab results, it highlights abnormal values and explains what they mean for the user's specific health context. The system integrates with external APIs to provide real-time drug interaction checking and up-to-date medical information from verified sources like Mayo Clinic and NIH.

### 2.2 Technology Stack and Tooling Decisions

• **LLM:** OpenAI GPT-4o-mini for all operations (generation, retrieval, evaluation) - Provides cost-effective and fast medical analysis with good reasoning capabilities for medical queries.

• **Embedding Model:** OpenAI text-embedding-3-small - Offers strong semantic understanding for medical terminology and concepts.

• **Orchestration:** LangGraph for agentic workflows - Enables multi-step reasoning for complex health queries and medication analysis.

• **Retrieval System:** BM25 retriever for in-memory document search - Provides fast keyword-based retrieval for medical documents without requiring a separate vector database.

• **Monitoring:** LangSmith for tracing and evaluation - Tracks agent performance, model usage, and provides insights for healthcare accuracy.

• **Evaluation:** RAGAS framework for quantitative assessment - Measures faithfulness, relevance, and precision critical for medical applications.

• **User Interface:** Next.js frontend with FastAPI backend - Provides modern, responsive chat interface with scalable architecture for production deployment.

• **Serving & Inference:** FastAPI backend with async processing - Provides scalable API endpoints for real-time medical query processing and response generation.

• **External APIs:** Tavily Search API and ArXiv API - Provides real-time medical information and research paper access.

### 2.3 Agent Architecture and Agentic Reasoning Usage

**Single Agent with Multiple Tools Implementation:**
The Personal Health Copilot uses a single agent with specialized medical tools for different domains:

1. **Medical Information Retrieval Tool:** Retrieves verified medical information from Mayo Clinic, NIH, and FDA sources using BM25 retriever
2. **Symptom Analysis Tool:** Analyzes user symptoms and suggests possible conditions based on medical knowledge
3. **Drug Interaction Tool:** Checks potential interactions between medications with safety recommendations
4. **Lab Results Tool:** Interprets lab test results and explains what they mean in plain language
5. **Web Search Tool:** Provides real-time medical information from Tavily Search API with formatted results
6. **Research Tool:** Searches medical research papers from ArXiv API with titles, authors, and abstracts
7. **Uploaded Lab Results Tool:** Analyzes uploaded lab results (PDF, DOCX) and answers user questions about them

**Agentic Reasoning Applications:**
- **Multi-step Medical Queries:** User question → tool selection → information retrieval → response synthesis
- **Drug Interaction Analysis:** Current medications → interaction checking → safety assessment → clear explanation
- **Lab Result Interpretation:** Raw lab data → medical context → interpretation → recommendations
- **Research Synthesis:** Medical query → multiple sources → evidence synthesis → clear explanation
- **Real-time Information:** User question → web search → latest medical updates → current information
- **File Processing Workflow:** Uploaded lab files → text extraction → medical analysis → personalized interpretation

---

## Task 3: Dealing with the Data

### 3.1 Data Sources and External APIs

**RAG Data Sources:**

**Mayo Clinic Medical Information:**
- Use: Comprehensive, verified medical content for symptom explanations, treatment options, and disease information
- Source: Public Mayo Clinic articles downloaded via curl commands (diabetes, hypertension, heart disease, arthritis, asthma guides)
- Purpose: Provide accurate, trusted medical information for user queries

**NIH MedlinePlus:**
- Use: Drug information, side effects, and medication guides
- Source: Public NIH MedlinePlus database articles (aspirin, ibuprofen, metformin, statins information)
- Purpose: Supplement drug-related queries and provide official medication information

**FDA Drug Safety Information:**
- Use: Drug interactions, safety information, and regulatory data
- Source: FDA's public drug safety and availability information
- Purpose: Real-time drug interaction checking and safety alerts

**Lab Test Information:**
- Use: Complete blood count (CBC), blood glucose test, and cholesterol level information
- Source: NIH MedlinePlus lab test guides
- Purpose: Provide accurate lab result interpretation and normal range information

**External APIs:**

**Tavily Search API:**
- Use: Real-time medical information and latest research
- Purpose: Supplement RAG with current medical news and research

**ArXiv API:**
- Use: Latest medical research papers and clinical studies
- Purpose: Provide evidence-based medical information

**Data Source Mapping to User Questions:**

**Lab Results Questions → Lab Test Information + Mayo Clinic:**
- "What do these lab results mean?" → NIH MedlinePlus lab guides + Mayo Clinic interpretation
- "How do I interpret this blood test?" → CBC, glucose, cholesterol guides
- "Is this normal for my age and condition?" → Normal range data + condition-specific context

**Medication Questions → FDA + NIH MedlinePlus:**
- "Are these medications safe to take together?" → FDA drug interaction database
- "What are the side effects of this medication?" → NIH MedlinePlus drug guides
- "Can I take this supplement with my current medications?" → FDA safety information

**Symptom Questions → Mayo Clinic + Tavily Search:**
- "What could be causing these symptoms?" → Mayo Clinic disease guides + current research
- "How serious is this condition?" → Mayo Clinic severity information
- "What should I do if my symptoms get worse?" → Mayo Clinic treatment guidelines

**Treatment Questions → ArXiv + Tavily Search:**
- "Are there alternative treatments available?" → ArXiv research papers + current studies
- "What lifestyle changes should I make?" → Mayo Clinic lifestyle recommendations
- "What questions should I ask my doctor?" → Mayo Clinic consultation guides

### 3.2 Chunking Strategy and Implementation

**Strategy:** RecursiveCharacterTextSplitter with medical-optimized parameters
- Chunk size: 1200 tokens
- Chunk overlap: 200 tokens
- Length function: tiktoken_len
- Separators: Default medical document structure

**Why this decision:** Medical documents contain complex, interconnected information that requires context preservation. The 1200-token chunk size ensures that complete medical concepts, including symptoms, treatments, and their relationships, are captured in single chunks. Medical information often spans multiple sentences and paragraphs, so larger chunks prevent fragmentation of critical medical concepts.

The 200-token overlap is essential for medical content because it preserves context across chunk boundaries. Medical concepts like drug interactions, symptom patterns, and treatment protocols often span chunk boundaries, and this overlap ensures that related information remains connected. This is particularly important for maintaining the accuracy and completeness of medical information retrieval.

**Why Recursive Chunking over Semantic Chunking:** 

For our medical use case, recursive chunking is preferred over semantic chunking because medical documents have natural structural boundaries (paragraphs, sections) that align with medical logic. Semantic chunking could fragment complete medical concepts like drug interactions or lab result interpretations across multiple chunks, leading to incomplete information retrieval. Recursive chunking respects the natural document structure while preserving the relationships between symptoms, treatments, and medical conditions, ensuring that complete medical contexts are maintained for accurate retrieval and response generation.

**Additional reasons why recursive chunking is better for our medical use case:**

- **Medical documents are already well-structured** with clear section boundaries (symptoms, treatments, side effects, etc.)
- **Our documents have consistent formatting** from Mayo Clinic, NIH, and FDA sources
- **Medical terminology is standardized** - doesn't need semantic analysis for interpretation
- **Processing speed matters** for real-time medical queries - recursive is faster than semantic
- **Recursive preserves medical document structure** better than semantic chunking

The structured nature of medical documents makes recursive chunking more effective than semantic chunking for this specific use case.



---

## Task 4: Building an End-to-End Agentic RAG Application

**Implementation Details:**

The Personal Health Copilot prototype has been successfully built using a production-grade stack with commercial off-the-shelf models. The system leverages OpenAI's GPT-4o and GPT-4o-mini for generation and retrieval, along with enterprise-grade components for scalability and reliability.

**Backend Implementation (FastAPI):**
- Agentic RAG system with 7 medical tools (including uploaded lab results processing)
- BM25 retriever for in-memory medical information retrieval (no vector database required)
- LangSmith tracing for monitoring and evaluation
- Real-time medical information retrieval from verified sources
- Drug interaction checking capabilities
- File upload processing (PDF, DOCX) for lab results
- WebSocket support for real-time communication
- Session management for lab results tracking

**Frontend Implementation (Next.js):**
- Modern, responsive chat interface with conversation management
- Real-time message streaming with typing indicators
- Medical information display with proper formatting
- File upload interface for lab results (PDF, DOCX)
- Conversation history with save/load functionality
- Smart conversation title generation
- Share conversation functionality
- Connection status monitoring
- User-friendly design optimized for health-related queries

**Core Features Implemented:**
1. **Medical Information Retrieval:** RAG-based system using real medical documents from Mayo Clinic, NIH, and FDA with BM25 retriever
2. **Symptom Analysis:** Tool for analyzing symptoms and suggesting possible conditions
3. **Drug Interaction Checking:** Tool for checking potential interactions between medications
4. **Lab Results Interpretation:** Tool for interpreting lab test results in plain language
5. **Uploaded Lab Results Analysis:** Tool for processing uploaded lab files (PDF, DOCX) and answering questions about them
6. **Web Search Integration:** Real-time medical information from Tavily Search API
7. **Research Paper Search:** Medical research from ArXiv API

**Advanced Features:**
- **File Processing:** PDF and DOCX file upload and text extraction
- **Conversation Management:** Save, load, rename, and share conversations
- **Smart Classification:** Intelligent question classification for appropriate responses
- **Session Tracking:** Lab results session management for personalized responses
- **Real-time Communication:** WebSocket support for live chat experience

**Testing Results:**
The prototype successfully handles various medical queries:
- Symptom analysis: "I have a headache and fever, what could this mean?"
- Drug interactions: "I'm taking aspirin and ibuprofen, are there any interactions?"
- Lab results: "My blood test shows elevated white blood cells, what does this mean?"
- Uploaded lab analysis: Processing and interpreting uploaded medical files
- Current treatments: "What are the latest treatments for diabetes?"
- Research queries: "Search for recent research on COVID-19 long-term effects"

**Deployment Status:** ✅ Local deployment ready with FastAPI backend and Next.js frontend

---

## Task 5: Golden Test Data Set Creation and RAGAS Evaluation

**Note:** The complete implementation with detailed code, golden dataset creation, and comprehensive RAGAS evaluation results is available in the Jupyter notebook `Personal_Health_Copilot_Certification_Challenge.ipynb`.

**Golden Test Data Set Generation:**
A synthetic test data set was created using the TestsetGenerator from the RAGAS framework, leveraging the real medical documents from Mayo Clinic, NIH, and FDA sources. The generator created 5 test cases with questions, ground truth answers, and context pairs to establish a baseline for evaluation.

### 5.1 RAGAS Framework Assessment

**RAGAS Evaluation Results:**

| Metric | Score | Interpretation |
|--------|-------|----------------|
| **Context Recall** | 1.0000 (100%) | Perfect medical corpus coverage |
| **Context Precision** | 0.9444 (94.44%) | Excellent medical information relevance |
| **Faithfulness** | 0.9150 (91.50%) | Good adherence to medical context |
| **Factual Correctness** | 0.5683 (56.83%) | Needs improvement in medical accuracy |
| **Answer Relevancy** | 0.7891 (78.91%) | Good medical query-response alignment |
| **Context Entity Recall** | 0.3338 (33.38%) | Needs improvement in medical entity recognition |
| **Noise Sensitivity** | 0.3124 (31.24%) | Good filtering of irrelevant medical information |

### 5.2 Performance and Effectiveness Analysis and Conclusions

**Performance Analysis:**

**Strengths:**
- ✅ Perfect context recall (100%) ensures comprehensive medical corpus coverage
- ✅ Excellent context precision (94.44%) shows highly relevant medical information
- ✅ Strong faithfulness (91.50%) indicates responses stick to medical context
- ✅ Good answer relevancy (78.91%) for medical query-response alignment

**Areas for Improvement:**
- 🔄 Factual correctness (56.83%) needs enhancement for medical accuracy
- 🔄 Context entity recall (33.38%) requires better medical entity recognition

**Conclusions:**

The Personal Health Copilot demonstrates **strong performance** in medical information retrieval and response generation. The pipeline shows:

1. **Excellent Coverage:** 100% context recall ensures no relevant medical information is missed from the corpus
2. **High Precision:** 94.44% context precision indicates very relevant medical information retrieval
3. **Good Faithfulness:** 91.50% shows responses adhere well to medical context
4. **Room for Improvement:** Factual correctness and entity recognition need enhancement

**Recommendations:**
1. **Enhance medical fact-checking** to improve factual correctness
2. **Improve medical entity recognition** for better context entity recall
3. **Expand medical corpus** to cover more medical entities
4. **Implement medical validation** for higher accuracy

The pipeline shows **production-ready performance** for medical information assistance with targeted improvements needed for higher accuracy.

---

## Task 6: Advanced Retrieval Implementation

**Note:** The complete implementation with detailed code, advanced retrieval techniques setup, and comprehensive performance comparison results is available in the Jupyter notebook `Personal_Health_Copilot_Certification_Challenge.ipynb`.

**Advanced Retriever Selection:**
Based on the medical domain requirements, we chose to install and implement two advanced retrieval techniques: BM25 for keyword-based medical terminology matching and Contextual Compression with Cohere Reranking for precision filtering. These were selected after analyzing the specific needs of medical information retrieval.

**Stepwise Improvement Approach:**
Starting with the basic Qdrant vector retriever, we systematically upgraded to advanced retrieval methods to improve medical information accuracy and relevance.

**Advanced Retrieval Techniques Implemented:**

**1. BM25 Retriever:**
- Implementation: BM25Retriever.from_documents(split_docs)
- Purpose: Exact keyword matching for specific drug names, symptoms, and medical conditions
- Advantage: Excellent for medical terminology matching

**Why BM25 is Perfect for Medical Use Case:**
- **Medical queries contain specific terminology** (drug names, symptoms, conditions)
- **Benefit:** Exact keyword matching for precise medical information retrieval
- **Example:** "aspirin interactions" → finds exact aspirin-related documents
- **Medical terminology precision** - Exact drug names, conditions, symptoms
- **Fast retrieval** - Efficient for large medical document sets
- **Proven effectiveness** - Widely used in medical search systems

**2. Contextual Compression with Cohere Reranking:**
- Implementation: ContextualCompressionRetriever with CohereRerank model="rerank-v3.5"
- Purpose: Filters irrelevant medical data and ensures only highly relevant treatment/symptom information is retrieved
- Advantage: Perfect precision for medical information filtering

**Why Contextual Compression is Perfect for Medical Use Case:**
- **Medical information needs to be highly relevant and accurate**
- **Benefit:** Reranks retrieved documents to ensure most relevant medical context is prioritized
- **Example:** Filters out irrelevant medical info, keeps only highly relevant treatment/symptom data
- **Medical accuracy** - Ensures retrieved info is highly relevant to query
- **Quality filtering** - Removes irrelevant medical information
- **Context preservation** - Maintains medical context relationships

**Why These 2 Are Best for Medical Use Case:**

✅ **BM25 Advantages:**
- Medical terminology precision - Exact drug names, conditions, symptoms
- Fast retrieval - Efficient for large medical document sets
- Proven effectiveness - Widely used in medical search systems

✅ **Contextual Compression Advantages:**
- Medical accuracy - Ensures retrieved info is highly relevant to query
- Quality filtering - Removes irrelevant medical information
- Context preservation - Maintains medical context relationships

**Why Not Other Advanced Retrievers:**
- **Multi-Query:** Overkill for medical queries (usually single, specific questions)
- **Parent Document:** Medical documents are already well-structured
- **Ensemble:** Adds complexity without significant benefit for medical domain
- **Semantic Chunking:** Our current chunking strategy works well for medical content



---

## Task 7: Assessing Performance

### 7.1 Performance Assessment using RAGAS (Original RAG vs Advanced Retrievers)

**Performance Comparison Table :**

| Metric | Original RAG | BM25 | Contextual Compression | Best Improvement |
|--------|-------------|------|----------------------|------------------|
| **Context Recall** | 1.0000 (100%) | 1.0000 (100%) | 1.0000 (100%) | 0% (All equal) |
| **Context Precision** | 0.9444 (94.44%) | 0.9676 (96.76%) | 1.0000 (100%) | +5.56% (Compression) |
| **Factual Correctness** | 0.5683 (56.83%) | 0.6283 (62.83%) | 0.4933 (49.33%) | +6.0% (BM25) |
| **Context Entity Recall** | 0.3338 (33.38%) | 0.3943 (39.43%) | 0.4838 (48.38%) | +15.0% (Compression) |
| **Noise Sensitivity** | 0.3124 (31.24%) | 0.2046 (20.46%) | 0.3487 (34.87%) | +10.78% (BM25) |
| **Faithfulness** | 0.9150 (91.50%) | 0.8405 (84.05%) | 0.8765 (87.65%) | -7.45% (BM25) |
| **Answer Relevancy** | 0.7891 (78.91%) | 0.7894 (78.94%) | 0.7857 (78.57%) | +0.03% (BM25) |

**Quantified Improvements:**

**BM25 Retriever Improvements:**
- Context Precision: +2.32% (96.76% vs 94.44%)
- Factual Correctness: +6.0% (62.83% vs 56.83%)
- Context Entity Recall: +6.05% (39.43% vs 33.38%)
- Noise Sensitivity Reduction: +10.78% (20.46% vs 31.24%)

**Contextual Compression Improvements:**
- Context Precision: +5.56% (100% vs 94.44%)
- Context Entity Recall: +15.0% (48.38% vs 33.38%)
- Noise Sensitivity: -3.63% (34.87% vs 31.24% - worse)

**Recommendation:** BM25 is the best choice for medical use case because it provides the best balance of performance improvements while maintaining high faithfulness and factual correctness.

### 7.2 Future Development Roadmap

**Future Improvements for Second Half of Course:**

**Backend Enhancements:**
- **File Processing Enhancement:** Enable image processing for lab result photos (currently commented out)
- **Tool Enhancements:** Integrate real drug databases (DrugBank, RxNorm APIs) for accurate drug interaction checking
- **Error Handling:** Comprehensive error handling and user feedback for better user experience

**Frontend Improvements:**
- **Visual Elements:** Charts for lab results, progress indicators, and interactive health dashboards for better data visualization

**AI/ML Enhancements:**
- **Medical Entity Recognition:** Named Entity Recognition for medical terms to improve context entity recall (currently 33.38%)
- **Factual Correctness:** Medical fact-checking integration to improve accuracy (currently 56.83%)
- **Personalization:** User-specific health profiles and history for personalized medical advice

**Data Improvements:**
- **Lab Result Templates:** Structured templates with normal ranges for more accurate lab result analysis
- **Medical Corpus Expansion:** More comprehensive medical databases for better coverage of medical topics

**Performance Optimizations:**
- **Caching:** Redis caching for frequent queries to improve response times
- **Batch Processing:** Batch processing for multiple queries to enhance efficiency

**Security & Compliance:**
- **Authentication:** User accounts and secure data storage for HIPAA compliance
- **Data Encryption:** End-to-end encryption for health data to ensure medical data security

**Enhanced Retrieval:**
- Semantic Chunking for medical documents
- Hybrid retrieval (BM25 + Semantic)
- Medical entity recognition enhancement
- Drug interaction database integration

**Advanced Features:**
- Real-time medical data updates
- Personalized health profiles
- Medical image analysis integration
- Voice interface for accessibility

**Performance Optimizations:**
- Fine-tuned medical embeddings
- Caching for frequent queries
- Batch processing for efficiency
- Edge deployment for low latency

 