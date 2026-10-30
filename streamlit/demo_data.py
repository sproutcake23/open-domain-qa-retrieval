"""
Demo data for RAG Retrieval Comparison
"""

# Fictional company corpus for demonstration
DEMO_CORPUS = [
    {
        "id": 0,
        "title": "TechFlow AI - Company Overview and Business Operations",
        "content": "TechFlow AI is a leading artificial intelligence company founded in January 2020 by five Stanford PhD graduates. Our company specializes in developing cutting-edge machine learning models, deep learning architectures, and AI-powered enterprise automation solutions. We focus on natural language processing (NLP), computer vision, and predictive analytics for Fortune 500 clients. TechFlow AI headquarters is located at 123 Innovation Drive, San Francisco, California, with regional offices in Manhattan, New York and downtown Austin, Texas. Our company has experienced rapid growth from 5 founding members to over 200 full-time employees including 85 engineers, 45 data scientists, and 70 business professionals worldwide."
    },
    {
        "id": 1,
        "title": "TechFlow AI - Product Suite and Technology Stack", 
        "content": "TechFlow AI offers four main AI-powered products for enterprise customers. FlowBot is our flagship intelligent workflow automation platform that uses advanced machine learning to optimize business processes, reduce manual tasks, and increase operational efficiency by 40-60%. ChatFlow provides customer service automation using large language models and conversational AI to handle 24/7 customer support with 95% accuracy. VisionFlow delivers computer vision solutions for image recognition, object detection, quality control, and visual inspection tasks in manufacturing. DataFlow enables predictive analytics, trend forecasting, and data-driven decision making using time-series analysis and statistical modeling. All products are built on FlowNet, our proprietary neural network architecture featuring transformer-based models, optimized inference engines, and scalable cloud deployment."
    },
    {
        "id": 2,
        "title": "TechFlow AI - Financial Performance and Recent Achievements",
        "content": "In 2024, TechFlow AI achieved remarkable business milestones and financial success. We successfully raised $50 million in Series B venture capital funding led by Andreessen Horowitz and Venture Capital Partners, bringing our total funding to $75 million. Our FlowBot automation platform now serves over 1,000 enterprise clients including Apple, Microsoft, Goldman Sachs, and other Fortune 500 companies, generating $25 million in annual recurring revenue. TechFlow AI was recognized as 'AI Startup of the Year' by TechCrunch Disrupt 2024 and received the 'Innovation Excellence Award' from MIT Technology Review. Our CEO and co-founder Sarah Chen was featured in Forbes' 30 Under 30 list for Enterprise Technology, while our CTO Dr. Michael Rodriguez won the 'AI Researcher of the Year' award from the Association for Computing Machinery."
    },
    {
        "id": 3,
        "title": "TechFlow AI - Research Initiatives and Scientific Publications",
        "content": "TechFlow AI conducts extensive research and development in artificial intelligence, machine learning, and deep learning technologies. Our research team, directed by Chief Technology Officer Dr. Michael Rodriguez (former MIT professor), focuses on advancing transformer architectures, attention mechanisms, and efficient neural network training algorithms. We research multimodal AI systems that combine text, image, and audio processing, federated learning for privacy-preserving machine learning, and sustainable AI computing to minimize environmental impact and carbon footprint. TechFlow AI has published 15 peer-reviewed research papers in top-tier AI conferences including NeurIPS (Neural Information Processing Systems), ICML (International Conference on Machine Learning), and ICLR (International Conference on Learning Representations). Our research contributions include novel attention mechanisms, efficient transformer training methods, and breakthrough algorithms for few-shot learning and domain adaptation."
    },
    {
        "id": 4,
        "title": "TechFlow AI - Employment Opportunities and Company Culture",
        "content": "TechFlow AI is actively recruiting talented professionals across multiple departments and technical specializations. Current open positions include Senior Machine Learning Engineers (PyTorch/TensorFlow expertise required), Lead Data Scientists with NLP experience, Full-Stack Software Engineers (Python/React), Product Managers for AI solutions, DevOps Engineers for MLOps, and Enterprise Sales Representatives for Fortune 500 accounts. We offer highly competitive compensation packages including base salaries ranging from $120k-$300k, comprehensive health benefits (medical, dental, vision), flexible remote work arrangements, professional development budgets, and equity participation through stock options. TechFlow AI company culture emphasizes innovation, continuous learning, collaboration, and work-life balance. We organize monthly hackathons, sponsor attendance at AI conferences like NeurIPS and ICML, provide mentorship programs, and maintain inclusive diversity initiatives to build a world-class team."
    }
]

# Extract just the text content for retrieval
CORPUS_TEXTS = [doc["content"] for doc in DEMO_CORPUS]

# Ground truth for evaluation (which documents are relevant for common queries)
GROUND_TRUTH = {
    "What does TechFlow AI do?": [0, 1],  # Company overview and products
    "Who is the CEO of TechFlow AI?": [2],  # Recent achievements mentions CEO Sarah Chen
    "What products does TechFlow AI offer?": [1],  # Product portfolio with FlowBot, ChatFlow, etc.
    "How much funding did TechFlow AI raise?": [2],  # Recent achievements mentions $50M Series B
    "What research does TechFlow AI do?": [3],  # R&D section with specific research areas
    "Is TechFlow AI hiring?": [4],  # Career opportunities with open positions
    "Where is TechFlow AI located?": [0],  # Company overview mentions SF, NY, Austin
    "Who leads the technology team?": [3],  # R&D mentions CTO Dr. Michael Rodriguez
    "What awards has TechFlow AI won?": [2],  # Recent achievements lists awards
    "What is FlowBot?": [1],  # Product portfolio describes FlowBot in detail
    "Tell me about TechFlow AI research": [3],  # Alternative phrasing for research
    "What AI research is TechFlow working on?": [3],  # More specific research query
    "What machine learning research does TechFlow do?": [3],  # ML-focused research query
    "Who founded TechFlow AI?": [0],  # Company overview mentions 5 Stanford PhD graduates
    "How many employees does TechFlow AI have?": [0],  # Company overview mentions 200 employees
    "What conferences has TechFlow published at?": [3],  # R&D mentions NeurIPS, ICML, ICLR
    "What is FlowNet?": [1],  # Product portfolio describes FlowNet architecture
    "What technology does TechFlow use?": [1, 3],  # Products and research sections
    "How much revenue does TechFlow AI make?": [2],  # Financial performance section
    "What kind of jobs are available at TechFlow?": [4],  # Career opportunities
    "How much did it raise in series B?": [2],  # Financial performance - Series B funding
    "How much funding in Series B?": [2],  # Alternative phrasing
    "Series B funding amount?": [2],  # Short form
}

# Sample queries for quick testing
SAMPLE_QUERIES = [
    "What does TechFlow AI do?",
    "What products does TechFlow AI offer?", 
    "Who is the CEO of TechFlow AI?",
    "How much funding did TechFlow AI raise?",
    "What research does TechFlow AI do?",
    "Is TechFlow AI hiring?",
    "Where is TechFlow AI located?",
    "What is FlowBot?",
    "Tell me about TechFlow AI research",
    "What conferences has TechFlow published at?",
    "How many employees does TechFlow AI have?",
    "What technology does TechFlow use?"
]
