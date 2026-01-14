"""
S.P.E.C. Valuation Engine - AI Oracle with RAG
===============================================
LLM-powered investment memo generation with RAG-enhanced market context.
V2.0 with ChromaDB vector store and semantic retrieval.
"""

import os
import logging
from typing import Dict, Optional, Any, List
from pathlib import Path

from dotenv import load_dotenv

# Import from sibling config module
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import (
    VECTOR_DB_DIR,
    VECTOR_COLLECTION_NAME,
    EMBEDDING_MODEL,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    TOP_K_RETRIEVAL,
    MARKET_REPORTS_DIR,
    COLORS,
)

# Import market context for interest rates
from src.market_context import get_rate_context, calculate_monthly_payment

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# ====================================
# RAG COMPONENTS - LAZY IMPORTS
# ====================================
def _get_chromadb():
    """Lazy import of ChromaDB."""
    try:
        import chromadb
        return chromadb
    except ImportError:
        logger.warning("ChromaDB not installed. RAG features disabled.")
        return None


def _get_sentence_transformer():
    """Lazy import of sentence-transformers."""
    try:
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer
    except ImportError:
        logger.warning("sentence-transformers not installed. Embedding disabled.")
        return None


def _get_pypdf():
    """Lazy import of pypdf."""
    try:
        from pypdf import PdfReader
        return PdfReader
    except ImportError:
        logger.warning("pypdf not installed. PDF parsing disabled.")
        return None


# ====================================
# VECTOR STORE CLASS
# ====================================
class MarketReportVectorStore:
    """
    ChromaDB-based vector store for market reports.
    
    Handles:
    - PDF ingestion and chunking
    - Embedding with sentence-transformers
    - Semantic retrieval
    """
    
    def __init__(self, persist_directory: Optional[Path] = None):
        """
        Initialize the vector store.
        
        Args:
            persist_directory: Directory to persist ChromaDB data.
        """
        self.persist_dir = persist_directory or VECTOR_DB_DIR
        self.collection_name = VECTOR_COLLECTION_NAME
        self.embedding_model_name = EMBEDDING_MODEL
        
        self.chromadb = _get_chromadb()
        self.SentenceTransformer = _get_sentence_transformer()
        
        self.client = None
        self.collection = None
        self.embedding_model = None
        
        self._is_available = False
        
        if self.chromadb and self.SentenceTransformer:
            self._initialize()
    
    def _initialize(self) -> None:
        """Initialize ChromaDB client and embedding model."""
        try:
            # Create persist directory
            self.persist_dir.mkdir(parents=True, exist_ok=True)
            
            # Initialize ChromaDB with persistence
            self.client = self.chromadb.PersistentClient(
                path=str(self.persist_dir)
            )
            
            # Get or create collection
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"description": "Market reports for real estate analysis"}
            )
            
            # Initialize embedding model
            self.embedding_model = self.SentenceTransformer(self.embedding_model_name)
            
            self._is_available = True
            logger.info(
                f"Vector store initialized. "
                f"Collection '{self.collection_name}' has {self.collection.count()} documents."
            )
            
        except Exception as e:
            logger.error(f"Failed to initialize vector store: {e}")
            self._is_available = False
    
    @property
    def is_available(self) -> bool:
        """Check if vector store is ready."""
        return self._is_available
    
    def _chunk_text(
        self,
        text: str,
        chunk_size: int = CHUNK_SIZE,
        overlap: int = CHUNK_OVERLAP
    ) -> List[str]:
        """
        Split text into overlapping chunks.
        
        Args:
            text: Input text to chunk.
            chunk_size: Maximum characters per chunk.
            overlap: Number of overlapping characters.
        
        Returns:
            List of text chunks.
        """
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            
            # Try to break at sentence boundary
            if end < len(text):
                last_period = chunk.rfind('.')
                if last_period > chunk_size * 0.5:
                    chunk = chunk[:last_period + 1]
                    end = start + last_period + 1
            
            chunks.append(chunk.strip())
            start = end - overlap
        
        return [c for c in chunks if len(c) > 50]  # Filter tiny chunks
    
    def _embed_text(self, text: str) -> List[float]:
        """
        Generate embedding for text.
        
        Args:
            text: Input text.
        
        Returns:
            Embedding vector as list of floats.
        """
        if self.embedding_model is None:
            return []
        
        embedding = self.embedding_model.encode(text)
        return embedding.tolist()
    
    def ingest_pdf(
        self,
        pdf_path: Path,
        metadata: Optional[Dict[str, Any]] = None
    ) -> int:
        """
        Ingest a PDF document into the vector store.
        
        Args:
            pdf_path: Path to PDF file.
            metadata: Optional metadata to attach to chunks.
        
        Returns:
            Number of chunks ingested.
        """
        if not self.is_available:
            logger.warning("Vector store not available. Cannot ingest PDF.")
            return 0
        
        PdfReader = _get_pypdf()
        if PdfReader is None:
            logger.error("pypdf not installed. Cannot parse PDF.")
            return 0
        
        try:
            # Read PDF
            reader = PdfReader(pdf_path)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
            
            logger.info(f"Extracted {len(text)} characters from {pdf_path.name}")
            
            # Chunk text
            chunks = self._chunk_text(text)
            logger.info(f"Created {len(chunks)} chunks")
            
            # Prepare for ChromaDB
            ids = [f"{pdf_path.stem}_{i}" for i in range(len(chunks))]
            embeddings = [self._embed_text(chunk) for chunk in chunks]
            
            # Prepare metadata
            metadatas = []
            base_metadata = metadata or {}
            for i, chunk in enumerate(chunks):
                chunk_meta = {
                    **base_metadata,
                    "source": pdf_path.name,
                    "chunk_index": i,
                }
                metadatas.append(chunk_meta)
            
            # Add to collection
            self.collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=chunks,
                metadatas=metadatas
            )
            
            logger.info(f"Ingested {len(chunks)} chunks from {pdf_path.name}")
            return len(chunks)
            
        except Exception as e:
            logger.error(f"Failed to ingest PDF {pdf_path}: {e}")
            return 0
    
    def ingest_text(
        self,
        text: str,
        source_name: str = "manual",
        metadata: Optional[Dict[str, Any]] = None
    ) -> int:
        """
        Ingest raw text into the vector store.
        
        Args:
            text: Text content to ingest.
            source_name: Name to identify the source.
            metadata: Optional metadata.
        
        Returns:
            Number of chunks ingested.
        """
        if not self.is_available:
            logger.warning("Vector store not available. Cannot ingest text.")
            return 0
        
        try:
            # Chunk text
            chunks = self._chunk_text(text)
            
            # Prepare for ChromaDB
            ids = [f"{source_name}_{i}" for i in range(len(chunks))]
            embeddings = [self._embed_text(chunk) for chunk in chunks]
            
            # Prepare metadata
            metadatas = []
            base_metadata = metadata or {}
            for i in range(len(chunks)):
                chunk_meta = {
                    **base_metadata,
                    "source": source_name,
                    "chunk_index": i,
                }
                metadatas.append(chunk_meta)
            
            # Add to collection
            self.collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=chunks,
                metadatas=metadatas
            )
            
            logger.info(f"Ingested {len(chunks)} chunks from {source_name}")
            return len(chunks)
            
        except Exception as e:
            logger.error(f"Failed to ingest text from {source_name}: {e}")
            return 0
    
    def query(
        self,
        query_text: str,
        n_results: int = TOP_K_RETRIEVAL,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Query the vector store for relevant chunks.
        
        Args:
            query_text: Query string.
            n_results: Number of results to return.
            filter_metadata: Optional metadata filter.
        
        Returns:
            List of result dictionaries with 'text', 'metadata', 'distance'.
        """
        if not self.is_available:
            logger.warning("Vector store not available for query.")
            return []
        
        try:
            # Generate query embedding
            query_embedding = self._embed_text(query_text)
            
            # Query ChromaDB
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                where=filter_metadata
            )
            
            # Format results
            formatted = []
            for i in range(len(results["documents"][0])):
                formatted.append({
                    "text": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                    "distance": results["distances"][0][i] if results["distances"] else None,
                })
            
            return formatted
            
        except Exception as e:
            logger.error(f"Vector store query failed: {e}")
            return []
    
    def clear(self) -> None:
        """Clear all documents from the collection."""
        if self.is_available and self.collection:
            try:
                # Delete the collection and recreate
                self.client.delete_collection(self.collection_name)
                self.collection = self.client.create_collection(
                    name=self.collection_name
                )
                logger.info("Vector store cleared.")
            except Exception as e:
                logger.error(f"Failed to clear vector store: {e}")


# ====================================
# SAMPLE MARKET REPORT (Mock Data)
# ====================================
SAMPLE_MARKET_REPORT = """
Q4 2025 San Francisco Real Estate Market Report

Executive Summary:
The San Francisco real estate market continues to show signs of stabilization after 
the disruption of remote work trends. Median home prices remain elevated at $1.45M, 
though down 8% from the 2022 peak. Inventory levels are at 2.1 months supply, 
indicating a balanced market.

Neighborhood Analysis:

SOMA (ZIP: 94103, 94107):
The SOMA district is experiencing a renaissance driven by new biotech campuses 
and mixed-use development. Condo prices have stabilized after significant declines 
in 2023. The area benefits from improved transit connections and new retail amenities.
Risk factors include ongoing office-to-residential conversions that may increase supply.

Financial District (ZIP: 94104):
The Financial District faces structural challenges with commercial vacancy rates 
at 29%. However, conversion projects are creating opportunities for residential 
repositioning. Investors should be cautious about timing, as absorption rates remain slow.

Mission District (ZIP: 94110):
The Mission continues to be a desirable neighborhood with strong price appreciation. 
Median prices are up 5% YoY driven by tech worker demand. Cultural amenities and 
dining scene remain key attractors. Risk: potential for rent control expansion.

Pacific Heights (ZIP: 94115):
Ultra-luxury segment showing resilience with trophy properties trading at premium.
Limited inventory driving competition. Buyers from tech sector remain active.
Average price per sqft: $1,850. Days on market: 22 (down from 35 last year).

Market Trends:
1. Interest rates stabilizing around 6.5% are bringing buyers back to market
2. First-time buyer programs expanding in California
3. Climate resilience becoming a factor in property valuations
4. AI company growth driving demand in certain neighborhoods

Investment Outlook:
We maintain a NEUTRAL stance on San Francisco residential for 2025.
Opportunities exist in undervalued neighborhoods like Outer Sunset and Excelsior.
Avoid over-concentration in downtown adjacent areas until office market stabilizes.
"""


def seed_sample_data(vector_store: MarketReportVectorStore) -> None:
    """
    Seed the vector store with sample market report data.
    
    Args:
        vector_store: Vector store instance.
    """
    if vector_store.collection.count() == 0:
        logger.info("Seeding vector store with sample market report...")
        vector_store.ingest_text(
            SAMPLE_MARKET_REPORT,
            source_name="sf_market_report_q4_2025",
            metadata={"type": "market_report", "region": "San Francisco", "quarter": "Q4 2025"}
        )


# ====================================
# RAG-ENHANCED CONTEXT RETRIEVAL
# ====================================
# Global vector store instance
_vector_store: Optional[MarketReportVectorStore] = None


def get_vector_store() -> MarketReportVectorStore:
    """Get or create the global vector store instance."""
    global _vector_store
    
    if _vector_store is None:
        _vector_store = MarketReportVectorStore()
        if _vector_store.is_available:
            seed_sample_data(_vector_store)
    
    return _vector_store


def get_market_context(zip_code: str, skip_rag: bool = False) -> Dict[str, Any]:
    """
    Retrieve market context for a given zip code using RAG.
    
    PERFORMANCE OPTIMIZATION: Set skip_rag=True to bypass the heavy
    vector store initialization for faster initial page loads.
    The RAG is only needed when generating investment memos.
    
    Args:
        zip_code: Property zip code.
        skip_rag: If True, skip RAG and use static database (faster).
    
    Returns:
        Dictionary with neighborhood info, news, and risk factors.
    """
    # Fast path: skip RAG for quick responses
    if skip_rag:
        return MARKET_NEWS_DATABASE.get(zip_code, DEFAULT_MARKET_NEWS)
    
    vector_store = get_vector_store()
    
    # Query vector store for relevant context
    query = f"Real estate market analysis for zip code {zip_code} San Francisco neighborhood investment risks trends"
    
    retrieved_chunks = []
    if vector_store.is_available:
        results = vector_store.query(query, n_results=TOP_K_RETRIEVAL)
        retrieved_chunks = [r["text"] for r in results]
    
    # Build context from retrieved chunks
    if retrieved_chunks:
        # Extract relevant info from retrieved chunks
        context = {
            "neighborhood": _extract_neighborhood_name(zip_code, retrieved_chunks),
            "sentiment": _extract_sentiment(retrieved_chunks),
            "news": _extract_key_points(retrieved_chunks),
            "risk_factors": _extract_risks(retrieved_chunks),
            "rag_context": "\n\n".join(retrieved_chunks),  # Full context for LLM
            "source": "rag_retrieval"
        }
        return context
    
    # Fallback to static database
    return MARKET_NEWS_DATABASE.get(zip_code, DEFAULT_MARKET_NEWS)


def _extract_neighborhood_name(zip_code: str, chunks: List[str]) -> str:
    """Extract neighborhood name from chunks."""
    zip_neighborhood_map = {
        "94102": "Tenderloin/Civic Center",
        "94103": "SoMa",
        "94104": "Financial District",
        "94105": "Rincon Hill/South Beach",
        "94107": "Potrero Hill/Dogpatch",
        "94109": "Nob Hill/Russian Hill",
        "94110": "Mission District",
        "94114": "Castro",
        "94115": "Pacific Heights",
        "94117": "Haight-Ashbury",
    }
    return zip_neighborhood_map.get(zip_code, "Greater San Francisco")


def _extract_sentiment(chunks: List[str]) -> str:
    """Extract market sentiment from chunks."""
    text = " ".join(chunks).lower()
    
    bullish_keywords = ["growth", "appreciation", "demand", "resilience", "premium", "up"]
    bearish_keywords = ["decline", "risk", "challenge", "vacancy", "concern", "down"]
    
    bullish_score = sum(1 for kw in bullish_keywords if kw in text)
    bearish_score = sum(1 for kw in bearish_keywords if kw in text)
    
    if bullish_score > bearish_score + 2:
        return "Bullish"
    elif bearish_score > bullish_score + 2:
        return "Bearish"
    else:
        return "Neutral"


def _extract_key_points(chunks: List[str]) -> List[str]:
    """Extract key points from chunks."""
    # Simple extraction - in production, use NLP
    points = []
    for chunk in chunks[:2]:
        sentences = chunk.split('.')
        for sentence in sentences[:2]:
            if len(sentence) > 30:
                points.append(sentence.strip() + '.')
    return points[:3]


def _extract_risks(chunks: List[str]) -> List[str]:
    """Extract risk factors from chunks."""
    text = " ".join(chunks).lower()
    
    potential_risks = [
        "Office vacancy",
        "Interest rate sensitivity",
        "Supply increase",
        "Rent control expansion",
        "Market volatility",
        "Commercial exodus",
    ]
    
    found_risks = []
    for risk in potential_risks:
        if risk.lower().split()[0] in text:
            found_risks.append(risk)
    
    return found_risks[:3] if found_risks else ["Market uncertainty"]


# ====================================
# LEGACY STATIC DATABASE (Fallback)
# ====================================
MARKET_NEWS_DATABASE: Dict[str, Dict[str, Any]] = {
    "94102": {
        "neighborhood": "Tenderloin/Civic Center",
        "sentiment": "Bearish",
        "news": [
            "City proposes new affordable housing mandates affecting investor returns",
            "Homeless services expansion planned for 2024",
            "Tech company announces office closure nearby"
        ],
        "risk_factors": ["High crime rate", "Rent control pressure", "Commercial vacancy"],
    },
    "94103": {
        "neighborhood": "SoMa",
        "sentiment": "Neutral",
        "news": [
            "Mixed-use development approved on Folsom Street",
            "Transit improvements planned for 2025",
            "Some tech companies returning to office"
        ],
        "risk_factors": ["Market volatility", "Zoning changes pending"],
    },
    "94104": {
        "neighborhood": "Financial District",
        "sentiment": "Bearish",
        "news": [
            "Office-to-residential conversions gaining momentum",
            "Major bank reducing downtown footprint",
            "Daytime foot traffic down 40% from 2019 levels"
        ],
        "risk_factors": ["Commercial exodus", "Structural market shift"],
    },
    "94105": {
        "neighborhood": "Rincon Hill/South Beach",
        "sentiment": "Bullish",
        "news": [
            "New waterfront park opening increases desirability",
            "Luxury condo sales up 15% YoY",
            "Oracle Park events driving local economy"
        ],
        "risk_factors": ["High HOA fees", "Earthquake liquefaction zone"],
    },
    "94107": {
        "neighborhood": "Potrero Hill/Dogpatch",
        "sentiment": "Bullish",
        "news": [
            "Biotech hub expansion creating jobs",
            "Historic warehouse conversions in demand",
            "New MUNI line improves connectivity"
        ],
        "risk_factors": ["Limited parking", "Industrial adjacency"],
    },
    "94109": {
        "neighborhood": "Nob Hill/Russian Hill",
        "sentiment": "Stable",
        "news": [
            "Historic preservation status limits new supply",
            "Cable car renovation completed",
            "Luxury rental market stabilizing"
        ],
        "risk_factors": ["Limited appreciation potential", "Aging building stock"],
    },
    "94110": {
        "neighborhood": "Mission District",
        "sentiment": "Neutral",
        "news": [
            "Small business revival gaining momentum",
            "Cultural district designation approved",
            "Gentrification concerns continue"
        ],
        "risk_factors": ["Political intervention risk", "Rent control"],
    },
    "94114": {
        "neighborhood": "Castro",
        "sentiment": "Stable",
        "news": [
            "Neighborhood retail occupancy improving",
            "Historic building renovation incentives available",
            "Strong community investment continues"
        ],
        "risk_factors": ["Limited inventory", "High entry price"],
    },
    "94115": {
        "neighborhood": "Pacific Heights",
        "sentiment": "Bullish",
        "news": [
            "Billionaires Row sees continued demand",
            "Trophy property sold for $30M+",
            "Private school enrollment driving family demand"
        ],
        "risk_factors": ["Ultra-high price point", "Limited buyer pool"],
    },
    "94117": {
        "neighborhood": "Haight-Ashbury",
        "sentiment": "Neutral",
        "news": [
            "Tourism recovery boosting local businesses",
            "Victorian restoration projects active",
            "Parking challenges persist"
        ],
        "risk_factors": ["Tourist impact", "Older housing stock"],
    },
}

DEFAULT_MARKET_NEWS = {
    "neighborhood": "Greater San Francisco",
    "sentiment": "Neutral",
    "news": [
        "Regional housing market showing mixed signals",
        "Interest rates impacting buyer behavior",
        "Inventory levels below historical average"
    ],
    "risk_factors": ["Market uncertainty", "Rate sensitivity"],
}


# ====================================
# OPENAI CLIENT
# ====================================
def _get_openai_client():
    """Get OpenAI client if API key is available."""
    api_key = os.getenv("OPENAI_API_KEY")
    
    if not api_key or api_key == "your_openai_api_key_here":
        logger.warning("OpenAI API key not configured. Running in simulation mode.")
        return None
    
    try:
        from openai import OpenAI
        return OpenAI(api_key=api_key)
    except ImportError:
        logger.error("OpenAI package not installed.")
        return None
    except Exception as e:
        logger.error(f"Failed to initialize OpenAI client: {e}")
        return None


# ====================================
# INVESTMENT MEMO GENERATION
# ====================================
def generate_investment_memo(
    price: float,
    shap_data: Dict[str, Any],
    zip_code: str,
    use_ai: bool = True,
) -> str:
    """
    Generate an investment memo using AI with RAG-enhanced context.
    
    The AI acts as a "Bearish Investment Analyst" who critically evaluates
    the property valuation and market conditions.
    
    Args:
        price: Predicted property price.
        shap_data: SHAP explanation data from the model.
        zip_code: Property zip code for market context.
        use_ai: Whether to attempt AI generation.
    
    Returns:
        Investment memo as a formatted string.
    """
    # Get RAG-enhanced market context
    market_context = get_market_context(zip_code)
    
    # Build the context for the memo
    shap_values = shap_data.get("shap_values", {})
    feature_values = shap_data.get("feature_values", {})
    base_value = shap_data.get("base_value", 0)
    
    # Sort SHAP values by absolute impact
    sorted_shap = sorted(
        shap_values.items(),
        key=lambda x: abs(x[1]),
        reverse=True
    )
    
    # Attempt AI generation
    if use_ai:
        client = _get_openai_client()
        
        if client:
            try:
                memo = _generate_ai_memo_with_rag(
                    client=client,
                    price=price,
                    base_value=base_value,
                    sorted_shap=sorted_shap,
                    feature_values=feature_values,
                    market_context=market_context,
                )
                return memo
            except Exception as e:
                logger.error(f"AI memo generation failed: {e}")
    
    # Fallback to template-based memo
    return _generate_template_memo(
        price=price,
        base_value=base_value,
        sorted_shap=sorted_shap,
        feature_values=feature_values,
        market_context=market_context,
    )


def _generate_ai_memo_with_rag(
    client,
    price: float,
    base_value: float,
    sorted_shap: list,
    feature_values: Dict,
    market_context: Dict,
) -> str:
    """Generate memo using OpenAI API with RAG context."""
    
    # Build the prompt with RAG context
    shap_explanation = "\n".join([
        f"  - {feat}: ${val:+,.0f} impact"
        for feat, val in sorted_shap
    ])
    
    news_items = "\n".join([f"  - {n}" for n in market_context.get("news", [])])
    risk_items = "\n".join([f"  - {r}" for r in market_context.get("risk_factors", [])])
    
    # Include RAG-retrieved context if available
    rag_context = market_context.get("rag_context", "")
    rag_section = ""
    if rag_context:
        rag_section = f"""
RETRIEVED MARKET INTELLIGENCE:
{rag_context[:1500]}  # Truncate to avoid token limits
"""
    
    prompt = f"""You are a Bearish Investment Analyst reviewing a residential property valuation. 
Your job is to critically evaluate the property and highlight potential risks while remaining data-driven.

PROPERTY DATA:
- Model Valuation: ${price:,.0f}
- Market Baseline: ${base_value:,.0f}
- Square Footage: {feature_values.get('sqft', 'N/A')} sqft
- Bedrooms: {feature_values.get('bedrooms', 'N/A')}
- Year Built: {feature_values.get('year_built', 'N/A')}
- Condition Rating: {feature_values.get('condition', 'N/A')}/5

VALUATION BREAKDOWN (SHAP Analysis):
{shap_explanation}

MARKET CONTEXT - {market_context.get('neighborhood', 'San Francisco')}:
Sentiment: {market_context.get('sentiment', 'Neutral')}

Recent News:
{news_items}

Risk Factors:
{risk_items}
{rag_section}

Write a concise investment memo (3-4 paragraphs) that:
1. Summarizes the valuation and key price drivers
2. Critically analyzes risks and bearish considerations  
3. Provides a recommendation with caveats

Use a professional, analytical tone. Include specific numbers. Be skeptical but fair."""

    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a senior real estate investment analyst with a bearish disposition. You use data-driven analysis and cite specific market intelligence when available."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        max_tokens=600,
    )
    
    return response.choices[0].message.content


def _generate_template_memo(
    price: float,
    base_value: float,
    sorted_shap: list,
    feature_values: Dict,
    market_context: Dict,
) -> str:
    """Generate fallback template-based memo."""
    
    # Determine overall assessment
    sentiment = market_context.get("sentiment", "Neutral")
    neighborhood = market_context.get("neighborhood", "Greater San Francisco")
    
    # Get top positive and negative drivers
    positive_drivers = [(f, v) for f, v in sorted_shap if v > 0]
    negative_drivers = [(f, v) for f, v in sorted_shap if v < 0]
    
    # Calculate price premium/discount
    price_diff = price - base_value
    price_diff_pct = (price_diff / base_value) * 100 if base_value > 0 else 0
    
    # Determine recommendation based on sentiment
    if sentiment == "Bullish":
        recommendation = "CAUTIOUS BUY"
        rec_color = "#00D47E"
    elif sentiment == "Bearish":
        recommendation = "HOLD / AVOID"
        rec_color = "#FF4757"
    else:
        recommendation = "NEUTRAL"
        rec_color = "#FFB946"
    
    # Check for RAG source
    source_note = ""
    if market_context.get("source") == "rag_retrieval":
        source_note = "Analysis enhanced with retrieved market intelligence."
    else:
        source_note = "Configure OPENAI_API_KEY for AI-powered insights."
    
    # Build the memo
    memo = f"""
<div style="background: #1a1d24; border: 1px solid #2d3139; border-radius: 8px; padding: 1.5rem; margin-top: 0.5rem;">

<div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1.5rem; padding-bottom: 1rem; border-bottom: 1px solid #2d3139;">
    <div>
        <p style="color: #6B7280; font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.1em; margin: 0;">Investment Recommendation</p>
        <p style="color: {rec_color}; font-size: 1.25rem; font-weight: 600; margin: 0.25rem 0 0 0;">{recommendation}</p>
    </div>
    <div style="text-align: right;">
        <p style="color: #6B7280; font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.1em; margin: 0;">Model Confidence</p>
        <p style="color: #E5E7EB; font-size: 1.25rem; font-weight: 600; margin: 0.25rem 0 0 0;">HIGH</p>
    </div>
</div>

<div style="margin-bottom: 1.25rem;">
    <p style="color: #9CA3AF; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.08em; margin: 0 0 0.5rem 0;">Valuation Summary</p>
    <p style="color: #D1D5DB; font-size: 0.9rem; line-height: 1.6; margin: 0;">
        The model values this property at <strong style="color: #fff;">${price:,.0f}</strong>, representing a 
        <span style="color: {'#00D47E' if price_diff >= 0 else '#FF4757'};">{price_diff_pct:+.1f}%</span> 
        ({'+' if price_diff >= 0 else ''}{price_diff:,.0f}) deviation from the market baseline of ${base_value:,.0f}. 
        The valuation is derived from {len(sorted_shap)} key property attributes analyzed via SHAP explainability.
    </p>
</div>

<div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin-bottom: 1.25rem;">
    <div>
        <p style="color: #9CA3AF; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.08em; margin: 0 0 0.5rem 0;">Value Drivers (+)</p>
"""
    
    if positive_drivers:
        for feat, val in positive_drivers[:3]:
            memo += f"""        <p style="color: #00D47E; font-size: 0.85rem; margin: 0.25rem 0;">{feat.replace('_', ' ').title()}: +${val:,.0f}</p>\n"""
    else:
        memo += """        <p style="color: #6B7280; font-size: 0.85rem; margin: 0.25rem 0;">No positive drivers identified</p>\n"""
    
    memo += """    </div>
    <div>
        <p style="color: #9CA3AF; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.08em; margin: 0 0 0.5rem 0;">Value Detractors (-)</p>
"""
    
    if negative_drivers:
        for feat, val in negative_drivers[:3]:
            memo += f"""        <p style="color: #FF4757; font-size: 0.85rem; margin: 0.25rem 0;">{feat.replace('_', ' ').title()}: ${val:,.0f}</p>\n"""
    else:
        memo += """        <p style="color: #6B7280; font-size: 0.85rem; margin: 0.25rem 0;">No negative drivers identified</p>\n"""
    
    memo += f"""    </div>
</div>

<div style="margin-bottom: 1.25rem; padding: 1rem; background: #12141a; border-radius: 6px;">
    <p style="color: #9CA3AF; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.08em; margin: 0 0 0.5rem 0;">Market Context: {neighborhood}</p>
    <div style="display: flex; gap: 2rem; margin-bottom: 0.75rem;">
        <div>
            <span style="color: #6B7280; font-size: 0.75rem;">Sentiment</span>
            <p style="color: {'#00D47E' if sentiment == 'Bullish' else '#FF4757' if sentiment == 'Bearish' else '#FFB946'}; font-size: 0.9rem; font-weight: 500; margin: 0.125rem 0 0 0;">{sentiment}</p>
        </div>
    </div>
    <p style="color: #6B7280; font-size: 0.75rem; margin: 0.5rem 0 0.25rem 0;">Recent Developments:</p>
"""
    for news in market_context.get("news", [])[:2]:
        memo += f"""    <p style="color: #9CA3AF; font-size: 0.8rem; margin: 0.2rem 0; padding-left: 0.75rem; border-left: 2px solid #2d3139;">{news}</p>\n"""
    
    memo += f"""</div>

<div style="margin-bottom: 1rem;">
    <p style="color: #9CA3AF; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.08em; margin: 0 0 0.5rem 0;">Risk Factors</p>
    <div style="display: flex; flex-wrap: wrap; gap: 0.5rem;">
"""
    for risk in market_context.get("risk_factors", []):
        memo += f"""        <span style="background: #FF475715; color: #FF4757; padding: 0.25rem 0.6rem; border-radius: 4px; font-size: 0.75rem;">{risk}</span>\n"""
    
    memo += f"""    </div>
</div>
"""
    
    # Add Interest Rate & Affordability Section
    try:
        rate_ctx = get_rate_context()
        payment = calculate_monthly_payment(price)
        
        current_rate = rate_ctx["current_rates"]["30_year_fixed"]
        rate_env = rate_ctx["impact"]["sentiment"]
        rate_color = "#00D47E" if rate_env == "Bullish" else "#FF4757" if rate_env == "Challenging" else "#FFB946"
        
        memo += f"""
<div style="margin-bottom: 1.25rem; padding: 1rem; background: #12141a; border-radius: 6px;">
    <p style="color: #9CA3AF; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.08em; margin: 0 0 0.75rem 0;">Interest Rate Environment</p>
    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem;">
        <div>
            <span style="color: #6B7280; font-size: 0.7rem;">30-Year Fixed</span>
            <p style="color: #fff; font-size: 1.1rem; font-weight: 600; margin: 0.125rem 0 0 0;">{current_rate:.2f}%</p>
        </div>
        <div>
            <span style="color: #6B7280; font-size: 0.7rem;">Environment</span>
            <p style="color: {rate_color}; font-size: 0.9rem; font-weight: 500; margin: 0.125rem 0 0 0;">{rate_env}</p>
        </div>
    </div>
    <div style="margin-top: 0.75rem; padding-top: 0.75rem; border-top: 1px solid #2d3139;">
        <p style="color: #6B7280; font-size: 0.7rem; margin: 0 0 0.25rem 0;">Est. Monthly Payment (20% down)</p>
        <p style="color: #fff; font-size: 1rem; font-weight: 600; margin: 0;">
            ${payment['total_monthly']:,.0f}<span style="color: #6B7280; font-size: 0.75rem; font-weight: 400;">/month</span>
        </p>
        <p style="color: #6B7280; font-size: 0.7rem; margin: 0.25rem 0 0 0;">
            P&I: ${payment['monthly_pi']:,.0f} + Tax: ${payment['monthly_tax']:,.0f} + Ins: ${payment['monthly_insurance']:,.0f}
        </p>
    </div>
</div>
"""
    except Exception as e:
        logger.debug(f"Could not add rate context: {e}")
    
    memo += f"""
<div style="padding-top: 1rem; border-top: 1px solid #2d3139;">
    <p style="color: #4B5563; font-size: 0.7rem; font-style: italic; margin: 0;">
        {source_note}
    </p>
</div>

</div>
"""
    
    return memo


# ====================================
# UTILITY FUNCTIONS
# ====================================
def ingest_market_report(pdf_path: Path) -> int:
    """
    Ingest a market report PDF into the vector store.
    
    Args:
        pdf_path: Path to the PDF file.
    
    Returns:
        Number of chunks ingested.
    """
    vector_store = get_vector_store()
    return vector_store.ingest_pdf(pdf_path)


def clear_vector_store() -> None:
    """Clear all data from the vector store."""
    vector_store = get_vector_store()
    vector_store.clear()


if __name__ == "__main__":
    # Test the oracle with RAG
    print("=" * 50)
    print("Testing AI Oracle with RAG v2.0")
    print("=" * 50)
    
    # Test vector store
    vs = get_vector_store()
    print(f"\nVector store available: {vs.is_available}")
    if vs.is_available:
        print(f"Documents in collection: {vs.collection.count()}")
    
    # Test RAG retrieval
    context = get_market_context("94107")
    print(f"\nMarket Context for 94107:")
    print(f"  Neighborhood: {context.get('neighborhood')}")
    print(f"  Sentiment: {context.get('sentiment')}")
    print(f"  Source: {context.get('source', 'static')}")
    
    # Test memo generation
    test_shap_data = {
        "predicted_price": 950000,
        "base_value": 850000,
        "shap_values": {
            "sqft": 75000,
            "bedrooms": 25000,
            "year_built": -10000,
            "condition": 10000,
        },
        "feature_values": {
            "sqft": 1800,
            "bedrooms": 3,
            "year_built": 1965,
            "condition": 4,
        }
    }
    
    memo = generate_investment_memo(
        price=950000,
        shap_data=test_shap_data,
        zip_code="94107",
    )
    
    print("\nGenerated Investment Memo (truncated):")
    print(memo[:500] + "...")
