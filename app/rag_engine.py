"""
RAG Engine - Answer questions using retrieved context from contracts.

Combines:
- Semantic search for relevant context retrieval
- LLM for answer generation
- Contract-specific prompt templates
"""

from typing import List, Optional, Tuple
import re

from app.config import settings
from app.models import DocumentChunk, SourceReference
from app.embedding_service import EmbeddingService
from app.vector_store import VectorStore


class RAGEngine:
    """
    Retrieval-Augmented Generation engine for contract Q&A.
    
    Handles:
    - Query understanding and classification
    - Context retrieval from vector store
    - Answer generation with citations
    """
    
    # Question category patterns for better retrieval (Azerbaijani, Russian, English)
    QUESTION_CATEGORIES = {
        "general_info": {
            "patterns": [
                # Azerbaijani
                r"status", r"vəziyyət", r"imzalan", r"bitir", r"müddət",
                r"uzadılma", r"təchizatçı", r"məbləğ", r"valyuta",
                r"sahibi", r"tender", r"tərəflər",
                # Russian
                r"статус", r"подписан", r"истекает",
                r"продлен", r"поставщик",
                r"сумма", r"валют", r"владелец", r"тендер",
                # English
                r"status", r"signed", r"expire", r"renewal", r"supplier", r"vendor",
                r"amount", r"total", r"currency", r"owner",
            ],
            "prompt_hint_az": "Müqavilə haqqında ümumi məlumat, tərəflər, tarixlər və əsas şərtlərə diqqət yetirin.",
            "prompt_hint_ru": "Сосредоточьтесь на общей информации о контракте, сторонах, датах и основных условиях.",
            "prompt_hint_en": "Focus on general contract metadata, parties, dates, and basic terms.",
        },
        "financial": {
            "patterns": [
                # Azerbaijani
                r"qiymət", r"ödəniş", r"məbləğ", r"dəyər", r"maliyyə",
                r"cərimə", r"gecikmə", r"indeksasiya", r"limit",
                # Russian
                r"стоимость", r"цен", r"индексац", r"оплат",
                r"платеж", r"штраф за задержку оплаты",
                r"лимит", r"объем закупки",
                # English
                r"cost", r"price", r"pricing", r"indexation", r"payment",
                r"late payment", r"limit", r"volume",
            ],
            "prompt_hint_az": "Maliyyə şərtləri, qiymətlər, ödəniş şərtləri və pul dəyərlərinə diqqət yetirin.",
            "prompt_hint_ru": "Сосредоточьтесь на финансовых условиях, ценах, условиях оплаты и денежных суммах.",
            "prompt_hint_en": "Focus on financial terms, pricing, payment conditions, and monetary values.",
        },
        "deadlines": {
            "patterns": [
                # Azerbaijani
                r"müddət", r"son tarix", r"çatdırılma", r"mərhələ",
                r"təhvil", r"gecikmə", r"sla",
                # Russian
                r"срок", r"milestone", r"этап",
                r"поставк", r"sla", r"уровень сервиса",
                r"задержк",
                # English
                r"deadline", r"milestone", r"stage", r"delivery", r"delay", r"penalty",
            ],
            "prompt_hint_az": "Son tarixlər, mərhələlər, çatdırılma qrafikləri və SLA şərtlərinə diqqət yetirin.",
            "prompt_hint_ru": "Сосредоточьтесь на сроках, этапах, графиках поставки и условиях SLA.",
            "prompt_hint_en": "Focus on deadlines, milestones, delivery schedules, and SLA terms.",
        },
        "risks": {
            "patterns": [
                # Azerbaijani
                r"cərimə", r"sanksiya", r"zəmanət", r"risk",
                r"ləğv", r"xitam", r"təminat",
                # Russian
                r"штраф", r"санкци", r"гарантия",
                r"расторжен", r"прекращен",
                # English
                r"penalty", r"sanction", r"warranty", r"guarantee", r"termination",
            ],
            "prompt_hint_az": "Cərimələr, risklər, zəmanətlər və müqavilənin ləğvi şərtlərinə diqqət yetirin.",
            "prompt_hint_ru": "Сосредоточьтесь на штрафах, рисках, гарантиях и условиях расторжения.",
            "prompt_hint_en": "Focus on penalties, risks, guarantees, and termination clauses.",
        },
        "scope": {
            "patterns": [
                # Azerbaijani
                r"mal", r"xidmət", r"miqdar", r"həcm",
                r"minimum", r"maksimum", r"təchizat",
                # Russian
                r"товар", r"услуг", r"количество", r"объем",
                r"миним", r"максим",
                # English
                r"goods", r"service", r"quantity", r"volume", r"minimum", r"maximum",
            ],
            "prompt_hint_az": "Təchizat həcmi, mal/xidmətlər, miqdarlar və limitlərə diqqət yetirin.",
            "prompt_hint_ru": "Сосредоточьтесь на объеме поставки, товарах/услугах, количестве и лимитах.",
            "prompt_hint_en": "Focus on scope of delivery, goods/services, quantities, and limits.",
        },
        "amendments": {
            "patterns": [
                # Azerbaijani
                r"dəyişiklik", r"əlavə", r"versiya", r"düzəliş", r"yeniləmə",
                # Russian
                r"изменен", r"дополн", r"версия", r"редакция",
                # English
                r"amendment", r"modification", r"version",
            ],
            "prompt_hint_az": "Müqavilə dəyişiklikləri, əlavələr və versiya tarixçəsinə diqqət yetirin.",
            "prompt_hint_ru": "Сосредоточьтесь на изменениях контракта, дополнениях и истории версий.",
            "prompt_hint_en": "Focus on contract amendments, changes, and version history.",
        },
    }
    
    # System prompt for contract Q&A (multilingual: Azerbaijani, Russian, English)
    SYSTEM_PROMPT = """You are an expert contract analyst assistant. Your task is to answer questions about contracts based on the provided context.

INSTRUCTIONS:
1. Answer ONLY based on the provided context from the contract documents
2. If the information is not in the context, clearly state:
   - Azerbaijani: "Təqdim olunan sənədlərdə məlumat tapılmadı"
   - Russian: "Информация не найдена в предоставленных документах"
   - English: "Information not found in the provided documents"
3. Be precise and cite specific sections/pages when possible
4. IMPORTANT: Match the response language to the question language:
   - If the question is in Azerbaijani, answer in Azerbaijani
   - If the question is in Russian, answer in Russian
   - If the question is in English, answer in English
5. Format monetary values, dates, and percentages clearly
6. If information is ambiguous or incomplete, note this explicitly

{category_hint}

CONTEXT FROM CONTRACT DOCUMENTS:
{context}

QUESTION: {question}

ANSWER:"""
    
    def __init__(
        self,
        embedding_service: EmbeddingService,
        vector_store: VectorStore,
    ):
        """
        Initialize RAG engine.
        
        Args:
            embedding_service: Service for generating embeddings
            vector_store: Store for document vectors
        """
        self.embedding_service = embedding_service
        self.vector_store = vector_store
        self._llm_client = None
    
    def _init_llm(self):
        """Initialize LLM client based on settings."""
        if self._llm_client is not None:
            return
        
        if settings.LLM_PROVIDER == "openai":
            try:
                from openai import OpenAI
                self._llm_client = OpenAI(api_key=settings.LLM_API_KEY)
                print(f"✅ Initialized OpenAI LLM: {settings.LLM_MODEL}")
            except Exception as e:
                print(f"⚠️ Failed to initialize OpenAI: {e}")
                self._llm_client = "mock"
        
        elif settings.LLM_PROVIDER == "anthropic":
            try:
                from anthropic import Anthropic
                self._llm_client = Anthropic(api_key=settings.ANTHROPIC_API_KEY)
                print(f"✅ Initialized Anthropic LLM: {settings.ANTHROPIC_MODEL}")
            except Exception as e:
                print(f"⚠️ Failed to initialize Anthropic: {e}")
                self._llm_client = "mock"
        
        else:
            self._llm_client = "mock"
    
    def classify_question(self, question: str, language: str = "auto") -> Tuple[str, str]:
        """
        Classify question into category and get prompt hint.
        
        Args:
            question: User's question
            language: Language code ("az", "ru", "en", or "auto")
            
        Returns:
            Tuple of (category_name, prompt_hint)
        """
        question_lower = question.lower()
        
        # Auto-detect language if not specified
        if language == "auto":
            language = self._detect_language(question)
        
        for category, config in self.QUESTION_CATEGORIES.items():
            for pattern in config["patterns"]:
                if re.search(pattern, question_lower):
                    # Get language-specific hint
                    hint_key = f"prompt_hint_{language}"
                    hint = config.get(hint_key, config.get("prompt_hint_en", ""))
                    return category, hint
        
        return "general", ""
    
    def _detect_language(self, text: str) -> str:
        """
        Detect language of text (Azerbaijani, Russian, or English).
        
        Returns:
            Language code: "az", "ru", or "en"
        """
        # Azerbaijani-specific characters
        az_chars = set("əğıöüçş")
        # Russian characters
        ru_chars = set("абвгдеёжзийклмнопрстуфхцчшщъыьэюя")
        
        text_lower = text.lower()
        
        # Check for Azerbaijani-specific characters
        if any(c in az_chars for c in text_lower):
            return "az"
        
        # Check for Russian characters
        if any(c in ru_chars for c in text_lower):
            return "ru"
        
        # Default to English
        return "en"
    
    def answer_question(
        self,
        session_id: str,
        question: str,
        top_k: int = 5,
        language: str = "auto",
    ) -> Tuple[str, List[SourceReference], float]:
        """
        Generate answer to question using RAG.
        
        Args:
            session_id: Session with uploaded documents
            question: User's question
            top_k: Number of chunks to retrieve
            language: Response language preference ("az", "ru", "en", or "auto")
            
        Returns:
            Tuple of (answer, sources, confidence)
        """
        # Detect language if auto
        detected_lang = self._detect_language(question) if language == "auto" else language
        
        # Classify question with language
        category, category_hint = self.classify_question(question, detected_lang)
        
        # Generate query embedding
        query_embedding = self.embedding_service.embed_text(question)
        
        # Retrieve relevant chunks
        results = self.vector_store.search(
            session_id=session_id,
            query_embedding=query_embedding,
            top_k=top_k,
            threshold=settings.SIMILARITY_THRESHOLD,
        )
        
        if not results:
            no_results_messages = {
                "az": "Təəssüf ki, yüklənmiş sənədlərdə uyğun məlumat tapılmadı. "
                      "Sualınızı yenidən formulə etməyə çalışın və ya lazımi sənədin yükləndiyinə əmin olun.",
                "ru": "К сожалению, релевантная информация не найдена в загруженных документах. "
                      "Попробуйте переформулировать вопрос или убедитесь, что нужный документ был загружен.",
                "en": "Unfortunately, no relevant information was found in the uploaded documents. "
                      "Try rephrasing your question or ensure the relevant document was uploaded.",
            }
            return (no_results_messages.get(detected_lang, no_results_messages["en"]), [], 0.0)
        
        # Build context from retrieved chunks
        context = self._build_context(results)
        
        # Calculate confidence based on relevance scores
        avg_score = sum(score for _, score in results) / len(results)
        confidence = min(avg_score, 1.0)
        
        # Generate answer
        answer = self._generate_answer(
            question=question,
            context=context,
            category_hint=category_hint,
            language=detected_lang,
        )
        
        # Build source references
        sources = [
            SourceReference(
                source_file=chunk.source_file,
                chunk_text=chunk.text[:500] + "..." if len(chunk.text) > 500 else chunk.text,
                relevance_score=round(score, 3),
                chunk_index=chunk.chunk_index,
            )
            for chunk, score in results
        ]
        
        return answer, sources, round(confidence, 3)
    
    def _build_context(self, results: List[Tuple[DocumentChunk, float]]) -> str:
        """Build context string from retrieved chunks."""
        context_parts = []
        
        for i, (chunk, score) in enumerate(results, 1):
            source_info = f"[Source: {chunk.source_file}, Chunk {chunk.chunk_index + 1}]"
            context_parts.append(f"{source_info}\n{chunk.text}")
        
        return "\n\n---\n\n".join(context_parts)
    
    def _generate_answer(
        self,
        question: str,
        context: str,
        category_hint: str,
        language: str,
    ) -> str:
        """Generate answer using LLM."""
        self._init_llm()
        
        # Build prompt
        prompt = self.SYSTEM_PROMPT.format(
            category_hint=f"\nCATEGORY FOCUS: {category_hint}" if category_hint else "",
            context=context,
            question=question,
        )
        
        # Add language instruction if specified
        language_instructions = {
            "az": "\n\n(Azərbaycan dilində cavab verin)",
            "ru": "\n\n(Отвечайте на русском языке)",
            "en": "\n\n(Please respond in English)",
        }
        prompt += language_instructions.get(language, "")
        
        # Generate with LLM
        if self._llm_client == "mock":
            return self._mock_answer(question, context, language)
        
        try:
            if settings.LLM_PROVIDER == "openai":
                response = self._llm_client.chat.completions.create(
                    model=settings.LLM_MODEL,
                    messages=[
                        {"role": "system", "content": "You are a contract analysis expert. You can respond in Azerbaijani, Russian, and English."},
                        {"role": "user", "content": prompt},
                    ],
                    max_tokens=settings.LLM_MAX_TOKENS,
                    temperature=settings.LLM_TEMPERATURE,
                )
                return response.choices[0].message.content.strip()
            
            elif settings.LLM_PROVIDER == "anthropic":
                response = self._llm_client.messages.create(
                    model=settings.ANTHROPIC_MODEL,
                    max_tokens=settings.LLM_MAX_TOKENS,
                    messages=[
                        {"role": "user", "content": prompt},
                    ],
                )
                return response.content[0].text.strip()
        
        except Exception as e:
            print(f"LLM generation error: {e}")
            return self._mock_answer(question, context, language)
    
    def _mock_answer(self, question: str, context: str, language: str = "auto") -> str:
        """Generate a mock answer when LLM is not available."""
        if language == "auto":
            language = self._detect_language(question)
        
        # Extract some relevant info from context
        context_preview = context[:500].replace("\n", " ")
        
        mock_responses = {
            "az": (
                f"Yüklənmiş sənədlərin təhlili əsasında:\n\n"
                f"Sənəd kontekstində uyğun məlumat tapıldı. "
                f"Dəqiq cavab almaq üçün LLM provayderi (OpenAI və ya Anthropic) "
                f"mühit dəyişənlərində konfiqurasiya edin.\n\n"
                f"Kontekst önizləməsi: {context_preview}..."
            ),
            "ru": (
                f"На основе анализа загруженных документов:\n\n"
                f"Найдена релевантная информация в контексте документов. "
                f"Для получения точного ответа настройте LLM провайдер (OpenAI или Anthropic) "
                f"в переменных окружения.\n\n"
                f"Предварительный контекст: {context_preview}..."
            ),
            "en": (
                f"Based on the analysis of uploaded documents:\n\n"
                f"Relevant information was found in the document context. "
                f"For accurate answers, please configure an LLM provider (OpenAI or Anthropic) "
                f"in environment variables.\n\n"
                f"Context preview: {context_preview}..."
            ),
        }
        
        return mock_responses.get(language, mock_responses["en"])
    
    def _detect_russian(self, text: str) -> bool:
        """Check if text contains Russian characters."""
        return bool(re.search(r'[а-яА-ЯёЁ]', text))
    
    def get_similar_questions(self, session_id: str, question: str, top_k: int = 3, language: str = "auto") -> List[str]:
        """
        Suggest similar questions based on document content.
        
        Args:
            session_id: Session ID
            question: Original question
            top_k: Number of suggestions to return
            language: Language for suggestions ("az", "ru", "en", or "auto")
        """
        if language == "auto":
            language = self._detect_language(question)
        
        category, _ = self.classify_question(question, language)
        
        # Return category-specific example questions in appropriate language
        suggestions = {
            "az": {
                "general_info": [
                    "Müqavilənin müddəti nə qədərdir?",
                    "Müqavilənin tərəfləri kimlərdir?",
                    "Müqavilənin ümumi məbləği nədir?",
                ],
                "financial": [
                    "Hansı ödəniş şərtləri nəzərdə tutulub?",
                    "Ödənişin gecikdirilməsinə görə cərimə varmı?",
                    "Maddələr üzrə hansı qiymətlər razılaşdırılıb?",
                ],
                "deadlines": [
                    "Hansı çatdırılma müddətləri göstərilib?",
                    "Mərhələlər və ya milestones varmı?",
                    "Müqavilədə hansı SLA-lar müəyyən edilib?",
                ],
                "risks": [
                    "Şərtlərin pozulmasına görə hansı cərimələr nəzərdə tutulub?",
                    "Müqavilənin ləğvi şərtləri hansılardır?",
                    "Zəmanət öhdəlikləri varmı?",
                ],
                "scope": [
                    "Müqaviləyə hansı mal/xidmətlər daxildir?",
                    "Minimum/maksimum təchizat həcmi nə qədərdir?",
                    "Hansı spesifikasiyalar göstərilib?",
                ],
                "amendments": [
                    "Müqavilədə dəyişikliklər olubmu?",
                    "Müqavilənin cari versiyası hansıdır?",
                    "Son əlavədə nə dəyişdirilib?",
                ],
            },
            "ru": {
                "general_info": [
                    "Какой срок действия контракта?",
                    "Кто является сторонами договора?",
                    "Какова общая сумма контракта?",
                ],
                "financial": [
                    "Какие условия оплаты предусмотрены?",
                    "Есть ли штрафы за просрочку платежа?",
                    "Какие цены согласованы по позициям?",
                ],
                "deadlines": [
                    "Какие сроки поставки указаны?",
                    "Есть ли этапы или milestones?",
                    "Какие SLA определены в договоре?",
                ],
                "risks": [
                    "Какие штрафы предусмотрены за нарушение условий?",
                    "Каковы условия расторжения договора?",
                    "Есть ли гарантийные обязательства?",
                ],
                "scope": [
                    "Какие товары/услуги входят в договор?",
                    "Каков минимальный/максимальный объем поставки?",
                    "Какие спецификации указаны?",
                ],
                "amendments": [
                    "Были ли изменения в договоре?",
                    "Какая текущая версия договора?",
                    "Что было изменено в последнем дополнении?",
                ],
            },
            "en": {
                "general_info": [
                    "What is the contract duration?",
                    "Who are the parties to the contract?",
                    "What is the total contract amount?",
                ],
                "financial": [
                    "What payment terms are specified?",
                    "Are there late payment penalties?",
                    "What prices are agreed per item?",
                ],
                "deadlines": [
                    "What delivery deadlines are specified?",
                    "Are there stages or milestones?",
                    "What SLAs are defined in the contract?",
                ],
                "risks": [
                    "What penalties are specified for breach of terms?",
                    "What are the contract termination conditions?",
                    "Are there warranty obligations?",
                ],
                "scope": [
                    "What goods/services are included in the contract?",
                    "What is the minimum/maximum delivery volume?",
                    "What specifications are indicated?",
                ],
                "amendments": [
                    "Have there been any amendments to the contract?",
                    "What is the current version of the contract?",
                    "What was changed in the last amendment?",
                ],
            },
        }
        
        lang_suggestions = suggestions.get(language, suggestions["en"])
        return lang_suggestions.get(category, lang_suggestions["general_info"])[:top_k]
