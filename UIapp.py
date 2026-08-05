import os
import time
from dotenv import load_dotenv

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

load_dotenv()

# ==========================================
# 1. RAG ENGINE INITIALIZATION
# ==========================================

PDF_FILE = "ML(doc).pdf"
PERSIST_DIR = "Chroma_db"

if not os.path.exists(PDF_FILE):
    raise FileNotFoundError(f"Source PDF '{PDF_FILE}' not found in working directory.")

embedding_model = HuggingFaceEmbeddings()

# Check if DB already exists to avoid re-indexing on every restart
if os.path.exists(PERSIST_DIR) and os.listdir(PERSIST_DIR):
    vectorstores = Chroma(
        persist_directory=PERSIST_DIR,
        embedding_function=embedding_model
    )
else:
    loader = PyPDFLoader(PDF_FILE)
    docs = loader.load()
    for doc in docs:
        doc.page_content = " ".join(doc.page_content.split())

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = text_splitter.split_documents(docs)

    vectorstores = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=PERSIST_DIR
    )

retriever = vectorstores.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5}
)

# Using gemini-2.5-flash for compatibility
model = ChatGoogleGenerativeAI(model="gemini-3.6-flash")

# ==========================================
# 2. FASTAPI BACKEND ENDPOINTS
# ==========================================

app = FastAPI(title="Enterprise ML RAG Copilot")

class QueryRequest(BaseModel):
    query: str

@app.post("/api/query")
async def process_query(payload: QueryRequest):
    query = payload.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query string cannot be empty.")

    # Retrieve context chunks
    result = retriever.invoke(query)
    if not result:
        return {
            "found": False,
            "answer": "No relevant context found in the Machine Learning document.",
            "sources": [],
            "latency": 0.0,
            "tokens": None
        }

    context = "\n\n".join([r.page_content for r in result])
    prompt = f"""
You are an AI assistant.

Answer the question only using the context below.

Context:
{context}

Question:
{query}
"""

    start = time.time()
    response = model.invoke([HumanMessage(content=prompt)])
    end = time.time()
    latency = round(end - start, 2)

    # Convert response content safely to string
    if isinstance(response.content, str):
        answer_text = response.content
    elif isinstance(response.content, list):
        answer_text = "".join([str(item.get("text", item)) if isinstance(item, dict) else str(item) for item in response.content])
    else:
        answer_text = str(response.content)

    # Extract Token Usage Telemetry
    token_usage = None
    try:
        if hasattr(response, "usage_metadata") and response.usage_metadata:
            token_usage = dict(response.usage_metadata)
    except Exception:
        token_usage = "Token usage metadata not available."

    sources = [{"content": doc.page_content, "metadata": doc.metadata} for doc in result]

    return {
        "found": True,
        "answer": answer_text,
        "sources": sources,
        "latency": latency,
        "tokens": token_usage
    }

# ==========================================
# 3. ENTERPRISE WEB INTERFACE
# ==========================================

@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    return """
<!DOCTYPE html>
<html lang="en" class="dark">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Enterprise ML Knowledge Base</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
    <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
    <script>
        tailwind.config = {
            darkMode: 'class',
            theme: {
                extend: {
                    fontFamily: {
                        sans: ['Plus Jakarta Sans', 'sans-serif'],
                        mono: ['JetBrains Mono', 'monospace'],
                    }
                }
            }
        }
    </script>
    <style>
        .prose pre { background-color: #0b0f19; padding: 1rem; border-radius: 0.75rem; border: 1px solid #1e293b; overflow-x: auto; }
        .prose code { color: #38bdf8; font-family: 'JetBrains Mono', monospace; font-size: 0.85em; }
        .prose p { margin-bottom: 0.75rem; line-height: 1.6; }
        .prose ul { list-style-type: disc; padding-left: 1.25rem; margin-bottom: 0.75rem; }
        .prose ol { list-style-type: decimal; padding-left: 1.25rem; margin-bottom: 0.75rem; }
        .prose h1, .prose h2, .prose h3 { color: #f8fafc; font-weight: 600; margin-top: 1rem; margin-bottom: 0.5rem; }
        .prose h2 { font-size: 1.15rem; border-bottom: 1px solid #1e293b; padding-bottom: 0.25rem; }
        .prose strong { color: #f1f5f9; font-weight: 600; }
        .custom-scrollbar::-webkit-scrollbar { width: 5px; }
        .custom-scrollbar::-webkit-scrollbar-track { background: #030712; }
        .custom-scrollbar::-webkit-scrollbar-thumb { background: #334155; border-radius: 4px; }
    </style>
</head>
<body class="bg-[#030712] text-slate-100 min-h-screen flex flex-col font-sans antialiased selection:bg-indigo-500/30">

    <!-- Navbar -->
    <header class="border-b border-slate-800/80 bg-slate-900/40 backdrop-blur-xl sticky top-0 z-50">
        <div class="max-w-7xl mx-auto px-6 h-16 flex items-center justify-between">
            <div class="flex items-center gap-3">
                <div class="h-9 w-9 rounded-xl bg-gradient-to-tr from-cyan-500 via-indigo-600 to-violet-600 flex items-center justify-center font-bold text-white shadow-lg shadow-indigo-500/20 ring-1 ring-white/20">
                    <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 10V3L4 14h7v7l9-11h-7z"/></svg>
                </div>
                <div>
                    <div class="flex items-center gap-2">
                        <span class="font-bold text-base tracking-tight text-white">ML Knowledge Engine</span>
                        <span class="bg-indigo-500/10 text-indigo-400 text-[10px] font-mono px-2 py-0.5 rounded border border-indigo-500/20">RAG Production</span>
                    </div>
                    <span class="block text-[11px] text-slate-400 font-mono">ChromaDB + Gemini 2.5 Flash</span>
                </div>
            </div>
            
            <div class="flex items-center gap-3">
                <div class="flex items-center gap-2 text-xs font-mono text-emerald-400 bg-emerald-950/40 border border-emerald-800/40 px-3 py-1.5 rounded-full">
                    <span class="inline-block w-2 h-2 rounded-full bg-emerald-400 animate-pulse"></span>
                    <span>ML(doc).pdf Ready</span>
                </div>
            </div>
        </div>
    </header>

    <!-- Main Grid Workspace -->
    <main class="flex-1 max-w-7xl w-full mx-auto p-6 grid grid-cols-1 lg:grid-cols-12 gap-6">

        <!-- Left Column: Input & AI Response -->
        <section class="lg:col-span-7 flex flex-col gap-6">
            
            <!-- Query Input Box -->
            <div class="bg-slate-900/60 border border-slate-800/80 rounded-2xl p-5 shadow-2xl backdrop-blur-md relative overflow-hidden">
                <div class="flex items-center justify-between mb-3">
                    <label for="queryInput" class="text-xs font-semibold uppercase tracking-wider text-cyan-400 flex items-center gap-2">
                        <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M8 10h.01M12 10h.01M16 10h.01M9 16H5a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v8a2 2 0 01-2 2h-4l-4 4z"/></svg>
                        Ask Question
                    </label>
                    <span class="text-[11px] text-slate-500 font-mono">Press Ctrl + Enter</span>
                </div>
                
                <textarea id="queryInput" rows="3" placeholder="Enter your question regarding Machine Learning models, optimization, architectures..." 
                    class="w-full bg-[#070b14] border border-slate-800 rounded-xl p-3.5 text-sm text-slate-100 placeholder-slate-600 focus:outline-none focus:border-cyan-500/60 focus:ring-1 focus:ring-cyan-500/60 transition-all resize-none font-sans"></textarea>
                
                <div class="flex justify-between items-center mt-3">
                    <div class="flex items-center gap-2 text-xs text-slate-500 font-mono">
                        <span class="w-1.5 h-1.5 rounded-full bg-slate-600"></span>
                        k=5 Similarity Vector Search
                    </div>
                    <button id="submitBtn" onclick="runQuery()" class="bg-gradient-to-r from-cyan-600 via-indigo-600 to-violet-600 hover:opacity-90 text-white font-medium text-xs px-5 py-2.5 rounded-xl transition-all shadow-lg shadow-indigo-600/20 flex items-center gap-2 active:scale-95">
                        <span id="btnText">Execute Query</span>
                        <svg id="spinner" class="hidden animate-spin h-4 w-4 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                            <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
                            <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                        </svg>
                    </button>
                </div>
            </div>

            <!-- Answer Display Panel -->
            <div class="bg-slate-900/60 border border-slate-800/80 rounded-2xl p-6 shadow-2xl backdrop-blur-md flex-1 flex flex-col min-h-[380px] relative">
                <div class="flex items-center justify-between pb-4 border-b border-slate-800/80 mb-4">
                    <h2 class="text-xs font-semibold uppercase tracking-wider text-slate-400 flex items-center gap-2">
                        <svg class="w-4 h-4 text-emerald-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"/></svg>
                        Synthesized Output
                    </h2>
                    <div class="flex items-center gap-2">
                        <button id="copyBtn" onclick="copyAnswer()" class="hidden text-xs font-mono bg-slate-800/80 hover:bg-slate-700 text-slate-300 px-2.5 py-1 rounded-md border border-slate-700 transition-all flex items-center gap-1">
                            <svg class="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M8 16H6a2 2 0 01-2-2V6a2 2 0 012-2h8a2 2 0 012 2v2m-6 12h8a2 2 0 002-2v-8a2 2 0 00-2-2h-8a2 2 0 00-2 2v8a2 2 0 002 2z"/></svg>
                            <span id="copyText">Copy</span>
                        </button>
                        <div id="statsBadge" class="hidden flex gap-2 text-xs font-mono">
                            <span id="latencyStat" class="bg-indigo-950/60 border border-indigo-800/50 text-indigo-300 px-2.5 py-1 rounded-md"></span>
                        </div>
                    </div>
                </div>
                
                <div id="answerContent" class="prose text-slate-300 text-sm flex-1 max-w-none overflow-y-auto custom-scrollbar pr-1">
                    <div class="h-full flex flex-col items-center justify-center text-slate-500 gap-3 py-16">
                        <svg class="w-10 h-10 opacity-30 text-indigo-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.5" d="M12 6.253v13m0-13C10.832 5.477 9.246 5 7.5 5S4.168 5.477 3 6.253v13C4.168 18.477 5.754 18 7.5 18s3.332.477 4.5 1.253m0-13C13.168 5.477 14.754 5 16.5 5c1.747 0 3.332.477 4.5 1.253v13C19.832 18.477 18.247 18 16.5 18c-1.746 0-3.332.477-4.5 1.253"/></svg>
                        <p class="text-xs font-mono">Submit a query to parse document context...</p>
                    </div>
                </div>

                <!-- Telemetry Panel -->
                <div id="tokenUsageContainer" class="hidden mt-4 pt-3 border-t border-slate-800/80 text-xs font-mono text-slate-400 flex items-center justify-between bg-[#070b14] p-3 rounded-xl border border-slate-800">
                    <span class="text-slate-500 text-[10px] uppercase tracking-wider font-semibold">Gemini Telemetry</span>
                    <span id="tokenUsageText" class="text-cyan-400"></span>
                </div>
            </div>
        </section>

        <!-- Right Column: Context Inspector -->
        <section class="lg:col-span-5 bg-slate-900/60 border border-slate-800/80 rounded-2xl p-6 shadow-2xl backdrop-blur-md flex flex-col h-[calc(100vh-120px)] sticky top-24">
            <div class="pb-4 border-b border-slate-800/80 mb-4 flex items-center justify-between">
                <h2 class="text-xs font-semibold uppercase tracking-wider text-slate-400 flex items-center gap-2">
                    <svg class="w-4 h-4 text-cyan-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10"/></svg>
                    Retrieved Context Chunks
                </h2>
                <span id="chunkCount" class="text-[10px] font-mono bg-slate-800 px-2 py-0.5 rounded text-slate-400">0 chunks</span>
            </div>
            
            <div id="sourcesContainer" class="overflow-y-auto flex-1 space-y-3.5 pr-1.5 custom-scrollbar">
                <div class="text-slate-500 text-xs italic text-center py-12 font-mono">
                    Retrieved document metadata and page chunks will appear here.
                </div>
            </div>
        </section>

    </main>

    <script>
        let rawAnswerText = "";

        async function runQuery() {
            const queryInput = document.getElementById("queryInput");
            const submitBtn = document.getElementById("submitBtn");
            const btnText = document.getElementById("btnText");
            const spinner = document.getElementById("spinner");
            const answerContent = document.getElementById("answerContent");
            const sourcesContainer = document.getElementById("sourcesContainer");
            const statsBadge = document.getElementById("statsBadge");
            const latencyStat = document.getElementById("latencyStat");
            const tokenUsageContainer = document.getElementById("tokenUsageContainer");
            const tokenUsageText = document.getElementById("tokenUsageText");
            const chunkCount = document.getElementById("chunkCount");
            const copyBtn = document.getElementById("copyBtn");

            const query = queryInput.value.trim();
            if (!query) return;

            // UI State updates
            submitBtn.disabled = true;
            btnText.innerText = "Processing...";
            spinner.classList.remove("hidden");
            copyBtn.classList.add("hidden");
            
            answerContent.innerHTML = `
                <div class="flex flex-col items-center justify-center py-12 gap-3 text-cyan-400 animate-pulse">
                    <svg class="w-6 h-6 animate-spin" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15"/></svg>
                    <span class="text-xs font-mono">Retrieving context & generating answer...</span>
                </div>`;
            
            sourcesContainer.innerHTML = `<div class="text-slate-500 text-xs font-mono animate-pulse text-center py-8">Searching Vector Index...</div>`;

            try {
                const response = await fetch("/api/query", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ query: query })
                });

                const data = await response.json();

                if (!response.ok) {
                    throw new Error(data.detail || "Error processing request.");
                }

                // Convert payload answer safely
                let answerText = "";
                if (typeof data.answer === "string") {
                    answerText = data.answer;
                } else if (Array.isArray(data.answer)) {
                    answerText = data.answer.map(i => typeof i === 'object' ? JSON.stringify(i) : String(i)).join("\\n");
                } else {
                    answerText = String(data.answer || "");
                }

                rawAnswerText = answerText;
                answerContent.innerHTML = marked.parse(answerText);
                copyBtn.classList.remove("hidden");

                // Metrics
                if (data.found) {
                    statsBadge.classList.remove("hidden");
                    latencyStat.innerText = `Latency: ${data.latency}s`;

                    if (data.tokens) {
                        tokenUsageContainer.classList.remove("hidden");
                        tokenUsageText.innerText = typeof data.tokens === 'object' 
                            ? JSON.stringify(data.tokens) 
                            : data.tokens;
                    } else {
                        tokenUsageContainer.classList.add("hidden");
                    }
                } else {
                    statsBadge.classList.add("hidden");
                    tokenUsageContainer.classList.add("hidden");
                }

                // Sources Rendering
                sourcesContainer.innerHTML = "";
                if (data.sources && data.sources.length > 0) {
                    chunkCount.innerText = `${data.sources.length} chunks`;
                    data.sources.forEach((source, index) => {
                        const card = document.createElement("div");
                        card.className = "bg-[#070b14] border border-slate-800 rounded-xl p-3.5 text-xs space-y-2 hover:border-slate-700 transition-all";
                        
                        const metaHeader = document.createElement("div");
                        metaHeader.className = "flex justify-between items-center text-cyan-400 font-mono text-[11px] pb-1 border-b border-slate-800/80";
                        
                        const pageNum = source.metadata && source.metadata.page !== undefined ? `Page ${source.metadata.page + 1}` : 'Chunk';
                        metaHeader.innerHTML = `<span class="font-semibold text-indigo-400">Match #${index + 1}</span><span class="bg-slate-800 px-2 py-0.5 rounded text-[10px] text-slate-300 font-mono">${pageNum}</span>`;
                        
                        const body = document.createElement("div");
                        body.className = "text-slate-300 leading-relaxed font-sans text-xs max-h-40 overflow-y-auto custom-scrollbar pr-1 pt-1";
                        body.innerText = source.content;

                        card.appendChild(metaHeader);
                        card.appendChild(body);
                        sourcesContainer.appendChild(card);
                    });
                } else {
                    chunkCount.innerText = "0 chunks";
                    sourcesContainer.innerHTML = "<div class='text-slate-500 text-xs italic text-center py-6 font-mono'>No context returned.</div>";
                }

            } catch (err) {
                answerContent.innerHTML = `<div class="text-rose-400 font-mono text-xs">Error: ${err.message}</div>`;
                sourcesContainer.innerHTML = "";
            } finally {
                submitBtn.disabled = false;
                btnText.innerText = "Execute Query";
                spinner.classList.add("hidden");
            }
        }

        function copyAnswer() {
            if (!rawAnswerText) return;
            navigator.clipboard.writeText(rawAnswerText);
            const copyText = document.getElementById("copyText");
            copyText.innerText = "Copied!";
            setTimeout(() => { copyText.innerText = "Copy"; }, 2000);
        }

        document.getElementById("queryInput").addEventListener("keydown", function(e) {
            if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) {
                runQuery();
            }
        });
    </script>
</body>
</html>
"""