"use client";

import { motion, AnimatePresence } from "framer-motion";
import { useState, useRef, useEffect } from "react";
import {
    Upload, Activity, Brain, Shield, BarChart3,
    AlertCircle, Download, FileText, Send, MessageSquare
} from "lucide-react";
import Link from "next/link";

type PredictionResult = {
    prediction_id: string;
    disease_type: string;
    prediction: string;
    confidence: string;
    confidence_score: number;
    risk_level: string;
    heatmap_url: string;
    guidance: string;
};

// ── Risk level colour helper ───────────────────────────────────────────────
function riskColour(risk: string): string {
    switch (risk?.toLowerCase()) {
        case "high": return "text-red-500";
        case "moderate": return "text-yellow-400";
        case "uncertain": return "text-orange-400";
        default: return "text-green-400";   // Low
    }
}

// ── Confidence bar colour helper ───────────────────────────────────────────
function confidenceBarColour(score: number): string {
    if (score >= 0.80) return "from-blue-600 to-cyan-400";
    if (score >= 0.65) return "from-yellow-500 to-amber-400";
    return "from-orange-500 to-red-400";
}

// ── Save JSON helper ───────────────────────────────────────────────────────
function saveJson(result: PredictionResult) {
    const blob = new Blob([JSON.stringify(result, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `diagnosis_${result.disease_type}_${result.prediction_id}.json`;
    a.click();
    URL.revokeObjectURL(url);
}

// ── Download PDF report helper ─────────────────────────────────────────────
async function downloadReport(result: PredictionResult, apiUrl: string) {
    try {
        const res = await fetch(`${apiUrl}/api/report/generate`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                prediction_id: result.prediction_id,
                disease_type: result.disease_type,
                prediction: result.prediction,
                confidence: result.confidence,
                risk_level: result.risk_level,
                guidance: result.guidance,
            }),
        });
        if (!res.ok) throw new Error("Report generation failed");
        const data = await res.json();
        const a = document.createElement("a");
        a.href = data.report_data;
        a.download = data.filename;
        a.click();
    } catch {
        alert("Could not generate report. Please try again.");
    }
}

// ─────────────────────────────────────────────────────────────────────────────

type ChatMessage = { role: "user" | "assistant"; content: string };

export default function Dashboard() {
    const [file, setFile] = useState<File | null>(null);
    const [preview, setPreview] = useState<string | null>(null);
    const [isAnalyzing, setIsAnalyzing] = useState(false);
    const [result, setResult] = useState<PredictionResult | null>(null);
    const [error, setError] = useState<string | null>(null);
    const [selectedType, setSelectedType] = useState<string>("brain");
    const fileInputRef = useRef<HTMLInputElement>(null);

    // ── Chat state ────────────────────────────────────────────────────────────
    const [chatMessages, setChatMessages] = useState<ChatMessage[]>([]);
    const [chatInput, setChatInput] = useState("");
    const [isChatLoading, setIsChatLoading] = useState(false);
    const chatEndRef = useRef<HTMLDivElement>(null);

    const apiUrl = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

    // Auto-scroll chat to bottom
    useEffect(() => {
        chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
    }, [chatMessages, isChatLoading]);

    // Reset chat when a new result comes in
    useEffect(() => {
        setChatMessages([]);
    }, [result]);

    const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        const selectedFile = e.target.files?.[0];
        if (selectedFile) {
            setFile(selectedFile);
            setPreview(URL.createObjectURL(selectedFile));
            setResult(null);
            setError(null);
            setChatMessages([]);
        }
    };

    // ── Chat send handler ─────────────────────────────────────────────────────
    const sendChatMessage = async () => {
        const text = chatInput.trim();
        if (!text || isChatLoading) return;

        const userMsg: ChatMessage = { role: "user", content: text };
        const updatedHistory = [...chatMessages, userMsg];

        setChatMessages(updatedHistory);
        setChatInput("");
        setIsChatLoading(true);

        // Build diagnosis context string from current result
        const diagCtx = result
            ? `Disease Type: ${result.disease_type} | Finding: ${result.prediction} | Confidence: ${result.confidence} | Risk: ${result.risk_level}`
            : undefined;

        try {
            const res = await fetch(`${apiUrl}/api/chat/ask`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    messages: updatedHistory,
                    diagnosis_context: diagCtx,
                }),
            });
            if (!res.ok) throw new Error(`Chat API error ${res.status}`);
            const data: ChatMessage = await res.json();
            setChatMessages(prev => [...prev, data]);
        } catch {
            setChatMessages(prev => [
                ...prev,
                { role: "assistant", content: "⚠️ I couldn't reach the server. Please check your connection and try again." },
            ]);
        } finally {
            setIsChatLoading(false);
        }
    };

    const handleAnalyze = async () => {
        if (!file) return;
        setIsAnalyzing(true);
        setError(null);
        setResult(null);

        const formData = new FormData();
        formData.append("file", file);

        try {
            const response = await fetch(`${apiUrl}/api/prediction/${selectedType}`, {
                method: "POST",
                body: formData,
            });

            if (!response.ok) {
                const errData = await response.json().catch(() => ({}));
                throw new Error(errData?.detail || `Server error ${response.status}`);
            }

            const data: PredictionResult = await response.json();

            // Small UX delay so the scanning animation is visible
            setTimeout(() => {
                setResult(data);
                setIsAnalyzing(false);
            }, 1500);

        } catch (err: unknown) {
            const message = err instanceof Error ? err.message : "Unknown error";
            setIsAnalyzing(false);
            setError(message);
        }
    };

    return (
        <div className="min-h-screen p-4 md:p-8 flex flex-col items-center">
            {/* Header */}
            <header className="max-w-7xl w-full flex justify-between items-center mb-12">
                <Link href="/" className="flex items-center gap-2">
                    <div className="w-10 h-10 bg-blue-600 rounded-lg flex items-center justify-center font-bold text-xl">G</div>
                    <span className="text-2xl font-bold tracking-tight">GoGenix-Med<span className="text-blue-500">AI</span></span>
                </Link>
                <div className="flex gap-4">
                    <button className="glass px-4 py-2 rounded-lg text-sm font-medium hover:bg-white/5 transition-all">Support</button>
                    <button className="px-4 py-2 bg-blue-600 rounded-lg text-sm font-medium hover:bg-blue-500 transition-all">My Reports</button>
                </div>
            </header>

            <div className="max-w-7xl w-full grid grid-cols-1 lg:grid-cols-12 gap-8">
                {/* Left Column */}
                <div className="lg:col-span-5 space-y-6">
                    <section className="glass p-8 rounded-3xl border-white/5">
                        <h2 className="text-2xl font-bold mb-6">Diagnostic Console</h2>

                        <div className="space-y-4 mb-8">
                            <label className="text-sm font-medium text-slate-400">Target Diagnosis</label>
                            <div className="grid grid-cols-2 gap-3">
                                {[
                                    { id: "brain", label: "Brain MRI", icon: Brain },
                                    { id: "lung", label: "Lung X-Ray", icon: Activity },
                                    { id: "cancer", label: "Histology", icon: Shield },
                                    { id: "renal", label: "Renal CT", icon: BarChart3 },
                                ].map((type) => (
                                    <button
                                        key={type.id}
                                        onClick={() => setSelectedType(type.id)}
                                        className={`flex items-center gap-3 p-4 rounded-xl border transition-all ${selectedType === type.id
                                            ? "bg-blue-600/20 border-blue-500 text-blue-400"
                                            : "glass border-white/5 text-slate-400 hover:border-white/10"
                                            }`}
                                    >
                                        <type.icon className="w-5 h-5" />
                                        <span className="font-medium text-sm">{type.label}</span>
                                    </button>
                                ))}
                            </div>
                        </div>

                        <div className="space-y-4">
                            <label className="text-sm font-medium text-slate-400">Source Medical Scan</label>
                            <div
                                onClick={() => fileInputRef.current?.click()}
                                className={`border-2 border-dashed rounded-2xl p-8 flex flex-col items-center justify-center gap-4 cursor-pointer transition-all ${preview ? "border-blue-500/50 bg-blue-500/5" : "border-slate-700 hover:border-slate-600"
                                    }`}
                            >
                                {preview ? (
                                    <div className="relative w-full aspect-video rounded-xl overflow-hidden glass">
                                        {/* eslint-disable-next-line @next/next/no-img-element */}
                                        <img src={preview} alt="Scan preview" className="w-full h-full object-cover" />
                                        <div className="absolute inset-0 bg-black/40 flex items-center justify-center opacity-0 hover:opacity-100 transition-opacity">
                                            <span className="text-sm font-medium">Click to Change</span>
                                        </div>
                                    </div>
                                ) : (
                                    <>
                                        <div className="w-16 h-16 bg-slate-800 rounded-full flex items-center justify-center text-slate-400">
                                            <Upload className="w-8 h-8" />
                                        </div>
                                        <div className="text-center">
                                            <p className="font-medium">Upload Image</p>
                                            <p className="text-xs text-slate-500 mt-1">MRI, CT, or X-Ray (PNG, JPG)</p>
                                        </div>
                                    </>
                                )}
                                <input
                                    type="file"
                                    hidden
                                    ref={fileInputRef}
                                    onChange={handleFileChange}
                                    accept="image/*"
                                />
                            </div>
                        </div>

                        <button
                            disabled={!file || isAnalyzing}
                            onClick={handleAnalyze}
                            className={`w-full mt-8 py-4 rounded-xl font-bold text-lg transition-all flex items-center justify-center gap-2 ${!file || isAnalyzing
                                ? "bg-slate-800 text-slate-500 cursor-not-allowed"
                                : "bg-blue-600 hover:bg-blue-500 text-white shadow-xl shadow-blue-600/20"
                                }`}
                        >
                            {isAnalyzing ? <>Inference Running...</> : <>Start Analysis</>}
                        </button>
                    </section>

                    {/* Error Card */}
                    {error && (
                        <motion.div
                            initial={{ opacity: 0, y: 10 }}
                            animate={{ opacity: 1, y: 0 }}
                            className="glass p-5 rounded-2xl border border-red-500/30 flex gap-3 items-start"
                        >
                            <AlertCircle className="w-5 h-5 text-red-400 mt-0.5 flex-shrink-0" />
                            <div>
                                <p className="font-bold text-red-400 text-sm">Analysis Failed</p>
                                <p className="text-slate-400 text-xs mt-1">{error}</p>
                                <p className="text-slate-500 text-xs mt-1">Check that the backend is running and try again.</p>
                            </div>
                        </motion.div>
                    )}

                    {/* Disclaimer */}
                    <div className="glass p-6 rounded-3xl border-white/5 flex gap-4 items-start">
                        <AlertCircle className="w-6 h-6 text-yellow-500 mt-1 flex-shrink-0" />
                        <div className="text-sm">
                            <p className="font-bold text-yellow-500">Medical Disclaimer</p>
                            <p className="text-slate-400 mt-1 italic">
                                This AI system is for research assistance and provides probabilistic analysis only.
                                Always consult a certified medical professional for diagnosis.
                            </p>
                        </div>
                    </div>
                </div>

                {/* Right Column – Results */}
                <div className="lg:col-span-7">
                    <AnimatePresence mode="wait">
                        {isAnalyzing ? (
                            <motion.div
                                key="analyzing"
                                initial={{ opacity: 0 }}
                                animate={{ opacity: 1 }}
                                exit={{ opacity: 0 }}
                                className="h-[600px] glass rounded-3xl flex flex-col items-center justify-center p-12 overflow-hidden relative"
                            >
                                <div className="relative w-full max-w-md aspect-square rounded-2xl overflow-hidden glass mb-8 border-white/10">
                                    {preview && (
                                        <>
                                            {/* eslint-disable-next-line @next/next/no-img-element */}
                                            <img src={preview} alt="Scanning" className="w-full h-full object-cover opacity-50 grayscale" />
                                            <div className="scan-line"></div>
                                        </>
                                    )}
                                </div>
                                <div className="text-center space-y-4">
                                    <h3 className="text-2xl font-bold animate-pulse text-indigo-400">Analysing Scan</h3>
                                    <div className="space-y-2">
                                        <p className="text-slate-400 text-sm">Calibrated Feature Analysis Engine</p>
                                        <p className="text-slate-500 text-[10px] uppercase tracking-widest">Bias-Reduced · Multi-Signal Detection</p>
                                    </div>
                                </div>
                            </motion.div>
                        ) : result ? (
                            <motion.div
                                key="result"
                                initial={{ opacity: 0, y: 20 }}
                                animate={{ opacity: 1, y: 0 }}
                                className="space-y-6"
                            >
                                {/* Summary Card */}
                                <div className="glass p-8 rounded-3xl border-white/5 space-y-4">
                                    <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                                        <div className="space-y-1">
                                            <span className="text-xs font-bold text-slate-500 uppercase">Detection</span>
                                            <p className="text-lg font-bold text-gradient leading-tight">{result.prediction}</p>
                                        </div>
                                        <div className="space-y-1 text-center md:border-x border-white/5">
                                            <span className="text-xs font-bold text-slate-500 uppercase">Confidence</span>
                                            <p className="text-2xl font-bold text-blue-400">{result.confidence}</p>
                                        </div>
                                        <div className="space-y-1 text-right">
                                            <span className="text-xs font-bold text-slate-500 uppercase">Risk Level</span>
                                            <p className={`text-2xl font-bold ${riskColour(result.risk_level)}`}>{result.risk_level}</p>
                                        </div>
                                    </div>

                                    {/* Confidence Progress Bar */}
                                    <div className="space-y-1">
                                        <div className="flex justify-between text-xs text-slate-500">
                                            <span>Confidence Score</span>
                                            <span>{result.confidence}</span>
                                        </div>
                                        <div className="h-2 w-full bg-slate-800 rounded-full overflow-hidden">
                                            <motion.div
                                                initial={{ width: 0 }}
                                                animate={{ width: `${(result.confidence_score ?? 0) * 100}%` }}
                                                transition={{ duration: 1, ease: "easeOut" }}
                                                className={`h-full bg-gradient-to-r ${confidenceBarColour(result.confidence_score ?? 0)}`}
                                            />
                                        </div>
                                    </div>
                                </div>

                                {/* Heatmap + LLM */}
                                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                                    <div className="glass p-6 rounded-3xl border-white/5 space-y-4">
                                        <div className="flex justify-between items-center">
                                            <h4 className="font-bold flex items-center gap-2">
                                                <Activity className="w-4 h-4 text-blue-500" /> Region Attention Map
                                            </h4>
                                            <span className="text-[10px] glass px-2 py-0.5 rounded-full text-slate-400">Heatmap</span>
                                        </div>
                                        <div className="rounded-2xl overflow-hidden relative aspect-square glass border-white/10">
                                            {/* eslint-disable-next-line @next/next/no-img-element */}
                                            <img src={result.heatmap_url} alt="Region Attention Heatmap" className="w-full h-full object-cover" />
                                        </div>
                                        <p className="text-[10px] text-slate-500 leading-tight">
                                            Attention circle is centred on the brightest detected region in the scan.
                                        </p>
                                    </div>

                                    <div className="glass p-6 rounded-3xl border-white/5 space-y-4 flex flex-col">
                                        <div className="flex justify-between items-center">
                                            <h4 className="font-bold flex items-center gap-2 text-cyan-400">
                                                <div className="w-2 h-2 rounded-full bg-cyan-400 animate-pulse" />
                                                Medical Assistant
                                            </h4>
                                            <MessageSquare className="w-4 h-4 text-slate-500" />
                                        </div>
                                        <div className="flex-1 bg-slate-900/50 rounded-2xl p-4 text-sm text-slate-300 overflow-y-auto max-h-[300px] prose prose-invert prose-sm">
                                            {result.guidance
                                                ? result.guidance.split('\n').map((line, i) => (
                                                    <p key={i} className="mb-2 leading-relaxed">{line}</p>
                                                ))
                                                : <p className="text-slate-500 italic">No guidance available for this scan.</p>
                                            }
                                        </div>
                                        <div className="flex gap-2">
                                            <button
                                                onClick={() => saveJson(result)}
                                                className="flex-1 py-3 glass hover:bg-white/5 rounded-xl text-xs font-bold flex items-center justify-center gap-2 transition-all"
                                            >
                                                <FileText className="w-3 h-3" /> Save JSON
                                            </button>
                                            <button
                                                onClick={() => downloadReport(result, apiUrl)}
                                                className="flex-1 py-3 bg-blue-600 hover:bg-blue-500 rounded-xl text-xs font-bold flex items-center justify-center gap-2 transition-all"
                                            >
                                                <Download className="w-3 h-3" /> PDF Report
                                            </button>
                                        </div>
                                    </div>
                                </div>

                                {/* ── Groq LLaMA Follow-up Chat ── */}
                                <div className="glass rounded-3xl border-white/5 overflow-hidden">
                                    {/* Chat Header */}
                                    <div className="flex items-center gap-3 px-5 py-4 border-b border-white/5">
                                        <div className="w-2 h-2 rounded-full bg-cyan-400 animate-pulse" />
                                        <span className="text-sm font-bold text-cyan-400">GoGenix-Med Chat</span>
                                        <span className="ml-auto text-[10px] text-slate-500 glass px-2 py-0.5 rounded-full">LLaMA 3.3 · 70B</span>
                                    </div>

                                    {/* Message History */}
                                    <div className="h-56 overflow-y-auto p-4 space-y-3 flex flex-col">
                                        {chatMessages.length === 0 && (
                                            <p className="text-slate-500 text-xs text-center mt-auto italic">
                                                Ask anything about your diagnosis result&hellip;
                                            </p>
                                        )}
                                        {chatMessages.map((msg, i) => (
                                            <div
                                                key={i}
                                                className={`flex gap-2 items-end ${msg.role === "user" ? "flex-row-reverse" : ""}`}
                                            >
                                                {/* Avatar */}
                                                <div className={`w-6 h-6 rounded-full flex-shrink-0 flex items-center justify-center text-[10px] font-bold ${msg.role === "user" ? "bg-blue-600" : "bg-cyan-700"
                                                    }`}>
                                                    {msg.role === "user" ? "U" : "AI"}
                                                </div>
                                                {/* Bubble */}
                                                <div className={`max-w-[75%] rounded-2xl px-3 py-2 text-xs leading-relaxed whitespace-pre-wrap ${msg.role === "user"
                                                        ? "bg-blue-600/30 text-slate-200 rounded-br-none"
                                                        : "bg-slate-800/70 text-slate-300 rounded-bl-none"
                                                    }`}>
                                                    {msg.content}
                                                </div>
                                            </div>
                                        ))}

                                        {/* Typing indicator */}
                                        {isChatLoading && (
                                            <div className="flex gap-2 items-end">
                                                <div className="w-6 h-6 rounded-full bg-cyan-700 flex-shrink-0 flex items-center justify-center text-[10px] font-bold">AI</div>
                                                <div className="bg-slate-800/70 rounded-2xl rounded-bl-none px-4 py-3 flex gap-1 items-center">
                                                    {[0, 0.15, 0.3].map((delay, i) => (
                                                        <motion.div
                                                            key={i}
                                                            className="w-1.5 h-1.5 rounded-full bg-cyan-400"
                                                            animate={{ y: [0, -5, 0] }}
                                                            transition={{ repeat: Infinity, duration: 0.6, delay }}
                                                        />
                                                    ))}
                                                </div>
                                            </div>
                                        )}
                                        <div ref={chatEndRef} />
                                    </div>

                                    {/* Input Row */}
                                    <div className="flex gap-3 items-center px-4 py-3 border-t border-white/5">
                                        <input
                                            type="text"
                                            value={chatInput}
                                            onChange={e => setChatInput(e.target.value)}
                                            onKeyDown={e => e.key === "Enter" && sendChatMessage()}
                                            placeholder="Ask a follow-up question…"
                                            disabled={isChatLoading}
                                            className="flex-1 bg-transparent border-none outline-none text-sm placeholder:text-slate-600 disabled:opacity-40"
                                        />
                                        <button
                                            onClick={sendChatMessage}
                                            disabled={isChatLoading || !chatInput.trim()}
                                            className="p-2 hover:bg-blue-600 rounded-lg transition-all text-blue-500 hover:text-white disabled:opacity-30"
                                        >
                                            <Send className="w-4 h-4" />
                                        </button>
                                    </div>
                                </div>

                            </motion.div>
                        ) : (
                            <motion.div
                                key="empty"
                                initial={{ opacity: 0 }}
                                animate={{ opacity: 1 }}
                                className="h-[600px] glass rounded-3xl border-white/5 flex flex-col items-center justify-center p-12 text-center"
                            >
                                <div className="w-24 h-24 bg-slate-800 rounded-full flex items-center justify-center text-slate-600 mb-6">
                                    <Upload className="w-12 h-12" />
                                </div>
                                <h3 className="text-2xl font-bold mb-2 text-slate-300">Awaiting Data</h3>
                                <p className="text-slate-500 max-w-sm">
                                    Upload a medical scan in the left panel to begin the AI diagnostic sequence.
                                </p>
                            </motion.div>
                        )}
                    </AnimatePresence>
                </div>
            </div>
        </div>
    );
}
