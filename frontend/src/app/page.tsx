"use client";

import { motion } from "framer-motion";
import { Brain, Activity, Shield, ChevronRight, Upload, BarChart3, MessageSquare } from "lucide-react";
import Link from "next/link";

export default function LandingPage() {
  return (
    <main className="flex flex-col items-center justify-between p-4 md:p-24 relative overflow-hidden">
      {/* Hero Section */}
      <section className="max-w-7xl w-full flex flex-col md:flex-row items-center gap-12 pt-10 md:pt-20">
        <motion.div
          initial={{ opacity: 0, x: -50 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.8 }}
          className="flex-1 space-y-8"
        >
          <div className="inline-flex items-center px-4 py-2 rounded-full glass text-blue-400 text-sm font-medium">
            <Activity className="w-4 h-4 mr-2" />
            Empowering Healthcare with GoGenix-AI Precision
          </div>

          <h1 className="text-5xl md:text-7xl font-bold leading-tight">
            Next-Gen <span className="text-gradient">Disease Diagnosis</span> with GoGenix-AI
          </h1>

          <p className="text-xl text-slate-400 max-w-xl">
            High-accuracy neural networks for MRI, CT, and X-ray analysis.
            Real-time insights powered by advanced ensemble models.
          </p>

          <div className="flex flex-wrap gap-4">
            <Link href="/dashboard" className="px-8 py-4 bg-blue-600 hover:bg-blue-500 rounded-xl font-semibold transition-all flex items-center shadow-lg shadow-blue-500/20">
              Launch Diagnostic Console <ChevronRight className="ml-2 w-5 h-5" />
            </Link>
            <button className="px-8 py-4 glass hover:bg-white/5 rounded-xl font-semibold transition-all">
              View Model Metrics
            </button>
          </div>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, scale: 0.8 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ duration: 1 }}
          className="flex-1 relative"
        >
          <div className="relative w-full aspect-square glass rounded-3xl overflow-hidden shadow-2xl">
            <div className="absolute inset-0 bg-blue-500/10 flex items-center justify-center">
              <Brain className="w-48 h-48 text-blue-500/30 animate-pulse" />
            </div>
            {/* Simulated UI overlay */}
            <div className="absolute bottom-6 right-6 left-6 glass p-6 rounded-2xl border-white/5">
              <div className="flex justify-between items-center mb-4">
                <span className="text-sm font-medium uppercase tracking-wider text-blue-400">Analysis Progress</span>
                <span className="text-sm font-bold">98% Match</span>
              </div>
              <div className="h-2 w-full bg-slate-800 rounded-full overflow-hidden">
                <motion.div
                  initial={{ width: 0 }}
                  animate={{ width: "98%" }}
                  transition={{ duration: 2, delay: 1 }}
                  className="h-full bg-gradient-to-right from-blue-600 to-cyan-400"
                ></motion.div>
              </div>
            </div>
          </div>
        </motion.div>
      </section>

      {/* Disease Categories */}
      <section className="max-w-7xl w-full py-24 grid grid-cols-1 md:grid-cols-4 gap-6">
        {[
          { icon: Brain, title: "Brain Tumor", desc: "MRI Segmentation & Analysis" },
          { icon: Activity, title: "Lung Health", desc: "Pneumonia & CT Screening" },
          { icon: Shield, title: "Cancer Detect", desc: "Early Malignancy Identification" },
          { icon: BarChart3, title: "Renal Stones", desc: "High-Accuracy Calculi Detection" }
        ].map((item, idx) => (
          <motion.div
            key={idx}
            whileHover={{ y: -10 }}
            className="p-8 glass rounded-3xl space-y-4 hover:border-blue-500/30 transition-all border-white/5"
          >
            <div className="w-12 h-12 bg-blue-500/10 rounded-xl flex items-center justify-center text-blue-400">
              <item.icon className="w-6 h-6" />
            </div>
            <h3 className="text-xl font-bold">{item.title}</h3>
            <p className="text-slate-400 text-sm">{item.desc}</p>
          </motion.div>
        ))}
      </section>
    </main>
  );
}
