"use client"

import { useState } from "react"
import { ChevronDown, ChevronRight, BrainCircuit } from "lucide-react"
import { motion, AnimatePresence } from "framer-motion"
import { cn } from "@/lib/utils"

interface ThinkingAccordionProps {
  thinking: string
}

export function ThinkingAccordion({ thinking }: ThinkingAccordionProps) {
  const [isOpen, setIsOpen] = useState(false)

  if (!thinking) return null

  // Show the last part of the thinking process to simulate streaming
  const preview = thinking.length > 180 ? "..." + thinking.slice(-180) : thinking

  return (
    <div className="flex gap-4 mb-6 group">
      {/* Timeline Line */}
      <div className="w-8 flex flex-col items-center mt-1 shrink-0">
         <div className={cn(
           "w-2 h-2 rounded-full transition-colors duration-300",
           isOpen ? "bg-[#d97706]" : "bg-stone-700 group-hover:bg-[#d97706]/50"
         )} />
         <div className="w-0.5 h-full bg-stone-800 my-1 min-h-[20px]" />
      </div>

      <div className="flex-1 min-w-0">
        <button 
          onClick={() => setIsOpen(!isOpen)}
          className="flex items-center gap-2 text-[#d97706] text-xs font-bold uppercase tracking-widest mb-2 hover:text-orange-400 transition-colors w-full text-left"
        >
          <span>Thought Process</span>
          {isOpen ? <ChevronDown size={12} /> : <ChevronRight size={12} />}
        </button>

        <div className="relative">
          <AnimatePresence initial={false} mode="wait">
            {isOpen ? (
              <motion.div
                key="content"
                initial={{ opacity: 0, height: 0 }}
                animate={{ opacity: 1, height: "auto" }}
                exit={{ opacity: 0, height: 0 }}
                transition={{ duration: 0.2 }}
                className="overflow-hidden"
              >
                <div className="text-stone-400 text-xs font-mono whitespace-pre-wrap pl-4 border-l border-stone-800 ml-1">
                  {thinking}
                </div>
              </motion.div>
            ) : (
              <motion.div
                key="preview"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                className="text-stone-600 text-xs font-mono pl-4 border-l border-stone-800 ml-1 line-clamp-2"
              >
                {preview}
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      </div>
    </div>
  )
}
