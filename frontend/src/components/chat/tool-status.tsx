"use client"

import { motion } from "framer-motion"
import { Loader2, CheckCircle2, Globe, FileText, Database, Newspaper, Terminal } from "lucide-react"
import { cn } from "@/lib/utils"

interface ToolStatusProps {
  toolName: string
  args: any
  status: "running" | "completed" | "error"
  result?: string
}

export function ToolStatus({ toolName, args, status, result }: ToolStatusProps) {
  const getIcon = () => {
    if (toolName.includes("web_search")) return <Globe className="h-3 w-3" />
    if (toolName.includes("fetch")) return <FileText className="h-3 w-3" />
    if (toolName.includes("vector")) return <Database className="h-3 w-3" />
    if (toolName.includes("news")) return <Newspaper className="h-3 w-3" />
    return <Terminal className="h-3 w-3" />
  }

  // Format args for display
  const formatArgs = (args: any) => {
    if (!args) return ""
    try {
      // If it's a string, try to parse it just in case, otherwise return as is
      if (typeof args === 'string') return args
      
      // If it's an object, format nicely
      const entries = Object.entries(args)
      if (entries.length === 0) return ""
      
      return entries.map(([key, value]) => {
        if (key === 'query' || key === 'keywords') return `"${value}"`
        if (key === 'url') return new URL(value as string).hostname
        return `${key}: ${value}`
      }).join(", ")
    } catch (e) {
      return JSON.stringify(args)
    }
  }

  return (
    <motion.div
      initial={{ opacity: 0, x: -10 }}
      animate={{ opacity: 1, x: 0 }}
      className="group"
    >
      <div className="flex items-center gap-2 text-stone-400 text-sm">
         <div className={cn(
           "opacity-50 font-mono text-xs",
           status === "running" ? "text-[#d97706]" : "text-stone-600"
         )}>
           {status === "running" ? "RUNNING" : "COMPLETE"}
         </div>
         <span className="font-medium text-stone-300 text-xs uppercase tracking-wider flex items-center gap-2">
            {getIcon()}
            {toolName.replace(/_/g, " ")}
         </span>
      </div>
      <p className="text-stone-600 text-xs pl-8 mt-1 border-l border-stone-800 ml-1.5 font-mono truncate max-w-md">
        {formatArgs(args)}
      </p>
    </motion.div>
  )
}
