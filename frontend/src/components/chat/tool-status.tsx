"use client"

import { motion } from "framer-motion"
import { Loader2, CheckCircle2, Globe, FileText, Database, Newspaper } from "lucide-react"

interface ToolStatusProps {
  toolName: string
  args: any
  status: "running" | "completed" | "error"
  result?: string
}

export function ToolStatus({ toolName, args, status, result }: ToolStatusProps) {
  const getIcon = () => {
    if (toolName.includes("web_search")) return <Globe className="h-4 w-4" />
    if (toolName.includes("fetch")) return <FileText className="h-4 w-4" />
    if (toolName.includes("vector")) return <Database className="h-4 w-4" />
    if (toolName.includes("news")) return <Newspaper className="h-4 w-4" />
    return <Loader2 className="h-4 w-4" />
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
      initial={{ opacity: 0, y: 5 }}
      animate={{ opacity: 1, y: 0 }}
      className="flex items-center gap-3 text-xs text-muted-foreground bg-card/50 p-2.5 rounded-lg my-2 border border-border/40 shadow-sm max-w-md"
    >
      <div className={`p-1.5 rounded-md flex-shrink-0 ${
        status === "running" 
          ? "bg-blue-500/10 text-blue-500" 
          : status === "error"
            ? "bg-red-500/10 text-red-500"
            : "bg-green-500/10 text-green-500"
      }`}>
        {status === "running" ? (
          <Loader2 className="h-3.5 w-3.5 animate-spin" />
        ) : (
          getIcon()
        )}
      </div>
      
      <div className="flex-1 min-w-0 flex flex-col">
        <div className="flex items-center gap-2 font-medium text-foreground/90">
          <span className="capitalize">{toolName.replace(/_/g, " ")}</span>
          {status === "completed" && <CheckCircle2 className="h-3 w-3 text-green-500/70" />}
        </div>
        <div className="truncate opacity-70 font-mono text-[10px] mt-0.5">
          {formatArgs(args)}
        </div>
      </div>
    </motion.div>
  )
}
