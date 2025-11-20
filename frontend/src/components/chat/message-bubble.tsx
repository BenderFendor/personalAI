"use client"

import { motion } from "framer-motion"
import { ThinkingAccordion } from "./thinking-accordion"
import ReactMarkdown from "react-markdown"
import remarkGfm from "remark-gfm"
import { Prism as SyntaxHighlighter } from "react-syntax-highlighter"
import { vscDarkPlus } from "react-syntax-highlighter/dist/esm/styles/prism"
import { Message } from "@/lib/store"
import { ToolStatus } from "./tool-status"
import { User, Bot } from "lucide-react"
import { cn } from "@/lib/utils"

interface MessageBubbleProps {
  message: Message
  isLast: boolean
}

export function MessageBubble({ message, isLast }: MessageBubbleProps) {
  const isUser = message.role === "user"
  const isTool = message.role === "tool"

  if (isTool) return null // Tools are handled inside the assistant message or separately

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3, ease: "easeOut" }}
      className={cn(
        "flex w-full mb-8 group",
        isUser ? "justify-end" : "justify-start"
      )}
    >
      <div className={cn(
        "flex max-w-[90%] md:max-w-[85%] gap-4",
        isUser ? "flex-row-reverse" : "flex-row"
      )}>
        
        {/* Avatar / Icon */}
        <div className={cn(
          "flex-shrink-0 w-8 h-8 rounded-lg flex items-center justify-center shadow-sm border",
          isUser 
            ? "bg-stone-800 border-stone-700 text-stone-300" 
            : "bg-[#0f291e] border-[#1a4533] text-emerald-500"
        )}>
          {isUser ? <User className="h-4 w-4" /> : <Bot className="h-4 w-4" />}
        </div>

        <div className="flex flex-col gap-2 min-w-0 w-full">
          {/* Name */}
          <span className={cn(
            "text-[10px] uppercase tracking-widest font-mono opacity-50",
            isUser ? "text-right" : "text-left"
          )}>
            {isUser ? "User" : "Assistant"}
          </span>

          {/* Thinking Process (Only for Assistant) */}
          {!isUser && message.thinking && (
            <ThinkingAccordion thinking={message.thinking} />
          )}

          {/* Tool Calls Visualization */}
          {!isUser && message.tool_calls && message.tool_calls.length > 0 && (
            <div className="mb-4 space-y-2 pl-4 border-l border-stone-800 ml-4">
              {message.tool_calls.map((tool, idx) => (
                <ToolStatus 
                  key={idx}
                  toolName={tool.function.name}
                  args={tool.function.arguments}
                  status="completed" 
                />
              ))}
            </div>
          )}

          {/* Content Bubble */}
          <div className={cn(
            "p-6 shadow-sm border rounded-lg",
            isUser 
              ? "bg-stone-800 border-stone-700 text-stone-200 rounded-tr-none" 
              : "bg-[#1a1a1a] border-stone-800 text-stone-300 rounded-tl-none"
          )}>
            <div className="prose prose-invert max-w-none prose-p:leading-relaxed prose-pre:bg-[#0a0a0a] prose-pre:border prose-pre:border-stone-800">
              <ReactMarkdown
                remarkPlugins={[remarkGfm]}
                components={{
                  code({node, inline, className, children, ...props}: any) {
                    const match = /language-(\w+)/.exec(className || '')
                    return !inline && match ? (
                      <div className="rounded-md overflow-hidden my-4 shadow-lg border border-stone-800">
                        <div className="bg-[#0a0a0a] px-4 py-1 text-xs text-stone-500 border-b border-stone-800 flex justify-between font-mono uppercase">
                          <span>{match[1]}</span>
                        </div>
                        <SyntaxHighlighter
                          {...props}
                          style={vscDarkPlus}
                          language={match[1]}
                          PreTag="div"
                          customStyle={{ margin: 0, borderRadius: 0, background: '#0a0a0a' }}
                        >
                          {String(children).replace(/\n$/, '')}
                        </SyntaxHighlighter>
                      </div>
                    ) : (
                      <code {...props} className={cn(className, "bg-stone-800 px-1.5 py-0.5 rounded text-sm font-mono text-[#d97706]")}>
                        {children}
                      </code>
                    )
                  },
                  a: ({node, ...props}) => (
                    <a {...props} className="text-[#d97706] hover:underline font-medium" target="_blank" rel="noopener noreferrer" />
                  ),
                  p: ({node, ...props}) => (
                    <p {...props} className="mb-4 last:mb-0" />
                  ),
                  ul: ({node, ...props}) => (
                    <ul {...props} className="list-disc pl-4 mb-4 space-y-1 marker:text-stone-600" />
                  ),
                  ol: ({node, ...props}) => (
                    <ol {...props} className="list-decimal pl-4 mb-4 space-y-1 marker:text-stone-600" />
                  ),
                  blockquote: ({node, ...props}) => (
                    <blockquote {...props} className="border-l-2 border-[#d97706] pl-4 italic text-stone-500 my-4" />
                  )
                }}
              >
                {message.content}
              </ReactMarkdown>
            </div>
          </div>
        </div>
      </div>
    </motion.div>
  )
}
