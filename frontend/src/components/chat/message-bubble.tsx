"use client"

import { motion } from "framer-motion"
import { Card } from "@/components/ui/card"
import { ThinkingAccordion } from "./thinking-accordion"
import ReactMarkdown from "react-markdown"
import remarkGfm from "remark-gfm"
import { Prism as SyntaxHighlighter } from "react-syntax-highlighter"
import { vscDarkPlus } from "react-syntax-highlighter/dist/esm/styles/prism"
import { Message } from "@/lib/store"
import { ToolStatus } from "./tool-status"
import { User, Bot } from "lucide-react"

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
      initial={{ opacity: 0, y: 20, scale: 0.95 }}
      animate={{ opacity: 1, y: 0, scale: 1 }}
      transition={{ duration: 0.3, ease: "easeOut" }}
      className={`flex w-full ${isUser ? "justify-end" : "justify-start"} mb-6 group`}
    >
      <div className={`flex max-w-[85%] md:max-w-[75%] gap-3 ${isUser ? "flex-row-reverse" : "flex-row"}`}>
        
        {/* Avatar */}
        <div className={`flex-shrink-0 w-8 h-8 rounded-full flex items-center justify-center shadow-sm ${
          isUser 
            ? "bg-primary text-primary-foreground" 
            : "bg-gradient-to-br from-indigo-500 to-purple-600 text-white"
        }`}>
          {isUser ? <User className="h-5 w-5" /> : <Bot className="h-5 w-5" />}
        </div>

        <div className="flex flex-col gap-1 min-w-0">
          {/* Name */}
          <span className={`text-xs text-muted-foreground ${isUser ? "text-right" : "text-left"}`}>
            {isUser ? "You" : "Personal AI"}
          </span>

          {/* Bubble */}
          <Card className={`p-4 shadow-md border-0 ${
            isUser 
              ? "bg-primary text-primary-foreground rounded-2xl rounded-tr-sm" 
              : "bg-card/80 backdrop-blur-sm border border-border/50 rounded-2xl rounded-tl-sm"
          }`}>
            
            {/* Thinking Process */}
            {message.thinking && (
              <div className="mb-4">
                <ThinkingAccordion thinking={message.thinking} />
              </div>
            )}

            {/* Tool Calls Visualization */}
            {message.tool_calls && message.tool_calls.length > 0 && (
              <div className="mb-4 space-y-2">
                {message.tool_calls.map((tool, idx) => (
                  <ToolStatus 
                    key={idx}
                    toolName={tool.function.name}
                    args={tool.function.arguments}
                    status="completed" // Or derive from state if we track it
                  />
                ))}
              </div>
            )}

            {/* Content */}
            <div className={`prose max-w-none break-words leading-relaxed ${
              isUser 
                ? "prose-invert text-primary-foreground" 
                : "dark:prose-invert prose-neutral"
            }`}>
              <ReactMarkdown
                remarkPlugins={[remarkGfm]}
                components={{
                  code({node, inline, className, children, ...props}: any) {
                    const match = /language-(\w+)/.exec(className || '')
                    return !inline && match ? (
                      <div className="rounded-md overflow-hidden my-4 shadow-lg border border-border/50">
                        <div className="bg-muted/50 px-4 py-1 text-xs text-muted-foreground border-b border-border/50 flex justify-between">
                          <span>{match[1]}</span>
                        </div>
                        <SyntaxHighlighter
                          {...props}
                          style={vscDarkPlus}
                          language={match[1]}
                          PreTag="div"
                          customStyle={{ margin: 0, borderRadius: 0 }}
                        >
                          {String(children).replace(/\n$/, '')}
                        </SyntaxHighlighter>
                      </div>
                    ) : (
                      <code {...props} className={`${className} bg-muted/50 px-1.5 py-0.5 rounded text-sm font-mono`}>
                        {children}
                      </code>
                    )
                  },
                  a: ({node, ...props}) => (
                    <a {...props} className="text-blue-500 hover:underline font-medium" target="_blank" rel="noopener noreferrer" />
                  ),
                  p: ({node, ...props}) => (
                    <p {...props} className="mb-4 last:mb-0" />
                  ),
                  ul: ({node, ...props}) => (
                    <ul {...props} className="list-disc pl-4 mb-4 space-y-1" />
                  ),
                  ol: ({node, ...props}) => (
                    <ol {...props} className="list-decimal pl-4 mb-4 space-y-1" />
                  )
                }}
              >
                {message.content}
              </ReactMarkdown>
            </div>
          </Card>
        </div>
      </div>
    </motion.div>
  )
}
