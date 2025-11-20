"use client"

import { useState, useRef, useEffect } from "react"
import { ScrollArea } from "@/components/ui/scroll-area"
import { Send, StopCircle, ChevronRight } from "lucide-react"
import { useAppStore, Message } from "@/lib/store"
import { MessageBubble } from "./message-bubble"
import { motion, AnimatePresence } from "framer-motion"
import { getConfig } from "@/lib/api"

export function ChatInterface() {
  const { messages, setMessages, addMessage, updateLastMessage, currentSessionId, reloadChatTrigger, clearChatTrigger } = useAppStore()
  const [input, setInput] = useState("")
  const [isLoading, setIsLoading] = useState(false)
  const [modelName, setModelName] = useState("LOADING...")
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const scrollAreaRef = useRef<HTMLDivElement>(null)
  const abortControllerRef = useRef<AbortController | null>(null)

  useEffect(() => {
    getConfig().then(config => {
      if (config.model) setModelName(config.model.toUpperCase())
    }).catch(err => console.error("Failed to load config", err))
  }, [])

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" })
  }

  useEffect(() => {
    scrollToBottom()
  }, [messages])

  // Clear chat
  useEffect(() => {
    if (clearChatTrigger > 0) {
      setMessages([])
    }
  }, [clearChatTrigger])

  // Reload chat
  useEffect(() => {
    if (reloadChatTrigger > 0) {
      fetch("http://localhost:8000/api/history")
        .then(res => res.json())
        .then(data => {
          setMessages(data)
        })
        .catch(err => console.error("Failed to load history", err))
    }
  }, [reloadChatTrigger])

  const handleStop = () => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort()
      abortControllerRef.current = null
      setIsLoading(false)
    }
  }

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!input.trim() || isLoading) return

    const userMsg: Message = { role: "user", content: input }
    addMessage(userMsg)
    setInput("")
    setIsLoading(true)

    abortControllerRef.current = new AbortController()

    try {
      const response = await fetch("http://localhost:8000/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ 
          message: userMsg.content,
          session_id: currentSessionId 
        }),
        signal: abortControllerRef.current.signal
      })

      if (!response.ok) throw new Error("Network response was not ok")

      const reader = response.body?.getReader()
      if (!reader) return

      const decoder = new TextDecoder()
      let buffer = "" // Buffer for incomplete lines
      
      // Create initial assistant message
      const assistantMsg: Message = { role: "assistant", content: "", thinking: "", tool_calls: [] }
      addMessage(assistantMsg)

      while (true) {
        const { done, value } = await reader.read()
        if (done) break

        buffer += decoder.decode(value, { stream: true })
        const lines = buffer.split("\n")
        
        // Keep last incomplete line in buffer
        buffer = lines.pop() || ""

        for (const line of lines) {
          if (!line.trim() || !line.startsWith("data: ")) continue
          
          const data = line.slice(6)
          if (data === "[DONE]") break

          try {
            const event = JSON.parse(data)
            
            updateLastMessage((prev) => {
              const newMsg = { ...prev }
              
              if (event.type === "token") {
                newMsg.content += event.content
              } else if (event.type === "thinking") {
                newMsg.thinking = (newMsg.thinking || "") + event.content
              } else if (event.type === "tool_calls") {
                // Initialize tool calls
                newMsg.tool_calls = event.data.map((tc: any) => ({
                  ...tc,
                  status: "pending"
                }))
              } else if (event.type === "tool_start") {
                // Mark tool as running
                if (newMsg.tool_calls) {
                  newMsg.tool_calls = newMsg.tool_calls.map(tc => 
                    tc.function.name === event.name ? { ...tc, status: "running" } : tc
                  )
                }
              } else if (event.type === "tool_result") {
                // Mark tool as completed
                if (newMsg.tool_calls) {
                  newMsg.tool_calls = newMsg.tool_calls.map(tc => 
                    tc.function.name === event.name ? { ...tc, status: "completed", result: event.result } : tc
                  )
                }
              }
              
              return newMsg
            })
            
          } catch (e) {
            console.error("Error parsing SSE event", e)
          }
        }
      }
    } catch (error: any) {
      if (error.name !== 'AbortError') {
        console.error("Error:", error)
        addMessage({ role: "assistant", content: "Sorry, something went wrong. Please try again." })
      }
    } finally {
      setIsLoading(false)
      abortControllerRef.current = null
    }
  }

  return (
    <div className="flex flex-col h-full max-w-3xl mx-auto pt-12 pb-6 px-8 w-full">
      {/* Header */}
      <header className="mb-12 border-b border-stone-800 pb-4">
        <h1 className="text-3xl font-serif text-stone-200">Discussion</h1>
        <div className="flex gap-2 mt-2">
           <span className="px-2 py-0.5 bg-stone-800 text-stone-400 text-[10px] rounded font-mono uppercase">{modelName}</span>
           <span className="px-2 py-0.5 bg-stone-800 text-stone-400 text-[10px] rounded font-mono uppercase">LOCAL</span>
           <span className="px-2 py-0.5 bg-stone-800 text-stone-400 text-[10px] rounded font-mono uppercase">SECURE</span>
        </div>
      </header>

      {/* Messages Area */}
      <div className="flex-1 overflow-y-auto space-y-8 pr-4 scrollbar-hide" ref={scrollAreaRef}>
        <AnimatePresence initial={false}>
          {messages.length === 0 ? (
            <motion.div 
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              className="flex flex-col items-center justify-center h-[40vh] text-center space-y-6"
            >
              <div className="w-16 h-16 border border-stone-700 rounded-full flex items-center justify-center mb-4">
                <div className="w-2 h-2 bg-[#d97706] rounded-full animate-pulse" />
              </div>
              <h1 className="text-2xl font-serif tracking-tight text-stone-400">
                Ready
              </h1>
              <p className="text-stone-600 max-w-md text-sm font-mono">
                How can I help you today?
              </p>
            </motion.div>
          ) : (
            messages.map((msg, i) => (
              <MessageBubble 
                key={i} 
                message={msg} 
                isLast={i === messages.length - 1} 
              />
            ))
          )}
        </AnimatePresence>
        <div ref={messagesEndRef} className="h-4" />
      </div>

      {/* Input Area */}
      <div className="mt-4 relative">
        <form onSubmit={handleSubmit}>
          <input 
            type="text" 
            value={input}
            onChange={e => setInput(e.target.value)}
            placeholder="Type a message..." 
            disabled={isLoading}
            className="w-full bg-[#0a0a0a] border border-stone-800 text-stone-300 p-4 pl-6 rounded-xl focus:outline-none focus:border-[#d97706] transition-colors font-mono text-sm"
          />
          <button 
            type="submit"
            disabled={!input.trim() || isLoading}
            className="absolute right-3 top-3 p-1.5 bg-[#d97706] text-black rounded-lg hover:bg-orange-500 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {isLoading ? (
              <StopCircle size={20} onClick={(e) => { e.preventDefault(); handleStop(); }} />
            ) : (
              <ChevronRight size={20} />
            )}
          </button>
        </form>
      </div>
    </div>
  )
}
