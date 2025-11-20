"use client"

import { useState, useRef, useEffect } from "react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { ScrollArea } from "@/components/ui/scroll-area"
import { Send, Sparkles, StopCircle } from "lucide-react"
import { useAppStore, Message } from "@/lib/store"
import { MessageBubble } from "./message-bubble"
import { motion, AnimatePresence } from "framer-motion"

export function ChatInterface() {
  const { messages, setMessages, addMessage, updateLastMessage, currentSessionId, reloadChatTrigger, clearChatTrigger } = useAppStore()
  const [input, setInput] = useState("")
  const [isLoading, setIsLoading] = useState(false)
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const scrollAreaRef = useRef<HTMLDivElement>(null)
  const abortControllerRef = useRef<AbortController | null>(null)

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
    <div className="flex flex-col h-full relative bg-gradient-to-b from-background to-muted/20">
      
      {/* Messages Area */}
      <ScrollArea className="flex-1 px-4 md:px-8 py-6" ref={scrollAreaRef}>
        <div className="max-w-4xl mx-auto min-h-[calc(100vh-10rem)]">
          <AnimatePresence initial={false}>
            {messages.length === 0 ? (
              <motion.div 
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                className="flex flex-col items-center justify-center h-[60vh] text-center space-y-6"
              >
                <div className="w-24 h-24 bg-primary/10 rounded-full flex items-center justify-center mb-4 animate-pulse">
                  <Sparkles className="w-12 h-12 text-primary" />
                </div>
                <h1 className="text-4xl font-bold tracking-tight bg-clip-text text-transparent bg-gradient-to-r from-primary to-purple-600">
                  How can I help you today?
                </h1>
                <p className="text-muted-foreground max-w-md text-lg">
                  I can help you with research, coding, analysis, and more. 
                  Just ask!
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
      </ScrollArea>

      {/* Input Area */}
      <div className="p-4 md:p-6 bg-gradient-to-t from-background via-background to-transparent z-10">
        <div className="max-w-4xl mx-auto relative">
          <form onSubmit={handleSubmit} className="relative group">
            <div className="absolute -inset-0.5 bg-gradient-to-r from-primary/50 to-purple-600/50 rounded-xl blur opacity-20 group-hover:opacity-40 transition duration-500"></div>
            <div className="relative flex items-center bg-card rounded-xl border border-border shadow-lg overflow-hidden">
              <Input
                value={input}
                onChange={e => setInput(e.target.value)}
                placeholder="Message Personal AI..."
                disabled={isLoading}
                className="flex-1 border-0 focus-visible:ring-0 focus-visible:ring-offset-0 py-6 px-4 text-base bg-transparent"
              />
              <div className="pr-2 flex items-center gap-2">
                {isLoading ? (
                  <Button 
                    type="button" 
                    onClick={handleStop}
                    variant="ghost" 
                    size="icon"
                    className="h-10 w-10 rounded-lg hover:bg-destructive/10 hover:text-destructive transition-colors"
                  >
                    <StopCircle className="h-5 w-5" />
                  </Button>
                ) : (
                  <Button 
                    type="submit" 
                    disabled={!input.trim()}
                    size="icon"
                    className="h-10 w-10 rounded-lg bg-primary hover:bg-primary/90 transition-all duration-300 shadow-sm"
                  >
                    <Send className="h-5 w-5" />
                  </Button>
                )}
              </div>
            </div>
          </form>
          <div className="text-center mt-2">
            <p className="text-xs text-muted-foreground/60">
              AI can make mistakes. Check important info.
            </p>
          </div>
        </div>
      </div>
    </div>
  )
}
