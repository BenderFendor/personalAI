"use client"

import React from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { ChevronRight, FileText, Plus, Folder } from 'lucide-react'
import { Session } from '@/lib/api'
import { cn } from "@/lib/utils"
import { formatDistanceToNow } from 'date-fns'

interface FolderStackProps {
  sessions: Session[]
  selectedSessionId: string | null
  onSelectSession: (sessionId: string) => void
  onNewChat: () => void
  activeCabinetName: string
}

export function FolderStack({ sessions, selectedSessionId, onSelectSession, onNewChat, activeCabinetName }: FolderStackProps) {
  // Dynamic height calculation
  const cardHeight = 65; // Height of the visible part when stacked
  const totalHeight = Math.max(sessions.length * cardHeight + 400, 800); // Buffer

  return (
    <div className="w-80 h-full bg-[#0c0c0c] relative border-r border-[#333] overflow-hidden flex flex-col shrink-0 z-10 font-sans">
      {/* Header */}
      <div className="p-6 border-b border-[#333] flex justify-between items-end bg-[#0c0c0c] z-30 sticky top-0">
        <div>
          <h2 className="text-[10px] uppercase tracking-[0.2em] text-[#666] font-mono mb-2">Archive Context</h2>
          <div className="text-[#e5e5e5] font-[family-name:var(--font-instrument-serif)] text-2xl leading-none italic">
            {activeCabinetName}
          </div>
        </div>
        <button 
          onClick={onNewChat}
          className="p-2 hover:bg-[#222] rounded-full text-[#666] hover:text-[#d97706] transition-all duration-300 border border-transparent hover:border-[#333]"
          title="New Discussion"
        >
          <Plus size={20} />
        </button>
      </div>
      
      {/* Scrollable Stack Area */}
      <div className="flex-1 relative overflow-y-auto custom-scrollbar">
        <div className="relative w-full px-4 pt-6 pb-32" style={{ height: `${totalHeight}px` }}>
          <AnimatePresence mode="popLayout">
            {sessions.length === 0 ? (
              <EmptyState onNewChat={onNewChat} />
            ) : (
              sessions.map((session, index) => (
                <FolderCard 
                  key={session.id} 
                  session={session} 
                  index={index} 
                  total={sessions.length}
                  isSelected={selectedSessionId === session.id}
                  onClick={() => onSelectSession(session.id)}
                />
              ))
            )}
          </AnimatePresence>
        </div>
      </div>
      
      {/* Bottom Gradient Fade */}
      <div className="absolute bottom-0 left-0 w-full h-32 bg-gradient-to-t from-[#0c0c0c] via-[#0c0c0c]/80 to-transparent pointer-events-none z-20" />
    </div>
  )
}

function EmptyState({ onNewChat }: { onNewChat: () => void }) {
  return (
    <motion.div 
      initial={{ opacity: 0, y: 20 }} 
      animate={{ opacity: 1, y: 0 }}
      className="flex flex-col items-center justify-center h-64 text-[#444] gap-4 mt-10"
    >
      <div className="w-16 h-16 rounded-2xl bg-[#1a1a1a] flex items-center justify-center border border-[#333]">
        <Folder className="opacity-20" size={32} />
      </div>
      <div className="text-center">
        <p className="font-[family-name:var(--font-instrument-serif)] text-xl text-[#666] italic">Empty Archive</p>
        <button onClick={onNewChat} className="text-[#d97706] hover:text-[#f59e0b] text-xs mt-2 font-mono uppercase tracking-wider hover:underline">
          Initialize Protocol
        </button>
      </div>
    </motion.div>
  )
}

interface FolderCardProps {
  session: Session
  index: number
  total: number
  isSelected: boolean
  onClick: () => void
}

function FolderCard({ session, index, total, isSelected, onClick }: FolderCardProps) {
  return (
    <motion.div
      onClick={onClick}
      layoutId={`folder-${session.id}`}
      initial={{ y: 50, opacity: 0 }}
      animate={{ 
        y: index * 65, // Consistent stacking spacing
        zIndex: isSelected ? 50 : index, // Selected pops up but respects stack order mostly
        scale: isSelected ? 1.05 : 1,
        x: isSelected ? 0 : 0,
        opacity: 1,
      }}
      whileHover={{ 
        x: isSelected ? 0 : 5, // Slide out slightly on hover
        transition: { duration: 0.2 }
      }}
      className={cn(
        "absolute left-4 right-4 h-40 cursor-pointer transition-all duration-500 ease-out group",
        "bg-transparent"
      )}
    >
      {/* Drop Shadow Wrapper for the composite shape */}
      <div className={cn(
        "relative w-full h-full transition-all duration-300",
        isSelected ? "drop-shadow-[0_20px_50px_rgba(217,119,6,0.3)]" : "drop-shadow-2xl"
      )}>
        
        {/* TAB */}
        <div className={cn(
          "absolute top-0 left-0 w-24 h-8 z-20 flex items-center justify-center",
          "rounded-t-xl border-t border-l border-r",
          "transition-colors duration-300",
          isSelected 
            ? "bg-[#d97706] border-[#f59e0b]" 
            : "bg-[#1a1a1a] border-[#333] group-hover:bg-[#222]"
        )}>
          <span className={cn(
            "text-[10px] font-mono font-bold",
            isSelected ? "text-[#0c0c0c]/50" : "text-[#333]"
          )}>
            {(index + 1).toString().padStart(2, '0')}
          </span>
        </div>

        {/* BODY */}
        <div className={cn(
          "absolute top-7 w-full h-[calc(100%)] z-10",
          "rounded-b-xl rounded-tr-xl rounded-tl-none border",
          "transition-colors duration-300",
          isSelected 
            ? "bg-[#d97706] border-[#f59e0b]" 
            : "bg-[#1a1a1a] border-[#333] group-hover:bg-[#222]"
        )}>
          <div className="p-5 flex flex-col h-full justify-between relative">
            {/* Content */}
            <div className="space-y-2">
              <div className="flex justify-between items-start gap-4">
                <h3 className={cn(
                  "font-[family-name:var(--font-instrument-serif)] text-xl leading-tight transition-colors duration-300",
                  isSelected ? "text-[#0c0c0c]" : "text-[#e5e5e5]"
                )}>
                  {session.title || "Untitled Entry"}
                </h3>
                {isSelected && <div className="w-2 h-2 rounded-full bg-[#0c0c0c] shrink-0 mt-2 animate-pulse" />}
              </div>
              
              <p className={cn(
                "text-xs font-mono uppercase tracking-wider",
                isSelected ? "text-[#0c0c0c]/60" : "text-[#666]"
              )}>
                {session.started_at ? formatDistanceToNow(new Date(session.started_at), { addSuffix: true }) : 'Unknown'}
              </p>
            </div>

            {/* Footer / Action */}
            {isSelected && (
              <motion.div 
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ delay: 0.1 }}
                className="flex items-center gap-2 text-[#0c0c0c] font-mono text-[10px] uppercase tracking-[0.15em] border-t border-[#0c0c0c]/10 pt-3 mt-2"
              >
                <span>Active Session</span>
                <ChevronRight size={12} />
              </motion.div>
            )}
            
            {/* Background Texture/Noise for tactile feel */}
            <div className="absolute inset-0 opacity-[0.03] pointer-events-none bg-noise mix-blend-overlay" />
          </div>
        </div>
      </div>
    </motion.div>
  )
}
